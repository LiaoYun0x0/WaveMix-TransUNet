"""R50-ViT-B/16 image encoder used by WaveMix-TransUNet.

The encoder consists of a pre-activation ResNetV2-50 feature extractor,
patch/position embeddings, and a ViT-B/16 Transformer. Decoder, WaveMix, and
DSM-guidance modules live in :mod:`wavemix_transunet`.
"""

from __future__ import annotations

import copy
import math
from collections import OrderedDict
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.utils import _pair


def np2th(weights: np.ndarray, conv: bool = False) -> torch.Tensor:
    """Convert NumPy weights, optionally from HWIO to OIHW layout."""
    if conv:
        weights = weights.transpose((3, 2, 0, 1))
    return torch.from_numpy(weights)


def get_r50_b16_config(num_classes: int = 6, image_size: int = 256):
    """Return the exact R50-ViT-B/16 configuration used in the paper."""
    return SimpleNamespace(
        patches={"size": (16, 16), "grid": (image_size // 16, image_size // 16)},
        hidden_size=768,
        transformer={
            "mlp_dim": 3072,
            "num_heads": 12,
            "num_layers": 12,
            "attention_dropout_rate": 0.0,
            "dropout_rate": 0.1,
        },
        resnet=SimpleNamespace(num_layers=(3, 4, 9), width_factor=1),
        classifier="seg",
        representation_size=None,
        cnn_out_channels=1024,
        decoder_channels=(256, 128, 64, 16),
        skip_channels=(512, 256, 64, 16),
        n_classes=num_classes,
        n_skip=3,
        activation="softmax",
    )


class StdConv2d(nn.Conv2d):
    """Weight-standardized convolution used by the TransUNet ResNetV2."""

    def forward(self, x):
        variance, mean = torch.var_mean(
            self.weight, dim=(1, 2, 3), keepdim=True, unbiased=False
        )
        weight = (self.weight - mean) / torch.sqrt(variance + 1e-5)
        return F.conv2d(
            x,
            weight,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )


def conv3x3(cin, cout, stride=1):
    return StdConv2d(cin, cout, 3, stride=stride, padding=1, bias=False)


def conv1x1(cin, cout, stride=1):
    return StdConv2d(cin, cout, 1, stride=stride, padding=0, bias=False)


class PreActBottleneck(nn.Module):
    """Pre-activation ResNetV2 bottleneck."""

    def __init__(self, cin, cout=None, cmid=None, stride=1):
        super().__init__()
        cout = cout or cin
        cmid = cmid or cout // 4
        self.gn1 = nn.GroupNorm(32, cmid, eps=1e-6)
        self.conv1 = conv1x1(cin, cmid)
        self.gn2 = nn.GroupNorm(32, cmid, eps=1e-6)
        self.conv2 = conv3x3(cmid, cmid, stride)
        self.gn3 = nn.GroupNorm(32, cout, eps=1e-6)
        self.conv3 = conv1x1(cmid, cout)
        self.relu = nn.ReLU(inplace=True)
        if stride != 1 or cin != cout:
            self.downsample = conv1x1(cin, cout, stride)
            self.gn_proj = nn.GroupNorm(cout, cout)

    def forward(self, x):
        residual = x
        if hasattr(self, "downsample"):
            residual = self.gn_proj(self.downsample(x))
        y = self.relu(self.gn1(self.conv1(x)))
        y = self.relu(self.gn2(self.conv2(y)))
        y = self.gn3(self.conv3(y))
        return self.relu(residual + y)

    def load_from(self, weights, n_block, n_unit):
        prefix = f"{n_block}/{n_unit}"
        with torch.no_grad():
            for name in ("conv1", "conv2", "conv3"):
                getattr(self, name).weight.copy_(
                    np2th(weights[f"{prefix}/{name}/kernel"], conv=True)
                )
            for name in ("gn1", "gn2", "gn3"):
                module = getattr(self, name)
                module.weight.copy_(np2th(weights[f"{prefix}/{name}/scale"]).view(-1))
                module.bias.copy_(np2th(weights[f"{prefix}/{name}/bias"]).view(-1))
            if hasattr(self, "downsample"):
                self.downsample.weight.copy_(
                    np2th(weights[f"{prefix}/conv_proj/kernel"], conv=True)
                )
                self.gn_proj.weight.copy_(
                    np2th(weights[f"{prefix}/gn_proj/scale"]).view(-1)
                )
                self.gn_proj.bias.copy_(
                    np2th(weights[f"{prefix}/gn_proj/bias"]).view(-1)
                )


class ResNetV2(nn.Module):
    """Pre-activation ResNetV2-50 encoder used by hybrid TransUNet."""

    def __init__(self, block_units=(3, 4, 9), width_factor=1):
        super().__init__()
        width = int(64 * width_factor)
        self.width = width
        self.root = nn.Sequential(
            OrderedDict(
                [
                    ("conv", StdConv2d(3, width, 7, stride=2, bias=False, padding=3)),
                    ("gn", nn.GroupNorm(32, width, eps=1e-6)),
                    ("relu", nn.ReLU(inplace=True)),
                ]
            )
        )
        self.body = nn.Sequential(
            OrderedDict(
                [
                    (
                        "block1",
                        self._make_block(width, width * 4, width, block_units[0], 1),
                    ),
                    (
                        "block2",
                        self._make_block(width * 4, width * 8, width * 2, block_units[1], 2),
                    ),
                    (
                        "block3",
                        self._make_block(width * 8, width * 16, width * 4, block_units[2], 2),
                    ),
                ]
            )
        )

    @staticmethod
    def _make_block(cin, cout, cmid, units, stride):
        layers = [("unit1", PreActBottleneck(cin, cout, cmid, stride))]
        layers.extend(
            (f"unit{index}", PreActBottleneck(cout, cout, cmid))
            for index in range(2, units + 1)
        )
        return nn.Sequential(OrderedDict(layers))

    def forward(self, x):
        features = []
        batch, _, input_size, _ = x.shape
        x = self.root(x)
        features.append(x)
        x = F.max_pool2d(x, kernel_size=3, stride=2, padding=0)
        for index in range(len(self.body) - 1):
            x = self.body[index](x)
            expected = int(input_size / 4 / (index + 1))
            if x.shape[2] != expected:
                feature = x.new_zeros((batch, x.shape[1], expected, expected))
                feature[:, :, : x.shape[2], : x.shape[3]] = x
            else:
                feature = x
            features.append(feature)
        x = self.body[-1](x)
        return x, features[::-1]


class TransformerMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.fc1 = nn.Linear(config.hidden_size, config.transformer["mlp_dim"])
        self.fc2 = nn.Linear(config.transformer["mlp_dim"], config.hidden_size)
        self.act_fn = F.gelu
        self.dropout = nn.Dropout(config.transformer["dropout_rate"])
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)

    def forward(self, x):
        x = self.dropout(self.act_fn(self.fc1(x)))
        return self.dropout(self.fc2(x))


class AttentionRGB(nn.Module):
    def __init__(self, config, vis=False):
        super().__init__()
        self.vis = vis
        self.num_attention_heads = config.transformer["num_heads"]
        self.attention_head_size = config.hidden_size // self.num_attention_heads
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)
        self.out = nn.Linear(config.hidden_size, config.hidden_size)
        rate = config.transformer["attention_dropout_rate"]
        self.attn_dropout = nn.Dropout(rate)
        self.proj_dropout = nn.Dropout(rate)
        self.softmax = nn.Softmax(dim=-1)

    def _transpose(self, x):
        shape = x.size()[:-1] + (
            self.num_attention_heads,
            self.attention_head_size,
        )
        return x.view(*shape).permute(0, 2, 1, 3)

    def forward(self, hidden_states):
        query = self._transpose(self.query(hidden_states))
        key = self._transpose(self.key(hidden_states))
        value = self._transpose(self.value(hidden_states))
        scores = torch.matmul(query, key.transpose(-1, -2))
        probabilities = self.softmax(scores / math.sqrt(self.attention_head_size))
        weights = probabilities if self.vis else None
        context = torch.matmul(self.attn_dropout(probabilities), value)
        context = context.permute(0, 2, 1, 3).contiguous()
        context = context.view(*context.size()[:-2], self.all_head_size)
        return self.proj_dropout(self.out(context)), weights


class TransformerBlock(nn.Module):
    def __init__(self, config, vis=False):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.attention_norm = nn.LayerNorm(config.hidden_size, eps=1e-6)
        self.ffn_norm = nn.LayerNorm(config.hidden_size, eps=1e-6)
        self.ffn = TransformerMLP(config)
        self.attn = AttentionRGB(config, vis)

    def forward(self, x):
        attention, weights = self.attn(self.attention_norm(x))
        x = x + attention
        x = x + self.ffn(self.ffn_norm(x))
        return x, weights

    def load_from(self, weights, n_block):
        root = f"Transformer/encoderblock_{n_block}"
        attention_names = {
            "query": "MultiHeadDotProductAttention_1/query",
            "key": "MultiHeadDotProductAttention_1/key",
            "value": "MultiHeadDotProductAttention_1/value",
            "out": "MultiHeadDotProductAttention_1/out",
        }
        with torch.no_grad():
            for module_name, weight_name in attention_names.items():
                module = getattr(self.attn, module_name)
                module.weight.copy_(
                    np2th(weights[f"{root}/{weight_name}/kernel"])
                    .view(self.hidden_size, self.hidden_size)
                    .t()
                )
                module.bias.copy_(
                    np2th(weights[f"{root}/{weight_name}/bias"]).view(-1)
                )
            # This follows the training code used for the reported model: the
            # ViT attention/norm weights are initialized from NPZ while FFN
            # weights retain their PyTorch initialization.
            self.attention_norm.weight.copy_(
                np2th(weights[f"{root}/LayerNorm_0/scale"])
            )
            self.attention_norm.bias.copy_(
                np2th(weights[f"{root}/LayerNorm_0/bias"])
            )
            self.ffn_norm.weight.copy_(np2th(weights[f"{root}/LayerNorm_2/scale"]))
            self.ffn_norm.bias.copy_(np2th(weights[f"{root}/LayerNorm_2/bias"]))


class EncoderRGB(nn.Module):
    def __init__(self, config, vis=False):
        super().__init__()
        self.vis = vis
        block = TransformerBlock(config, vis)
        self.layer = nn.ModuleList(
            [copy.deepcopy(block) for _ in range(config.transformer["num_layers"])]
        )
        self.encoder_norm = nn.LayerNorm(config.hidden_size, eps=1e-6)

    def forward(self, hidden_states):
        attention_weights = []
        for block in self.layer:
            hidden_states, weights = block(hidden_states)
            if self.vis:
                attention_weights.append(weights)
        return self.encoder_norm(hidden_states), attention_weights


class SharedEmbeddings(nn.Module):
    """R50 feature pyramid and ViT patch/position embeddings."""

    def __init__(self, config, img_size=256, in_channels=3):
        super().__init__()
        self.config = config
        img_size = _pair(img_size)
        grid_size = config.patches.get("grid")
        if grid_size is not None:
            patch_size = (
                img_size[0] // 16 // grid_size[0],
                img_size[1] // 16 // grid_size[1],
            )
            patch_size_real = (patch_size[0] * 16, patch_size[1] * 16)
            n_patches = (img_size[0] // patch_size_real[0]) * (
                img_size[1] // patch_size_real[1]
            )
            self.hybrid_model = ResNetV2(
                block_units=config.resnet.num_layers,
                width_factor=config.resnet.width_factor,
            )
            in_channels = self.hybrid_model.width * 16
        else:
            patch_size = _pair(config.patches["size"])
            n_patches = (img_size[0] // patch_size[0]) * (
                img_size[1] // patch_size[1]
            )
            self.hybrid_model = None
        self.patch_embeddings = nn.Conv2d(
            in_channels,
            config.hidden_size,
            kernel_size=patch_size,
            stride=patch_size,
        )
        self.position_embeddings = nn.Parameter(
            torch.zeros(1, n_patches, config.hidden_size)
        )
        self.dropout = nn.Dropout(config.transformer["dropout_rate"])

    def forward(self, x):
        features = None
        if self.hybrid_model is not None:
            x, features = self.hybrid_model(x)
            features.append(x)
        x = self.patch_embeddings(x).flatten(2).transpose(-1, -2)
        return self.dropout(x + self.position_embeddings), features


__all__ = [
    "EncoderRGB",
    "SharedEmbeddings",
    "StdConv2d",
    "get_r50_b16_config",
    "np2th",
]
