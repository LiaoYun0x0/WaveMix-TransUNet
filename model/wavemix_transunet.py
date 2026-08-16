"""WaveMix-TransUNet model for IRRG + DSM semantic segmentation.

All non-backbone model code is kept in this file: WaveMix blocks, raw-DSM
geometry priors, the progressive decoder, segmentation heads, and final model
assembly. DSM is encoder-free and is used only as a relation prior at 1/16,
1/8, and 1/4 scales.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .r50_vit_encoder import (
    EncoderRGB,
    SharedEmbeddings,
    get_r50_b16_config,
    np2th,
)


class ConvBNReLU(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1,
                 stride=1, bias=False):
        padding = ((stride - 1) + dilation * (kernel_size - 1)) // 2
        super().__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride,
                      padding, dilation, bias=bias),
            nn.BatchNorm2d(out_channels),
            nn.ReLU6(),
        )


class ConvBN(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1,
                 stride=1, bias=False):
        padding = ((stride - 1) + dilation * (kernel_size - 1)) // 2
        super().__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride,
                      padding, dilation, bias=bias),
            nn.BatchNorm2d(out_channels),
        )


class Conv(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1,
                 stride=1, bias=False):
        padding = ((stride - 1) + dilation * (kernel_size - 1)) // 2
        super().__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride,
                      padding, dilation, bias=bias)
        )


class SeparableConvBN(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 dilation=1):
        padding = ((stride - 1) + dilation * (kernel_size - 1)) // 2
        super().__init__(
            nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding,
                      dilation, groups=in_channels, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
        )


class AuxHead(nn.Module):
    def __init__(self, in_channels=64, num_classes=6):
        super().__init__()
        self.conv = ConvBNReLU(in_channels, in_channels)
        self.drop = nn.Dropout(0.1)
        self.conv_out = Conv(in_channels, num_classes, kernel_size=1)

    def forward(self, x, height, width):
        x = self.conv_out(self.drop(self.conv(x)))
        return F.interpolate(x, (height, width), mode="bilinear",
                             align_corners=False)


class RGBFeatureRefinementHead(nn.Module):
    def __init__(self, in_channels_skip, in_channels_x, out_channels):
        super().__init__()
        self.decode_channels = out_channels
        self.pre_conv_res = Conv(in_channels_skip, out_channels, kernel_size=1)
        self.pre_conv_x = ConvBN(in_channels_x, out_channels, kernel_size=1)
        self.weights = nn.Parameter(torch.ones(2))
        self.eps = 1e-8
        self.post_conv = ConvBNReLU(out_channels, out_channels)
        self.pa = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, padding=1,
                      groups=out_channels),
            nn.Sigmoid(),
        )
        self.ca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            Conv(out_channels, out_channels // 16, kernel_size=1),
            nn.ReLU6(),
            Conv(out_channels // 16, out_channels, kernel_size=1),
            nn.Sigmoid(),
        )
        self.shortcut = ConvBN(out_channels, out_channels, kernel_size=1)
        self.proj = SeparableConvBN(out_channels, out_channels)
        self.act = nn.ReLU6()

    def forward(self, x, rgb_skip):
        x = F.interpolate(x, rgb_skip.shape[-2:], mode="bilinear",
                          align_corners=False)
        weights = F.relu(self.weights)
        weights = weights / (weights.sum() + self.eps)
        x = self.post_conv(
            weights[0] * self.pre_conv_res(rgb_skip)
            + weights[1] * self.pre_conv_x(x)
        )
        refined = self.pa(x) * x + self.ca(x) * x
        return self.act(self.proj(refined) + self.shortcut(x))


class LayerNorm2d(nn.Module):
    def __init__(self, channels, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(1, keepdim=True)
        variance = (x - mean).pow(2).mean(1, keepdim=True)
        x = (x - mean) / torch.sqrt(variance + self.eps)
        return self.weight[:, None, None] * x + self.bias[:, None, None]


class ChannelMLP(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None,
                 act_layer=nn.GELU, drop=0.1):
        super().__init__()
        hidden_features = hidden_features or in_features
        out_features = out_features or in_features
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1)
        self.act = act_layer()
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1)
        self.drop = nn.Dropout(drop)
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.normal_(self.fc1.bias, std=1e-6)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.normal_(self.fc2.bias, std=1e-6)

    def forward(self, x):
        x = self.drop(self.act(self.fc1(x)))
        return self.drop(self.fc2(x))


class WaveletBranch(nn.Module):
    """Haar-DWT branch from WaveMixFormer."""

    def __init__(self, dim, expansion_ratio=0.5, se_ratio=16):
        super().__init__()
        hidden_ll = int(dim * expansion_ratio)
        hidden_hf = max(dim // se_ratio, 32)
        self.process_LL = nn.Sequential(
            nn.Conv2d(dim, dim, 5, padding=2, groups=dim, bias=False),
            LayerNorm2d(dim),
            nn.Conv2d(dim, hidden_ll, 1, bias=False),
            nn.GELU(),
            nn.Conv2d(hidden_ll, dim, 1, bias=False),
        )
        self.se_HF = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim * 3, hidden_hf, 1),
            nn.ReLU(),
            nn.Conv2d(hidden_hf, dim * 3, 1),
            nn.Sigmoid(),
        )

    @staticmethod
    def dwt_haar(x):
        if x.shape[-2] % 2 or x.shape[-1] % 2:
            x = F.pad(x, (0, x.shape[-1] % 2, 0, x.shape[-2] % 2),
                      mode="replicate")
        x01, x02 = x[:, :, 0::2, :] / 2, x[:, :, 1::2, :] / 2
        x1, x2 = x01[:, :, :, 0::2], x02[:, :, :, 0::2]
        x3, x4 = x01[:, :, :, 1::2], x02[:, :, :, 1::2]
        ll = x1 + x2 + x3 + x4
        return ll, torch.cat((-x1 - x2 + x3 + x4,
                              -x1 + x2 - x3 + x4,
                              x1 - x2 - x3 + x4), dim=1)

    @staticmethod
    def idwt_haar(ll, hf):
        channels = ll.shape[1]
        lh, hl, hh = torch.split(hf, channels, dim=1)
        x1, x2 = ll - lh - hl + hh, ll - lh + hl - hh
        x3, x4 = ll + lh - hl - hh, ll + lh + hl + hh
        output = ll.new_zeros((ll.shape[0], channels,
                               ll.shape[2] * 2, ll.shape[3] * 2))
        output[:, :, 0::2, 0::2] = x1
        output[:, :, 0::2, 1::2] = x2
        output[:, :, 1::2, 0::2] = x3
        output[:, :, 1::2, 1::2] = x4
        return output

    def forward(self, x):
        original_size = x.shape[-2:]
        ll, hf = self.dwt_haar(x)
        output = self.idwt_haar(ll + self.process_LL(ll), hf * self.se_HF(hf))
        return output[:, :, : original_size[0], : original_size[1]]


class WaveMixTokenMixer(nn.Module):
    """Spatial-frequency token mixer used by WaveMixFormer."""

    def __init__(self, dim, dwt_expansion=0.5, se_ratio=16):
        super().__init__()
        self.spatial = nn.Sequential(
            nn.Conv2d(dim, dim, 7, padding=3, groups=dim, bias=False),
            nn.BatchNorm2d(dim),
        )
        self.wavelet = WaveletBranch(dim, dwt_expansion, se_ratio)
        self.spatial_weight = nn.Parameter(torch.full((1, dim, 1, 1), 0.5))
        self.wavelet_weight = nn.Parameter(torch.full((1, dim, 1, 1), 0.5))

    def forward(self, x):
        return (self.spatial_weight * self.spatial(x)
                + self.wavelet_weight * self.wavelet(x))


class WaveMixFormer(nn.Module):
    """WaveMix token mixing followed by a channel MLP."""

    def __init__(self, dim, mlp_ratio=4.0, drop=0.1, act_layer=nn.GELU,
                 dwt_expansion=0.5, dwt_se_ratio=16):
        super().__init__()
        self.norm1 = LayerNorm2d(dim)
        self.mixer = WaveMixTokenMixer(dim, dwt_expansion, dwt_se_ratio)
        self.norm2 = LayerNorm2d(dim)
        self.mlp = ChannelMLP(dim, int(dim * mlp_ratio), act_layer=act_layer,
                              drop=drop)

    def forward(self, x):
        x = x + self.mixer(self.norm1(x))
        return x + self.mlp(self.norm2(x))


def _make_wavemix_stack(channels, depth, dwt_expansion, dwt_se_ratio):
    if depth <= 0:
        return nn.Identity()
    return nn.Sequential(*[
        WaveMixFormer(channels, dwt_expansion=dwt_expansion,
                      dwt_se_ratio=dwt_se_ratio)
        for _ in range(depth)
    ])


def _as_dsm_map(dsm):
    if dsm is None:
        raise ValueError("WaveMix-TransUNet requires a DSM tensor")
    if dsm.dim() == 3:
        dsm = dsm.unsqueeze(1)
    if dsm.dim() != 4 or dsm.shape[1] != 1:
        raise ValueError(f"DSM must be [B,H,W] or [B,1,H,W], got {dsm.shape}")
    return dsm


class DSMGeometryAttention(nn.Module):
    """Global 1/16 RGB self-attention biased by DSM height and distance."""

    def __init__(self, channels, num_heads=8, attn_drop=0.0):
        super().__init__()
        self.num_heads = min(num_heads, channels)
        while channels % self.num_heads:
            self.num_heads -= 1
        self.head_dim = channels // self.num_heads
        self.scale = self.head_dim ** -0.5
        self.attn_drop = nn.Dropout(attn_drop)
        self.qkv = nn.Conv2d(channels, channels * 3, 1, bias=False)
        self.proj = nn.Sequential(
            nn.Conv2d(channels, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.height_scale = nn.Parameter(torch.zeros(self.num_heads))
        self.spatial_scale = nn.Parameter(torch.zeros(self.num_heads))
        self.gamma = nn.Parameter(torch.zeros(1))

    @staticmethod
    def _spatial_distance(height, width, device, dtype):
        rows = torch.arange(height, device=device, dtype=dtype)
        cols = torch.arange(width, device=device, dtype=dtype)
        grid_y, grid_x = torch.meshgrid(rows, cols, indexing="ij")
        coords = torch.stack((grid_y.flatten(), grid_x.flatten()), dim=-1)
        distance = (coords[:, None] - coords[None, :]).abs().sum(-1)
        return distance / max(height + width - 2, 1)

    def forward(self, rgb, raw_dsm):
        batch, channels, height, width = rgb.shape
        dsm = F.adaptive_avg_pool2d(
            _as_dsm_map(raw_dsm).to(rgb), (height, width)
        ).flatten(2).squeeze(1)
        height_distance = (dsm[:, :, None] - dsm[:, None, :]).abs()
        height_distance /= height_distance.amax((1, 2), keepdim=True) + 1e-6
        spatial_distance = self._spatial_distance(
            height, width, rgb.device, rgb.dtype
        )
        qkv = self.qkv(rgb).reshape(
            batch, 3, self.num_heads, self.head_dim, height * width
        )
        q, k, v = [item.transpose(-2, -1) for item in qkv.unbind(1)]
        logits = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        logits -= F.softplus(self.height_scale)[None, :, None, None] * \
            height_distance[:, None]
        logits -= F.softplus(self.spatial_scale)[None, :, None, None] * \
            spatial_distance[None, None]
        output = torch.matmul(self.attn_drop(F.softmax(logits, -1)), v)
        output = output.transpose(-2, -1).reshape(batch, channels, height, width)
        return rgb + self.gamma * self.proj(output)


class DSMLocalBoundaryAttention(nn.Module):
    """Windowed RGB attention biased by DSM relative height and gradients."""

    def __init__(self, channels, num_heads=8, window_size=8, attn_drop=0.0):
        super().__init__()
        self.num_heads = min(num_heads, channels)
        while channels % self.num_heads:
            self.num_heads -= 1
        self.head_dim = channels // self.num_heads
        self.scale = self.head_dim ** -0.5
        self.window_size = int(window_size)
        self.attn_drop = nn.Dropout(attn_drop)
        self.qkv = nn.Conv2d(channels, channels * 3, 1, bias=False)
        self.proj = nn.Sequential(
            nn.Conv2d(channels, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.height_scale = nn.Parameter(torch.zeros(self.num_heads))
        self.gradient_scale = nn.Parameter(torch.zeros(self.num_heads))
        self.spatial_scale = nn.Parameter(torch.zeros(self.num_heads))
        self.gamma = nn.Parameter(torch.zeros(1))
        sobel_x = torch.tensor(
            [[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]]
        ).view(1, 1, 3, 3)
        self.register_buffer("sobel_x", sobel_x, persistent=False)
        self.register_buffer("sobel_y", sobel_x.transpose(-1, -2).contiguous(),
                             persistent=False)
        coords = torch.arange(self.window_size, dtype=torch.float32)
        grid_y, grid_x = torch.meshgrid(coords, coords, indexing="ij")
        coords = torch.stack((grid_y.flatten(), grid_x.flatten()), -1)
        distance = (coords[:, None] - coords[None, :]).abs().sum(-1)
        self.register_buffer(
            "window_spatial_distance",
            distance / max(2 * (self.window_size - 1), 1),
            persistent=False,
        )

    def _partition_windows(self, x):
        batch, channels, height, width = x.shape
        window = self.window_size
        pad_h, pad_w = (-height) % window, (-width) % window
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode="replicate")
        padded_h, padded_w = x.shape[-2:]
        x = x.view(batch, channels, padded_h // window, window,
                   padded_w // window, window)
        windows = x.permute(0, 2, 4, 3, 5, 1).contiguous()
        return windows.view(-1, window * window, channels), \
            (batch, height, width, padded_h, padded_w)

    def _reverse_windows(self, windows, shape):
        batch, height, width, padded_h, padded_w = shape
        window, channels = self.window_size, windows.shape[-1]
        x = windows.view(batch, padded_h // window, padded_w // window,
                         window, window, channels)
        x = x.permute(0, 5, 1, 3, 2, 4).contiguous()
        return x.view(batch, channels, padded_h, padded_w)[:, :, :height, :width]

    def _dsm_structure(self, raw_dsm, size, dtype, device):
        dsm = F.adaptive_avg_pool2d(
            _as_dsm_map(raw_dsm).to(device=device, dtype=dtype), size
        )
        relative = dsm - F.avg_pool2d(dsm, 5, stride=1, padding=2)
        grad_x = F.conv2d(relative, self.sobel_x.to(dtype), padding=1)
        grad_y = F.conv2d(relative, self.sobel_y.to(dtype), padding=1)
        relative /= relative.abs().amax((2, 3), keepdim=True) + 1e-6
        gradient_norm = (grad_x.abs() + grad_y.abs()).amax((2, 3), keepdim=True)
        return torch.cat((relative, grad_x / (gradient_norm + 1e-6),
                          grad_y / (gradient_norm + 1e-6)), dim=1)

    def forward(self, rgb, raw_dsm):
        batch, channels, height, width = rgb.shape
        structure = self._dsm_structure(raw_dsm, (height, width), rgb.dtype,
                                        rgb.device)
        qkv_windows, shape = self._partition_windows(self.qkv(rgb))
        structure_windows, structure_shape = self._partition_windows(structure)
        if shape != structure_shape:
            raise RuntimeError("RGB and DSM window layouts do not match")
        window_batch, token_count, _ = qkv_windows.shape
        qkv = qkv_windows.view(window_batch, token_count, 3, self.num_heads,
                               self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        height_values, gx, gy = structure_windows.unbind(-1)
        height_distance = (height_values[:, :, None]
                           - height_values[:, None, :]).abs()
        gradient_distance = ((gx[:, :, None] - gx[:, None, :]).abs()
                             + (gy[:, :, None] - gy[:, None, :]).abs())
        logits = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        logits -= F.softplus(self.height_scale)[None, :, None, None] * \
            height_distance[:, None]
        logits -= F.softplus(self.gradient_scale)[None, :, None, None] * \
            gradient_distance[:, None]
        logits -= F.softplus(self.spatial_scale)[None, :, None, None] * \
            self.window_spatial_distance.to(rgb)[None, None]
        output = torch.matmul(self.attn_drop(F.softmax(logits, -1)), v)
        output = output.transpose(1, 2).reshape(window_batch, token_count,
                                                channels)
        return rgb + self.gamma * self.proj(self._reverse_windows(output, shape))


class DSMFineBoundaryAttention(DSMLocalBoundaryAttention):
    def __init__(self, channels, num_heads=8, window_size=4, attn_drop=0.0):
        super().__init__(channels, num_heads, window_size, attn_drop)
        nn.init.constant_(self.height_scale, -2.0)
        nn.init.constant_(self.gradient_scale, 0.0)
        nn.init.constant_(self.spatial_scale, 0.0)

    def _dsm_structure(self, raw_dsm, size, dtype, device):
        dsm = F.adaptive_avg_pool2d(
            _as_dsm_map(raw_dsm).to(device=device, dtype=dtype), size
        )
        relative = dsm - F.avg_pool2d(dsm, 3, stride=1, padding=1)
        grad_x = F.conv2d(relative, self.sobel_x.to(dtype), padding=1)
        grad_y = F.conv2d(relative, self.sobel_y.to(dtype), padding=1)
        relative /= relative.abs().amax((2, 3), keepdim=True) + 1e-6
        gradient_norm = (grad_x.abs() + grad_y.abs()).amax((2, 3), keepdim=True)
        return torch.cat((relative, grad_x / (gradient_norm + 1e-6),
                          grad_y / (gradient_norm + 1e-6)), dim=1)


class DSMLocalPriorStage(nn.Module):
    def __init__(self, in_channels, out_channels, skip_channels, depth,
                 dwt_expansion, dwt_se_ratio, num_heads=8, window_size=8):
        super().__init__()
        self.reduce = ConvBNReLU(in_channels + skip_channels, out_channels, 1)
        self.local_attention = DSMLocalBoundaryAttention(
            out_channels, num_heads, window_size
        )
        self.blocks = _make_wavemix_stack(
            out_channels, depth, dwt_expansion, dwt_se_ratio
        )

    def forward(self, x, rgb_skip, raw_dsm):
        x = F.interpolate(x, rgb_skip.shape[-2:], mode="bilinear",
                          align_corners=False)
        x = self.reduce(torch.cat((x, rgb_skip), 1))
        return self.blocks(self.local_attention(x, raw_dsm))


class DSMFinePriorStage(DSMLocalPriorStage):
    def __init__(self, in_channels, out_channels, skip_channels, depth,
                 dwt_expansion, dwt_se_ratio, num_heads=8, window_size=4):
        super().__init__(in_channels, out_channels, skip_channels, depth,
                         dwt_expansion, dwt_se_ratio, num_heads, window_size)
        self.local_attention = DSMFineBoundaryAttention(
            out_channels, num_heads, window_size
        )


class RGBConcatStage(nn.Module):
    """Used during construction to retain the reported initialization order."""

    def __init__(self, in_channels, out_channels, skip_channels, is_up, depth,
                 dwt_expansion, dwt_se_ratio):
        super().__init__()
        self.is_up = is_up
        self.reduce = ConvBNReLU(in_channels + skip_channels, out_channels, 1)
        self.blocks = _make_wavemix_stack(
            out_channels, depth, dwt_expansion, dwt_se_ratio
        )

    def forward(self, x, skip):
        if self.is_up:
            x = F.interpolate(x, skip.shape[-2:], mode="bilinear",
                              align_corners=False)
        return self.blocks(self.reduce(torch.cat((x, skip), 1)))


class WaveMixDecoder(nn.Module):
    def __init__(self, config, block_depths=(1, 1, 1, 1),
                 dwt_expansion=0.5, dwt_se_ratio=16, num_classes=6,
                 geometry_heads=8, local_window_size=8, fine_window_size=4):
        super().__init__()
        depths = list(block_depths) + [1] * max(0, 4 - len(block_depths))
        head_channels = 512
        stage8_channels, stage4_channels = config.decoder_channels[:2]
        self.output_channels = config.decoder_channels[2]
        self.vit_rgb_proj = ConvBNReLU(config.hidden_size, head_channels)
        self.global_attention16 = DSMGeometryAttention(head_channels,
                                                       geometry_heads)
        self.vit16_blocks = _make_wavemix_stack(
            head_channels, depths[0], dwt_expansion, dwt_se_ratio
        )

        # Construct then remove the legacy 1/16 RGB-CNN concat branch. This
        # preserves the module initialization sequence used for paper runs.
        self.cnn16_rgb_proj = ConvBNReLU(config.cnn_out_channels, head_channels)
        self.cnn16_stage = RGBConcatStage(
            head_channels, head_channels, head_channels, False, depths[3],
            dwt_expansion, dwt_se_ratio
        )
        self.stage8 = DSMLocalPriorStage(
            head_channels, stage8_channels, config.skip_channels[0], depths[1],
            dwt_expansion, dwt_se_ratio, geometry_heads, local_window_size
        )
        self.stage4 = RGBConcatStage(
            stage8_channels, stage4_channels, config.skip_channels[1], True,
            depths[2], dwt_expansion, dwt_se_ratio
        )
        self.refinement2 = RGBFeatureRefinementHead(
            config.skip_channels[2], stage4_channels, self.output_channels
        )
        self.aux16_proj = Conv(head_channels, stage4_channels, 1)
        self.aux8_proj = Conv(stage8_channels, stage4_channels, 1)
        self.aux_head = AuxHead(stage4_channels, num_classes)

        self.stage4 = DSMFinePriorStage(
            stage8_channels, stage4_channels, config.skip_channels[1], depths[2],
            dwt_expansion, dwt_se_ratio, geometry_heads, fine_window_size
        )
        del self.cnn16_rgb_proj
        del self.cnn16_stage

    def forward(self, hidden, rgb_skips, raw_dsm, output_shape):
        batch, token_count, channels = hidden.shape
        height = int(math.sqrt(token_count))
        if height * height != token_count:
            raise ValueError(f"ViT token count must be square, got {token_count}")
        x = hidden.transpose(1, 2).reshape(batch, channels, height, height)
        x = self.vit16_blocks(self.global_attention16(
            self.vit_rgb_proj(x), raw_dsm
        ))
        feature16 = x
        x = self.stage8(x, rgb_skips[0], raw_dsm)
        feature8 = x
        x = self.stage4(x, rgb_skips[1], raw_dsm)
        feature4 = x
        auxiliary = None
        if self.training:
            aux16 = F.interpolate(self.aux16_proj(feature16),
                                  feature4.shape[-2:], mode="bilinear",
                                  align_corners=False)
            aux8 = F.interpolate(self.aux8_proj(feature8),
                                 feature4.shape[-2:], mode="bilinear",
                                 align_corners=False)
            auxiliary = self.aux_head(aux16 + aux8 + feature4, *output_shape)
        return self.refinement2(x, rgb_skips[2]), auxiliary


class WaveMixTransUNet(nn.Module):
    """Final paper model: R50-ViT-B/16 + WaveMix decoder + DSM priors."""

    input_modalities = ("IRRG", "DSM")

    def __init__(self, config=None, img_size=256, num_classes=6,
                 zero_head=False, vis=False, block_depths=(1, 1, 1, 1),
                 dwt_expansion=0.5, dwt_se_ratio=16, dropout=0.1,
                 geometry_heads=8, local_window_size=8, fine_window_size=4):
        super().__init__()
        config = config or get_r50_b16_config(num_classes, img_size)
        self.num_classes = num_classes
        self.zero_head = zero_head
        self.config = config
        self.embeddings = SharedEmbeddings(config, img_size)
        self.encoder = EncoderRGB(config, vis)
        self.decoder = WaveMixDecoder(
            config, block_depths, dwt_expansion, dwt_se_ratio, num_classes,
            geometry_heads, local_window_size, fine_window_size
        )
        channels = self.decoder.output_channels
        self.segmentation_head = nn.Sequential(
            ConvBNReLU(channels, channels),
            nn.Dropout2d(dropout, inplace=True),
            Conv(channels, num_classes, kernel_size=1),
        )

    def forward(self, rgb, dsm):
        output_shape = rgb.shape[-2:]
        embeddings, rgb_skips = self.embeddings(rgb)
        encoded, _ = self.encoder(embeddings)
        x, auxiliary = self.decoder(encoded, rgb_skips, _as_dsm_map(dsm),
                                    output_shape)
        x = F.interpolate(x, output_shape, mode="bilinear", align_corners=False)
        logits = self.segmentation_head(x)
        return (logits, auxiliary) if self.training else logits

    def load_from(self, weights):
        """Load the ImageNet R50+ViT-B/16 NPZ initialization."""
        with torch.no_grad():
            self.encoder.encoder_norm.weight.copy_(
                np2th(weights["Transformer/encoder_norm/scale"])
            )
            self.encoder.encoder_norm.bias.copy_(
                np2th(weights["Transformer/encoder_norm/bias"])
            )
            for index, block in enumerate(self.encoder.layer):
                block.load_from(weights, str(index))
            position = np2th(weights["Transformer/posembed_input/pos_embedding"])
            target = self.embeddings.position_embeddings
            if position.shape != target.shape:
                position = position[:, 1:, :]
                old_grid = int(math.sqrt(position.shape[1]))
                new_grid = int(math.sqrt(target.shape[1]))
                position = position.permute(0, 2, 1).reshape(
                    1, -1, old_grid, old_grid
                )
                position = F.interpolate(position, (new_grid, new_grid),
                                         mode="bilinear", align_corners=False)
                position = position.flatten(2).transpose(1, 2)
            target.copy_(position)
            root = self.embeddings.hybrid_model.root
            root.conv.weight.copy_(np2th(weights["conv_root/kernel"], conv=True))
            root.gn.weight.copy_(np2th(weights["gn_root/scale"]).view(-1))
            root.gn.bias.copy_(np2th(weights["gn_root/bias"]).view(-1))
            for block_name, block in self.embeddings.hybrid_model.body.named_children():
                for unit_name, unit in block.named_children():
                    unit.load_from(weights, block_name, unit_name)

    def load_pretrained(self, npz_path):
        npz_path = Path(npz_path)
        if not npz_path.is_file():
            raise FileNotFoundError(f"TransUNet NPZ not found: {npz_path}")
        with np.load(npz_path) as weights:
            self.load_from(weights)
        return self


def build_wavemix_transunet(num_classes=6, image_size=256,
                            pretrained_path=None, **kwargs):
    config = get_r50_b16_config(num_classes, image_size)
    model = WaveMixTransUNet(config, image_size, num_classes, **kwargs)
    if pretrained_path:
        model.load_pretrained(pretrained_path)
    return model


__all__ = [
    "WaveMixTransUNet",
    "WaveMixFormer",
    "WaveMixTokenMixer",
    "build_wavemix_transunet",
]
