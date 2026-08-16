"""Train and validate WaveMix-TransUNet on ISPRS Vaihingen.

Example:
    python train/train_vaihingen.py \
        --data-root ./ISPRS_dataset/Vaihingen \
        --pretrained ./pretrain/R50+ViT-B_16.npz
"""

from __future__ import annotations

import argparse
import random
import sys
import time
from pathlib import Path

import albumentations as A
import cv2
import numpy as np
import torch
from skimage import io
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from model.wavemix_transunet import build_wavemix_transunet
from utils import (
    CeAndDiceLoss,
    accuracy,
    append_to_file,
    convert_from_color,
    cross_entropy_2d,
    format_metrics,
    grouper,
    metrics,
    set_seed,
    sliding_window,
)


LABELS = ("impervious", "building", "low_vegetation", "tree", "car", "clutter")
TRAIN_IDS = ("1", "3", "23", "26", "7", "11", "13", "28", "17", "32", "34", "37")
TEST_IDS = ("5", "21", "15", "30")
IGNORE_INDEX = 6


def normalize_dsm(dsm):
    dsm = np.asarray(dsm, dtype=np.float32)
    minimum, maximum = float(dsm.min()), float(dsm.max())
    if maximum - minimum <= np.finfo(np.float32).eps:
        return np.zeros_like(dsm)
    return (dsm - minimum) / (maximum - minimum)


class VaihingenPaths:
    def __init__(self, root):
        self.root = Path(root)

    def image(self, area):
        return self.root / "top" / f"top_mosaic_09cm_area{area}.tif"

    def dsm(self, area):
        return self.root / "dsm" / f"dsm_09cm_matching_area{area}.tif"

    def label(self, area):
        return self.root / "gts_for_participants" / f"top_mosaic_09cm_area{area}.tif"

    def eroded_label(self, area):
        return (self.root / "gts_eroded_for_participants"
                / f"top_mosaic_09cm_area{area}_noBoundary.tif")

    def validate(self, areas, require_eroded=False):
        required = []
        for area in areas:
            required.extend((self.image(area), self.dsm(area), self.label(area)))
            if require_eroded:
                required.append(self.eroded_label(area))
        missing = [str(path) for path in required if not path.is_file()]
        if missing:
            raise FileNotFoundError("Missing Vaihingen files:\n" + "\n".join(missing))


class VaihingenDataset(Dataset):
    """Random 256x256 IRRG/DSM crops with the paper's augmentation policy."""

    def __init__(self, paths, areas, patch_size=256, batch_size=10,
                 iterations=300, cache=True):
        self.paths = paths
        self.areas = tuple(areas)
        self.patch_size = int(patch_size)
        self.length = int(batch_size) * int(iterations)
        self.cache = cache
        self.enable_mosaic = True
        self._cache = {}
        self.spatial = A.Compose(
            [
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                A.Affine(
                    scale=(0.9, 1.1),
                    translate_percent=0.1,
                    rotate=(-15, 15),
                    interpolation=cv2.INTER_LINEAR,
                    mask_interpolation=cv2.INTER_NEAREST,
                    fill=0,
                    fill_mask=IGNORE_INDEX,
                    p=0.7,
                ),
            ],
            additional_targets={"dsm": "image"},
        )
        self.color = A.Compose(
            [
                A.RandomBrightnessContrast(0.2, 0.2, p=0.5),
                A.HueSaturationValue(20, 30, 20, p=0.3),
                A.GaussNoise(p=0.2),
                A.GaussianBlur(p=0.3),
            ]
        )
        self.dropout = A.Compose(
            [
                A.CoarseDropout(
                    num_holes_range=(1, 6),
                    hole_height_range=(4, 12),
                    hole_width_range=(4, 12),
                    fill=0,
                    fill_mask=IGNORE_INDEX,
                    p=0.5,
                )
            ],
            additional_targets={"dsm": "image"},
        )

    def __len__(self):
        return self.length

    def disable_mosaic(self):
        self.enable_mosaic = False

    def _load(self, area):
        if area in self._cache:
            return self._cache[area]
        image = np.asarray(io.imread(self.paths.image(area)), dtype=np.float32)
        image = image.transpose(2, 0, 1) / 255.0
        dsm = normalize_dsm(io.imread(self.paths.dsm(area)))
        label = convert_from_color(io.imread(self.paths.label(area))).astype(np.int64)
        value = image, dsm, label
        if self.cache:
            self._cache[area] = value
        return value

    def _crop(self, area):
        image, dsm, label = self._load(area)
        size = self.patch_size
        row = random.randint(0, image.shape[1] - size)
        col = random.randint(0, image.shape[2] - size)
        return (image[:, row:row + size, col:col + size],
                dsm[row:row + size, col:col + size],
                label[row:row + size, col:col + size])

    def _mosaic(self):
        size = self.patch_size
        image = np.zeros((3, size * 2, size * 2), np.float32)
        dsm = np.zeros((size * 2, size * 2), np.float32)
        label = np.full((size * 2, size * 2), IGNORE_INDEX, np.int64)
        areas = [random.choice(self.areas) for _ in range(4)]
        for area, (row, col) in zip(areas, ((0, 0), (0, size), (size, 0),
                                                     (size, size))):
            tile_image, tile_dsm, tile_label = self._crop(area)
            image[:, row:row + size, col:col + size] = tile_image
            dsm[row:row + size, col:col + size] = tile_dsm
            label[row:row + size, col:col + size] = tile_label
        offset = size // 2
        return (image[:, offset:offset + size, offset:offset + size],
                dsm[offset:offset + size, offset:offset + size],
                label[offset:offset + size, offset:offset + size])

    def __getitem__(self, _):
        if self.enable_mosaic and random.random() < 0.5:
            image, dsm, label = self._mosaic()
        else:
            image, dsm, label = self._crop(random.choice(self.areas))
        transformed = self.spatial(
            image=image.transpose(1, 2, 0), dsm=dsm, mask=label
        )
        transformed["image"] = self.color(image=transformed["image"])["image"]
        transformed = self.dropout(**transformed)
        image = transformed["image"].transpose(2, 0, 1).astype(np.float32)
        return (torch.from_numpy(np.ascontiguousarray(image)),
                torch.from_numpy(np.ascontiguousarray(transformed["dsm"],
                                                       dtype=np.float32)),
                torch.from_numpy(np.ascontiguousarray(transformed["mask"],
                                                       dtype=np.int64)))


def build_validation_epochs(epochs):
    silent_end = int(epochs * 0.70)
    dense_start = epochs - max(1, int(epochs * 0.10)) + 1
    selected = {epoch for epoch in range(10, epochs + 1, 10)
                if silent_end < epoch < dense_start}
    selected.update(range(dense_start, epochs + 1))
    selected.add(epochs)
    return sorted(selected)


@torch.no_grad()
def validate(model, paths, areas, device, patch_size=256, stride=32,
             batch_size=10):
    model.eval()
    all_predictions, all_targets = [], []
    for area in tqdm(areas, desc="Validation images"):
        image = np.asarray(io.imread(paths.image(area)), dtype=np.float32) / 255.0
        dsm = normalize_dsm(io.imread(paths.dsm(area)))
        target = convert_from_color(io.imread(paths.eroded_label(area)))
        score = np.zeros((*image.shape[:2], len(LABELS)), dtype=np.float32)
        windows = sliding_window(image, stride, (patch_size, patch_size))
        for coordinates in tqdm(grouper(batch_size, windows), leave=False,
                                desc=f"Area {area}"):
            rgb_batch = np.stack([
                image[row:row + height, col:col + width].transpose(2, 0, 1)
                for row, col, height, width in coordinates
            ])
            dsm_batch = np.stack([
                dsm[row:row + height, col:col + width]
                for row, col, height, width in coordinates
            ])
            logits = model(
                torch.from_numpy(rgb_batch).to(device),
                torch.from_numpy(dsm_batch).to(device),
            ).cpu().numpy()
            for output, (row, col, height, width) in zip(logits, coordinates):
                score[row:row + height, col:col + width] += output.transpose(1, 2, 0)
        all_predictions.append(score.argmax(-1))
        all_targets.append(target)
    return metrics(
        np.concatenate([item.ravel() for item in all_predictions]),
        np.concatenate([item.ravel() for item in all_targets]),
        len(LABELS),
    )


def train(args):
    set_seed(args.seed)
    device = torch.device(args.device if args.device else
                          ("cuda" if torch.cuda.is_available() else "cpu"))
    if device.type != "cuda":
        print("Warning: training on CPU will be very slow.")
    paths = VaihingenPaths(args.data_root)
    paths.validate(TRAIN_IDS)
    paths.validate(TEST_IDS, require_eroded=True)
    model = build_wavemix_transunet(
        num_classes=len(LABELS),
        image_size=args.patch_size,
        pretrained_path=args.pretrained,
    ).to(device)
    dataset = VaihingenDataset(
        paths, TRAIN_IDS, args.patch_size, args.batch_size,
        args.iterations_per_epoch, cache=not args.no_cache,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
    )
    optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate,
                                momentum=0.9, weight_decay=5e-4)
    warmup_epochs = min(args.warmup_epochs, max(args.epochs - 1, 0))
    if warmup_epochs:
        warmup = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=0.01, total_iters=warmup_epochs
        )
        cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(args.epochs - warmup_epochs, 1), eta_min=1e-5
        )
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer, (warmup, cosine), milestones=(warmup_epochs,)
        )
    else:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(args.epochs, 1), eta_min=1e-5
        )
    criterion = CeAndDiceLoss(
        torch.ones(len(LABELS), dtype=torch.float32), IGNORE_INDEX
    ).to(device)
    run_dir = Path(args.output_dir) / time.strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=False)
    record_path = run_dir / "training_records.txt"
    validation_epochs = build_validation_epochs(args.epochs)
    append_to_file(record_path, f"arguments: {vars(args)}\nvalidation_epochs: "
                                  f"{validation_epochs}\n")
    best_miou, best_epoch = float("-inf"), None
    no_mosaic_epoch = max(1, args.epochs - args.no_mosaic_epochs + 1)
    for epoch in range(1, args.epochs + 1):
        if epoch == no_mosaic_epoch:
            dataset.disable_mosaic()
        model.train()
        epoch_losses = []
        progress = tqdm(loader, desc=f"Epoch {epoch}/{args.epochs}")
        for rgb, dsm, target in progress:
            rgb, dsm, target = rgb.to(device), dsm.to(device), target.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits, auxiliary = model(rgb, dsm)
            loss = criterion(logits, target)
            loss = loss + args.aux_weight * cross_entropy_2d(
                auxiliary, target, criterion.ce_weight, IGNORE_INDEX
            )
            loss.backward()
            optimizer.step()
            epoch_losses.append(loss.item())
            progress.set_postfix(loss=f"{loss.item():.4f}")
        scheduler.step()
        mean_loss = float(np.mean(epoch_losses))
        message = f"epoch={epoch}, train_loss={mean_loss:.6f}, lr={optimizer.param_groups[0]['lr']:.8f}\n"
        print(message.strip())
        append_to_file(record_path, message)
        if epoch in validation_epochs:
            result = validate(model, paths, TEST_IDS, device, args.patch_size,
                              args.val_stride, args.batch_size)
            report = format_metrics(epoch, result, LABELS)
            print(report)
            append_to_file(record_path, report + "\n")
            if result["mean_iou"] > best_miou:
                best_miou, best_epoch = result["mean_iou"], epoch
                if not args.no_save_checkpoint:
                    torch.save(
                        {
                            "epoch": epoch,
                            "best_miou": best_miou,
                            "model_version": "WaveMix-TransUNet",
                            "model_state_dict": model.state_dict(),
                            "arguments": vars(args),
                        },
                        run_dir / "best_miou_model_weights.pth",
                    )
    summary = f"best_epoch={best_epoch}, best_miou={best_miou:.6f}\n"
    append_to_file(record_path, summary)
    print(summary.strip())


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True,
                        help="Vaihingen directory containing top/, dsm/, and labels")
    parser.add_argument("--pretrained", type=Path, required=True,
                        help="R50+ViT-B_16.npz path")
    parser.add_argument("--output-dir", type=Path, default=Path("result/WaveMix-TransUNet"))
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--iterations-per-epoch", type=int, default=300)
    parser.add_argument("--patch-size", type=int, default=256)
    parser.add_argument("--val-stride", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--warmup-epochs", type=int, default=5)
    parser.add_argument("--no-mosaic-epochs", type=int, default=15)
    parser.add_argument("--aux-weight", type=float, default=0.4)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default=None, help="for example cuda:0 or cpu")
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--no-save-checkpoint", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
