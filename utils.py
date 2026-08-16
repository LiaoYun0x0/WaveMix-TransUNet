"""Utilities required by the public Vaihingen training script."""

from __future__ import annotations

import itertools
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


PALETTE = {
    0: (255, 255, 255),
    1: (0, 0, 255),
    2: (0, 255, 255),
    3: (0, 255, 0),
    4: (255, 255, 0),
    5: (255, 0, 0),
    6: (0, 0, 0),
}
INVERT_PALETTE = {color: label for label, color in PALETTE.items()}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def convert_from_color(image, palette=INVERT_PALETTE):
    labels = np.zeros(image.shape[:2], dtype=np.uint8)
    for color, index in palette.items():
        labels[np.all(image == np.asarray(color), axis=2)] = index
    return labels


def get_random_pos(image, window_shape):
    height, width = window_shape
    image_height, image_width = image.shape[-2:]
    if height > image_height or width > image_width:
        raise ValueError(
            f"Window {window_shape} exceeds image {(image_height, image_width)}"
        )
    row = random.randint(0, image_height - height)
    col = random.randint(0, image_width - width)
    return row, row + height, col, col + width


def _window_starts(size, window, step):
    if step <= 0:
        raise ValueError("step must be positive")
    if window > size:
        raise ValueError(f"Window {window} exceeds image dimension {size}")
    starts = list(range(0, size - window + 1, step))
    if starts[-1] != size - window:
        starts.append(size - window)
    return starts


def sliding_window(image, step, window_size):
    for row in _window_starts(image.shape[0], window_size[0], step):
        for col in _window_starts(image.shape[1], window_size[1], step):
            yield row, col, window_size[0], window_size[1]


def grouper(size, iterable):
    iterator = iter(iterable)
    while True:
        chunk = tuple(itertools.islice(iterator, size))
        if not chunk:
            return
        yield chunk


def accuracy(prediction, target):
    valid = target != 6
    return 100.0 * float(np.count_nonzero(prediction[valid] == target[valid])) / max(
        int(np.count_nonzero(valid)), 1
    )


def confusion_matrix(predictions, targets, num_classes=6):
    valid = ((targets >= 0) & (targets < num_classes)
             & (predictions >= 0) & (predictions < num_classes))
    indices = num_classes * targets[valid].astype(np.int64) \
        + predictions[valid].astype(np.int64)
    return np.bincount(indices, minlength=num_classes ** 2).reshape(
        num_classes, num_classes
    )


def metrics(predictions, targets, num_classes=6):
    matrix = confusion_matrix(predictions, targets, num_classes)
    diagonal = np.diag(matrix).astype(np.float64)
    row_sum = matrix.sum(axis=1)
    col_sum = matrix.sum(axis=0)
    per_class_accuracy = np.divide(
        diagonal, row_sum, out=np.full(num_classes, np.nan), where=row_sum != 0
    )
    f1 = np.divide(
        2 * diagonal,
        row_sum + col_sum,
        out=np.full(num_classes, np.nan),
        where=(row_sum + col_sum) != 0,
    )
    iou = np.divide(
        diagonal,
        row_sum + col_sum - diagonal,
        out=np.full(num_classes, np.nan),
        where=(row_sum + col_sum - diagonal) != 0,
    )
    overall_accuracy = 100.0 * diagonal.sum() / max(matrix.sum(), 1)
    # ISPRS convention excludes clutter from the reported mean scores.
    return {
        "confusion_matrix": matrix,
        "overall_accuracy": overall_accuracy,
        "per_class_accuracy": per_class_accuracy,
        "per_class_f1": f1,
        "mean_f1": float(np.nanmean(f1[:5])),
        "per_class_iou": iou,
        "mean_iou": float(np.nanmean(iou[:5])),
    }


def cross_entropy_2d(inputs, targets, weight=None, ignore_index=6):
    return F.cross_entropy(inputs, targets, weight=weight,
                           ignore_index=ignore_index)


class MulticlassDiceLoss(nn.Module):
    def __init__(self, ignore_index=6, smooth=1.0):
        super().__init__()
        self.ignore_index = ignore_index
        self.smooth = smooth

    def forward(self, inputs, targets):
        probabilities = F.softmax(inputs, dim=1)
        valid = targets != self.ignore_index
        safe_targets = targets.masked_fill(~valid, 0)
        one_hot = F.one_hot(safe_targets, inputs.shape[1]).permute(0, 3, 1, 2)
        mask = valid[:, None].to(probabilities.dtype)
        probabilities = probabilities * mask
        one_hot = one_hot.to(probabilities.dtype) * mask
        intersection = (probabilities * one_hot).sum((0, 2, 3))
        cardinality = probabilities.sum((0, 2, 3)) + one_hot.sum((0, 2, 3))
        score = (2 * intersection + self.smooth) / (cardinality + self.smooth)
        return 1 - score.mean()


class CeAndDiceLoss(nn.Module):
    def __init__(self, weight=None, ignore_index=6):
        super().__init__()
        self.register_buffer("ce_weight", weight)
        self.ignore_index = ignore_index
        self.dice = MulticlassDiceLoss(ignore_index)

    def forward(self, inputs, targets):
        ce = cross_entropy_2d(inputs, targets, self.ce_weight, self.ignore_index)
        return ce + self.dice(inputs, targets)


def append_to_file(path, content):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(content)


def format_metrics(epoch, result, labels):
    lines = [f"epoch: {epoch}", f"confusion_matrix:\n{result['confusion_matrix']}"]
    lines.append(f"overall_accuracy: {result['overall_accuracy']:.4f}")
    lines.append(f"mean_f1: {result['mean_f1']:.6f}")
    lines.append(f"mean_iou: {result['mean_iou']:.6f}")
    for index, label in enumerate(labels):
        lines.append(
            f"{label}: accuracy={result['per_class_accuracy'][index]:.6f}, "
            f"f1={result['per_class_f1'][index]:.6f}, "
            f"iou={result['per_class_iou'][index]:.6f}"
        )
    return "\n".join(lines) + "\n"
