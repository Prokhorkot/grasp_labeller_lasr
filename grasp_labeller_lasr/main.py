from __future__ import annotations

from pathlib import Path

import cv2
import torch
from torch.utils.data import random_split
from torchvision.transforms import v2

from grasp_labeller_lasr.dataset import TactileDataset
from grasp_labeller_lasr.transforms import (
    MultiFingerTransform,
    Cv2Resize,
    RemoveBackground,
    Identity
)


FRAMES_TO_STACK = 16
DOWNSAMPLING_SIZE = 256
CROP_SIZE = 224

N_AUG_COPIES = 2
COL_JITTER_BR = 0.2
REMOVE_BACKGROUND = False

TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
SPLIT_SEED = 42


def get_iteration_paths(base_dir: Path) -> list[Path]:
    return sorted(
        iteration_path
        for iteration_path in base_dir.iterdir()
        if iteration_path.is_dir()
        and (iteration_path / "grasp/grasp_phases.csv").exists()
    )


def main() -> None:
    package_dir = Path(__file__).resolve().parent
    repo_root = package_dir.parent
    dataset_root = repo_root / "dataset"

    val_image_transform = v2.Compose(
        [
            RemoveBackground() if REMOVE_BACKGROUND else Identity(),
            Cv2Resize((DOWNSAMPLING_SIZE, DOWNSAMPLING_SIZE), cv2.INTER_AREA),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.CenterCrop((CROP_SIZE, CROP_SIZE)),
        ]
    )
    train_image_transform = v2.Compose(
        [
            RemoveBackground() if REMOVE_BACKGROUND else Identity(),
            Cv2Resize((DOWNSAMPLING_SIZE, DOWNSAMPLING_SIZE), cv2.INTER_AREA),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.RandomCrop(CROP_SIZE),
            v2.ColorJitter(brightness=COL_JITTER_BR),
        ]
    )

    val_transform = MultiFingerTransform(val_image_transform)
    train_transform = MultiFingerTransform(train_image_transform)

    iteration_paths = get_iteration_paths(dataset_root)
    train_size = int(len(iteration_paths) * TRAIN_RATIO)
    val_size = int(len(iteration_paths) * VAL_RATIO)
    test_size = len(iteration_paths) - train_size - val_size

    train_paths, val_paths, test_paths = random_split(
        iteration_paths,
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(SPLIT_SEED),
    )

    train_dataset = TactileDataset(
        iteration_paths=list(train_paths),
        train_transform=train_transform,
        val_transform=val_transform,
        n_aug_copies=N_AUG_COPIES,
        include_original=True,
        train=True,
        frames_to_stack=FRAMES_TO_STACK,
    )
    val_dataset = TactileDataset(
        iteration_paths=list(val_paths),
        val_transform=val_transform,
        train=False,
        frames_to_stack=FRAMES_TO_STACK,
    )
    test_dataset = TactileDataset(
        iteration_paths=list(test_paths),
        val_transform=val_transform,
        train=False,
        frames_to_stack=FRAMES_TO_STACK,
    )

    sample, label = train_dataset[0]
    first_finger = next(iter(sample))
    first_phase = next(iter(sample[first_finger]))
    first_image = sample[first_finger][first_phase]

    print(
        "Loaded "
        f"{len(train_dataset)} train, "
        f"{len(val_dataset)} val, "
        f"{len(test_dataset)} test items."
    )
    print(f"First label: {label}")
    print(f"First tensor: {first_finger}/{first_phase} -> {tuple(first_image.shape)}")


if __name__ == "__main__":
    main()
