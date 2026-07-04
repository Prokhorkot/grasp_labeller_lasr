from __future__ import annotations

from pathlib import Path

import cv2
import torch
from torch.utils.data import random_split
from torchvision.transforms import v2

from grasp_labeller_lasr.data.dataset import TactileDataset
from grasp_labeller_lasr.data.loaders import (
    PHASE_DATA_RELATIVE_PATH,
    DirectoryIterationLoader,
    H5IterationLoader,
)
from grasp_labeller_lasr.data.transforms import (
    MultiFingerTransform,
    Cv2Resize,
    RemoveBackground,
    Identity
)


FRAMES_TO_STACK = 16
DOWNSAMPLING_SIZE = 256
CROP_SIZE = 224
FINGER_NAMES = ("thumb", "index", "middle", "ring")
LABEL_METHOD = "manual"

N_AUG_COPIES = 2
COL_JITTER_BR = 0.2
REMOVE_BACKGROUND = False

TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
SPLIT_SEED = 42


def get_iteration_paths(base_dir: Path) -> tuple[list[dict], type]:
    data_types = set()
    samples = []

    def add_data_type(data_type: str) -> None:
        data_types.add(data_type)
        if len(data_types) > 1:
            raise ValueError(
                f"Mixed iteration formats are not supported: {sorted(data_types)}."
            )

    for object_dir in sorted(path for path in base_dir.iterdir() if path.is_dir()):
        iteration_paths = []
        for iteration_path in sorted(object_dir.iterdir()):
            if iteration_path.suffix == ".h5":
                add_data_type("h5")
                iteration_paths.append(iteration_path)
            elif (
                iteration_path.is_dir()
                and (iteration_path / PHASE_DATA_RELATIVE_PATH).exists()
            ):
                add_data_type("directory")
                iteration_paths.append(iteration_path)

        for iteration_path in iteration_paths:
            samples.append(
                {
                    "object": object_dir.name,
                    "path": iteration_path,
                }
            )

    if not samples:
        raise ValueError(f"No object-level iterations found in {base_dir}.")

    data_type = next(iter(data_types))
    loader_cls = H5IterationLoader(
        frames_to_stack=FRAMES_TO_STACK,
        finger_names=FINGER_NAMES,
        label_method=LABEL_METHOD
    ) if data_type == "h5" else DirectoryIterationLoader(
        frames_to_stack=FRAMES_TO_STACK,
        finger_names=FINGER_NAMES,
        label_method=LABEL_METHOD
    )
    return samples, loader_cls


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

    iteration_paths, loader = get_iteration_paths(dataset_root)
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
        n_aug_copies=N_AUG_COPIES,
        include_original=True,
        train=True,
        loader=loader,
    )
    val_dataset = TactileDataset(
        iteration_paths=list(val_paths),
        val_transform=val_transform,
        train=False,
        loader=loader,
    )
    test_dataset = TactileDataset(
        iteration_paths=list(test_paths),
        val_transform=val_transform,
        train=False,
        loader=loader,
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
