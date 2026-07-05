from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import lightning as L
import torch
from torch.utils.data import DataLoader, random_split
from torchvision.transforms import v2

from grasp_labeller_lasr.data.dataset import TactileDataset
from grasp_labeller_lasr.data.loaders import (
    PHASE_DATA_RELATIVE_PATH,
    CachedIterationLoader,
    DirectoryIterationLoader,
    H5IterationLoader,
)
from grasp_labeller_lasr.data.transforms import (
    Cv2Resize,
    Identity,
    MultiFingerTransform,
    RemoveBackground,
)


class GraspDataModule(L.LightningDataModule):
    def __init__(
        self,
        dataset_root: str | Path,
        frames_to_stack: int,
        downsampling_size: int,
        crop_size: int,
        finger_names: tuple[str, ...],
        label_method: str,
        n_aug_copies: int,
        include_original: bool,
        color_jitter_brightness: float,
        remove_background: bool,
        train_ratio: float,
        val_ratio: float,
        split_seed: int,
        batch_size: int,
        num_workers: int,
        cache_enabled: bool = False,
        cache_dir: str | Path = ".cache/grasp_iterations",
    ) -> None:
        super().__init__()

        self.dataset_root = Path(dataset_root)
        self.frames_to_stack = frames_to_stack
        self.downsampling_size = downsampling_size
        self.crop_size = crop_size
        self.finger_names = finger_names
        self.label_method = label_method
        self.n_aug_copies = n_aug_copies
        self.include_original = include_original
        self.color_jitter_brightness = color_jitter_brightness
        self.remove_background = remove_background
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.split_seed = split_seed
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.cache_enabled = cache_enabled
        self.cache_dir = Path(cache_dir)

        self.train_dataset: TactileDataset | None = None
        self.val_dataset: TactileDataset | None = None
        self.test_dataset: TactileDataset | None = None

    def setup(self, stage: str | None = None) -> None:
        if self.train_dataset is not None:
            return

        train_transform, val_transform = self._build_transforms()
        iteration_paths, loader = self._get_iteration_paths()
        train_size, val_size, test_size = self._split_sizes(len(iteration_paths))
        train_paths, val_paths, test_paths = random_split(
            iteration_paths,
            [train_size, val_size, test_size],
            generator=torch.Generator().manual_seed(self.split_seed),
        )

        self.train_dataset = TactileDataset(
            iteration_paths=list(train_paths),
            train_transform=train_transform,
            val_transform=val_transform,
            n_aug_copies=self.n_aug_copies,
            include_original=self.include_original,
            train=True,
            loader=loader,
        )
        self.val_dataset = TactileDataset(
            iteration_paths=list(val_paths),
            val_transform=val_transform,
            train=False,
            loader=loader,
        )
        self.test_dataset = TactileDataset(
            iteration_paths=list(test_paths),
            val_transform=val_transform,
            train=False,
            loader=loader,
        )

    def train_dataloader(self) -> DataLoader:
        if self.train_dataset is None:
            raise RuntimeError("Call setup() before requesting train_dataloader().")
        return self._dataloader(self.train_dataset, shuffle=True)

    def val_dataloader(self) -> DataLoader:
        if self.val_dataset is None:
            raise RuntimeError("Call setup() before requesting val_dataloader().")
        return self._dataloader(self.val_dataset)

    def test_dataloader(self) -> DataLoader:
        if self.test_dataset is None:
            raise RuntimeError("Call setup() before requesting test_dataloader().")
        return self._dataloader(self.test_dataset)

    def _build_transforms(self) -> tuple[MultiFingerTransform, MultiFingerTransform]:
        val_image_transform = v2.Compose(
            [
                RemoveBackground() if self.remove_background else Identity(),
                Cv2Resize(
                    (self.downsampling_size, self.downsampling_size),
                    cv2.INTER_AREA,
                ),
                v2.ToImage(),
                v2.ToDtype(torch.float32, scale=True),
                v2.CenterCrop((self.crop_size, self.crop_size)),
            ]
        )
        train_image_transform = v2.Compose(
            [
                RemoveBackground() if self.remove_background else Identity(),
                Cv2Resize(
                    (self.downsampling_size, self.downsampling_size),
                    cv2.INTER_AREA,
                ),
                v2.ToImage(),
                v2.ToDtype(torch.float32, scale=True),
                v2.RandomCrop(self.crop_size),
                v2.ColorJitter(brightness=self.color_jitter_brightness),
            ]
        )

        return (
            MultiFingerTransform(train_image_transform),
            MultiFingerTransform(val_image_transform),
        )

    def _get_iteration_paths(self) -> tuple[list[dict], Any]:
        data_types = set()
        samples = []
        cache_dir = self.cache_dir
        if not cache_dir.is_absolute():
            cache_dir = self.dataset_root / cache_dir
        try:
            cache_root = (
                cache_dir.resolve()
                .relative_to(self.dataset_root.resolve())
                .parts[0]
            )
        except ValueError:
            cache_root = None

        def add_data_type(data_type: str) -> None:
            data_types.add(data_type)
            if len(data_types) > 1:
                raise ValueError(
                    "Mixed iteration formats are not supported: "
                    f"{sorted(data_types)}."
                )

        for object_dir in sorted(
            path
            for path in self.dataset_root.iterdir()
            if path.is_dir()
            and path.name != cache_root
        ):
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
            raise ValueError(f"No object-level iterations found in {self.dataset_root}.")

        data_type = next(iter(data_types))
        loader = (
            H5IterationLoader(
                frames_to_stack=self.frames_to_stack,
                finger_names=self.finger_names,
                label_method=self.label_method,
            )
            if data_type == "h5"
            else DirectoryIterationLoader(
                frames_to_stack=self.frames_to_stack,
                finger_names=self.finger_names,
                label_method=self.label_method,
            )
        )
        if self.cache_enabled:
            cache_dir = self.cache_dir
            if not cache_dir.is_absolute():
                cache_dir = self.dataset_root / cache_dir
            loader = CachedIterationLoader(loader, cache_dir=cache_dir)
        return samples, loader

    def _split_sizes(self, num_items: int) -> tuple[int, int, int]:
        train_size = int(num_items * self.train_ratio)
        val_size = int(num_items * self.val_ratio)
        test_size = num_items - train_size - val_size
        return train_size, val_size, test_size

    def _dataloader(self, dataset: TactileDataset, shuffle: bool = False) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
        )
