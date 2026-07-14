from __future__ import annotations

from typing import Callable, Mapping, Sequence

from torch.utils.data import Dataset


IMAGES_KEY = "images"
PROPRIOCEPTORS_KEY = "proprioceptors"


class TactileDataset(Dataset):
    """Dataset for one grasp iteration per item.

    Each returned sample is a nested dictionary:

    {
        "images": {
            "thumb": {
                "background": image,
                "lift_start": image,
                "lift_end": image,
            },
            ...
        },
        "proprioceptors": {
            "commanded": action,
            "observed": pose,
        },
    }

    Images are RGB HWC numpy arrays before transforms. The label is loaded from
    grasp/grasp_label.csv using label_method, usually "manual".
    """

    def __init__(
        self,
        iteration_paths: Sequence[Mapping],
        loader,
        train_transform: Callable | None = None,
        val_transform: Callable | None = None,
        n_aug_copies: int = 1,
        include_original: bool = True,
        train: bool = True,
    ) -> None:
        if n_aug_copies < 0:
            raise ValueError("n_aug_copies must be non-negative.")

        self.iteration_paths = list(iteration_paths)
        self.train_transform = train_transform
        self.val_transform = val_transform
        self.train = train
        self.include_original = include_original
        self.loader = loader

        if self.train:
            self.copies_per_iteration = n_aug_copies + int(include_original)
            if self.copies_per_iteration == 0:
                raise ValueError(
                    "Training dataset is empty when n_aug_copies=0 and "
                    "include_original=False."
                )
        else:
            self.copies_per_iteration = 1
            if not include_original:
                raise ValueError("Evaluation data must include the original sample.")

        if not self.iteration_paths:
            raise ValueError("No iteration paths were provided.")

    def __len__(self) -> int:
        return len(self.iteration_paths) * self.copies_per_iteration

    def __getitem__(self, index: int):
        real_index = index // self.copies_per_iteration
        copy_index = index % self.copies_per_iteration

        iteration_path = self.iteration_paths[real_index]["path"]
        sample, label = self.loader.load_iteration(iteration_path)
        transform = self._select_transform(copy_index)
        if transform is not None:
            sample[IMAGES_KEY] = transform(sample[IMAGES_KEY])

        return sample, label

    def _select_transform(self, copy_index: int):
        if not self.train:
            return self.val_transform

        if self.include_original and copy_index == 0:
            return self.val_transform

        return self.train_transform
