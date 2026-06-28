from __future__ import annotations

from pathlib import Path
from typing import Callable, Sequence

import cv2
import numpy as np
import pandas as pd
from torch.utils.data import Dataset


DEFAULT_FINGER_NAMES = ("thumb", "index", "middle", "ring")


class TactileDataset(Dataset):
    """Dataset for one grasp iteration per item.

    Each returned sample is a nested dictionary:

    {
        "thumb": {
            "background": image,
            "lift_start": image,
            "lift_end": image,
        },
        ...
    }

    Images are RGB HWC numpy arrays before transforms. The label is loaded from
    grasp/grasp_label.csv using label_method, usually "manual".
    """

    def __init__(
        self,
        root_dir: str | Path | None = None,
        iteration_paths: Sequence[str | Path] | None = None,
        train_transform: Callable | None = None,
        val_transform: Callable | None = None,
        n_aug_copies: int = 1,
        include_original: bool = True,
        train: bool = True,
        frames_to_stack: int = 16,
        finger_names: Sequence[str] = DEFAULT_FINGER_NAMES,
        label_method: str = "manual",
    ) -> None:
        if root_dir is None and iteration_paths is None:
            raise ValueError("Pass either root_dir or iteration_paths.")
        if n_aug_copies < 0:
            raise ValueError("n_aug_copies must be non-negative.")
        if frames_to_stack <= 0:
            raise ValueError("frames_to_stack must be positive.")

        self.iteration_paths = self._resolve_iteration_paths(root_dir, iteration_paths)
        self.train_transform = train_transform
        self.val_transform = val_transform
        self.train = train
        self.include_original = include_original
        self.frames_to_stack = frames_to_stack
        self.finger_names = tuple(finger_names)
        self.label_method = label_method

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
            raise ValueError("No valid iteration directories found.")

    def __len__(self) -> int:
        return len(self.iteration_paths) * self.copies_per_iteration

    def __getitem__(self, index: int):
        real_index = index // self.copies_per_iteration
        copy_index = index % self.copies_per_iteration

        sample, label = self.load_iteration(self.iteration_paths[real_index])
        transform = self._select_transform(copy_index)
        if transform is not None:
            sample = transform(sample)

        return sample, label

    def load_iteration(self, iteration_path: str | Path):
        iteration_path = Path(iteration_path)
        phase_data = self._read_phase_data(iteration_path)
        lift_start_time = self._get_phase_time(phase_data, phase="lift")
        label = self._read_label(iteration_path)

        sample = {}
        for finger in self.finger_names:
            camera_meta_path = (
                iteration_path / f"opentouch/digit360/{finger}/camera/camera.csv"
            )
            frames_path = iteration_path / f"opentouch/digit360/{finger}/camera/frames"
            if not camera_meta_path.exists() or not frames_path.exists():
                raise FileNotFoundError(
                    f"Missing camera data for finger {finger!r} in {iteration_path}"
                )

            frames_meta = pd.read_csv(camera_meta_path).sort_values(
                by="time_perf", axis=0
            )
            first_frame_path = frames_path / frames_meta["filename"].iloc[0]
            lift_start_paths = self._paths_from_filenames(
                frames_path,
                frames_meta.loc[
                    frames_meta["time_perf"] >= lift_start_time,
                    "filename",
                ].iloc[: self.frames_to_stack],
                context=f"{iteration_path.name}/{finger} lift_start",
            )
            lift_end_paths = self._paths_from_filenames(
                frames_path,
                frames_meta["filename"].iloc[-self.frames_to_stack :],
                context=f"{iteration_path.name}/{finger} lift_end",
            )

            sample[finger] = {
                "background": self._load_rgb_frame(first_frame_path),
                "lift_start": self._average_frames(lift_start_paths),
                "lift_end": self._average_frames(lift_end_paths),
            }

        return sample, label

    def _select_transform(self, copy_index: int):
        if not self.train:
            return self.val_transform

        if self.include_original and copy_index == 0:
            return self.val_transform

        return self.train_transform

    def _resolve_iteration_paths(
        self,
        root_dir: str | Path | None,
        iteration_paths: Sequence[str | Path] | None,
    ) -> list[Path]:
        if iteration_paths is not None:
            paths = [Path(path) for path in iteration_paths]
        else:
            root = Path(root_dir).expanduser()
            paths = sorted(path for path in root.iterdir() if path.is_dir())

        return [path for path in paths if (path / "grasp/grasp_phases.csv").exists()]

    def _read_phase_data(self, iteration_path: Path) -> pd.DataFrame:
        phase_data_path = iteration_path / "grasp/grasp_phases.csv"
        if not phase_data_path.exists():
            raise FileNotFoundError(phase_data_path)
        return pd.read_csv(phase_data_path)

    def _get_phase_time(self, phase_data: pd.DataFrame, phase: str) -> float:
        phase_times = phase_data.loc[phase_data["phase"] == phase, "perf_time"]
        if phase_times.empty:
            raise ValueError(f"Missing phase {phase!r} in grasp_phases.csv.")
        return float(phase_times.iloc[0])

    def _read_label(self, iteration_path: Path) -> int:
        label_path = iteration_path / "grasp/grasp_label.csv"
        if not label_path.exists():
            raise FileNotFoundError(label_path)

        labels = pd.read_csv(label_path)
        selected_labels = labels.loc[labels["method"] == self.label_method, "label"]
        if selected_labels.empty:
            raise ValueError(
                f"Missing label method {self.label_method!r} in {label_path}."
            )

        return int(selected_labels.iloc[0])

    def _paths_from_filenames(
        self,
        frames_path: Path,
        filenames: pd.Series,
        context: str,
    ) -> list[Path]:
        paths = [frames_path / filename for filename in filenames]
        if len(paths) < self.frames_to_stack:
            raise ValueError(
                f"Expected at least {self.frames_to_stack} frames for {context}, "
                f"got {len(paths)}."
            )
        return paths

    def _average_frames(self, frame_paths: Sequence[Path]) -> np.ndarray:
        frames = np.stack([self._load_rgb_frame(path) for path in frame_paths], axis=0)
        return np.round(frames.astype(np.float32).mean(axis=0)).astype(np.uint8)

    def _load_rgb_frame(self, path: Path) -> np.ndarray:
        img_bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if img_bgr is None:
            raise FileNotFoundError(path)
        return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
