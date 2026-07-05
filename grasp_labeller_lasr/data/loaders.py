from __future__ import annotations

import hashlib
import os
from pathlib import Path, PurePosixPath
import tempfile
from typing import Sequence

import cv2
import h5py
import numpy as np
import pandas as pd
import torch


DEFAULT_FINGER_NAMES = ("thumb", "index", "middle", "ring")
PHASE_DATA_RELATIVE_PATH = Path("grasp/grasp_phases.csv")
LABEL_RELATIVE_PATH = Path("grasp/grasp_label.csv")
CAMERA_META_RELATIVE_TEMPLATE = "opentouch/digit360/{finger}/camera/camera.csv"
FRAMES_RELATIVE_TEMPLATE = "opentouch/digit360/{finger}/camera/frames"

LIFT_PHASE = "lift"

BACKGROUND_KEY = "background"
LIFT_START_KEY = "lift_start"
LIFT_END_KEY = "lift_end"


class CachedIterationLoader:
    """Cache loaded iteration samples while preserving the loader interface."""

    CACHE_VERSION = 1

    def __init__(self, loader, cache_dir: str | Path) -> None:
        self.loader = loader
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def load_iteration(self, iteration_path: str | Path):
        iteration_path = Path(iteration_path)
        cache_path = self._cache_path(iteration_path)
        if cache_path.exists():
            cached = self._load_cache(cache_path)
            return cached["sample"], cached["label"]

        sample, label = self.loader.load_iteration(iteration_path)
        self._save_cache(cache_path, {"sample": sample, "label": label})
        return sample, label

    def _cache_path(self, iteration_path: Path) -> Path:
        return self.cache_dir / f"{self._cache_key(iteration_path)}.pt"

    def _cache_key(self, iteration_path: Path) -> str:
        stat = iteration_path.stat()
        payload = "|".join(
            [
                f"v={self.CACHE_VERSION}",
                f"loader={type(self.loader).__name__}",
                f"path={iteration_path.resolve()}",
                f"mtime_ns={stat.st_mtime_ns}",
                f"size={stat.st_size}",
                f"frames_to_stack={getattr(self.loader, 'frames_to_stack', None)}",
                f"finger_names={getattr(self.loader, 'finger_names', None)}",
                f"label_method={getattr(self.loader, 'label_method', None)}",
            ]
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    @staticmethod
    def _load_cache(cache_path: Path) -> dict:
        try:
            return torch.load(cache_path, weights_only=False)
        except TypeError:
            return torch.load(cache_path)

    def _save_cache(self, cache_path: Path, payload: dict) -> None:
        fd, temp_path = tempfile.mkstemp(
            prefix=f".{cache_path.stem}.",
            suffix=".tmp",
            dir=self.cache_dir,
        )
        os.close(fd)
        temp_cache_path = Path(temp_path)
        try:
            torch.save(payload, temp_cache_path)
            os.replace(temp_cache_path, cache_path)
        finally:
            if temp_cache_path.exists():
                temp_cache_path.unlink()


class DirectoryIterationLoader:
    def __init__(
        self,
        frames_to_stack: int = 16,
        finger_names: Sequence[str] = DEFAULT_FINGER_NAMES,
        label_method: str = "manual",
    ) -> None:
        self.frames_to_stack = frames_to_stack
        self.finger_names = tuple(finger_names)
        self.label_method = label_method

    def load_iteration(self, iteration_path: str | Path):
        iteration_path = Path(iteration_path)
        phase_data = self._read_phase_data(iteration_path)
        lift_start_time = self._get_phase_time(phase_data, phase=LIFT_PHASE)
        label = self._read_label(iteration_path)

        sample = {}
        for finger in self.finger_names:
            camera_meta_path = iteration_path / CAMERA_META_RELATIVE_TEMPLATE.format(
                finger=finger
            )
            frames_path = iteration_path / FRAMES_RELATIVE_TEMPLATE.format(
                finger=finger
            )
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
                BACKGROUND_KEY: self._load_rgb_frame(first_frame_path),
                LIFT_START_KEY: self._average_frames(lift_start_paths),
                LIFT_END_KEY: self._average_frames(lift_end_paths),
            }

        return sample, label

    def _read_phase_data(self, iteration_path: Path) -> pd.DataFrame:
        phase_data_path = iteration_path / PHASE_DATA_RELATIVE_PATH
        if not phase_data_path.exists():
            raise FileNotFoundError(phase_data_path)
        return pd.read_csv(phase_data_path)

    def _get_phase_time(self, phase_data: pd.DataFrame, phase: str) -> float:
        phase_times = phase_data.loc[
            phase_data["phase"] == phase,
            "perf_time",
        ]
        if phase_times.empty:
            raise ValueError(f"Missing phase {phase!r} in grasp_phases.csv.")
        return float(phase_times.iloc[0])

    def _read_label(self, iteration_path: Path) -> int:
        label_path = iteration_path / LABEL_RELATIVE_PATH
        if not label_path.exists():
            raise FileNotFoundError(label_path)

        labels = pd.read_csv(label_path)
        selected_labels = labels.loc[
            labels["method"] == self.label_method,
            "label",
        ]
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


class H5IterationLoader:
    def __init__(
        self,
        frames_to_stack: int = 16,
        finger_names: Sequence[str] = DEFAULT_FINGER_NAMES,
        label_method: str = "manual",
    ) -> None:
        self.frames_to_stack = frames_to_stack
        self.finger_names = tuple(finger_names)
        self.label_method = label_method

    def load_iteration(self, iteration_path: str | Path):
        iteration_path = Path(iteration_path)
        with h5py.File(iteration_path, "r") as h5_file:
            phase_data = self._read_phase_data(h5_file, iteration_path)
            lift_start_time = self._get_phase_time(phase_data, phase=LIFT_PHASE)
            label = self._read_label(h5_file, iteration_path)

            sample = {}
            for finger in self.finger_names:
                camera_meta_key = self._h5_key(
                    iteration_path,
                    CAMERA_META_RELATIVE_TEMPLATE.format(finger=finger),
                )
                frames_key = self._h5_key(
                    iteration_path,
                    FRAMES_RELATIVE_TEMPLATE.format(finger=finger),
                )
                if camera_meta_key not in h5_file or frames_key not in h5_file:
                    raise FileNotFoundError(
                        f"Missing camera data for finger {finger!r} in "
                        f"{iteration_path}"
                    )

                frames_meta = self._read_table(h5_file[camera_meta_key]).sort_values(
                    by="time_perf", axis=0
                )
                first_frame_idx = int(frames_meta.index[0])
                lift_start_indices = frames_meta.loc[
                    frames_meta["time_perf"] >= lift_start_time,
                ].index[: self.frames_to_stack]
                lift_end_indices = frames_meta.index[-self.frames_to_stack :]

                self._validate_frame_count(
                    lift_start_indices,
                    context=f"{iteration_path.name}/{finger} lift_start",
                )
                self._validate_frame_count(
                    lift_end_indices,
                    context=f"{iteration_path.name}/{finger} lift_end",
                )

                frames_ds = h5_file[frames_key]
                sample[finger] = {
                    BACKGROUND_KEY: self._read_encoded_frame(
                        frames_ds,
                        first_frame_idx,
                    ),
                    LIFT_START_KEY: self._average_frames(
                        frames_ds,
                        lift_start_indices,
                    ),
                    LIFT_END_KEY: self._average_frames(
                        frames_ds,
                        lift_end_indices,
                    ),
                }

        return sample, label

    def _read_phase_data(
        self,
        h5_file: h5py.File,
        iteration_path: Path,
    ) -> pd.DataFrame:
        key = self._h5_key(iteration_path, PHASE_DATA_RELATIVE_PATH)
        if key not in h5_file:
            raise FileNotFoundError(f"{iteration_path}:{key}")
        return self._read_table(h5_file[key])

    def _get_phase_time(self, phase_data: pd.DataFrame, phase: str) -> float:
        phase_times = phase_data.loc[
            phase_data["phase"] == phase,
            "perf_time",
        ]
        if phase_times.empty:
            raise ValueError(f"Missing phase {phase!r} in grasp_phases.csv.")
        return float(phase_times.iloc[0])

    def _read_label(self, h5_file: h5py.File, iteration_path: Path) -> int:
        key = self._h5_key(iteration_path, LABEL_RELATIVE_PATH)
        if key not in h5_file:
            raise FileNotFoundError(f"{iteration_path}:{key}")

        labels = self._read_table(h5_file[key])
        selected_labels = labels.loc[
            labels["method"] == self.label_method,
            "label",
        ]
        if selected_labels.empty:
            raise ValueError(
                f"Missing label method {self.label_method!r} in "
                f"{iteration_path}:{key}."
            )

        return int(selected_labels.iloc[0])

    def _h5_key(self, iteration_path: Path, relative_path: str | Path) -> str:
        return str(PurePosixPath(iteration_path.stem) / PurePosixPath(str(relative_path)))

    def _read_table(self, h5_dataset: h5py.Dataset) -> pd.DataFrame:
        df = pd.DataFrame.from_records(h5_dataset[()])
        for col in df.columns:
            if df[col].dtype == object:
                df[col] = df[col].apply(
                    lambda value: value.decode("utf-8")
                    if isinstance(value, (bytes, bytearray))
                    else value
                )
        return df

    def _read_encoded_frame(
        self,
        frames_dataset: h5py.Dataset,
        frame_idx: int,
        *,
        rgb: bool = True,
    ) -> np.ndarray:
        encoded = np.asarray(frames_dataset[frame_idx], dtype=np.uint8).ravel()
        img = cv2.imdecode(encoded, cv2.IMREAD_UNCHANGED)

        if img is None:
            raise ValueError(
                f"Could not decode frame {frame_idx} from {frames_dataset.name}"
            )

        if rgb and img.ndim == 3:
            if img.shape[2] == 3:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            elif img.shape[2] == 4:
                img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGBA)

        return img

    def _validate_frame_count(self, frame_indices: Sequence[int], context: str) -> None:
        if len(frame_indices) < self.frames_to_stack:
            raise ValueError(
                f"Expected at least {self.frames_to_stack} frames for {context}, "
                f"got {len(frame_indices)}."
            )

    def _average_frames(
        self,
        frames_ds: h5py.Dataset,
        frame_indices: Sequence[int],
    ) -> np.ndarray:
        frames = np.stack(
            [
                self._read_encoded_frame(frames_ds, int(frame_idx))
                for frame_idx in frame_indices
            ],
            axis=0,
        )
        return np.round(frames.astype(np.float32).mean(axis=0)).astype(np.uint8)
