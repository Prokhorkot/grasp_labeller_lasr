from __future__ import annotations

import hashlib
import io
import os
import tempfile
import wave
from pathlib import Path, PurePosixPath
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
AUDIO_META_RELATIVE_TEMPLATE = "opentouch/digit360/{finger}/audio/chunks.csv"
AUDIO_WAV_RELATIVE_TEMPLATE = "opentouch/digit360/{finger}/audio/wav"

PROPRIOCEPTORS_ACTIONS = Path("tilburg/tilburg_action.csv")
PROPRIOCEPTORS_POSES = Path("tilburg/tilburg_pos.csv")
PROPRIOCEPTION_DIM = 16
N_POSES_TO_AVERAGE = 3

LIFT_PHASE = "lift"
POST_LIFT_PHASE = "post_lift"

BACKGROUND_KEY = "background"
LIFT_START_KEY = "lift_start"
LIFT_END_KEY = "lift_end"


def _summarize_proprioceptors(
    actions: pd.DataFrame,
    poses: pd.DataFrame,
    proprio_window: float,
) -> tuple[np.ndarray, np.ndarray]:
    if actions.empty:
        raise ValueError("The proprioceptor action table is empty.")
    if poses.empty:
        raise ValueError("The proprioceptor pose table is empty.")
    if actions.shape[1] != PROPRIOCEPTION_DIM:
        raise ValueError(
            f"Expected {PROPRIOCEPTION_DIM} commanded proprioceptor values, "
            f"got {actions.shape[1]}."
        )
    if poses.shape[1] != PROPRIOCEPTION_DIM:
        raise ValueError(
            f"Expected {PROPRIOCEPTION_DIM} observed proprioceptor values, "
            f"got {poses.shape[1]}."
        )

    action = actions.iloc[-1].to_numpy()
    last_perf_time = poses.index[-1]
    poses_to_average = []

    for i in range(N_POSES_TO_AVERAGE):
        target_time = last_perf_time - i * proprio_window
        idx = poses.index.searchsorted(target_time, side="right") - 1
        if idx < 0:
            raise ValueError(
                f"No pose is available at or before perf_time={target_time}."
            )
        poses_to_average.append(poses.iloc[idx].to_numpy())

    observed_pose = np.median(np.asarray(poses_to_average), axis=0)
    return action, observed_pose


def _read_pcm16_wav(wav_bytes: bytes, *, context: str) -> np.ndarray:
    """Read a 16-bit PCM WAV file into its unprocessed sample array."""
    try:
        with wave.open(io.BytesIO(wav_bytes), "rb") as wav_file:
            channel_count = max(1, int(wav_file.getnchannels()))
            sample_width = int(wav_file.getsampwidth())
            frame_count = int(wav_file.getnframes())
            raw_pcm = wav_file.readframes(frame_count)
    except (EOFError, wave.Error) as exc:
        raise ValueError(f"Could not decode WAV data from {context}.") from exc

    if sample_width != 2:
        raise ValueError(
            f"Expected 16-bit PCM audio in {context}, got "
            f"{sample_width * 8}-bit samples."
        )

    pcm = np.frombuffer(raw_pcm, dtype="<i2")
    if pcm.size == 0:
        raise ValueError(f"The WAV dataset is empty in {context}.")
    if pcm.size % channel_count != 0:
        raise ValueError(
            f"PCM sample count {pcm.size} is not divisible by the "
            f"channel count {channel_count} in {context}."
        )

    if channel_count > 1:
        return pcm.reshape(-1, channel_count)
    return pcm


def _select_lifting_audio(
    waveform: np.ndarray,
    chunks: pd.DataFrame,
    *,
    finger: str,
    t0: float,
    t1: float,
    context: str,
) -> np.ndarray:
    """Return the raw PCM samples belonging to the lifting time window."""
    required_columns = ["time_perf", "start_sample", "end_sample"]
    missing_columns = [
        column for column in required_columns if column not in chunks.columns
    ]
    if missing_columns:
        raise ValueError(f"Missing columns {missing_columns} in {context}.")

    metadata = chunks[required_columns].apply(pd.to_numeric, errors="coerce")
    metadata = metadata.replace([np.inf, -np.inf], np.nan)
    selected = metadata.loc[
        metadata.notna().all(axis=1)
        & metadata["time_perf"].between(t0, t1, inclusive="both")
    ]
    if selected.empty:
        raise ValueError(
            f"No audio chunks for finger {finger!r} fall in the time window "
            f"[{t0}, {t1}] in {context}."
        )

    start_sample = max(0, int(selected["start_sample"].min()))
    end_sample = min(len(waveform), int(selected["end_sample"].max()))
    if not 0 <= start_sample < end_sample <= len(waveform):
        raise ValueError(
            f"Invalid audio sample interval [{start_sample}, {end_sample}) "
            f"for waveform length {len(waveform)} in {context}."
        )
    return waveform[start_sample:end_sample]


class CachedIterationLoader:
    """Cache loaded iteration samples while preserving the loader interface."""

    CACHE_VERSION = 7

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
                f"proprio_window={getattr(self.loader, 'proprio_window', None)}",
                f"load_audio={getattr(self.loader, 'load_audio', None)}",
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
        proprio_window: float = 0.020,
        load_audio: bool = True,
    ) -> None:
        self.frames_to_stack = frames_to_stack
        self.finger_names = tuple(finger_names)
        self.label_method = label_method
        self.proprio_window = proprio_window
        self.load_audio = bool(load_audio)

    def load_iteration(self, iteration_path: str | Path):
        iteration_path = Path(iteration_path)
        phase_data = self._read_phase_data(iteration_path)
        lift_start_time = self._get_phase_time(phase_data, phase=LIFT_PHASE)
        label = self._read_label(iteration_path)

        lift_end_time = None
        if self.load_audio:
            lift_end_time = self._get_phase_time(
                phase_data,
                phase=POST_LIFT_PHASE,
            )
            if lift_end_time <= lift_start_time:
                raise ValueError(
                    f"Phase {POST_LIFT_PHASE!r} must occur after "
                    f"{LIFT_PHASE!r} in {iteration_path}."
                )
        sample = {"opentouch": {}, "proprioceptors": {}}
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

            sample["opentouch"][finger] = {
                "images": {
                    BACKGROUND_KEY: self._load_rgb_frame(first_frame_path),
                    LIFT_START_KEY: self._average_frames(lift_start_paths),
                    LIFT_END_KEY: self._average_frames(lift_end_paths),
                },
            }
            if self.load_audio:
                if lift_end_time is None:
                    raise RuntimeError("Lift end time was not initialized.")
                sample["opentouch"][finger]["audio"] = self._read_lifting_audio(
                    iteration_path,
                    finger=finger,
                    t0=lift_start_time,
                    t1=lift_end_time,
                )

        action, observed_pose = self.read_proprioceptors(iteration_path)
        sample["proprioceptors"] = {"commanded": action, "observed": observed_pose}

        return sample, label

    def read_proprioceptors(self, iteration_path: Path):
        actions = pd.read_csv(
            iteration_path / PROPRIOCEPTORS_ACTIONS,
            index_col="perf_time",
        ).sort_index()
        poses = pd.read_csv(
            iteration_path / PROPRIOCEPTORS_POSES,
            index_col="perf_time",
        ).sort_index()

        return _summarize_proprioceptors(
            actions,
            poses,
            self.proprio_window,
        )

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

    def _read_lifting_audio(
        self,
        iteration_path: Path,
        *,
        finger: str,
        t0: float,
        t1: float,
    ) -> np.ndarray:
        chunks_path = iteration_path / AUDIO_META_RELATIVE_TEMPLATE.format(
            finger=finger
        )
        wav_path = iteration_path / AUDIO_WAV_RELATIVE_TEMPLATE.format(finger=finger)
        if not chunks_path.exists() or not wav_path.exists():
            raise FileNotFoundError(
                f"Missing audio data for finger {finger!r} in {iteration_path}."
            )
        if not wav_path.is_file():
            raise ValueError(f"Expected WAV file at {wav_path}.")

        waveform = _read_pcm16_wav(wav_path.read_bytes(), context=str(wav_path))
        return _select_lifting_audio(
            waveform,
            pd.read_csv(chunks_path),
            finger=finger,
            t0=t0,
            t1=t1,
            context=str(chunks_path),
        )


class H5IterationLoader:
    def __init__(
        self,
        frames_to_stack: int = 16,
        finger_names: Sequence[str] = DEFAULT_FINGER_NAMES,
        label_method: str = "manual",
        proprio_window: float = 0.020,
        load_audio: bool = True,
    ) -> None:
        self.frames_to_stack = frames_to_stack
        self.finger_names = tuple(finger_names)
        self.label_method = label_method
        self.proprio_window = proprio_window
        self.load_audio = bool(load_audio)

    def load_iteration(self, iteration_path: str | Path):
        iteration_path = Path(iteration_path)
        with h5py.File(iteration_path, "r") as h5_file:
            phase_data = self._read_phase_data(h5_file, iteration_path)
            lift_start_time = self._get_phase_time(phase_data, phase=LIFT_PHASE)
            label = self._read_label(h5_file, iteration_path)

            lift_end_time = None
            if self.load_audio:
                lift_end_time = self._get_phase_time(
                    phase_data,
                    phase=POST_LIFT_PHASE,
                )
                if lift_end_time <= lift_start_time:
                    raise ValueError(
                        f"Phase {POST_LIFT_PHASE!r} must occur after "
                        f"{LIFT_PHASE!r} in {iteration_path}."
                    )

            sample = {"opentouch": {}, "proprioceptors": {}}
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
                finger_sample = {
                    "images": {
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
                    },
                }

                if self.load_audio:
                    if lift_end_time is None:
                        raise RuntimeError("Lift end time was not initialized.")
                    finger_sample["audio"] = self._read_lifting_audio(
                        h5_file,
                        iteration_path,
                        finger=finger,
                        t0=lift_start_time,
                        t1=lift_end_time,
                    )

                sample["opentouch"][finger] = finger_sample

            action, observed_pose = self.read_proprioceptors(
                h5_file,
                iteration_path,
            )
            sample["proprioceptors"] = {
                "commanded": action,
                "observed": observed_pose,
            }

        return sample, label

    def _read_lifting_audio(
        self,
        h5_file: h5py.File,
        iteration_path: Path,
        *,
        finger: str,
        t0: float,
        t1: float,
    ) -> np.ndarray:
        chunks_key = self._h5_key(
            iteration_path,
            AUDIO_META_RELATIVE_TEMPLATE.format(finger=finger),
        )
        wav_key = self._h5_key(
            iteration_path,
            AUDIO_WAV_RELATIVE_TEMPLATE.format(finger=finger),
        )
        if chunks_key not in h5_file or wav_key not in h5_file:
            raise FileNotFoundError(
                f"Missing audio data for finger {finger!r} in {iteration_path}."
            )

        wav_bytes = self._read_bytes_dataset(h5_file[wav_key])
        waveform = _read_pcm16_wav(
            wav_bytes,
            context=f"{iteration_path}:{wav_key}",
        )
        chunks = self._read_table(h5_file[chunks_key])
        return _select_lifting_audio(
            waveform,
            chunks,
            finger=finger,
            t0=t0,
            t1=t1,
            context=f"{iteration_path}:{chunks_key}",
        )

    def read_proprioceptors(
        self,
        h5_file: h5py.File,
        iteration_path: Path,
    ) -> tuple[np.ndarray, np.ndarray]:
        actions = self._read_indexed_table(
            h5_file,
            iteration_path,
            PROPRIOCEPTORS_ACTIONS,
        )
        poses = self._read_indexed_table(
            h5_file,
            iteration_path,
            PROPRIOCEPTORS_POSES,
        )
        return _summarize_proprioceptors(
            actions,
            poses,
            self.proprio_window,
        )

    def _read_indexed_table(
        self,
        h5_file: h5py.File,
        iteration_path: Path,
        relative_path: str | Path,
    ) -> pd.DataFrame:
        key = self._h5_key(iteration_path, relative_path)
        if key not in h5_file:
            raise FileNotFoundError(f"{iteration_path}:{key}")

        table = self._read_table(h5_file[key])
        if "perf_time" not in table.columns:
            raise ValueError(f"Missing 'perf_time' column in {iteration_path}:{key}.")
        return table.set_index("perf_time").sort_index()

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

    def _read_bytes_dataset(self, h5_dataset: h5py.Dataset) -> bytes:
        if len(h5_dataset) == 0:
            return b""
        raw = h5_dataset[0]
        if isinstance(raw, np.ndarray):
            return raw.tobytes()
        if isinstance(raw, (bytes, bytearray)):
            return bytes(raw)
        return np.asarray(raw, dtype=np.uint8).tobytes()

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
