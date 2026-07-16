from __future__ import annotations

from typing import Any, Callable, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


IMAGES_KEY = "images"
PROPRIOCEPTORS_KEY = "proprioceptors"
AUDIO_KEY = "audio"
OPENTOUCH_KEY = "opentouch"

DEFAULT_AUDIO_SEQUENCE_LENGTH = 224
DEFAULT_AUDIO_FREQUENCY_BINS = 256
STFT_N_FFT = 512
STFT_HOP_LENGTH = 128


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
        "audio": {
            "thumb": audio_features,
            ...
        },
    }

    Images are RGB HWC numpy arrays before transforms. The label is loaded from
    grasp/grasp_label.csv using label_method, usually "manual". The ``audio``
    key is present when the loader was created with ``load_audio=True``.

    Loaders provide raw PCM arrays for the lifting portion of each finger's
    ``audio`` data. This dataset converts those arrays to normalized log-STFT
    features before returning the sample.
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
        audio_sequence_length: int = DEFAULT_AUDIO_SEQUENCE_LENGTH,
        audio_frequency_bins: int = DEFAULT_AUDIO_FREQUENCY_BINS,
    ) -> None:
        if n_aug_copies < 0:
            raise ValueError("n_aug_copies must be non-negative.")

        self.iteration_paths = list(iteration_paths)
        self.train_transform = train_transform
        self.val_transform = val_transform
        self.train = train
        self.include_original = include_original
        self.loader = loader
        self.audio_sequence_length = int(audio_sequence_length)
        self.audio_frequency_bins = int(audio_frequency_bins)

        if self.audio_sequence_length <= 0:
            raise ValueError("audio_sequence_length must be positive.")
        if self.audio_frequency_bins <= 0:
            raise ValueError("audio_frequency_bins must be positive.")

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
        sample = self._prepare_opentouch(sample)
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

    def _prepare_opentouch(self, sample: dict[str, Any]) -> dict[str, Any]:
        """Normalize a loader's per-finger tactile data into the dataset schema."""
        try:
            opentouch = sample.pop(OPENTOUCH_KEY)
        except KeyError as exc:
            raise KeyError(f"Loaded sample is missing {OPENTOUCH_KEY!r} data.") from exc

        images = {}
        audio = {}
        for finger, finger_sample in opentouch.items():
            try:
                images[finger] = finger_sample[IMAGES_KEY]
            except KeyError as exc:
                raise KeyError(
                    f"Loaded sample is missing image data for finger {finger!r}."
                ) from exc

            raw_audio = finger_sample.get(AUDIO_KEY)
            if raw_audio is not None:
                audio[finger] = self._log_stft_features(raw_audio)

        sample[IMAGES_KEY] = images
        if audio:
            sample[AUDIO_KEY] = audio
        return sample

    def _log_stft_features(self, raw_waveform: np.ndarray) -> np.ndarray:
        """Normalize a raw PCM waveform and compute [time, frequency] features."""
        waveform = np.asarray(raw_waveform)
        if waveform.size == 0:
            raise ValueError("Cannot compute an STFT from an empty waveform.")
        if waveform.ndim == 2:
            waveform = waveform.mean(axis=1)
        elif waveform.ndim != 1:
            raise ValueError(
                "Expected a one- or two-dimensional raw WAV sample array, got "
                f"shape {waveform.shape}."
            )
        waveform = waveform.astype(np.float32) / 32768.0
        if waveform.size < STFT_N_FFT:
            waveform = np.pad(
                waveform,
                (0, STFT_N_FFT - waveform.size),
                mode="constant",
            )

        wav_tensor = torch.from_numpy(waveform)
        window = torch.hann_window(STFT_N_FFT, dtype=torch.float32)
        spectrum = torch.stft(
            wav_tensor,
            n_fft=STFT_N_FFT,
            hop_length=STFT_HOP_LENGTH,
            win_length=STFT_N_FFT,
            window=window,
            return_complex=True,
        )
        log_magnitude = torch.abs(spectrum).clamp_min(1e-6).log()
        resized = F.interpolate(
            log_magnitude[None, None],
            size=(self.audio_frequency_bins, self.audio_sequence_length),
            mode="bilinear",
            align_corners=False,
        )[0, 0]
        features = resized.transpose(0, 1)
        features = (features - features.mean()) / features.std().clamp_min(1e-6)
        return features.cpu().numpy().astype(np.float32)
