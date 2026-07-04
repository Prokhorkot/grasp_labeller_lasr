from __future__ import annotations

from typing import Callable

import cv2
import numpy as np

from grasp_labeller_lasr.data.loaders import BACKGROUND_KEY, LIFT_END_KEY, LIFT_START_KEY


class RemoveBackground:
    """Subtract a background image without uint8 wraparound."""

    def __call__(
        self,
        image: np.ndarray,
        background: np.ndarray | None = None,
    ) -> np.ndarray:
        if background is None:
            if not isinstance(image, tuple):
                raise ValueError("RemoveBackground needs an image and a background.")
            image, background = image

        if image.shape != background.shape:
            raise ValueError(
                f"Image shape {image.shape} does not match background shape "
                f"{background.shape}."
            )

        return image.astype(np.float32) - background.astype(np.float32)


class Cv2Resize:
    """Resize one HWC image with OpenCV.

    OpenCV expects size as (width, height), matching cv2.resize.
    """

    def __init__(self, size: tuple[int, int], interpolation: int = cv2.INTER_AREA):
        self.size = size
        self.interpolation = interpolation

    def __call__(self, image: np.ndarray) -> np.ndarray:
        return cv2.resize(image, self.size, interpolation=self.interpolation)
    

class Identity:
    def __call__(
        self,
        image: np.ndarray,
        background: np.ndarray | None = None,
    ) -> np.ndarray:
        if background is None and isinstance(image, tuple):
            image, _ = image

        return image


class MultiFingerTransform:
    """Apply an image transform to selected contact phases for each finger."""

    def __init__(
        self,
        transform: Callable,
        phase_keys: tuple[str, ...] = (LIFT_START_KEY, LIFT_END_KEY),
        background_key: str = BACKGROUND_KEY,
    ) -> None:
        self.transform = transform
        self.phase_keys = phase_keys
        self.background_key = background_key

    def __call__(self, sample: dict) -> dict:
        transformed = {}
        for finger, images in sample.items():
            background = images[self.background_key]
            transformed[finger] = {
                phase_key: self.transform((images[phase_key], background))
                for phase_key in self.phase_keys
            }

        return transformed
