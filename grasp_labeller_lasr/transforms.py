from __future__ import annotations

from collections.abc import Iterable
from typing import Callable

import cv2
import numpy as np


class RemoveBackground:
    """Subtract a background image without uint8 wraparound."""

    def __call__(
        self,
        image: np.ndarray | tuple[np.ndarray, np.ndarray],
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
        image: np.ndarray | tuple[np.ndarray, np.ndarray],
        background: np.ndarray | None = None,
    ) -> np.ndarray:
        if background is None and isinstance(image, tuple):
            image, _ = image

        return image


class MultiFingerTransform:
    """Apply an image transform to every selected image in a tactile sample."""

    def __init__(
        self,
        transform: Callable,
        image_keys: Iterable[str] | None = None,
        background_key: str = "background",
    ) -> None:
        self.transform = transform
        self.image_keys = tuple(image_keys) if image_keys is not None else None
        self.background_key = background_key

    def __call__(self, sample: dict) -> dict:
        transformed = {
            finger: {
                key: image
                for key, image in images.items()
            }
            for finger, images in sample.items()
        }
        for finger, images in sample.items():
            for key, image in images.items():
                if self.image_keys is None or key in self.image_keys:
                    image = (image, images[self.background_key])
                    transformed[finger][key] = self.transform(image)

        return transformed
