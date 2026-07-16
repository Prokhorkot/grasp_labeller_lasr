from __future__ import annotations

import torch

from grasp_labeller_lasr.data.dataset import AUDIO_KEY, IMAGES_KEY, PROPRIOCEPTORS_KEY
from grasp_labeller_lasr.data.loaders import DEFAULT_FINGER_NAMES


class GraspClassifier(torch.nn.Module):
    """Fuse image, temporal-audio, and proprioceptive grasp features."""

    def __init__(
        self,
        encoder: torch.nn.Module,
        head: torch.nn.Module,
        temporal_encoder: torch.nn.Module | None = None,
        finger_names: tuple[str, ...] = DEFAULT_FINGER_NAMES,
    ) -> None:
        super().__init__()

        self.encoder = encoder
        self.head = head
        self.temporal_encoder = temporal_encoder
        self.finger_names = tuple(finger_names)

    def forward(self, sample: dict[str, object]) -> torch.Tensor:
        images = sample[IMAGES_KEY]
        image_features = self._encode_images(images)
        audio_features = self._encode_audio(sample, image_features)
        proprioceptor_features = self._encode_proprioceptors(sample, image_features)

        feature_groups = [image_features]
        if audio_features is not None:
            feature_groups.append(audio_features)
        feature_groups.append(proprioceptor_features)
        features = torch.cat(feature_groups, dim=1)
        return self.head(features)

    def _encode_images(self, images: dict[str, dict[str, torch.Tensor]]) -> torch.Tensor:
        return torch.cat(
            [self.encoder(images[finger]) for finger in self.finger_names],
            dim=1,
        )

    def _encode_proprioceptors(
        self,
        sample: dict[str, object],
        image_features: torch.Tensor,
    ) -> torch.Tensor:
        proprioceptors = sample[PROPRIOCEPTORS_KEY]
        commanded = torch.as_tensor(
            proprioceptors["commanded"],
            device=image_features.device,
            dtype=image_features.dtype,
        )
        observed = torch.as_tensor(
            proprioceptors["observed"],
            device=image_features.device,
            dtype=image_features.dtype,
        )
        difference = commanded - observed
        return torch.cat(
            [observed.flatten(start_dim=1), difference.flatten(start_dim=1)],
            dim=1,
        )

    def _encode_audio(
        self,
        sample: dict[str, object],
        image_features: torch.Tensor,
    ) -> torch.Tensor | None:
        if self.temporal_encoder is None:
            return None
        audio = sample[AUDIO_KEY]
        sequences = torch.stack(
            [
                torch.as_tensor(
                    audio[finger],
                    device=image_features.device,
                    dtype=image_features.dtype,
                )
                for finger in self.finger_names
            ],
            dim=1,
        ).flatten(0, 1)
        encoded = self.temporal_encoder(sequences)
        return encoded.mean(dim=1).reshape(image_features.shape[0], -1)
