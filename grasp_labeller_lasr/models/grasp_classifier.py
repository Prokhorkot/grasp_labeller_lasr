from __future__ import annotations

import torch

from grasp_labeller_lasr.data.dataset import IMAGES_KEY, PROPRIOCEPTORS_KEY


class GraspClassifier(torch.nn.Module):
    """Compose a grasp feature encoder with a classification head."""

    def __init__(
        self,
        encoder: torch.nn.Module,
        head: torch.nn.Module,
    ) -> None:
        super().__init__()

        self.encoder = encoder
        self.head = head

    def forward(self, sample: dict[str, object]) -> torch.Tensor:
        images = sample[IMAGES_KEY]
        image_features = self.encoder(images).flatten(start_dim=1)

        proprioceptors = sample[PROPRIOCEPTORS_KEY]
        proprio_commanded = torch.as_tensor(
            proprioceptors["commanded"],
            device=image_features.device,
            dtype=image_features.dtype,
        )
        proprio_observed = torch.as_tensor(
            proprioceptors["observed"],
            device=image_features.device,
            dtype=image_features.dtype,
        )
        if proprio_commanded.shape != proprio_observed.shape:
            raise ValueError(
                "Commanded and observed proprioceptors must have the same shape, "
                f"got {tuple(proprio_commanded.shape)} and "
                f"{tuple(proprio_observed.shape)}."
            )
        proprio_difference = proprio_commanded - proprio_observed
        proprio_difference = proprio_difference.flatten(start_dim=1)

        if proprio_difference.shape[0] != image_features.shape[0]:
            raise ValueError(
                "Image and proprioceptor features must have the same batch size."
            )

        features = torch.cat([image_features, proprio_difference], dim=1)
        return self.head(features)
