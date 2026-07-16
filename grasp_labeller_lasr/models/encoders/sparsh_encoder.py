from __future__ import annotations

from collections.abc import Callable

import torch
from huggingface_hub import hf_hub_download
from tactile_ssl.model import vit_base

from grasp_labeller_lasr.data.loaders import LIFT_END_KEY, LIFT_START_KEY


PatchPooling = Callable[[torch.Tensor, torch.nn.Module | None], torch.Tensor]


def mean_patch_pooling(
    features: torch.Tensor,
    pooler: torch.nn.Module | None = None,
) -> torch.Tensor:
    if features.ndim == 2:
        return features
    if features.ndim != 3:
        raise ValueError(
            "mean_patch_pooling expects features shaped [B, P, D] "
            f"or [B, D], got {tuple(features.shape)}."
        )
    return features.mean(dim=1)


def max_patch_pooling(
    features: torch.Tensor,
    pooler: torch.nn.Module | None = None,
) -> torch.Tensor:
    if features.ndim == 2:
        return features
    if features.ndim != 3:
        raise ValueError(
            "max_patch_pooling expects features shaped [B, P, D] "
            f"or [B, D], got {tuple(features.shape)}."
        )
    return features.max(dim=1).values


def attentive_patch_pooling(
    features: torch.Tensor,
    pooler: torch.nn.Module | None = None,
) -> torch.Tensor:
    if features.ndim == 2:
        return features
    if features.ndim != 3:
        raise ValueError(
            "attentive_patch_pooling expects features shaped [B, P, D] "
            f"or [B, D], got {tuple(features.shape)}."
        )
    if pooler is None:
        raise ValueError("attentive_patch_pooling requires a pooler module.")

    pooled = pooler(features)
    if pooled.ndim == 3 and pooled.shape[1] == 1:
        pooled = pooled.squeeze(1)
    return pooled


class SparshEncoder(torch.nn.Module):
    """Encode the lift-start and lift-end images from one tactile finger."""

    def __init__(
        self,
        encoder_name: str,
        encoder_path: str,
        device: str,
        image_size: tuple[int, int] = (224, 224),
        patch_pooling: PatchPooling = mean_patch_pooling,
        patch_pooler: torch.nn.Module | None = None,
    ) -> None:
        super().__init__()

        self.device = device
        self.patch_pooling = patch_pooling
        self.patch_pooler = patch_pooler

        self.encoder = vit_base(
            img_size=image_size,
            in_chans=6,
            pos_embed_fn="sinusoidal",
            num_register_tokens=1,
        ).to(device)
        self.embedding_dim = int(self.encoder.embed_dim)
        self.load_encoder(encoder_name, encoder_path, device)

    def load_encoder(self, encoder_name: str, encoder_path: str, device: str) -> None:
        ckpt_path = hf_hub_download(
            repo_id=encoder_name,
            filename=encoder_path,
        )

        ckpt = torch.load(ckpt_path, map_location=device)

        model_state = ckpt["model"]
        encoder_key = "target_encoder"
        target_keys = [key for key in model_state if key.startswith(f"{encoder_key}.")]
        if not target_keys:
            raise ValueError(f"No {encoder_key!r} weights found in {ckpt_path}.")
        if target_keys[0].startswith(f"{encoder_key}.backbone."):
            encoder_key = f"{encoder_key}.backbone"

        state = {
            key.removeprefix(f"{encoder_key}."): model_state[key]
            for key in target_keys
        }
        self.encoder.load_state_dict(state, strict=False)
        self.encoder.eval()
        self.encoder.requires_grad_(False)

    def forward(
        self,
        images: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        x = self._build_encoder_input(images)
        self.encoder.eval()
        with torch.no_grad():
            features = self.encoder(x)
        return self.patch_pooling(features, self.patch_pooler)

    def _build_encoder_input(
        self,
        images: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        lift_start = images[LIFT_START_KEY].to(self.device)
        lift_end = images[LIFT_END_KEY].to(self.device)
        if lift_start.ndim != 4 or lift_end.ndim != 4:
            raise ValueError(
                "SparshEncoder expects batched tensors shaped [B, C, H, W]."
            )
        return torch.cat([lift_end, lift_start], dim=1)
