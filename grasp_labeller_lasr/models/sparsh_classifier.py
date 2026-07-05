from __future__ import annotations

import torch
from huggingface_hub import hf_hub_download
from tactile_ssl.model import vit_base

from grasp_labeller_lasr.data.loaders import (
    DEFAULT_FINGER_NAMES,
    LIFT_END_KEY,
    LIFT_START_KEY,
)


class SparshGraspClassifier(torch.nn.Module):
    def __init__(
        self,
        encoder_name: str,
        encoder_path: str,
        device: str,
        head: torch.nn.Module,
        image_size: tuple[int, int] = (224, 224),
        finger_names: tuple[str, ...] = DEFAULT_FINGER_NAMES,
    ) -> None:
        super().__init__()

        self.device = device
        self.finger_names = finger_names

        self.encoder = vit_base(
            img_size=image_size,
            in_chans=6,
            pos_embed_fn="sinusoidal",
            num_register_tokens=1,
        ).to(device)
        self.load_encoder(encoder_name, encoder_path, device)

        self.head = head.to(device)

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
        if target_keys and target_keys[0].startswith(f"{encoder_key}.backbone."):
            encoder_key = f"{encoder_key}.backbone"

        state = {
            key.removeprefix(f"{encoder_key}."): model_state[key]
            for key in target_keys
        }
        self.encoder.load_state_dict(state, strict=False)
        self.encoder.eval()
        self.encoder.requires_grad_(False)

    def forward(self, sample: dict[str, dict[str, torch.Tensor]]) -> torch.Tensor:
        x = self._build_encoder_input(sample)
        b, n, c, h, w = x.shape

        x = x.reshape(b * n, c, h, w)
        self.encoder.eval()
        with torch.no_grad():
            features = self.encoder(x)

        features = features.reshape(b, n, *features.shape[1:])
        return self.head(features)

    def _build_encoder_input(
        self,
        sample: dict[str, dict[str, torch.Tensor]],
    ) -> torch.Tensor:
        finger_inputs = []
        for finger in self.finger_names:
            lift_start = sample[finger][LIFT_START_KEY].to(self.device)
            lift_end = sample[finger][LIFT_END_KEY].to(self.device)
            if lift_start.ndim != 4 or lift_end.ndim != 4:
                raise ValueError(
                    "SparshGraspClassifier expects batched tensors shaped "
                    "[B, C, H, W]."
                )
            finger_inputs.append(torch.cat([lift_end, lift_start], dim=1))

        return torch.stack(finger_inputs, dim=1)
