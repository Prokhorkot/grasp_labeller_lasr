from __future__ import annotations

from collections.abc import Callable

import torch


PatchPooling = Callable[[torch.Tensor, torch.nn.Module | None], torch.Tensor]


def mean_patch_pooling(
    features: torch.Tensor,
    pooler: torch.nn.Module | None = None,
) -> torch.Tensor:
    if features.ndim == 3:
        return features
    if features.ndim != 4:
        raise ValueError(
            "mean_patch_pooling expects features shaped [B, N, P, D] "
            f"or [B, N, D], got {tuple(features.shape)}."
        )
    return features.mean(dim=2)


def max_patch_pooling(
    features: torch.Tensor,
    pooler: torch.nn.Module | None = None,
) -> torch.Tensor:
    if features.ndim == 3:
        return features
    if features.ndim != 4:
        raise ValueError(
            "max_patch_pooling expects features shaped [B, N, P, D] "
            f"or [B, N, D], got {tuple(features.shape)}."
        )
    return features.max(dim=2).values


def attentive_patch_pooling(
    features: torch.Tensor,
    pooler: torch.nn.Module | None = None,
) -> torch.Tensor:
    if features.ndim == 3:
        return features
    if features.ndim != 4:
        raise ValueError(
            "attentive_patch_pooling expects features shaped [B, N, P, D] "
            f"or [B, N, D], got {tuple(features.shape)}."
        )
    if pooler is None:
        raise ValueError("attentive_patch_pooling requires a pooler module.")

    batch_size, num_inputs, num_patches, embedding_dim = features.shape
    features = features.reshape(batch_size * num_inputs, num_patches, embedding_dim)
    pooled = pooler(features)
    if pooled.ndim == 3 and pooled.shape[1] == 1:
        pooled = pooled.squeeze(1)
    return pooled.reshape(batch_size, num_inputs, -1)


class MLPClassifierHead(torch.nn.Module):
    def __init__(
        self,
        num_inputs: int,
        embedding_dim: int,
        hidden_dim: int = 512,
        dropout: float = 0.2,
        output_dim: int = 1,
        patch_pooling: PatchPooling = mean_patch_pooling,
        patch_pooler: torch.nn.Module | None = None,
    ) -> None:
        super().__init__()

        input_dim = num_inputs * embedding_dim
        self.input_dim = input_dim
        self.patch_pooling = patch_pooling
        self.patch_pooler = patch_pooler
        self.net = torch.nn.Sequential(
            torch.nn.LayerNorm(input_dim),
            torch.nn.Linear(input_dim, hidden_dim),
            torch.nn.GELU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        features = self.patch_pooling(features, self.patch_pooler)
        if features.ndim != 3:
            raise ValueError(
                "MLPClassifierHead expects pooled features shaped [B, N, D], "
                f"got {tuple(features.shape)}."
            )

        features = features.flatten(start_dim=1)
        if features.shape[1] != self.input_dim:
            raise ValueError(
                f"MLPClassifierHead expected flattened feature dim {self.input_dim}, "
                f"got {features.shape[1]}."
            )

        return self.net(features)
