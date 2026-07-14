from __future__ import annotations

import torch


class MLPClassifierHead(torch.nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 512,
        dropout: float = 0.2,
        output_dim: int = 1,
    ) -> None:
        super().__init__()

        self.input_dim = input_dim
        self.net = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dim),
            torch.nn.LayerNorm(hidden_dim),
            torch.nn.GELU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim != 2:
            raise ValueError(
                "MLPClassifierHead expects flattened features shaped [B, D], "
                f"got {tuple(features.shape)}."
            )

        if features.shape[1] != self.input_dim:
            raise ValueError(
                f"MLPClassifierHead expected feature dim {self.input_dim}, "
                f"got {features.shape[1]}."
            )

        return self.net(features)
