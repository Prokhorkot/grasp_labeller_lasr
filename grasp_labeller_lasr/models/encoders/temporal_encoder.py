import torch
from torch import nn
from torch.nn import functional as F


def _resample_positional(pos_embed: torch.Tensor, target_len: int) -> torch.Tensor:
    # pos_embed: [1, L, D]
    if int(target_len) <= int(pos_embed.shape[1]):
        return pos_embed[:, : int(target_len), :]
    return F.interpolate(
        pos_embed.transpose(1, 2),
        size=int(target_len),
        mode="linear",
        align_corners=False,
    ).transpose(1, 2)


class TemporalSequenceEncoder(nn.Module):
    def __init__(
        self,
        *,
        in_dim: int,
        dim: int,
        n_layers: int,
        n_heads: int,
        dropout: float,
        max_len: int,
    ) -> None:
        super().__init__()
        self.dim = int(dim)
        self.in_proj = nn.Linear(int(in_dim), self.dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, max(8, int(max_len)), self.dim))
        layer = nn.TransformerEncoderLayer(
            d_model=self.dim,
            nhead=max(1, int(n_heads)),
            dim_feedforward=self.dim * 4,
            dropout=float(dropout),
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.temporal = nn.TransformerEncoder(layer, num_layers=max(1, int(n_layers)))
        self.norm = nn.LayerNorm(self.dim)

    def forward(self, seq: torch.Tensor) -> torch.Tensor:
        # seq: [N, L, C]
        if seq.ndim == 2:
            seq = seq.unsqueeze(-1)
        _, l, _ = seq.shape
        x = self.in_proj(seq)
        x = x + _resample_positional(self.pos_embed, l)
        x = self.temporal(x)
        return self.norm(x)
