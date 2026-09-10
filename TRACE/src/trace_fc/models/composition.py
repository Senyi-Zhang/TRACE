"""PMA and FiLM modules for bottom-up tree composition."""

from __future__ import annotations

import torch
from torch import nn


class PMA(nn.Module):
    """Pooling by multi-head attention with one learnable seed."""

    def __init__(self, hidden_size: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        if hidden_size % num_heads:
            raise ValueError("hidden_size must be divisible by num_heads")
        self.seed = nn.Parameter(torch.randn(1, hidden_size))
        self.attention = nn.MultiheadAttention(
            hidden_size, num_heads, dropout=dropout, batch_first=True,
        )
        self.attention_norm = nn.LayerNorm(hidden_size)
        self.feed_forward = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden_size * 4, hidden_size),
        )
        self.output_norm = nn.LayerNorm(hidden_size)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        if values.ndim != 2 or values.shape[0] == 0:
            raise ValueError("PMA expects a non-empty [nodes, hidden] tensor")
        values = values.unsqueeze(0)
        seed = self.seed.unsqueeze(0)
        attended, _ = self.attention(seed, values, values, need_weights=False)
        hidden = self.attention_norm(seed + attended)
        hidden = self.output_norm(hidden + self.feed_forward(hidden))
        return hidden[0, 0]


class FiLM(nn.Module):
    """Equation (6-7): child-conditioned feature-wise modulation."""

    def __init__(self, hidden_size: int, dropout: float = 0.1):
        super().__init__()
        self.gamma_beta = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 2), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden_size * 2, hidden_size * 2),
        )

    def forward(self, parent: torch.Tensor, child_summary: torch.Tensor) -> torch.Tensor:
        delta_gamma, beta = self.gamma_beta(child_summary).chunk(2, dim=-1)
        gamma = 1.0 + torch.tanh(delta_gamma)
        return gamma * parent + beta

