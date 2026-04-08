"""Shared prediction head for TimesFM downstream models."""

from __future__ import annotations

import torch
import torch.nn as nn


class PredictionHead(nn.Module):
    def __init__(
        self,
        embed_dim: int = 1280,
        hidden_dim: int = 256,
        output_dim: int = 1,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, output_dim),
        )

    def forward(self, sequence_embedding: torch.Tensor) -> torch.Tensor:
        last_patch = sequence_embedding[-1]
        return self.head(last_patch)
