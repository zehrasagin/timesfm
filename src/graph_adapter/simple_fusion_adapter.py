"""
Simple Graph-to-Temporal Fusion Adapter
=======================================

Amaç:
- Graph branch çıktısını TimesFM temporal embedding'ine enjekte etmek
- Bunu cross-attention'dan daha sade ve stabil yapmak

Form:
    E'_i = E_i + gate(E_i, h_i) * proj(h_i)

Burada:
- E_i: (P, D) target commodity temporal embedding'i
- h_i: (D,) target commodity graph embedding'i
- gate: graph bilgisinin ne kadar enjekte edileceğini öğrenir
"""

from __future__ import annotations

import torch
import torch.nn as nn


class GatedGraphFusionAdapter(nn.Module):
    def __init__(self, embed_dim: int = 1280, hidden_dim: int = 256, dropout: float = 0.1):
        super().__init__()
        self.graph_proj = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.gate_net = nn.Sequential(
            nn.Linear(embed_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
            nn.Sigmoid(),
        )
        self.out_norm = nn.LayerNorm(embed_dim)

    def forward(self, temporal_embedding: torch.Tensor, graph_embedding: torch.Tensor) -> torch.Tensor:
        # temporal_embedding: (P, D)
        # graph_embedding: (D,)
        pooled_temporal = temporal_embedding.mean(dim=0)  # (D,)
        gate = self.gate_net(torch.cat([pooled_temporal, graph_embedding], dim=-1))  # (D,)
        graph_token = self.graph_proj(graph_embedding).unsqueeze(0)  # (1, D)
        fused = temporal_embedding + gate.unsqueeze(0) * graph_token
        return self.out_norm(fused)
