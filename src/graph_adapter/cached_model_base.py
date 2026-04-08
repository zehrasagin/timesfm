"""Shared base for cached TimesFM downstream models."""

from __future__ import annotations

import torch.nn as nn
from typing import Dict, List

from .embedding_extractor import TimesFMEmbeddingExtractor


class CachedTimesFMModelBase(nn.Module):
    """Common utilities for graph and embedding-only cached models."""

    def __init__(
        self,
        timesfm_model,
        target_idx: int = 0,
        max_context: int = 1024,
        embed_dim: int = 1280,
    ):
        super().__init__()
        self.target_idx = target_idx
        self.max_context = max_context
        self.embed_dim = embed_dim
        self.embedding_extractor = TimesFMEmbeddingExtractor(timesfm_model)

    def get_trainable_params(self) -> List[nn.Parameter]:
        return [
            p for n, p in self.named_parameters()
            if "embedding_extractor" not in n
        ]

    def count_parameters(self) -> Dict[str, int]:
        trainable = sum(
            p.numel() for n, p in self.named_parameters()
            if "embedding_extractor" not in n and p.requires_grad
        )
        frozen = sum(p.numel() for p in self.embedding_extractor.module.parameters())
        total = trainable + frozen
        return {
            "trainable": trainable,
            "frozen": frozen,
            "total": total,
            "trainable_pct": 100.0 * trainable / max(total, 1),
        }
