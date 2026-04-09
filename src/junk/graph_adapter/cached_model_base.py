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
        max_context: int = 1024, # kaç günlük geçmişe bakılacağı
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
            if "embedding_extractor" not in n and p.requires_grad # timesfm backbone parametreleri hariç 
        )
        frozen = sum(p.numel() for p in self.embedding_extractor.module.parameters())
        total = trainable + frozen
        return {
            "trainable": trainable,
            "frozen": frozen,
            "total": total,
            "trainable_pct": 100.0 * trainable / max(total, 1),
        }
    
"""
Bir parametrenin trainable (eğitilebilir) olup olmadığını, PyTorch’ta requires_grad özelliği ile anlar:

Eğer bir parametrenin requires_grad=True ise, bu parametre eğitim sırasında gradient alır ve güncellenir (trainable).
Eğer requires_grad=False ise, bu parametre dondurulmuştur (ör. TimesFM backbone gibi) ve eğitim sırasında değişmez (frozen).

"""