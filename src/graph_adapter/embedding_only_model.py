"""
Embedding-Only Baseline — TSFM-Graph Adapter
============================================

Frozen TimesFM backbone'dan gelen target sequence embedding'lerini alır,
graph branch kullanmadan doğrudan prediction head'e verir.

Amaç:
- Graph branch'in gerçekten ek değer katıp katmadığını ölçmek
- Aynı Torch embedding-store training pipeline üzerinde temiz bir baseline sağlamak
"""

from __future__ import annotations

import numpy as np
import torch
from typing import List, Optional

from .timesfm_model_base import TimesFMDownstreamModelBase
from .prediction_head import PredictionHead


class TSFMEmbeddingOnlyModel(TimesFMDownstreamModelBase):
    """Graph'siz embedding-only baseline.

    Interface'i TSFMGraphAdapterModelV2 ile uyumludur; böylece aynı embedding-store
    training ve rolling forecast fonksiyonlarında kullanılabilir.
    """

    def __init__(
        self,
        timesfm_model,
        target_idx: int = 0,
        max_context: int = 1024,
        embed_dim: int = 1280,
        hidden_dim: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__(
            timesfm_model=timesfm_model,
            target_idx=target_idx,
            max_context=max_context,
            embed_dim=embed_dim,
        )
        self.prediction_head = PredictionHead(
            embed_dim=embed_dim,
            hidden_dim=hidden_dim,
            output_dim=1,
            dropout=dropout,
        )

    def forward(
        self,
        multi_asset_series: List[np.ndarray],
        price_history: Optional[np.ndarray] = None,
        target_idx: Optional[int] = None,
    ) -> torch.Tensor:
        del price_history
        if target_idx is None:
            target_idx = self.target_idx

        all_seq_emb = self.embedding_extractor.extract_embeddings(
            multi_asset_series, self.max_context
        )
        target_seq_emb = all_seq_emb[target_idx]
        return self.prediction_head(target_seq_emb)

    def forward_with_embeddings(
        self,
        target_seq_embeddings: torch.Tensor,
        price_history: Optional[np.ndarray] = None,
    ) -> torch.Tensor:
        del price_history
        return self.prediction_head(target_seq_embeddings)
