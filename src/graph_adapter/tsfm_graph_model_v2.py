"""
Simplified TSFM-Graph Adapter Model (V2)
=========================================

Branch A: Frozen TimesFM backbone → temporal embeddings
Branch B: Handcrafted node features + sparse correlation graph + GAT
Fusion:   Gated late fusion (target commodity only)
Output:   Prediction head → scalar forecast

V1'den farklar:
  - LearnedAdjacency + α kaldırıldı → sadece korelasyon bazlı sparse graph
  - CrossAttention → GatedFusion (gradient doğal akar, zero-init sorunu yok)
  - ~%35 daha az parametre → overfitting azalır
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Optional

from .embedding_extractor import TimesFMEmbeddingExtractor
from .node_features import NodeFeatureBuilder
from .gat_layer import GATNetwork
from .graph_structure_v2 import CorrelationGraphStructure
from .simple_fusion_adapter import GatedGraphFusionAdapter


class PredictionHead(nn.Module):
    def __init__(self, embed_dim: int = 1280, hidden_dim: int = 256, output_dim: int = 1, dropout: float = 0.1):
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

    def forward(self, enhanced_embedding: torch.Tensor) -> torch.Tensor:
        last_patch = enhanced_embedding[-1]
        return self.head(last_patch)


class TSFMGraphAdapterModelV2(nn.Module):
    def __init__(
        self,
        timesfm_model,
        asset_names: List[str],
        target_idx: int = 0,
        max_context: int = 1024,
        embed_dim: int = 1280,
        graph_dim: int = 256,
        num_gat_heads: int = 4,
        num_gat_layers: int = 2,
        dropout: float = 0.1,
        corr_window: int = 60,
        corr_threshold: float = 0.35,
        corr_top_k: int = 3,
    ):
        super().__init__()
        self.asset_names = asset_names
        self.n_assets = len(asset_names)
        self.target_idx = target_idx
        self.max_context = max_context
        self.embed_dim = embed_dim

        self.embedding_extractor = TimesFMEmbeddingExtractor(timesfm_model)
        self.node_feature_builder = NodeFeatureBuilder(
            asset_names=asset_names,
            corr_window=corr_window,
        )
        self.graph_structure = CorrelationGraphStructure(
            corr_window=corr_window,
            corr_threshold=corr_threshold,
            top_k=corr_top_k,
            use_absolute_corr=True,
            add_self_loops=False,
        )
        self.gat_network = GATNetwork(
            embed_dim=embed_dim,
            node_feature_dim=self.node_feature_builder.feature_dim,
            graph_dim=graph_dim,
            num_heads=num_gat_heads,
            num_layers=num_gat_layers,
            dropout=dropout,
        )
        self.fusion_adapter = GatedGraphFusionAdapter(
            embed_dim=embed_dim,
            hidden_dim=graph_dim,
            dropout=dropout,
        )
        self.prediction_head = PredictionHead(
            embed_dim=embed_dim,
            hidden_dim=graph_dim,
            output_dim=1,
            dropout=dropout,
        )

    def get_trainable_params(self) -> List[nn.Parameter]:
        """Sadece trainable parametreleri döndür (backbone hariç)."""
        return [p for n, p in self.named_parameters()
                if "embedding_extractor" not in n]

    def count_parameters(self) -> Dict[str, int]:
        """Parametre sayımı: trainable vs frozen."""
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

    def _build_price_history(self, multi_asset_series: List[np.ndarray]) -> np.ndarray:
        return np.column_stack([series[-self.max_context:] for series in multi_asset_series])

    def forward(
        self,
        multi_asset_series: List[np.ndarray],
        price_history: Optional[np.ndarray] = None,
        target_idx: Optional[int] = None,
    ) -> torch.Tensor:
        if target_idx is None:
            target_idx = self.target_idx

        # Branch A: Frozen TimesFM
        all_seq_emb = self.embedding_extractor.extract_embeddings(
            multi_asset_series, self.max_context
        )

        # Branch B: handcrafted node features + sparse corr graph
        if price_history is None:
            price_history = self._build_price_history(multi_asset_series)

        node_feats = self.node_feature_builder.build_tensor(
            price_history, device=all_seq_emb.device
        )
        adj = self.graph_structure(price_history, device=all_seq_emb.device)
        graph_context = self.gat_network(node_feats, adj)

        # Fusion sadece target commodity için
        target_seq_emb = all_seq_emb[target_idx]
        target_graph_emb = graph_context[target_idx]
        enhanced_emb = self.fusion_adapter(target_seq_emb, target_graph_emb)

        return self.prediction_head(enhanced_emb)

    def forward_cached(
        self,
        target_seq_embeddings: torch.Tensor,
        price_history: np.ndarray,
    ) -> torch.Tensor:
        """Cache'den gelen embedding'lerle forward (TimesFM çalışmaz)."""
        node_feats = self.node_feature_builder.build_tensor(
            price_history, device=target_seq_embeddings.device
        )
        adj = self.graph_structure(price_history, device=target_seq_embeddings.device)
        graph_context = self.gat_network(node_feats, adj)
        target_graph_emb = graph_context[self.target_idx]
        enhanced_emb = self.fusion_adapter(target_seq_embeddings, target_graph_emb)
        return self.prediction_head(enhanced_emb)
