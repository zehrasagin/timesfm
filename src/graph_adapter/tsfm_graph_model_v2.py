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
from typing import List, Optional

from .cached_model_base import CachedTimesFMModelBase
from .node_features import NodeFeatureBuilder
from .gat_layer import GATNetwork
from .graph_structure_v2 import CorrelationGraphStructure
from .prediction_head import PredictionHead
from .simple_fusion_adapter import GatedGraphFusionAdapter


class TSFMGraphAdapterModelV2(CachedTimesFMModelBase): 
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
        corr_threshold: float = 0.25,
        corr_top_k: int = 5,
        use_absolute_corr: bool = True,
        add_self_loops: bool = True,
    ):
        super().__init__(
            timesfm_model=timesfm_model,
            target_idx=target_idx,
            max_context=max_context,
            embed_dim=embed_dim,
        )
        self.asset_names = asset_names
        self.n_assets = len(asset_names)

        self.node_feature_builder = NodeFeatureBuilder( # korelasyon, momentum, volatilite, sektör one-hot, supply chain degree gibi feature'ları üretir
            asset_names=asset_names,
            corr_window=corr_window,
        )
        self.graph_structure = CorrelationGraphStructure( # korelasyon matrisinden sparse adjacency üretir
            corr_window=corr_window,
            corr_threshold=corr_threshold,
            top_k=corr_top_k,
            use_absolute_corr=use_absolute_corr,
            add_self_loops=add_self_loops,
        )
        self.gat_network = GATNetwork( # GATv2Conv ile node feature'ları ve adjacency'yi işleyerek graph context üretir
            embed_dim=embed_dim,
            node_feature_dim=self.node_feature_builder.feature_dim,
            graph_dim=graph_dim,
            num_heads=num_gat_heads,
            num_layers=num_gat_layers,
            dropout=dropout,
        )
        self.fusion_adapter = GatedGraphFusionAdapter( # target commodity'nin sequence embedding'i ile graph embedding'ini gated fusion ile birleştirir
            embed_dim=embed_dim,
            hidden_dim=graph_dim,
            dropout=dropout,
        )
        self.prediction_head = PredictionHead( # GAT'ten gelen zenginleştirilmiş embedding'i alıp tek boyutlu tahmine çevirir
            embed_dim=embed_dim,
            hidden_dim=graph_dim,
            output_dim=1,
            dropout=dropout,
        )

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
            multi_asset_series, self.max_context # (N, P, D) boyutunda bir tensor (N: asset sayısı, P: patch sayısı, D: embedding boyutu).
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
        target_seq_emb = all_seq_emb[target_idx] # timesfmden gelen embedding
        target_graph_emb = graph_context[target_idx] # GNN den gelen graph embedding
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
        target_graph_emb = graph_context[self.target_idx] # GNN den gelen graph embedding
        enhanced_emb = self.fusion_adapter(target_seq_embeddings, target_graph_emb) # fusion adapter ile temporal embedding ve graph embedding'i birleştirir
        return self.prediction_head(enhanced_emb)
