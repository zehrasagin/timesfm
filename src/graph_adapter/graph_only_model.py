"""
Graph-Only Baseline — TSFM-Graph Adapter
========================================

Handcrafted node features + weighted correlation graph + GAT kullanır,
TimesFM temporal embedding branch'ini hiç kullanmaz.

Amaç:
- Graph branch'in tek başına ne kadar sinyal taşıdığını ölçmek
- Embedding-only ve fused total mimariyle aynı training pipeline'da kıyaslamak
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn

from .gat_layer import GATNetwork
from .graph_structure_v2 import CorrelationGraphStructure
from .node_features import NodeFeatureBuilder
from .prediction_head import PredictionHead


class TSFMGraphOnlyModel(nn.Module):
    """TimesFM embedding kullanmayan graph-only baseline."""

    uses_temporal_embeddings = False

    def __init__(
        self,
        asset_names: List[str],
        target_idx: int = 0,
        max_context: int = 1024,
        embed_dim: int = 1280,
        graph_dim: int = 256,
        num_gat_heads: int = 4,
        num_gat_layers: int = 2,
        dropout: float = 0.1,
        corr_window: int = 60,
        use_absolute_corr: bool = True,
        add_self_loops: bool = True,
    ):
        super().__init__()
        self.asset_names = asset_names
        self.n_assets = len(asset_names)
        self.target_idx = target_idx
        self.max_context = max_context
        self.embed_dim = embed_dim

        self.node_feature_builder = NodeFeatureBuilder(
            asset_names=asset_names,
            corr_window=corr_window,
        )
        self.graph_structure = CorrelationGraphStructure(
            corr_window=corr_window,
            use_absolute_corr=use_absolute_corr,
            add_self_loops=add_self_loops,
        )
        self.gat_network = GATNetwork(
            embed_dim=embed_dim,
            node_feature_dim=self.node_feature_builder.feature_dim,
            graph_dim=graph_dim,
            num_heads=num_gat_heads,
            num_layers=num_gat_layers,
            dropout=dropout,
        )
        self.prediction_head = PredictionHead(
            embed_dim=embed_dim,
            hidden_dim=graph_dim,
            output_dim=1,
            dropout=dropout,
        )

    def _device(self) -> torch.device:
        return next(self.parameters()).device

    def _build_price_history(self, multi_asset_series: List[np.ndarray]) -> np.ndarray:
        return np.column_stack([series[-self.max_context:] for series in multi_asset_series])

    def get_trainable_params(self) -> List[nn.Parameter]:
        return [p for p in self.parameters() if p.requires_grad]

    def count_parameters(self) -> Dict[str, float]:
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        frozen = sum(p.numel() for p in self.parameters() if not p.requires_grad)
        total = trainable + frozen
        return {
            "trainable": trainable,
            "frozen": frozen,
            "total": total,
            "trainable_pct": 100.0 * trainable / max(total, 1),
        }

    def forward(
        self,
        multi_asset_series: List[np.ndarray],
        price_history: Optional[np.ndarray] = None,
        target_idx: Optional[int] = None,
    ) -> torch.Tensor:
        if target_idx is None:
            target_idx = self.target_idx
        if price_history is None:
            price_history = self._build_price_history(multi_asset_series)

        node_feats = self.node_feature_builder.build_tensor(
            price_history, device=self._device()
        )
        adj = self.graph_structure(price_history, device=self._device())
        graph_context = self.gat_network(node_feats, adj)
        target_graph_emb = graph_context[target_idx]
        return self.prediction_head(target_graph_emb.unsqueeze(0))

    def forward_with_embeddings(
        self,
        target_seq_embeddings: Optional[torch.Tensor] = None,
        price_history: Optional[np.ndarray] = None,
    ) -> torch.Tensor:
        del target_seq_embeddings
        if price_history is None:
            raise ValueError("Graph-only forward requires price_history.")

        node_feats = self.node_feature_builder.build_tensor(
            price_history, device=self._device()
        )
        adj = self.graph_structure(price_history, device=self._device())
        graph_context = self.gat_network(node_feats, adj)
        target_graph_emb = graph_context[self.target_idx]
        return self.prediction_head(target_graph_emb.unsqueeze(0))
