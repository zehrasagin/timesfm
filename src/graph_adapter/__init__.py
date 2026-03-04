"""
TSFM-Graph Adapter Architecture
================================

Frozen TimesFM backbone üzerine Graph Neural Network (GAT) ve
Cross-Attention Adapter ekleyerek multi-asset zaman serisi
tahminini graph-aware hale getiren mimari.

Mimari:
  Stream A (Frozen): TimesFM → Patch Embeddings E_i
  Stream B (Trainable): Graph Learner → GAT → Cross-sectional context H_i
  Fusion: E'_i = E_i + Adapter(E_i, H_i)
  Output: Prediction Head → Forecast ŷ

Trainable parameters: ~5% (GNN + Adapter + Head)
Frozen: TimesFM Backbone (~95%)
"""

from .graph_structure import (
    CorrelationAdjacency,
    SectorAdjacency,
    SupplyChainAdjacency,
    LearnedAdjacency,
    HybridGraphStructure,
    COMMODITY_SECTORS,
    SUPPLY_CHAIN_EDGES,
)
from .gat_layer import GATNetwork
from .cross_attention_adapter import CrossAttentionAdapter
from .embedding_extractor import TimesFMEmbeddingExtractor
from .embedding_cache import EmbeddingCache
from .node_features import NodeFeatureBuilder
from .tsfm_graph_model import TSFMGraphAdapterModel, PredictionHead
from .dataset import MultiAssetDataset
from .cached_dataset import CachedEmbeddingDataset

__all__ = [
    "CorrelationAdjacency",
    "SectorAdjacency",
    "SupplyChainAdjacency",
    "LearnedAdjacency",
    "HybridGraphStructure",
    "GATNetwork",
    "CrossAttentionAdapter",
    "TimesFMEmbeddingExtractor",
    "EmbeddingCache",
    "NodeFeatureBuilder",
    "TSFMGraphAdapterModel",
    "PredictionHead",
    "MultiAssetDataset",
    "CachedEmbeddingDataset",
    "COMMODITY_SECTORS",
    "SUPPLY_CHAIN_EDGES",
]
