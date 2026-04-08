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

from .gat_layer import GATNetwork
from .embedding_extractor import TimesFMEmbeddingExtractor
from .embedding_cache import EmbeddingCache
from .node_features import NodeFeatureBuilder, COMMODITY_SECTORS, SUPPLY_CHAIN_EDGES
from .cached_dataset import CachedEmbeddingDataset
from .graph_structure_v2 import CorrelationGraphStructure
from .simple_fusion_adapter import GatedGraphFusionAdapter
from .tsfm_graph_model_v2 import TSFMGraphAdapterModelV2
from .embedding_only_model import TSFMEmbeddingOnlyModel
from .prediction_head import PredictionHead
from .cached_model_base import CachedTimesFMModelBase

__all__ = [
    "GATNetwork",
    "TimesFMEmbeddingExtractor",
    "EmbeddingCache",
    "NodeFeatureBuilder",
    "CachedEmbeddingDataset",
    "COMMODITY_SECTORS",
    "SUPPLY_CHAIN_EDGES",
    "CorrelationGraphStructure",
    "GatedGraphFusionAdapter",
    "TSFMGraphAdapterModelV2",
    "TSFMEmbeddingOnlyModel",
    "PredictionHead",
    "CachedTimesFMModelBase",
]
