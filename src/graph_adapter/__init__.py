"""
TSFM-Graph Adapter Architecture
================================

Frozen TimesFM backbone üzerine Graph Neural Network (GAT) ve
gated fusion adapter ekleyerek multi-asset zaman serisi
tahminini graph-aware hale getiren mimari.

Mimari:
  Stream A (Frozen): TimesFM → Patch Embeddings E_i
  Stream B (Trainable): Weighted Correlation Graph → GAT → Cross-sectional context H_i
  Fusion: E'_i = E_i + Adapter(E_i, H_i)
  Output: Prediction Head → Forecast ŷ

Trainable parameters: ~5% (GNN + Adapter + Head)
Frozen: TimesFM Backbone (~95%)
"""

import importlib.util

from .gat_layer import GATNetwork
from .embedding_extractor import TimesFMEmbeddingExtractor
from .embedding_store import TorchEmbeddingStore, EmbeddingStore
from .node_features import NodeFeatureBuilder, COMMODITY_SECTORS, SUPPLY_CHAIN_EDGES
from .embedding_dataset import EmbeddingPositionDataset
from .graph_structure_v2 import CorrelationGraphStructure
from .simple_fusion_adapter import GatedGraphFusionAdapter
from .tsfm_graph_model_v2 import TSFMGraphAdapterModelV2
from .embedding_only_model import TSFMEmbeddingOnlyModel
from .graph_only_model import TSFMGraphOnlyModel
from .prediction_head import PredictionHead
from .timesfm_model_base import TimesFMDownstreamModelBase

LIGHTNING_AVAILABLE = (
    importlib.util.find_spec("lightning") is not None
    or importlib.util.find_spec("pytorch_lightning") is not None
)


def __getattr__(name: str):
    if name == "EmbeddingStoreLightningModule":
        from .lightning_module import EmbeddingStoreLightningModule

        return EmbeddingStoreLightningModule
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "GATNetwork",
    "TimesFMEmbeddingExtractor",
    "TorchEmbeddingStore",
    "EmbeddingStore",
    "NodeFeatureBuilder",
    "EmbeddingPositionDataset",
    "COMMODITY_SECTORS",
    "SUPPLY_CHAIN_EDGES",
    "CorrelationGraphStructure",
    "GatedGraphFusionAdapter",
    "TSFMGraphAdapterModelV2",
    "TSFMEmbeddingOnlyModel",
    "TSFMGraphOnlyModel",
    "PredictionHead",
    "TimesFMDownstreamModelBase",
    "EmbeddingStoreLightningModule",
    "LIGHTNING_AVAILABLE",
]
