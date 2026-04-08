"""
Graph Attention Network (GAT) — PyTorch Geometric
===================================================

PyG'nin GATv2Conv katmanıyla production-quality GAT implementasyonu.

GATv2 (dynamic attention) vs GATv1 (static attention):
  GATv1:  e_ij = LeakyReLU(a^T · [Wh_i || Wh_j])     → query-independent
  GATv2:  e_ij = a^T · LeakyReLU(W · [h_i || h_j])    → query-dependent ✓

GATv2 daha expressive: attention score hem source hem destination'a
bağlı olarak değişir. Bu financial graph'larda önemli çünkü
aynı edge (CO1→CL1) farklı rejimde farklı ağırlık taşımalı.

Ref: Brody et al., "How Attentive are Graph Attention Networks?" ICLR 2022.

Pipeline:
  Input Projection: node_feature_dim (F) → graph_dim (256)
  GATv2 Layers: Multi-head dynamic attention over graph neighbors (×2)
  Output Projection: graph_dim → embed_dim (1280)

ÖNEMLİ: GNN'e TimesFM embedding'i GİRMEZ.
  Node feature'ları tamamen handcrafted'tır (korelasyon, volatilite,
  momentum, sektör one-hot, supply chain degree — 14-d).
  TimesFM embedding'leri SADECE Gated Fusion'da kullanılır.
"""

import torch
import torch.nn as nn
from torch_geometric.nn import GATv2Conv


class GATNetwork(nn.Module):
    """Multi-layer GATv2 with residual connections (PyG-based).

    GATv2Conv: Dynamic attention — attention score query-dependent.
    (Brody et al. 2022, statik GATv1'den kanıtlanmış şekilde daha expressive)

    Akış:
      Handcrafted node features (N, node_feature_dim)
        → Input projection → Graph space (graph_dim=256)
        → GATv2 layers × num_layers (residual + LayerNorm)
        → Output projection → Embedding space (embed_dim=1280)

    NOT: GNN'e TimesFM embedding'i GİRMEZ.
    Node feature'ları NodeFeatureBuilder tarafından üretilir
    (korelasyon, momentum, volatilite, sektör one-hot, supply chain).

    Edge weight'ler adjacency değerlerinden gelir (signed korelasyon vb.).
    GATv2Conv bunları attention hesabında ek bilgi olarak kullanır.

    Args:
        embed_dim: Çıkış embedding boyutu (1280) — cross-attention ile uyumlu.
        node_feature_dim: GNN giriş boyutu — handcrafted feature dim.
        graph_dim: GAT iç boyutu (256). Daha küçük = daha verimli.
        num_heads: GAT attention head sayısı.
        num_layers: GAT katman sayısı.
        dropout: Dropout oranı.
    """

    def __init__(
        self,
        embed_dim: int = 1280,
        node_feature_dim: int = 14,
        graph_dim: int = 256,
        num_heads: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.embed_dim = embed_dim
        self.node_feature_dim = node_feature_dim
        self.graph_dim = graph_dim
        head_dim = graph_dim // num_heads

        assert graph_dim % num_heads == 0, (
            f"graph_dim ({graph_dim}) must be divisible by "
            f"num_heads ({num_heads})"
        )

        # Input projection: node_feature_dim → graph_dim
        self.input_proj = nn.Sequential(
            nn.Linear(node_feature_dim, graph_dim),
            nn.LayerNorm(graph_dim),
            nn.GELU(),
        )

        # GATv2 katmanları (PyG)
        self.gat_layers = nn.ModuleList()
        self.layer_norms = nn.ModuleList()

        for _ in range(num_layers):
            self.gat_layers.append(
                GATv2Conv(
                    in_channels=graph_dim,
                    out_channels=head_dim,
                    heads=num_heads,
                    concat=True,           # concat heads → graph_dim
                    dropout=dropout,
                    edge_dim=1,            # edge weight desteği
                    add_self_loops=False,  # self-loop'lar adjacency'de tanımlanır
                    share_weights=False,   # GATv2 full expressiveness
                )
            )
            self.layer_norms.append(nn.LayerNorm(graph_dim))

        # Output projection: graph_dim → embed_dim
        self.output_proj = nn.Sequential(
            nn.Linear(graph_dim, embed_dim),
            nn.LayerNorm(embed_dim),
        )

        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def _build_full_edge_inputs(adj: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Create directed full-graph edges and keep adjacency as edge weights."""
        n_nodes = adj.shape[0]
        node_ids = torch.arange(n_nodes, device=adj.device)
        src = node_ids.repeat_interleave(n_nodes)
        dst = node_ids.repeat(n_nodes)
        edge_weight = adj.reshape(-1)
        keep_mask = (src != dst) | (edge_weight != 0)
        edge_index = torch.stack([src[keep_mask], dst[keep_mask]], dim=0)
        edge_attr = edge_weight[keep_mask].unsqueeze(-1)
        return edge_index, edge_attr

    def forward(
        self,
        node_features: torch.Tensor,
        adj: torch.Tensor,
    ) -> torch.Tensor:
        """GATv2 network forward pass.

        Args:
            node_features: (N, node_feature_dim) handcrafted node features.
                NodeFeatureBuilder tarafından üretilir.
            adj: (N, N) dense weighted adjacency matrix. Off-diagonal node
                pairs are kept as edges even when the current weight is zero.

        Returns:
            (N, embed_dim) graph-enhanced node representations.
        """
        # ── Dense adj → full PyG edge format ──
        # Off-diagonal pairs are always candidate edges; adjacency values are
        # passed as edge_attr so the model can learn from the current weights.
        edge_index, edge_attr = self._build_full_edge_inputs(adj)

        # ── Embed → Graph space ──
        h = self.input_proj(node_features)

        # ── GATv2 katmanları (residual + norm) ──
        for gat_layer, layer_norm in zip(self.gat_layers, self.layer_norms):
            h_new = gat_layer(h, edge_index, edge_attr=edge_attr)
            h_new = self.dropout(h_new)
            h = layer_norm(h + h_new)  # Residual + LayerNorm

        # ── Graph → Embed space ──
        h = self.output_proj(h)

        return h
