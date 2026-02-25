"""
Graph Attention Network (GAT) — TSFM-Graph Adapter
====================================================

Her düğüm (asset) için komşularının bilgisini attention mekanizmasıyla
toplayarak cross-sectional context üretir.

H_t = GAT(E_{1,t}, ..., E_{n,t}; A_t)

"CO1'in embedding'ini hesaplarken, bağlı olduğu CL1, HO1, NG1'in
embedding'lerini de attention ile ağırlıklandırarak topla."

Yapı:
  Input Projection: 1280 → graph_dim (256)
  GAT Layers: Multi-head attention over graph neighbors (×2)
  Output Projection: graph_dim → 1280
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ═══════════════════════════════════════════════════════════════════════════════
# GAT LAYER (Tek Katman)
# ═══════════════════════════════════════════════════════════════════════════════
class GATLayer(nn.Module):
    """Tek Graph Attention katmanı.

    Her düğüm i için:
      1. Feature projection: h_i = W · x_i
      2. Attention score:    e_ij = LeakyReLU(a_src · h_i + a_dst · h_j)
      3. Adjacency mask:     Sadece bağlı düğümler (+ self-loop)
      4. Normalize:          α_ij = softmax_j(e_ij)
      5. Aggregate:          h'_i = Σ_j α_ij · h_j

    Multi-head: K head parallel çalışır, sonuçlar concatenate edilir.

    Args:
        in_features: Giriş feature boyutu.
        out_features: Çıkış feature boyutu.
        num_heads: Attention head sayısı.
        dropout: Dropout oranı.
        negative_slope: LeakyReLU negative slope.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        num_heads: int = 4,
        dropout: float = 0.1,
        negative_slope: float = 0.2,
    ):
        super().__init__()

        self.num_heads = num_heads
        self.head_dim = out_features // num_heads
        assert out_features % num_heads == 0, (
            f"out_features ({out_features}) must be divisible by num_heads ({num_heads})"
        )

        # Node feature projection: in_features → num_heads * head_dim
        self.W = nn.Linear(in_features, num_heads * self.head_dim, bias=False)

        # Attention parametreleri (her head için ayrı)
        self.a_src = nn.Parameter(torch.empty(num_heads, self.head_dim))
        self.a_dst = nn.Parameter(torch.empty(num_heads, self.head_dim))
        nn.init.xavier_normal_(self.a_src)
        nn.init.xavier_normal_(self.a_dst)

        self.leaky_relu = nn.LeakyReLU(negative_slope)
        self.dropout = nn.Dropout(dropout)
        self.attn_dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """GAT layer forward pass.

        Args:
            x: (N, in_features) node feature'ları.
            adj: (N, N) adjacency matrix (soft veya binary).

        Returns:
            (N, out_features) güncellenmiş node feature'ları.
        """
        N = x.size(0)

        # Feature projection: (N, num_heads * head_dim)
        h = self.W(x)
        # Reshape: (N, H, head_dim) burada H = num_heads
        h = h.view(N, self.num_heads, self.head_dim)

        # ── Attention Score Hesaplama ──
        # Source score: her düğümün "gönderici" skoru
        # (N, H, hd) * (H, hd) → sum → (N, H)
        attn_src = (h * self.a_src.unsqueeze(0)).sum(dim=-1)
        # Destination score: her düğümün "alıcı" skoru
        attn_dst = (h * self.a_dst.unsqueeze(0)).sum(dim=-1)

        # Pairwise score: e_ij = src_i + dst_j → (N, N, H)
        attn_scores = attn_src.unsqueeze(1) + attn_dst.unsqueeze(0)
        attn_scores = self.leaky_relu(attn_scores)

        # ── Adjacency Mask ──
        # Self-loop ekle + adjacency mask
        mask = adj + torch.eye(N, device=x.device)
        mask = (mask > 0).float()

        # Bağlantısız düğüm çiftlerini -inf yap
        attn_scores = attn_scores.masked_fill(
            mask.unsqueeze(-1) == 0, float("-inf")
        )

        # ── Softmax + Dropout ──
        attn_weights = F.softmax(attn_scores, dim=1)  # source üzerinden normalize
        attn_weights = torch.nan_to_num(attn_weights, nan=0.0)  # izole düğümler
        attn_weights = self.attn_dropout(attn_weights)

        # ── Aggregation ──
        # attn: (N_dst, N_src, H), h: (N_src, H, hd) → out: (N_dst, H, hd)
        out = torch.einsum("ijh,jhd->ihd", attn_weights, h)

        # Concatenate heads: (N, H * hd) = (N, out_features)
        out = out.reshape(N, -1)
        out = self.dropout(out)

        return out


# ═══════════════════════════════════════════════════════════════════════════════
# GAT NETWORK (Multi-Layer)
# ═══════════════════════════════════════════════════════════════════════════════
class GATNetwork(nn.Module):
    """Multi-layer GAT with residual connections.

    TimesFM embedding space (1280) → Graph space (graph_dim) →
    GAT layers → Graph space → Embedding space (1280).

    Bu network her asset'in embedding'ini diğer asset'lerin bilgisiyle zenginleştirir.

    Args:
        embed_dim: TimesFM embedding boyutu (1280).
        graph_dim: GAT iç boyutu (256). Daha küçük = daha verimli.
        num_heads: GAT attention head sayısı.
        num_layers: GAT katman sayısı.
        dropout: Dropout oranı.
    """

    def __init__(
        self,
        embed_dim: int = 1280,
        graph_dim: int = 256,
        num_heads: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.embed_dim = embed_dim
        self.graph_dim = graph_dim

        # Input projection: embed_dim → graph_dim
        self.input_proj = nn.Sequential(
            nn.Linear(embed_dim, graph_dim),
            nn.LayerNorm(graph_dim),
            nn.GELU(),
        )

        # GAT katmanları + LayerNorm
        self.gat_layers = nn.ModuleList()
        self.layer_norms = nn.ModuleList()

        for _ in range(num_layers):
            self.gat_layers.append(
                GATLayer(
                    in_features=graph_dim,
                    out_features=graph_dim,
                    num_heads=num_heads,
                    dropout=dropout,
                )
            )
            self.layer_norms.append(nn.LayerNorm(graph_dim))

        # Output projection: graph_dim → embed_dim
        self.output_proj = nn.Sequential(
            nn.Linear(graph_dim, embed_dim),
            nn.LayerNorm(embed_dim),
        )

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        node_features: torch.Tensor,
        adj: torch.Tensor,
    ) -> torch.Tensor:
        """GAT network forward pass.

        Args:
            node_features: (N, embed_dim) TimesFM'den gelen node embeddings.
            adj: (N, N) adjacency matrix.

        Returns:
            (N, embed_dim) graph-enhanced node features.
        """
        # Embed → Graph space
        h = self.input_proj(node_features)

        # GAT katmanları (residual + norm)
        for gat_layer, layer_norm in zip(self.gat_layers, self.layer_norms):
            h_new = gat_layer(h, adj)
            h_new = self.dropout(h_new)
            h = layer_norm(h + h_new)  # Pre-LN residual

        # Graph → Embed space
        h = self.output_proj(h)

        return h
