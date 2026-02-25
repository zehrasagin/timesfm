"""
Cross-Attention Adapter — TSFM-Graph Adapter
=============================================

Temporal embedding'leri (TimesFM backbone) ile cross-sectional
graph context'i (GAT çıktısı) birleştirir.

E'_i = E_i + Adapter(E_i, H_i)

Burada:
  E_i: (num_patches, D) — Asset i'nin temporal patch embedding'leri
  H:   (N, D) — Tüm asset'lerin graph-enhanced feature'ları
  Adapter: Cross-Attention + FFN (bottleneck yapı)

Her temporal patch, tüm asset'lerin graph context'ine attend eder.
Böylece "CO1'in bugünkü patch embedding'i, GC1'deki değişimi de hesaba katar."

Output projections sıfıra yakın initialize edilir (stable training).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossAttentionAdapter(nn.Module):
    """Temporal ↔ Graph cross-attention adapter.

    Mimari:
      1. Pre-LayerNorm
      2. Q = proj(E_i), K = proj(H), V = proj(H)  (down-projection: D → adapter_dim)
      3. Multi-head cross-attention: her patch tüm asset'lere attend eder
      4. Up-projection: adapter_dim → D
      5. Residual: E'_i = E_i + attn_output
      6. FFN bottleneck + Residual

    Output projection'lar sıfıra yakın başlar → başlangıçta adapter etkisiz,
    training ilerledikçe graph bilgisi yavaşça eklenir.

    Args:
        embed_dim: TimesFM embedding boyutu (1280).
        adapter_dim: Bottleneck boyutu (256). Parametreyi azaltır.
        num_heads: Attention head sayısı.
        dropout: Dropout oranı.
    """

    def __init__(
        self,
        embed_dim: int = 1280,
        adapter_dim: int = 256,
        num_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.embed_dim = embed_dim
        self.adapter_dim = adapter_dim
        self.num_heads = num_heads
        self.head_dim = adapter_dim // num_heads

        assert adapter_dim % num_heads == 0, (
            f"adapter_dim ({adapter_dim}) must be divisible by num_heads ({num_heads})"
        )

        # ── Down-Projection (D → adapter_dim) ──
        self.q_proj = nn.Linear(embed_dim, adapter_dim)
        self.k_proj = nn.Linear(embed_dim, adapter_dim)
        self.v_proj = nn.Linear(embed_dim, adapter_dim)

        # ── Up-Projection (adapter_dim → D) ──
        self.out_proj = nn.Linear(adapter_dim, embed_dim)

        # ── Layer Norms ──
        self.norm_q = nn.LayerNorm(embed_dim)
        self.norm_kv = nn.LayerNorm(embed_dim)
        self.norm_ffn = nn.LayerNorm(embed_dim)

        # ── Feed-Forward Network (bottleneck) ──
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, adapter_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(adapter_dim, embed_dim),
            nn.Dropout(dropout),
        )

        self.attn_dropout = nn.Dropout(dropout)
        self.scale = self.head_dim ** -0.5

        # ── Sıfıra yakın initialization (stable training) ──
        # Başlangıçta adapter etkisiz: E'_i ≈ E_i
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)
        nn.init.zeros_(self.ffn[-2].weight)
        nn.init.zeros_(self.ffn[-2].bias)

    def forward(
        self,
        temporal_embedding: torch.Tensor,
        graph_context: torch.Tensor,
    ) -> torch.Tensor:
        """Cross-attention adapter forward.

        Args:
            temporal_embedding: (P, D) target asset'in temporal patch embeddings.
                P = num_patches, D = embed_dim (1280).
            graph_context: (N, D) tüm asset'lerin graph-enhanced features.
                N = asset sayısı.

        Returns:
            (P, D) graph bilgisiyle zenginleştirilmiş temporal embedding.
        """
        P = temporal_embedding.size(0)
        N = graph_context.size(0)
        H = self.num_heads
        hd = self.head_dim

        # ── Pre-LayerNorm ──
        q_in = self.norm_q(temporal_embedding)
        kv_in = self.norm_kv(graph_context)

        # ── Projection ──
        Q = self.q_proj(q_in).view(P, H, hd).transpose(0, 1)   # (H, P, hd)
        K = self.k_proj(kv_in).view(N, H, hd).transpose(0, 1)  # (H, N, hd)
        V = self.v_proj(kv_in).view(N, H, hd).transpose(0, 1)  # (H, N, hd)

        # ── Scaled Dot-Product Cross-Attention ──
        # (H, P, hd) × (H, hd, N) → (H, P, N)
        attn_weights = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)

        # (H, P, N) × (H, N, hd) → (H, P, hd)
        attn_out = torch.matmul(attn_weights, V)

        # Reshape: (H, P, hd) → (P, adapter_dim)
        attn_out = attn_out.transpose(0, 1).contiguous().view(P, self.adapter_dim)

        # ── Up-Projection + Residual ──
        attn_out = self.out_proj(attn_out)
        output = temporal_embedding + attn_out

        # ── FFN + Residual ──
        output = output + self.ffn(self.norm_ffn(output))

        return output
