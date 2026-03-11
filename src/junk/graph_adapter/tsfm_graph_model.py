"""
TSFM-Graph Adapter Model — Full Pipeline
==========================================

Tüm bileşenleri birleştiren ana model sınıfı.

Pipeline:
  1. [FROZEN]    TimesFM Backbone → Per-asset temporal embeddings E_i (P, 1280)
  2. [TRAINABLE] NodeFeatureBuilder → Handcrafted features X (N, F)
  3. [TRAINABLE] Graph Learner → A_t (Hybrid: Corr ∪ Sector ∪ Supply ∪ Learned(X))
  4. [TRAINABLE] GNN(X; A) → Cross-sectional Context H (N, 1280)
  5. [TRAINABLE] Cross-Attention Adapter(Q=E_target, KV=H) → Enhanced E'_target
  6. [TRAINABLE] Prediction Head → Forecast ŷ

ÖNEMLİ: GNN'e TimesFM embedding'i GİRMEZ.
  Branch A (temporal): TimesFM her asset'i bağımsız çalıştırır → sadece fusion'da Q.
  Branch B (graph): Tamamen handcrafted feature + kural tabanlı/learned adjacency.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import List, Optional, Tuple, Dict

from .graph_structure import HybridGraphStructure
from .gat_layer import GATNetwork
from ..junk.cross_attention_adapter import CrossAttentionAdapter
from .embedding_extractor import TimesFMEmbeddingExtractor
from .node_features import NodeFeatureBuilder


# ═══════════════════════════════════════════════════════════════════════════════
# PREDICTION HEAD
# ═══════════════════════════════════════════════════════════════════════════════
class PredictionHead(nn.Module):
    """Graph-enhanced embedding'den nihai tahmin üreten MLP.

    Son patch'in enhanced embedding'ini alır ve fiyat tahmini yapar.

    Mimari: D → hidden → hidden/2 → 1

    Args:
        embed_dim: Giriş embedding boyutu (1280).
        hidden_dim: Gizli katman boyutu.
        output_dim: Çıkış boyutu (1 = point forecast).
        dropout: Dropout oranı.
    """

    def __init__(
        self,
        embed_dim: int = 1280,
        hidden_dim: int = 256,
        output_dim: int = 1,
        dropout: float = 0.1,
    ):
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
        """Prediction head forward.

        Args:
            enhanced_embedding: (P, D) enhanced temporal embedding.
                Son patch kullanılır.

        Returns:
            (1,) veya (output_dim,) forecast değeri.
        """
        # Son patch'in embedding'i
        if enhanced_embedding.dim() == 2:
            last_patch = enhanced_embedding[-1]  # (D,)
        else:
            last_patch = enhanced_embedding  # (D,)

        return self.head(last_patch)


# ═══════════════════════════════════════════════════════════════════════════════
# TSFM-GRAPH ADAPTER MODEL
# ═══════════════════════════════════════════════════════════════════════════════
class TSFMGraphAdapterModel(nn.Module):
    """Tam TSFM-Graph Adapter pipeline.

    Bu model TimesFM'in frozen embedding'lerini alır, graph yapısıyla
    zenginleştirir ve nihai tahmin üretir.

    Two-Branch Flow:
      Branch A (Temporal, frozen):
        TimesFM tüm asset'leri bağımsız çalıştırır → E_i (P, 1280)
        Bu embedding GNN'e GİRMEZ — sadece fusion'da temporal query.

      Branch B (Cross-sectional Graph, trainable):
        NodeFeatureBuilder → X (N, F) handcrafted features
        Graph Learner → A_t (hybrid: corr/sector/supply/learned)
        GNN(X; A) → H (N, 1280) cross-sectional context

      Fusion:
        Cross-Attention(Q=E_target, KV=H) → E'_target (P, 1280)

      Output:
        Prediction Head(E'_target[-1]) → ŷ

    Args:
        timesfm_model: Initialize edilmiş TimesFM_2p5_200M_torch.
        asset_names: Asset isim listesi (graph düğümleri).
        target_idx: Hedef asset'in indeksi.
        max_context: Maksimum context uzunluğu.
        embed_dim: TimesFM embedding boyutu (1280).
        graph_dim: GAT iç boyutu.
        adapter_dim: Cross-attention adapter bottleneck boyutu.
        num_gat_heads: GAT head sayısı.
        num_gat_layers: GAT katman sayısı.
        num_adapter_heads: Cross-attention head sayısı.
        dropout: Global dropout oranı.
        corr_window: Korelasyon pencere uzunluğu.
        corr_threshold: Korelasyon eşiği.
        initial_alpha: Hybrid adjacency α başlangıcı.
    """

    def __init__(
        self,
        timesfm_model,
        asset_names: List[str],
        target_idx: int = 0,
        max_context: int = 1024,
        embed_dim: int = 1280,
        graph_dim: int = 256,
        adapter_dim: int = 256,
        num_gat_heads: int = 4,
        num_gat_layers: int = 2,
        num_adapter_heads: int = 4,
        dropout: float = 0.1,
        corr_window: int = 60,
        corr_threshold: float = 0.3,
        initial_alpha: float = 0.7,
    ):
        super().__init__()

        self.asset_names = asset_names
        self.n_assets = len(asset_names)
        self.target_idx = target_idx
        self.max_context = max_context
        self.embed_dim = embed_dim

        # ── Stream A: Frozen Backbone (Embedding Extractor) ──
        self.embedding_extractor = TimesFMEmbeddingExtractor(timesfm_model)

        # ── Node Feature Builder (handcrafted, TimesFM-free) ──
        self.node_feature_builder = NodeFeatureBuilder(
            asset_names=asset_names,
            corr_window=corr_window,
        )

        # ── Stream B: Trainable Graph Path ──
        self.graph_structure = HybridGraphStructure(
            asset_names=asset_names,
            node_feature_dim=self.node_feature_builder.feature_dim,
            corr_window=corr_window,
            corr_threshold=corr_threshold,
            initial_alpha=initial_alpha,
        )

        self.gat_network = GATNetwork(
            embed_dim=embed_dim,
            node_feature_dim=self.node_feature_builder.feature_dim,
            graph_dim=graph_dim,
            num_heads=num_gat_heads,
            num_layers=num_gat_layers,
            dropout=dropout,
        )

        # ── Fusion: Cross-Attention Adapter ──
        self.cross_attention_adapter = CrossAttentionAdapter(
            embed_dim=embed_dim,
            adapter_dim=adapter_dim,
            num_heads=num_adapter_heads,
            dropout=dropout,
        )

        # ── Output: Prediction Head ──
        self.prediction_head = PredictionHead(
            embed_dim=embed_dim,
            hidden_dim=graph_dim,
            output_dim=1,
            dropout=dropout,
        )

    def get_trainable_params(self) -> List[nn.Parameter]:
        """Sadece trainable parametreleri döndür (backbone hariç)."""
        params = []
        for name, param in self.named_parameters():
            if "embedding_extractor" not in name:
                params.append(param)
        return params

    def count_parameters(self) -> Dict[str, int]:
        """Parametre sayımı: trainable vs frozen."""
        trainable = sum(
            p.numel() for n, p in self.named_parameters()
            if "embedding_extractor" not in n and p.requires_grad
        )
        # Frozen backbone'daki parametre sayısını hesapla
        frozen = sum(
            p.numel() for p in self.embedding_extractor.module.parameters()
        )
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
        static_adj: Optional[torch.Tensor] = None,
        target_idx: Optional[int] = None,
    ) -> torch.Tensor:
        """Tam forward pass.

        Args:
            multi_asset_series: N adet zaman serisi listesi, her biri (T,).
            price_history: (T_hist, N) fiyat geçmişi (korelasyon adj için).
            static_adj: Önceden hesaplanmış statik adjacency (cache).
            target_idx: Hedef asset indeksi (None ise self.target_idx).

        Returns:
            (1,) point forecast.
        """
        if target_idx is None:
            target_idx = self.target_idx

        # ═══ STREAM A: Frozen Backbone ═══
        # TimesFM tüm asset'leri bağımsız olarak çalıştırır
        all_seq_emb = self.embedding_extractor.extract_embeddings(
            multi_asset_series, self.max_context
        )
        # all_seq_emb: (N, num_patches, 1280) — detached (no grad)

        # ═══ STREAM B: Trainable Graph Path (handcrafted, TimesFM-free) ═══
        # 1. Handcrafted node features
        if price_history is not None:
            node_feats = self.node_feature_builder.build_tensor(
                price_history, device=all_seq_emb.device
            )
        else:
            ph = np.column_stack([
                s[-self.max_context:] for s in multi_asset_series
            ])
            node_feats = self.node_feature_builder.build_tensor(
                ph, device=all_seq_emb.device
            )

        # 2. Hybrid adjacency matrix (learned kısmı da handcrafted feature kullanır)
        adj = self.graph_structure(
            node_feats, price_history=price_history, static_adj=static_adj
        )

        # 3. GAT: handcrafted features girer, 1280-d graph context çıkar
        graph_context = self.gat_network(node_feats, adj)
        # graph_context: (N, 1280) — cross-sectional context

        # ═══ FUSION: Cross-Attention Adapter ═══
        # Target asset'in temporal embedding'i (TimesFM'den)
        target_seq_emb = all_seq_emb[target_idx]  # (num_patches, 1280)

        # Cross-attention: temporal patches attend to all assets' graph context
        enhanced_emb = self.cross_attention_adapter(target_seq_emb, graph_context)
        # enhanced_emb: (num_patches, 1280)

        # ═══ OUTPUT: Prediction Head ═══
        forecast = self.prediction_head(enhanced_emb)
        # forecast: (1,)

        return forecast

    def forward_batch(
        self,
        batch_multi_asset_series: List[List[np.ndarray]],
        batch_price_histories: Optional[List[np.ndarray]] = None,
        batch_static_adjs: Optional[List[torch.Tensor]] = None,
        target_idx: Optional[int] = None,
    ) -> torch.Tensor:
        """Batch forward pass (training için).

        Her sample: multi-asset context window → single forecast.

        Args:
            batch_multi_asset_series: B adet [N × (T,)] listesi.
            batch_price_histories: B adet (T_hist, N) fiyat geçmişi.
            batch_static_adjs: B adet (N, N) static adjacency.
            target_idx: Hedef asset indeksi.

        Returns:
            (B, 1) batch forecast.
        """
        B = len(batch_multi_asset_series)
        forecasts = []

        for i in range(B):
            price_hist = batch_price_histories[i] if batch_price_histories else None
            static_adj = batch_static_adjs[i] if batch_static_adjs else None

            fc = self.forward(
                batch_multi_asset_series[i],
                price_history=price_hist,
                static_adj=static_adj,
                target_idx=target_idx,
            )
            forecasts.append(fc)

        return torch.stack(forecasts, dim=0)
    def forward_cached(
        self,
        target_seq_embeddings: torch.Tensor,
        price_history: np.ndarray,
        static_adj: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Cache'den gelen embedding'lerle forward pass (TimesFM ÇALIŞMAZ).

        Training loop'ta kullanılır — sadece trainable bileşenler çalışır.
        ~50x daha hızlı (TimesFM forward pass yok).

        Args:
            target_seq_embeddings: (P, 1280) pre-computed target sequence embeddings.
                Cross-attention Q olarak kullanılır.
            price_history: (T, N) fiyat geçmişi (handcrafted node features + adj için).
            static_adj: Önceden hesaplanmış statik adjacency.

        Returns:
            (1,) point forecast.
        """
        # ═══ STREAM B: Trainable Graph Path (handcrafted) ═══
        # 1. Handcrafted node features (TimesFM-free!)
        node_feats = self.node_feature_builder.build_tensor(
            price_history, device=target_seq_embeddings.device
        )

        # 2. Hybrid adjacency matrix
        adj = self.graph_structure(
            node_feats, price_history=price_history, static_adj=static_adj
        )

        # 3. GAT: handcrafted features girer, 1280-d graph context çıkar
        graph_context = self.gat_network(node_feats, adj)

        # ═══ FUSION: Cross-Attention Adapter ═══
        enhanced_emb = self.cross_attention_adapter(
            target_seq_embeddings, graph_context
        )

        # ═══ OUTPUT: Prediction Head ═══
        forecast = self.prediction_head(enhanced_emb)
        return forecast