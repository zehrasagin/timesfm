"""
Graph Structure Module — TSFM-Graph Adapter
============================================

Graph yapısı G = (V, E):
  Nodes V: Her commodity bir düğüm (CO1, CL1, GC1, ...)
  Edges E: 4 farklı kenar tipi

Kenar Tipleri:
  1. Correlation:   A_ij = |Corr(r_i, r_j)|  (rolling 60-90 gün)
  2. Sector:        A_ij = 1 if same GICS sector
  3. Supply Chain:  A_ij = 1 if i supplies j
  4. Learned:       A_t = σ(E_t·W_q · (E_t·W_k)^T)  (data-driven)

Hybrid Formula:
  A_final = α · (A_corr ∪ A_sector ∪ A_supply) + (1-α) · A_learned
  α ≈ 0.7 (learnable, statik bilgi ağırlıklı başla)
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional, Tuple


# ─── Commodity Sektör Tanımları ────────────────────────────────────────────────
COMMODITY_SECTORS: Dict[str, List[str]] = {
    "Energy": ["CO1 Comdty", "CL1 Comdty", "HO1 Comdty", "NG1 Comdty"],
    "Precious Metals": ["GC1 Comdty", "PA1 Comdty", "PL1 Comdty", "SI1 Comdty"],
    "Industrial Metals": ["HG1 Comdty"],
    "Agriculture": ["C 1 Comdty"],
}

# ─── Supply Chain İlişkileri (ekonomik yapı) ───────────────────────────────────
SUPPLY_CHAIN_EDGES: List[Tuple[str, str]] = [
    # Crude Oil → Refined Products (rafineri zinciri)
    ("CO1 Comdty", "HO1 Comdty"),
    ("CL1 Comdty", "HO1 Comdty"),
    # Brent ↔ WTI (benchmark paritesi)
    ("CO1 Comdty", "CL1 Comdty"),
    ("CL1 Comdty", "CO1 Comdty"),
    # Precious metals pair trading
    ("GC1 Comdty", "SI1 Comdty"),
    ("SI1 Comdty", "GC1 Comdty"),
    # PGM pair (Palladium ↔ Platinum)
    ("PA1 Comdty", "PL1 Comdty"),
    ("PL1 Comdty", "PA1 Comdty"),
    # Crude → Natural Gas (enerji ikamesi)
    ("CO1 Comdty", "NG1 Comdty"),
    ("CL1 Comdty", "NG1 Comdty"),
    # Copper (endüstriyel talep indikatörü) → Energy
    ("HG1 Comdty", "CO1 Comdty"),
    ("HG1 Comdty", "CL1 Comdty"),
]


# ═══════════════════════════════════════════════════════════════════════════════
# 1. CORRELATION ADJACENCY
# ═══════════════════════════════════════════════════════════════════════════════
class CorrelationAdjacency:
    """Rolling return korelasyonundan adjacency matrix hesaplar.

    A_ij = |Corr(r_i, r_j)| burada r_i = log-return.
    Threshold veya Top-K ile sparsify edilir.

    Args:
        window: Korelasyon pencere uzunluğu (gün).
        threshold: Minimum korelasyon eşiği (altı 0 yapılır).
        top_k: Her düğüm için maksimum bağlantı sayısı.
    """

    def __init__(
        self,
        window: int = 60,
        threshold: Optional[float] = 0.3,
        top_k: Optional[int] = None,
    ):
        self.window = window
        self.threshold = threshold
        self.top_k = top_k

    def compute(self, price_history: np.ndarray) -> np.ndarray:
        """Fiyat geçmişinden korelasyon adjacency matrix hesapla.

        Args:
            price_history: (T, N) fiyat matrisi. T=gün, N=asset.

        Returns:
            (N, N) adjacency matrix, değerler [0, 1] arasında.
        """
        # Log-return hesapla (np.maximum ile güvenli — RuntimeWarning önlenir)
        returns = np.diff(np.log(np.maximum(price_history, 1e-8)), axis=0)

        # Son `window` günü kullan
        if returns.shape[0] > self.window:
            returns = returns[-self.window:]

        # Korelasyon matrisi
        if returns.shape[0] < 2:
            return np.zeros((returns.shape[1], returns.shape[1]))

        corr_matrix = np.corrcoef(returns.T)
        corr_matrix = np.nan_to_num(corr_matrix, nan=0.0)
        adj = np.abs(corr_matrix)

        # Self-loop kaldır
        np.fill_diagonal(adj, 0.0)

        # Threshold ile sparsify
        if self.threshold is not None:
            adj = np.where(adj >= self.threshold, adj, 0.0)

        # Top-K ile sparsify
        if self.top_k is not None:
            for i in range(adj.shape[0]):
                if np.sum(adj[i] > 0) > self.top_k:
                    sorted_idx = np.argsort(adj[i])[::-1]
                    adj[i, sorted_idx[self.top_k:]] = 0.0

        return adj


# ═══════════════════════════════════════════════════════════════════════════════
# 2. SECTOR ADJACENCY
# ═══════════════════════════════════════════════════════════════════════════════
class SectorAdjacency:
    """Sektör üyeliğine dayalı statik adjacency matrix.

    A_ij = 1 eğer asset i ve j aynı sektördeyse.
    Interpretable prior: Enerji emtiaları birbirine bağlı, metaller birbirine bağlı.

    Args:
        asset_names: Asset isim listesi.
        sector_map: Sektör → asset listesi eşleşmesi.
    """

    def __init__(
        self,
        asset_names: List[str],
        sector_map: Dict[str, List[str]] = COMMODITY_SECTORS,
    ):
        self.asset_names = asset_names
        self.sector_map = sector_map
        self._adj = self._build()

    def _build(self) -> np.ndarray:
        n = len(self.asset_names)

        # Reverse map: asset → sector label
        asset_to_sector = {}
        for sector, assets in self.sector_map.items():
            for asset in assets:
                asset_to_sector[asset] = sector

        # Vectorized: numpy broadcasting instead of O(n²) Python loop
        labels = np.array([
            asset_to_sector.get(name, f"_no_sector_{i}")
            for i, name in enumerate(self.asset_names)
        ])
        adj = (labels[:, None] == labels[None, :]).astype(np.float64)
        np.fill_diagonal(adj, 0.0)
        return adj

    def compute(self) -> np.ndarray:
        return self._adj.copy()


# ═══════════════════════════════════════════════════════════════════════════════
# 3. SUPPLY CHAIN ADJACENCY
# ═══════════════════════════════════════════════════════════════════════════════
class SupplyChainAdjacency:
    """Tedarik zinciri / ekonomik yapı adjacency matrix.

    A_ij = 1 eğer asset i, asset j'ye supply ediyorsa.
    Örnek: Crude Oil → Heating Oil (rafineri ilişkisi).

    Args:
        asset_names: Asset isim listesi.
        edges: (source, destination) tuple listesi.
    """

    def __init__(
        self,
        asset_names: List[str],
        edges: List[Tuple[str, str]] = SUPPLY_CHAIN_EDGES,
    ):
        self.asset_names = asset_names
        self.edges = edges
        self._adj = self._build()

    def _build(self) -> np.ndarray:
        n = len(self.asset_names)
        adj = np.zeros((n, n))
        name_to_idx = {name: idx for idx, name in enumerate(self.asset_names)}

        for src, dst in self.edges:
            if src in name_to_idx and dst in name_to_idx:
                adj[name_to_idx[src], name_to_idx[dst]] = 1.0
        return adj

    def compute(self) -> np.ndarray:
        return self._adj.copy()


# ═══════════════════════════════════════════════════════════════════════════════
# 4. LEARNED ADJACENCY
# ═══════════════════════════════════════════════════════════════════════════════
class LearnedAdjacency(nn.Module):
    """Handcrafted feature'lardan öğrenilen zaman-bağımlı adjacency matrix.

    A_t = σ(X_t·W_q · (X_t·W_k)^T)

    X_t = handcrafted node features (F-d).
    Data-driven: Rejim değişikliklerine adapte olur.
    Örneğin kriz dönemlerinde tüm emtialar arası korelasyon artar →
    learned edges bunu otomatik yakalar.

    ÖNEMLİ: TimesFM embedding'den türetilmez — handcrafted feature kullanır.

    Args:
        input_dim: Node feature boyutu (handcrafted feature dim).
        key_dim: Attention key boyutu (daha küçük = daha az parametre).
    """

    def __init__(self, input_dim: int, key_dim: int = 64):
        super().__init__()
        self.W_q = nn.Linear(input_dim, key_dim, bias=False)
        self.W_k = nn.Linear(input_dim, key_dim, bias=False)
        self.scale = key_dim ** -0.5

    def forward(self, node_features: torch.Tensor) -> torch.Tensor:
        """Learned adjacency hesapla.

        Args:
            node_features: (N, F) veya (B, N, F) handcrafted node features.

        Returns:
            (N, N) veya (B, N, N) soft adjacency, değerler [0, 1].
        """
        Q = self.W_q(node_features)  # (..., N, key_dim)
        K = self.W_k(node_features)

        # Scaled dot-product
        attn = torch.matmul(Q, K.transpose(-2, -1)) * self.scale

        # Sigmoid → [0, 1] soft edges
        adj = torch.sigmoid(attn)

        # Self-loop kaldır
        eye = torch.eye(adj.size(-1), device=adj.device)
        if adj.dim() == 3:
            eye = eye.unsqueeze(0)
        adj = adj * (1.0 - eye)

        return adj


# ═══════════════════════════════════════════════════════════════════════════════
# 5. HYBRID GRAPH STRUCTURE
# ═══════════════════════════════════════════════════════════════════════════════
class HybridGraphStructure(nn.Module):
    """Statik prior + learned adjacency birleşimi.

    A_final = α · A_static + (1-α) · A_learned

    Burada:
      A_static = max(A_corr, A_sector, A_supply)  (union/OR operasyonu)
      α ≈ 0.7: Statik bilgi ağırlıklı başla, zamanla ayarla.

    Args:
        asset_names: Asset isim listesi (graph düğümleri).
        node_feature_dim: Handcrafted node feature boyutu (LearnedAdjacency için).
        corr_window: Korelasyon pencere uzunluğu.
        corr_threshold: Korelasyon eşiği.
        initial_alpha: α başlangıç değeri.
        key_dim: Learned adjacency key boyutu.
    """

    def __init__(
        self,
        asset_names: List[str],
        node_feature_dim: int,
        corr_window: int = 60,
        corr_threshold: float = 0.3,
        initial_alpha: float = 0.7,
        key_dim: int = 64,
    ):
        super().__init__()

        self.asset_names = asset_names
        self.n_assets = len(asset_names)

        # Statik adjacency builders
        self.corr_builder = CorrelationAdjacency(
            window=corr_window, threshold=corr_threshold
        )
        self.sector_builder = SectorAdjacency(asset_names)
        self.supply_builder = SupplyChainAdjacency(asset_names)

        # Learned adjacency (trainable)
        self.learned_adj = LearnedAdjacency(node_feature_dim, key_dim)

        # α parametresi: sigmoid(0.847) ≈ 0.7
        self.alpha_logit = nn.Parameter(torch.tensor(0.847))

    @property
    def alpha(self) -> torch.Tensor:
        """Mevcut α değeri (0-1 arası, sigmoid ile sınırlandırılmış)."""
        return torch.sigmoid(self.alpha_logit)

    def compute_static_adjacency(
        self, price_history: Optional[np.ndarray] = None
    ) -> torch.Tensor:
        """Statik adjacency hesapla: A_static = A_corr ∪ A_sector ∪ A_supply.

        Args:
            price_history: (T, N) fiyat geçmişi. None ise sadece sektör+supply kullanılır.

        Returns:
            (N, N) statik adjacency tensor.
        """
        A_sector = torch.tensor(self.sector_builder.compute(), dtype=torch.float32)
        A_supply = torch.tensor(self.supply_builder.compute(), dtype=torch.float32)

        # Union: element-wise max
        A_static = torch.maximum(A_sector, A_supply)

        if price_history is not None:
            A_corr = torch.tensor(
                self.corr_builder.compute(price_history), dtype=torch.float32
            )
            A_static = torch.maximum(A_static, A_corr)

        return A_static

    def forward(
        self,
        node_features: torch.Tensor,
        price_history: Optional[np.ndarray] = None,
        static_adj: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Hybrid adjacency hesapla.

        Args:
            node_features: (N, F) handcrafted node features.
            price_history: (T, N) fiyat geçmişi (statik adj için).
            static_adj: Önceden hesaplanmış statik adjacency (verimlilik için cache).

        Returns:
            (N, N) hybrid adjacency matrix.
        """
        # Statik bileşen
        if static_adj is None:
            A_static = self.compute_static_adjacency(price_history)
        else:
            A_static = static_adj

        A_static = A_static.to(node_features.device)

        # Learned bileşen (trainable)
        A_learned = self.learned_adj(node_features)

        # Hybrid birleştirme
        alpha = self.alpha
        A_final = alpha * A_static + (1.0 - alpha) * A_learned

        return A_final
