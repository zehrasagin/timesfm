"""
Node Feature Builder — TSFM-Graph Adapter
===========================================

GNN node feature'ları: tamamen handcrafted, TimesFM-free.

TimesFM embedding'leri GNN'e GİRMEZ. GNN'in node input'u bu modül
tarafından üretilen handcrafted feature vektörüdür.

Feature Grupları (14-d toplam):
  1. Return İstatistikleri (3-d):
     - 5-gün log-return ortalaması
     - 20-gün log-return ortalaması
     - 60-gün log-return ortalaması

  2. Volatilite (2-d):
     - 20-gün gerçekleşen volatilite (annualize)
     - 60-gün gerçekleşen volatilite

  3. Momentum / Trend (3-d):
     - MA5 / MA20 oranı (kısa/orta trend)
     - MA20 / MA60 oranı (orta/uzun trend)
     - Fiyat / MA60 oranı (mean-reversion sinyali)

  4. Cross-Correlation (1-d):
     - Diğer tüm asset'lerle ortalama korelasyon

  5. Sector One-Hot (4-d):
     - Energy, Precious Metals, Industrial Metals, Agriculture

  6. Supply Chain Degree (1-d):
     - Tedarik zincirindeki bağlantı sayısı (normalize)

  Her feature z-score normalize edilir (outlier dayanıklılığı için clamp ±3).

Toplam: 3 + 2 + 3 + 1 + 4 + 1 = 14-d (feature sayısı dinamik, sector sayısına bağlı)
"""

import numpy as np
import torch
from typing import List, Dict, Optional, Tuple


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


class NodeFeatureBuilder:
    """Handcrafted node feature builder for GNN.

    Fiyat geçmişinden (T, N) her asset için F-boyutlu feature vektörü üretir.
    TimesFM embedding'leri KULLANILMAZ.

    Args:
        asset_names: Asset isim listesi.
        sector_map: Sektör → asset listesi eşleşmesi.
        supply_edges: Tedarik zinciri kenarları.
        corr_window: Korelasyon hesaplama penceresi.
    """

    def __init__(
        self,
        asset_names: List[str],
        sector_map: Dict[str, List[str]] = COMMODITY_SECTORS,
        supply_edges: Optional[List] = None,
        corr_window: int = 60,
    ):
        self.asset_names = asset_names
        self.n_assets = len(asset_names)
        self.sector_map = sector_map
        self.supply_edges = supply_edges or SUPPLY_CHAIN_EDGES
        self.corr_window = corr_window

        # Pre-compute statik feature'lar
        self._sector_onehot = self._build_sector_onehot()
        self._supply_degree = self._build_supply_degree()

        # Feature boyutu
        n_sectors = len(sector_map)
        # returns(3) + vol(2) + momentum(3) + corr(1) + sector(n_sec) + supply(1)
        self.feature_dim = 3 + 2 + 3 + 1 + n_sectors + 1

    def _build_sector_onehot(self) -> np.ndarray:
        """(N, n_sectors) sector one-hot encoding."""
        sectors = list(self.sector_map.keys())
        asset_to_sector = {}
        for sector, assets in self.sector_map.items():
            for asset in assets:
                asset_to_sector[asset] = sector

        onehot = np.zeros((self.n_assets, len(sectors)), dtype=np.float32)
        for i, name in enumerate(self.asset_names):
            sec = asset_to_sector.get(name)
            if sec and sec in sectors:
                onehot[i, sectors.index(sec)] = 1.0
        return onehot

    def _build_supply_degree(self) -> np.ndarray:
        """(N,) her asset'in supply chain bağlantı sayısı (normalize)."""
        degree = np.zeros(self.n_assets, dtype=np.float32)
        name_to_idx = {name: i for i, name in enumerate(self.asset_names)}
        for src, dst in self.supply_edges:
            if src in name_to_idx:
                degree[name_to_idx[src]] += 1
            if dst in name_to_idx:
                degree[name_to_idx[dst]] += 1

        # Normalize to [0, 1]
        max_deg = degree.max()
        if max_deg > 0:
            degree = degree / max_deg
        return degree

    def build(self, price_history: np.ndarray) -> np.ndarray:
        """Fiyat geçmişinden handcrafted node feature matrisi üret.

        Args:
            price_history: (T, N) fiyat matrisi.

        Returns:
            (N, F) node feature matrisi.
        """
        T, N = price_history.shape
        assert N == self.n_assets

        # Log-returns (güvenli)
        safe_prices = np.maximum(price_history, 1e-8)
        log_returns = np.diff(np.log(safe_prices), axis=0)  # (T-1, N)

        features = []

        # ── 1. Return İstatistikleri (3-d) ──
        for window in [5, 20, 60]:
            if log_returns.shape[0] >= window:
                feat = log_returns[-window:].mean(axis=0)  # (N,)
            else:
                feat = log_returns.mean(axis=0)
            features.append(feat)

        # ── 2. Volatilite (2-d) ──
        for window in [20, 60]:
            if log_returns.shape[0] >= window:
                vol = log_returns[-window:].std(axis=0) * np.sqrt(252)
            else:
                vol = log_returns.std(axis=0) * np.sqrt(252)
            features.append(vol)

        # ── 3. Momentum / Trend (3-d) ──
        for (short_w, long_w) in [(5, 20), (20, 60)]:
            short_ma = safe_prices[-short_w:].mean(axis=0) if T >= short_w else safe_prices.mean(axis=0)
            long_ma = safe_prices[-long_w:].mean(axis=0) if T >= long_w else safe_prices.mean(axis=0)
            ratio = short_ma / np.maximum(long_ma, 1e-8)
            features.append(ratio)

        # Price / MA60
        ma60 = safe_prices[-60:].mean(axis=0) if T >= 60 else safe_prices.mean(axis=0)
        price_ma_ratio = safe_prices[-1] / np.maximum(ma60, 1e-8)
        features.append(price_ma_ratio)

        # ── 4. Cross-Correlation (1-d) ──
        corr_window = min(self.corr_window, log_returns.shape[0])
        if corr_window >= 2:
            recent_returns = log_returns[-corr_window:]
            corr_matrix = np.corrcoef(recent_returns.T)
            corr_matrix = np.nan_to_num(corr_matrix, nan=0.0)
            np.fill_diagonal(corr_matrix, 0.0)
            avg_corr = corr_matrix.sum(axis=1) / max(N - 1, 1)
        else:
            avg_corr = np.zeros(N)
        features.append(avg_corr)

        # ── 5. Sector One-Hot (4-d) ──
        for col in range(self._sector_onehot.shape[1]):
            features.append(self._sector_onehot[:, col])

        # ── 6. Supply Chain Degree (1-d) ──
        features.append(self._supply_degree)

        # Stack: (F, N) → (N, F)
        node_features = np.stack(features, axis=0).T.astype(np.float32)

        # Z-score normalize (dinamik feature'lar için)
        n_dynamic = 3 + 2 + 3 + 1  # returns + vol + momentum + corr = 9
        dynamic_part = node_features[:, :n_dynamic]
        mu = dynamic_part.mean(axis=0, keepdims=True)
        sigma = dynamic_part.std(axis=0, keepdims=True)
        sigma = np.where(sigma < 1e-8, 1.0, sigma)
        dynamic_part = (dynamic_part - mu) / sigma
        dynamic_part = np.clip(dynamic_part, -3.0, 3.0)
        node_features[:, :n_dynamic] = dynamic_part

        return node_features

    def build_tensor( # featureları tensor olarak döner
        self,
        price_history: np.ndarray,
        device: torch.device = torch.device("cpu"),
    ) -> torch.Tensor:
        """Fiyat geçmişinden node feature tensor'ü üret.

        Args:
            price_history: (T, N) fiyat matrisi.
            device: Hedef device.

        Returns:
            (N, F) node feature tensor.
        """
        features = self.build(price_history)
        return torch.from_numpy(features).to(device)
