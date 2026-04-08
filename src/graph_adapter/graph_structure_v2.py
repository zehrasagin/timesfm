"""
Weighted Correlation Graph Structure — TSFM-Graph Adapter V2
============================================================

Amaç:
- Her node = bir commodity
- Edge weight'ler = rolling return korelasyon ağırlıkları
- Her asset çifti arasında edge vardır; model ağırlıklara göre öğrenir

Özellikler:
- Fully-connected correlation-weighted adjacency
- Threshold / top-k pruning yok
- Symmetry guaranteed
- Optional self-loop

Not:
- Sector / supply-chain bilgisi edge yerine node feature içinde tutulur (NodeFeatureBuilder).
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from typing import Optional


class CorrelationAdjacency:
    """Return korelasyonundan weighted adjacency üretir.

    Adımlar:
    1. log-return hesapla
    2. corr matrix çıkar
    3. signed veya absolute edge weight olarak kullan
    4. diagonal self-loop ayarını uygula

    Args:
        window: kaç son günün return'ü kullanılacak.
        use_absolute_corr: True ise edge weight=|corr|, False ise sign korunur.
        add_self_loops: diagonal 1 yapılsın mı.
    """

    def __init__(
        self,
        window: int = 60,
        use_absolute_corr: bool = True,
        add_self_loops: bool = True,
    ):
        self.window = window
        self.use_absolute_corr = use_absolute_corr
        self.add_self_loops = add_self_loops

    def _compute_returns(self, price_history: np.ndarray) -> np.ndarray:
        safe_prices = np.maximum(price_history, 1e-8)
        returns = np.diff(np.log(safe_prices), axis=0)
        if returns.shape[0] > self.window:
            returns = returns[-self.window :]
        return returns

    def compute(self, price_history: np.ndarray) -> np.ndarray:
        returns = self._compute_returns(price_history)
        n_assets = price_history.shape[1]

        if returns.shape[0] < 2:
            adj = np.zeros((n_assets, n_assets), dtype=np.float32)
            if self.add_self_loops:
                np.fill_diagonal(adj, 1.0)
            return adj

        corr = np.corrcoef(returns.T)
        corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
        np.fill_diagonal(corr, 0.0)

        if self.use_absolute_corr:
            adj = np.abs(corr)
        else:
            adj = corr

        if self.add_self_loops:
            np.fill_diagonal(adj, 1.0)
        else:
            np.fill_diagonal(adj, 0.0)

        return adj.astype(np.float32)


class CorrelationGraphStructure(nn.Module):
    """Model içinde doğrudan kullanılabilecek sade graph builder."""

    def __init__(
        self,
        corr_window: int = 60,
        use_absolute_corr: bool = True,
        add_self_loops: bool = True,
    ):
        super().__init__()
        self.builder = CorrelationAdjacency(
            window=corr_window,
            use_absolute_corr=use_absolute_corr,
            add_self_loops=add_self_loops,
        )

    def forward(
        self,
        price_history: np.ndarray,
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        adj = self.builder.compute(price_history)
        tensor = torch.from_numpy(adj)
        if device is not None:
            tensor = tensor.to(device)
        return tensor
