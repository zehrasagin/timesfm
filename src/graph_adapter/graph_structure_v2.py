"""
Correlation-Based Graph Structure — TSFM-Graph Adapter V2
==========================================================

Amaç:
- Her node = bir commodity
- Edge'ler = yüksek korelasyonlu bağlantılar
- Graph sparse ve yorumlanabilir kalsın

Özellikler:
- Correlation-based adjacency (threshold |corr| üstünden uygulanır)
- top-k sparsification per node
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
    """Return korelasyonundan sparse adjacency üretir.

    Adımlar:
    1. log-return hesapla
    2. corr matrix çıkar
    3. |corr| al
    4. threshold uygula
    5. her node için top-k komşu bırak
    6. simetrikleştir

    Args:
        window: kaç son günün return'ü kullanılacak.
        threshold: minimum |corr| eşiği.
        top_k: her node için maksimum komşu sayısı.
        use_absolute_corr: True ise edge weight=|corr|, False ise sign korunur.
        add_self_loops: diagonal 1 yapılsın mı.
    """

    def __init__(
        self,
        window: int = 60,
        threshold: float = 0.25,
        top_k: int = 5,
        use_absolute_corr: bool = True,
        add_self_loops: bool = True,
    ):
        self.window = window
        self.threshold = threshold
        self.top_k = top_k
        self.use_absolute_corr = use_absolute_corr
        self.add_self_loops = add_self_loops

    def _compute_returns(self, price_history: np.ndarray) -> np.ndarray:
        safe_prices = np.maximum(price_history, 1e-8)
        returns = np.diff(np.log(safe_prices), axis=0)
        if returns.shape[0] > self.window:
            returns = returns[-self.window :]
        return returns

    def _topk_per_row(self, adj: np.ndarray) -> np.ndarray:
        if self.top_k is None or self.top_k <= 0:
            return adj

        pruned = np.zeros_like(adj)
        n = adj.shape[0]
        for i in range(n):
            row = adj[i].copy()
            row[i] = 0.0
            row_magnitude = np.abs(row)
            nonzero_idx = np.flatnonzero(row_magnitude > 0)
            if nonzero_idx.size == 0:
                continue

            if nonzero_idx.size <= self.top_k:
                pruned[i, nonzero_idx] = row[nonzero_idx]
                continue

            keep_idx_local = np.argpartition(row_magnitude, -self.top_k)[-self.top_k :]
            keep_idx = keep_idx_local[row_magnitude[keep_idx_local] > 0]
            pruned[i, keep_idx] = row[keep_idx]
        return pruned

    def _symmetrize_adj(self, adj: np.ndarray) -> np.ndarray:
        """Daha güçlü kenarı iki yöne de taşıyarak simetrileştir."""
        stronger = np.where(np.abs(adj) > np.abs(adj.T), adj, adj.T)
        tie_mask = np.isclose(np.abs(adj), np.abs(adj.T))
        averaged = 0.5 * (adj + adj.T)
        return np.where(tie_mask, averaged, stronger)

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
            threshold_ref = adj
        else:
            adj = corr
            threshold_ref = np.abs(corr)

        if self.threshold is not None:
            adj = np.where(threshold_ref >= self.threshold, adj, 0.0)

        adj = self._topk_per_row(adj)
        adj = self._symmetrize_adj(adj)
        adj = self._topk_per_row(adj)
        adj = self._symmetrize_adj(adj)

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
        corr_threshold: float = 0.25,
        top_k: int = 5,
        use_absolute_corr: bool = True,
        add_self_loops: bool = True,
    ):
        super().__init__()
        self.builder = CorrelationAdjacency(
            window=corr_window,
            threshold=corr_threshold,
            top_k=top_k,
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
