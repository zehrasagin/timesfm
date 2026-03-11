"""
Cached Embedding Dataset — TSFM-Graph Adapter V2
==================================================

Pre-computed embedding cache ile çalışan hafif dataset.
TimesFM forward pass YAPILMAZ — sadece cache'den okur.

Her sample:
  Input:  target_seq_embeddings (P, 1280) + price_history (korelasyon adjacency için)
  Target: Hedef asset'in log-return'ü: ln(P_t / P_{t-1})

Training loop sadece graph adapter bileşenlerini çalıştırır (~3.6M param).
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple, Dict, Optional


class CachedEmbeddingDataset(Dataset):
    """Pre-computed embedding'lerle çalışan training dataset.

    EmbeddingCache'den target seq embedding'lerini okur.
    Training loop'ta TimesFM hiç çalışmaz → ~50x hızlanma.

    Args:
        embedding_cache: Dolu EmbeddingCache instance.
        all_data: (T_total, N) tüm asset fiyatları.
        positions: Context window bitiş indeksleri.
        target_idx: Hedef asset indeksi.
        corr_lookback: Korelasyon adj için ekstra lookback.
    """

    def __init__(
        self,
        embedding_cache,
        all_data: np.ndarray,
        positions: List[int],
        target_idx: int = 0,
        corr_lookback: int = 90,
        target_mode: str = "log_return",
    ):
        self.cache = embedding_cache
        self.all_data = all_data
        self.positions = positions
        self.target_idx = target_idx
        self.corr_lookback = corr_lookback
        self.target_mode = target_mode  # "delta" or "log_return"

    def __len__(self) -> int:
        return len(self.positions)

    def __getitem__(self, idx: int) -> Dict:
        """Tek bir training sample döndür (cache'den).

        Returns:
            dict with:
                'position': int — cache key
                'price_history': (T, N) numpy — korelasyon hesabı için
                'target': float — log_return ln(P_t/P_{t-1}) veya delta
                'last_price': float — P_{t-1} (fiyat reconstruction için)
        """
        t = self.positions[idx]

        # Price history (korelasyon adjacency için)
        hist_start = max(0, t - self.corr_lookback)
        price_history = self.all_data[hist_start:t]

        # Target calculation
        last_price = float(self.all_data[t - 1, self.target_idx])
        current_price = float(self.all_data[t, self.target_idx])

        if self.target_mode == "log_return":
            target = float(np.log(current_price / last_price))
        else:  # delta
            target = current_price - last_price

        return {
            "position": t,
            "price_history": price_history,
            "target": target,
            "last_price": last_price,
        }


def cached_collate_fn(batch: List[Dict]) -> Dict:
    """Cached dataset için collate function."""
    return {
        "positions": [item["position"] for item in batch],
        "price_histories": [item["price_history"] for item in batch],
        "targets": torch.tensor(
            [item["target"] for item in batch], dtype=torch.float32
        ),
        "last_prices": torch.tensor(
            [item["last_price"] for item in batch], dtype=torch.float32
        ),
    }


def create_cached_dataloaders(
    embedding_cache,
    all_data: np.ndarray,
    train_positions: List[int],
    val_positions: List[int],
    target_idx: int = 0,
    batch_size: int = 64,
    corr_lookback: int = 90,
    target_mode: str = "log_return",
) -> Tuple[DataLoader, DataLoader]:
    """Cache-based train ve val DataLoader'ları oluştur.

    Args:
        embedding_cache: Dolu EmbeddingCache.
        all_data: Tüm fiyat verisi.
        train_positions: Train window bitiş indeksleri.
        val_positions: Validation window bitiş indeksleri.
        target_idx: Hedef asset indeksi.
        batch_size: Batch boyutu (cache'de olduğu için daha büyük olabilir).
        corr_lookback: Korelasyon lookback.
        target_mode: "log_return" or "delta".

    Returns:
        train_loader, val_loader
    """
    train_dataset = CachedEmbeddingDataset(
        embedding_cache, all_data, train_positions,
        target_idx, corr_lookback, target_mode,
    )
    val_dataset = CachedEmbeddingDataset(
        embedding_cache, all_data, val_positions,
        target_idx, corr_lookback, target_mode,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=cached_collate_fn,
        num_workers=0,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=cached_collate_fn,
        num_workers=0,
    )

    return train_loader, val_loader
