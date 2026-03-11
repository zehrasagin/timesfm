"""
Multi-Asset Time Series Dataset — TSFM-Graph Adapter
=====================================================

Sliding window yaklaşımı ile multi-asset zaman serisi verisi hazırlar.
Her sample:
  Input:  N asset × context_length fiyat penceresi
  Target: Hedef asset'in bir sonraki günkü fiyatı

Training için kullanılır: Graph Adapter sadece bu data üzerinde eğitilir.
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple, Optional, Dict


class MultiAssetDataset(Dataset):
    """Multi-asset sliding window dataset.

    CSV'den multi-asset fiyat verisi yükler ve training için
    sliding window sample'ları oluşturur.

    Her sample:
      context: [asset_1_prices, asset_2_prices, ..., asset_N_prices]
               Her biri (context_length,) boyutunda
      price_history: (context_length, N) — korelasyon hesabı için
      target: Hedef asset'in context_length+1'inci günkü fiyatı

    Args:
        prices_df: Multi-asset fiyat DataFrame'i.
        target_col: Hedef asset kolon adı.
        asset_cols: Tüm asset kolon adları (target dahil).
        context_length: Context window uzunluğu.
        horizon: Forecast horizon (şimdilik 1).
        corr_lookback: Korelasyon adjacency için ekstra lookback.
    """

    def __init__(
        self,
        prices_df: pd.DataFrame,
        target_col: str,
        asset_cols: List[str],
        context_length: int = 1024,
        horizon: int = 1,
        corr_lookback: int = 90,
    ):
        self.target_col = target_col
        self.asset_cols = asset_cols
        self.context_length = context_length
        self.horizon = horizon
        self.corr_lookback = corr_lookback

        # Target indeksi
        self.target_idx = asset_cols.index(target_col)
        self.n_assets = len(asset_cols)

        # Veriyi numpy'a çevir
        self.data = prices_df[asset_cols].values.astype(np.float32)  # (T, N)
        self.total_length = len(self.data)

        # Geçerli sample sayısı
        # corr_lookback + context_length + horizon kadar veri lazım
        self.start_offset = max(0, corr_lookback)
        self.n_samples = self.total_length - self.start_offset - context_length - horizon + 1

        if self.n_samples <= 0:
            raise ValueError(
                f"Yetersiz veri: {self.total_length} satır, "
                f"en az {self.start_offset + context_length + horizon} lazım."
            )

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int) -> Dict:
        """Tek bir training sample döndür.

        Returns:
            dict with:
                'context_series': List of N numpy arrays, each (context_length,)
                'price_history': (corr_lookback + context_length, N) — korelasyon için
                'target': float — hedef asset'in bir sonraki fiyatı
                'target_context_last': float — hedef asset'in son context fiyatı
        """
        # Absolute başlangıç indeksi
        abs_start = self.start_offset + idx
        context_end = abs_start + self.context_length
        target_end = context_end + self.horizon

        # Context: her asset'in fiyat penceresi
        context_data = self.data[abs_start:context_end]  # (context_length, N)

        # Korelasyon için price history (context dahil + lookback)
        hist_start = max(0, abs_start - self.corr_lookback)
        price_history = self.data[hist_start:context_end]  # (lookback+context, N)

        # Target: hedef asset'in bir sonraki fiyatı
        target = self.data[context_end:target_end, self.target_idx]  # (horizon,)

        # Her asset'i ayrı numpy array olarak listele
        context_series = [context_data[:, i] for i in range(self.n_assets)]

        return {
            "context_series": context_series,
            "price_history": price_history,
            "target": target[0] if self.horizon == 1 else target,
            "target_context_last": context_data[-1, self.target_idx],
        }


def collate_fn(batch: List[Dict]) -> Dict:
    """Custom collate: Liste bazlı batch oluştur.

    DataLoader'ın default collate'i numpy array listelerini
    düzgün handle edemez. Bu fonksiyon batch'i uygun formata getirir.

    Returns:
        dict with:
            'context_series': List of B items, each is List of N numpy arrays
            'price_histories': List of B numpy arrays
            'targets': (B,) tensor
            'target_context_lasts': (B,) tensor
    """
    return {
        "context_series": [item["context_series"] for item in batch],
        "price_histories": [item["price_history"] for item in batch],
        "targets": torch.tensor(
            [item["target"] for item in batch], dtype=torch.float32
        ),
        "target_context_lasts": torch.tensor(
            [item["target_context_last"] for item in batch], dtype=torch.float32
        ),
    }


def create_dataloaders(
    prices_df: pd.DataFrame,
    target_col: str,
    asset_cols: List[str],
    context_length: int = 1024,
    test_ratio: float = 0.1,
    batch_size: int = 16,
    num_workers: int = 0,
) -> Tuple[DataLoader, DataLoader, int]:
    """Train ve validation DataLoader'ları oluştur.

    Zaman serisi split: Son %test_ratio validation, geri kalan train.

    Args:
        prices_df: Fiyat DataFrame.
        target_col: Hedef asset.
        asset_cols: Tüm asset'ler.
        context_length: Context uzunluğu.
        test_ratio: Validation oranı.
        batch_size: Batch boyutu.
        num_workers: DataLoader worker sayısı.

    Returns:
        train_loader, val_loader, target_idx
    """
    total = len(prices_df)
    split_idx = int(total * (1 - test_ratio))

    train_df = prices_df.iloc[:split_idx].reset_index(drop=True)
    val_df = prices_df.iloc[split_idx:].reset_index(drop=True)  # Sadece test kısmı

    target_idx = asset_cols.index(target_col)

    train_dataset = MultiAssetDataset(
        train_df, target_col, asset_cols, context_length
    )

    # Validation: sadece son kısımdaki sample'lar
    val_dataset = MultiAssetDataset(
        val_df, target_col, asset_cols, context_length
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=num_workers,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=num_workers,
    )

    return train_loader, val_loader, target_idx
