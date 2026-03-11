
"""
Commodity Graph Dataset v2
==========================

Bu dataset, "her node = bir commodity" varsayımına uygun olacak şekilde
CSV içindeki kolonları ikiye ayırır:

1) node_cols   : graph node'ları olacak commodity fiyat serileri
2) global_cols : macro / FX / equity / index gibi opsiyonel exogenous seriler

Temel kararlar:
- Target default olarak raw price DEĞİL, log-return'dür.
- Correlation graph sadece node_cols üstünden kurulur.
- Eğer global_cols verilirse, bunlar graph node'u yapılmaz; ayrı context olarak döner.
- Validation/Test split'i dataframe'i parçalayarak değil, sample-position üzerinden yapılır.

Beklenen kullanım:
    dataset = CommodityGraphDataset(
        df,
        target_col="GC1 Comdty",
        node_cols=[... commodity columns ...],
        global_cols=[... optional exogenous columns ...],
        context_length=96,
        horizon=1,
        corr_lookback=90,
        target_mode="log_return",
    )

    sample = dataset[0]
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, Subset


def infer_column_groups(columns: Sequence[str]) -> Dict[str, List[str]]:
    """Basit suffix kurallarıyla kolonları kaba gruplara ayır."""
    cols = list(columns)
    return {
        "commodity": [c for c in cols if c.endswith("Comdty")],
        "equity": [c for c in cols if c.endswith("Equity")],
        "currency": [c for c in cols if c.endswith("Curncy")],
        "index_or_govt": [c for c in cols if c.endswith("Index") or c.endswith("Govt")],
    }


def default_node_cols_from_csv(columns: Sequence[str]) -> List[str]:
    """
    Node'lar için default seçim:
    Sadece '... Comdty' kolonlarını alır.
    """
    groups = infer_column_groups(columns)
    node_cols = groups["commodity"]
    if not node_cols:
        raise ValueError("Hiç commodity kolonu bulunamadı. node_cols'u elle ver.")
    return node_cols


def clean_and_align_dataframe(
    df: pd.DataFrame,
    feature_cols: Sequence[str],
    date_col: Optional[str] = "date",
    fill_method: str = "ffill_bfill",
) -> pd.DataFrame:
    """
    Tarih sıralama, numeric cast ve NaN doldurma yapar.
    """
    out = df.copy()

    if date_col is not None and date_col in out.columns:
        out[date_col] = pd.to_datetime(out[date_col], errors="coerce")
        out = out.sort_values(date_col).reset_index(drop=True)

    missing = [c for c in feature_cols if c not in out.columns]
    if missing:
        raise ValueError(f"DataFrame'de bulunmayan kolonlar var: {missing[:10]}")

    for c in feature_cols:
        out[c] = pd.to_numeric(out[c], errors="coerce")

    if fill_method == "ffill_bfill":
        out[list(feature_cols)] = out[list(feature_cols)].ffill().bfill()
    elif fill_method == "drop":
        out = out.dropna(subset=list(feature_cols)).reset_index(drop=True)
    else:
        raise ValueError(f"Bilinmeyen fill_method: {fill_method}")

    # Hâlâ NaN kaldıysa sorun çıkar
    if out[list(feature_cols)].isna().any().any():
        bad_cols = out[list(feature_cols)].columns[out[list(feature_cols)].isna().any()].tolist()
        raise ValueError(f"Temizlik sonrası hâlâ NaN var. Problemli kolonlar: {bad_cols[:10]}")

    return out


def make_time_split_indices(
    n_samples: int,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Sample indexleri üzerinde zaman sıralı split yapar.
    """
    if not (0 < train_ratio < 1 and 0 < val_ratio < 1 and train_ratio + val_ratio < 1):
        raise ValueError("train_ratio ve val_ratio geçersiz.")

    train_end = int(n_samples * train_ratio)
    val_end = int(n_samples * (train_ratio + val_ratio))

    train_idx = np.arange(0, train_end)
    val_idx = np.arange(train_end, val_end)
    test_idx = np.arange(val_end, n_samples)
    return train_idx, val_idx, test_idx


class CommodityGraphDataset(Dataset):
    """
    Commodity graph forecasting dataset.

    Her sample şunları döner:
        - node_context: (N, T)
        - price_history: (L, N)  -> graph/correlation için
        - global_context: (T, G) veya None
        - target_value: scalar (log_return / delta / price)
        - last_price: scalar
        - target_price: scalar
        - target_idx: int
        - node_cols, global_cols
        - sample_end_pos: int
    """

    def __init__(
        self,
        prices_df: pd.DataFrame,
        target_col: str,
        node_cols: Optional[List[str]] = None,
        global_cols: Optional[List[str]] = None,
        context_length: int = 96,
        horizon: int = 1,
        corr_lookback: int = 90,
        target_mode: str = "log_return",   # {"log_return", "delta", "price"}
        date_col: Optional[str] = "date",
        fill_method: str = "ffill_bfill",
        eps: float = 1e-8,
    ):
        if horizon != 1:
            raise NotImplementedError("Şimdilik sadece horizon=1 destekleniyor.")

        if node_cols is None:
            node_cols = default_node_cols_from_csv(prices_df.columns)

        if target_col not in node_cols:
            raise ValueError(
                f"target_col='{target_col}' node_cols içinde olmalı. "
                f"Şu an node_cols içinde değil."
            )

        if global_cols is None:
            global_cols = []

        # Aynı kolon hem node hem global olmasın
        overlap = set(node_cols).intersection(global_cols)
        if overlap:
            raise ValueError(f"Aynı kolon hem node hem global verildi: {sorted(overlap)[:10]}")

        self.target_col = target_col
        self.node_cols = list(node_cols)
        self.global_cols = list(global_cols)
        self.context_length = context_length
        self.horizon = horizon
        self.corr_lookback = corr_lookback
        self.target_mode = target_mode
        self.date_col = date_col
        self.eps = eps

        all_feature_cols = self.node_cols + self.global_cols
        self.df = clean_and_align_dataframe(
            prices_df,
            feature_cols=all_feature_cols,
            date_col=date_col,
            fill_method=fill_method,
        )

        self.node_data = self.df[self.node_cols].to_numpy(dtype=np.float32)     # (T_total, N)
        self.global_data = (
            self.df[self.global_cols].to_numpy(dtype=np.float32) if self.global_cols else None
        )

        self.target_idx = self.node_cols.index(self.target_col)
        self.n_nodes = len(self.node_cols)
        self.n_global = 0 if self.global_data is None else self.global_data.shape[1]
        self.total_length = len(self.df)

        # sample_end_pos = target zamanının bir önceki index'i değil,
        # context'in bittiği anın index-exclusive sınırı gibi düşünülebilir.
        # context = [start : context_end)
        # target  = context_end
        self.start_offset = max(1, self.corr_lookback)
        self.n_samples = self.total_length - self.start_offset - self.context_length - self.horizon + 1

        if self.n_samples <= 0:
            need = self.start_offset + self.context_length + self.horizon
            raise ValueError(
                f"Yetersiz veri: toplam {self.total_length} satır var, "
                f"en az {need} satır gerekli."
            )

    def __len__(self) -> int:
        return self.n_samples

    def _compute_target(self, last_price: float, target_price: float) -> float:
        if self.target_mode == "price":
            return float(target_price)
        if self.target_mode == "delta":
            return float(target_price - last_price)
        if self.target_mode == "log_return":
            return float(np.log((target_price + self.eps) / (last_price + self.eps)))
        raise ValueError(f"Bilinmeyen target_mode: {self.target_mode}")

    def __getitem__(self, idx: int) -> Dict:
        abs_start = self.start_offset + idx
        context_end = abs_start + self.context_length
        target_t = context_end  # horizon=1

        node_context = self.node_data[abs_start:context_end].T                 # (N, T)
        hist_start = max(0, abs_start - self.corr_lookback)
        price_history = self.node_data[hist_start:context_end]                 # (L, N)

        if self.global_data is not None:
            global_context = self.global_data[abs_start:context_end]           # (T, G)
        else:
            global_context = None

        last_price = float(self.node_data[context_end - 1, self.target_idx])
        target_price = float(self.node_data[target_t, self.target_idx])
        target_value = self._compute_target(last_price, target_price)

        out = {
            "node_context": node_context,              # (N, T)
            "context_series": [node_context[i] for i in range(self.n_nodes)],  # legacy-friendly
            "price_history": price_history,            # (L, N)
            "global_context": global_context,          # (T, G) or None
            "target_value": target_value,
            "target": target_value,                    # legacy alias
            "last_price": last_price,
            "target_price": target_price,
            "target_idx": self.target_idx,
            "sample_end_pos": context_end,
            "node_cols": self.node_cols,
            "global_cols": self.global_cols,
        }

        if self.date_col is not None and self.date_col in self.df.columns:
            out["target_date"] = self.df.iloc[target_t][self.date_col]

        return out


def commodity_collate_fn(batch: List[Dict]) -> Dict:
    """
    Heterogeneous alanlar içerdiği için özel collate.
    price_history uzunluğu batch içinde farklı olabilir; list olarak bırakıyoruz.
    """
    global_contexts = [item["global_context"] for item in batch]
    has_global = all(gc is not None for gc in global_contexts)

    out = {
        "node_contexts": torch.tensor(
            np.stack([item["node_context"] for item in batch], axis=0),
            dtype=torch.float32,
        ),  # (B, N, T)
        "context_series_list": [item["context_series"] for item in batch],  # legacy
        "price_histories": [item["price_history"] for item in batch],
        "targets": torch.tensor([item["target_value"] for item in batch], dtype=torch.float32),
        "last_prices": torch.tensor([item["last_price"] for item in batch], dtype=torch.float32),
        "target_prices": torch.tensor([item["target_price"] for item in batch], dtype=torch.float32),
        "target_indices": torch.tensor([item["target_idx"] for item in batch], dtype=torch.long),
        "sample_end_positions": torch.tensor([item["sample_end_pos"] for item in batch], dtype=torch.long),
        "node_cols": batch[0]["node_cols"],
        "global_cols": batch[0]["global_cols"],
    }

    if has_global:
        out["global_contexts"] = torch.tensor(
            np.stack(global_contexts, axis=0),
            dtype=torch.float32,
        )  # (B, T, G)
    else:
        out["global_contexts"] = None

    if "target_date" in batch[0]:
        out["target_dates"] = [item["target_date"] for item in batch]

    return out


def create_dataloaders_from_dataset(
    dataset: CommodityGraphDataset,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    batch_size: int = 32,
    num_workers: int = 0,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Sample-position tabanlı time split.
    """
    train_idx, val_idx, test_idx = make_time_split_indices(
        len(dataset),
        train_ratio=train_ratio,
        val_ratio=val_ratio,
    )

    train_ds = Subset(dataset, train_idx.tolist())
    val_ds = Subset(dataset, val_idx.tolist())
    test_ds = Subset(dataset, test_idx.tolist())

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=commodity_collate_fn,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=commodity_collate_fn,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=commodity_collate_fn,
    )

    return train_loader, val_loader, test_loader
