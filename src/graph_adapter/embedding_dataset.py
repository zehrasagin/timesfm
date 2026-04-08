"""Dataset helpers for training from pre-computed Torch embeddings."""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


class EmbeddingPositionDataset(Dataset):
    """Training samples keyed by embedding-store position."""

    def __init__(
        self,
        all_data: np.ndarray,
        positions: List[int],
        target_idx: int = 0,
        corr_lookback: int = 90,
        target_mode: str = "log_return",
    ):
        self.all_data = all_data
        self.positions = positions
        self.target_idx = target_idx
        self.corr_lookback = corr_lookback
        self.target_mode = target_mode

    def __len__(self) -> int:
        return len(self.positions)

    def __getitem__(self, idx: int) -> Dict:
        t = self.positions[idx]

        hist_start = max(0, t - self.corr_lookback)
        price_history = self.all_data[hist_start:t]

        last_price = float(self.all_data[t - 1, self.target_idx])
        current_price = float(self.all_data[t, self.target_idx])

        if self.target_mode == "log_return":
            target = float(np.log(current_price / last_price))
        else:
            target = current_price - last_price

        return {
            "position": t,
            "price_history": price_history,
            "target": target,
            "last_price": last_price,
        }


def embedding_collate_fn(batch: List[Dict]) -> Dict:
    return {
        "positions": [item["position"] for item in batch],
        "price_histories": [item["price_history"] for item in batch],
        "targets": torch.tensor(
            [item["target"] for item in batch],
            dtype=torch.float32,
        ),
        "last_prices": torch.tensor(
            [item["last_price"] for item in batch],
            dtype=torch.float32,
        ),
    }


def create_embedding_dataloaders(
    all_data: np.ndarray,
    train_positions: List[int],
    val_positions: List[int],
    target_idx: int = 0,
    batch_size: int = 64,
    corr_lookback: int = 90,
    target_mode: str = "log_return",
    seed: Optional[int] = None,
) -> Tuple[DataLoader, DataLoader]:
    """Create train/val DataLoaders keyed by embedding-store positions."""
    train_dataset = EmbeddingPositionDataset(
        all_data,
        train_positions,
        target_idx,
        corr_lookback,
        target_mode,
    )
    val_dataset = EmbeddingPositionDataset(
        all_data,
        val_positions,
        target_idx,
        corr_lookback,
        target_mode,
    )
    generator = None
    if seed is not None:
        generator = torch.Generator()
        generator.manual_seed(int(seed))

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=embedding_collate_fn,
        num_workers=0,
        drop_last=True,
        generator=generator,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=embedding_collate_fn,
        num_workers=0,
    )

    return train_loader, val_loader
