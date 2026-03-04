"""
Embedding Cache — TSFM-Graph Adapter
======================================

Frozen TimesFM backbone'dan SADECE target asset'in sequence embedding'ini
ÖNCEden hesaplar ve cache'ler.

Neden?
  Backbone frozen → aynı input → aynı output. Her epoch'ta tekrar
  hesaplamak gereksiz. Bir kez hesapla, N epoch boyunca yeniden kullan.

Kazanım:
  Training: ~1.75s/batch → ~0.01s/batch (sadece graph adapter çalışır)
  50 epoch: ~9 saat → ~15 dakika (pre-compute dahil)

Akış:
  1. Tüm sliding window pozisyonları için TimesFM'i çalıştır (SADECE target asset)
  2. Her pozisyon için kaydet:
     - target_seq_emb: (P, 1280) — target asset'in patch embedding'leri
  3. Training loop cache'den okur, TimesFM çalıştırmaz

Kullanım:
  - Cross-attention Q: target_seq_emb → (P, 1280)
  - GNN node features: cache'den GELMEZ — NodeFeatureBuilder handcrafted üretir
"""

import os
import numpy as np
import torch
from typing import List, Tuple, Optional
from tqdm import tqdm


class EmbeddingCache:
    """Pre-computed embedding cache for efficient training.

    TimesFM'den çıkan SADECE target asset'in sequence embedding'ini tüm
    training/test window'ları için önceden hesaplar ve memory'de tutar.

    Her pozisyon t için:
      target_seq[t]: (P, 1280) — target asset'in patch embedding'leri

    GNN node features bu cache'den GELMEZ.
    GNN tamamen handcrafted feature kullanır (NodeFeatureBuilder).

    Args:
        embedding_extractor: TimesFMEmbeddingExtractor instance.
        max_context: Maksimum context uzunluğu.
        target_idx: Hedef asset indeksi.
    """

    def __init__(
        self,
        embedding_extractor,
        max_context: int = 1024,
        target_idx: int = 0,
    ):
        self.extractor = embedding_extractor
        self.max_context = max_context
        self.target_idx = target_idx

        # Cache storage
        self.target_seq_cache = {}  # t → (P, 1280) numpy
        self.is_built = False

    def build(
        self,
        all_data: np.ndarray,
        positions: List[int],
        batch_size: int = 32,
        save_path: Optional[str] = None,
    ) -> None:
        """Tüm pozisyonlar için embedding'leri pre-compute et.

        Args:
            all_data: (T_total, N) tüm asset fiyatları.
            positions: Context window bitiş indeksleri listesi.
                       Her pozisyon t: context = all_data[t-ctx:t]
            batch_size: Kaç pozisyonu aynı anda işle (memory vs speed).
            save_path: Disk'e kaydet (opsiyonel).
        """
        N = all_data.shape[1]
        ctx = self.max_context

        print(f"  Pre-computing {len(positions)} embeddings...")
        print(f"  Context: {ctx}, Target idx: {self.target_idx}")

        for t in tqdm(positions, desc="  Caching embeddings"):
            if t in self.target_seq_cache:
                continue  # Zaten cache'lenmiş

            # Context window
            start = max(0, t - ctx)
            context_data = all_data[start:t]  # (ctx, N) veya daha kısa

            # Sadece target asset'in serisini TimesFM'e ver
            target_series = context_data[:, self.target_idx]

            # TimesFM forward (frozen, no_grad inside) — tek asset
            seq_emb = self.extractor.extract_single(
                target_series, ctx
            )
            # seq_emb: (P, 1280)

            # CPU'ya taşı ve numpy'a çevir (memory tasarrufu)
            self.target_seq_cache[t] = seq_emb.cpu().numpy()

        self.is_built = True

        # Memory raporu
        n_pos = len(self.target_seq_cache)
        total_mb = sum(v.nbytes for v in self.target_seq_cache.values()) / 1e6
        print(f"  Cache built: {n_pos} positions")
        print(f"  Memory: {total_mb:.1f}MB (target asset only)")

        if save_path:
            self.save(save_path)

    def get(
        self, position: int, device: torch.device = torch.device("cpu")
    ) -> torch.Tensor:
        """Cache'den target asset'in sequence embedding'ini çek.

        Args:
            position: Context window bitiş indeksi.
            device: Hedef device.

        Returns:
            target_seq: (P, 1280) target asset'in full sequence (cross-attn Q).
        """
        target_seq = torch.from_numpy(
            self.target_seq_cache[position].copy()
        ).to(device)
        return target_seq

    def save(self, path: str) -> None:
        """Cache'i disk'e kaydet."""
        np.savez_compressed(
            path,
            seq_keys=np.array(list(self.target_seq_cache.keys())),
            seq_values=np.array(list(self.target_seq_cache.values())),
        )
        print(f"  Cache saved to {path}")

    def load(self, path: str) -> None:
        """Cache'i disk'ten yükle."""
        data = np.load(path)
        for k, v in zip(data["seq_keys"], data["seq_values"]):
            self.target_seq_cache[int(k)] = v
        self.is_built = True
        print(f"  Cache loaded from {path}: {len(self.target_seq_cache)} positions")

    def __contains__(self, position: int) -> bool:
        return position in self.target_seq_cache

    def __len__(self) -> int:
        return len(self.target_seq_cache)
