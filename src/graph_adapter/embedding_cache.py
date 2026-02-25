"""
Embedding Cache — TSFM-Graph Adapter
======================================

Frozen TimesFM backbone'dan embedding'leri ÖNCEden hesaplar ve cache'ler.

Neden?
  Backbone frozen → aynı input → aynı output. Her epoch'ta tekrar
  hesaplamak gereksiz. Bir kez hesapla, N epoch boyunca yeniden kullan.

Kazanım:
  Training: ~1.75s/batch → ~0.01s/batch (sadece graph adapter çalışır)
  50 epoch: ~9 saat → ~15 dakika (pre-compute dahil)

Akış:
  1. Tüm sliding window pozisyonları için TimesFM'i çalıştır
  2. Her pozisyon için kaydet:
     - pooled_embeddings: (N, 1280) — GAT input
     - target_seq_embeddings: (P, 1280) — Cross-attention query
  3. Training loop cache'den okur, TimesFM çalıştırmaz
"""

import os
import numpy as np
import torch
from typing import List, Tuple, Optional
from tqdm import tqdm


class EmbeddingCache:
    """Pre-computed embedding cache for efficient training.

    TimesFM'den çıkan embedding'leri tüm training/test window'ları için
    önceden hesaplar ve memory'de tutar.

    Her pozisyon t için:
      pooled[t]: (N, 1280) — tüm asset'lerin pooled embedding'leri (GAT input)
      target_seq[t]: (P, 1280) — target asset'in sequence embedding'leri

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
        self.pooled_cache = {}     # t → (N, 1280) numpy
        self.target_seq_cache = {} # t → (P, 1280) numpy
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
        print(f"  Context: {ctx}, Assets: {N}, Target idx: {self.target_idx}")

        for t in tqdm(positions, desc="  Caching embeddings"):
            if t in self.pooled_cache:
                continue  # Zaten cache'lenmiş

            # Context window
            start = max(0, t - ctx)
            context_data = all_data[start:t]  # (ctx, N) veya daha kısa

            # Her asset'in serisi
            series_list = [context_data[:, i] for i in range(N)]

            # TimesFM forward (frozen, no_grad inside)
            seq_emb, pooled = self.extractor.extract_embeddings(
                series_list, ctx
            )
            # seq_emb: (N, P, 1280), pooled: (N, 1280)

            # CPU'ya taşı ve numpy'a çevir (memory tasarrufu)
            self.pooled_cache[t] = pooled.cpu().numpy()
            self.target_seq_cache[t] = seq_emb[self.target_idx].cpu().numpy()

        self.is_built = True

        # Memory raporu
        n_pos = len(self.pooled_cache)
        pooled_mb = sum(v.nbytes for v in self.pooled_cache.values()) / 1e6
        seq_mb = sum(v.nbytes for v in self.target_seq_cache.values()) / 1e6
        print(f"  Cache built: {n_pos} positions")
        print(f"  Memory: pooled={pooled_mb:.1f}MB + seq={seq_mb:.1f}MB = {pooled_mb+seq_mb:.1f}MB")

        if save_path:
            self.save(save_path)

    def get(
        self, position: int, device: torch.device = torch.device("cpu")
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Cache'den embedding çek.

        Args:
            position: Context window bitiş indeksi.
            device: Hedef device.

        Returns:
            pooled: (N, 1280) tensor
            target_seq: (P, 1280) tensor
        """
        pooled = torch.tensor(self.pooled_cache[position], device=device)
        target_seq = torch.tensor(self.target_seq_cache[position], device=device)
        return pooled, target_seq

    def save(self, path: str) -> None:
        """Cache'i disk'e kaydet."""
        np.savez_compressed(
            path,
            pooled_keys=np.array(list(self.pooled_cache.keys())),
            pooled_values=np.array(list(self.pooled_cache.values())),
            seq_keys=np.array(list(self.target_seq_cache.keys())),
            seq_values=np.array(list(self.target_seq_cache.values())),
        )
        print(f"  Cache saved to {path}")

    def load(self, path: str) -> None:
        """Cache'i disk'ten yükle."""
        data = np.load(path)
        for k, v in zip(data["pooled_keys"], data["pooled_values"]):
            self.pooled_cache[int(k)] = v
        for k, v in zip(data["seq_keys"], data["seq_values"]):
            self.target_seq_cache[int(k)] = v
        self.is_built = True
        print(f"  Cache loaded from {path}: {len(self.pooled_cache)} positions")

    def __contains__(self, position: int) -> bool:
        return position in self.pooled_cache

    def __len__(self) -> int:
        return len(self.pooled_cache)
