"""
Torch Embedding Store — TSFM-Graph Adapter V2
=============================================

Frozen TimesFM backbone'dan SADECE target asset'in sequence embedding'ini
önceden hesaplar ve Torch `.pt` dosyası olarak diske kaydeder.

Akış:
  1. Tüm sliding window pozisyonları için TimesFM'i çalıştır.
  2. Her pozisyon için target_seq_emb: (P, 1280) CPU tensor olarak tut.
  3. `torch.save` ile tek `.pt` dosyasına yaz.
  4. Training loop `.pt` store'dan okur, TimesFM çalıştırmaz.
"""

"CACHE YERİNE"


from __future__ import annotations

from typing import List, Optional

import torch
from tqdm import tqdm

EMBEDDING_STORE_FORMAT_VERSION = 2


class TorchEmbeddingStore:
    """Pre-computed target embeddings stored in a Torch `.pt` file."""

    def __init__(
        self,
        embedding_extractor,
        max_context: int = 1024,
        target_idx: int = 0,
    ):
        self.extractor = embedding_extractor
        self.max_context = max_context
        self.target_idx = target_idx
        self.target_seq_embeddings: dict[int, torch.Tensor] = {}
        self.is_built = False

    def build(
        self,
        all_data,
        positions: List[int],
        save_path: Optional[str] = None,
    ) -> None:
        """Compute embeddings for positions and optionally save to `.pt`."""
        ctx = self.max_context

        print(f"  Pre-computing {len(positions)} embeddings...")
        print(f"  Context: {ctx}, Target idx: {self.target_idx}")

        for t in tqdm(positions, desc="  Writing embeddings"):
            if t in self.target_seq_embeddings:
                continue

            start = max(0, t - ctx)
            context_data = all_data[start:t]
            target_series = context_data[:, self.target_idx]
            seq_emb = self.extractor.extract_single(target_series, ctx)
            self.target_seq_embeddings[t] = seq_emb.detach().cpu().contiguous()

        self.is_built = True
        total_mb = sum(
            tensor.numel() * tensor.element_size()
            for tensor in self.target_seq_embeddings.values()
        ) / 1e6
        print(f"  Embedding store built: {len(self.target_seq_embeddings)} positions")
        print(f"  Memory: {total_mb:.1f}MB (target asset only, torch tensors)")

        if save_path:
            self.save(save_path)

    def get(
        self,
        position: int,
        device: torch.device = torch.device("cpu"),
    ) -> torch.Tensor:
        """Return target sequence embedding for one position."""
        return self.target_seq_embeddings[position].to(device)

    def save(self, path: str) -> None:
        """Save the store directly as a Torch `.pt` file."""
        payload = {
            "format_version": EMBEDDING_STORE_FORMAT_VERSION,
            "max_context": self.max_context,
            "target_idx": self.target_idx,
            "target_seq_embeddings": {
                int(k): v.detach().cpu()
                for k, v in self.target_seq_embeddings.items()
            },
        }
        torch.save(payload, path)
        print(f"  Embeddings saved to {path}")

    def load(self, path: str) -> None:
        """Load embeddings from a Torch `.pt` file."""
        try:
            payload = torch.load(path, map_location="cpu", weights_only=True)
        except TypeError:
            payload = torch.load(path, map_location="cpu")

        if not isinstance(payload, dict):
            raise ValueError("Embedding store payload is not a dict.")
        if payload.get("format_version") != EMBEDDING_STORE_FORMAT_VERSION:
            raise ValueError(
                f"Embedding store format mismatch: file has "
                f"{payload.get('format_version')}, current code needs "
                f"{EMBEDDING_STORE_FORMAT_VERSION}"
            )
        if "max_context" in payload and payload["max_context"] != self.max_context:
            raise ValueError(
                f"Embedding store max_context mismatch: file has "
                f"{payload['max_context']}, current run needs {self.max_context}"
            )
        if "target_idx" in payload and payload["target_idx"] != self.target_idx:
            raise ValueError(
                f"Embedding store target_idx mismatch: file has "
                f"{payload['target_idx']}, current run needs {self.target_idx}"
            )

        embeddings = payload.get("target_seq_embeddings", payload)
        self.target_seq_embeddings = {
            int(k): v.detach().cpu()
            for k, v in embeddings.items()
        }
        self.is_built = True
        print(
            f"  Embeddings loaded from {path}: "
            f"{len(self.target_seq_embeddings)} positions"
        )

    def __contains__(self, position: int) -> bool:
        return position in self.target_seq_embeddings

    def __len__(self) -> int:
        return len(self.target_seq_embeddings)


EmbeddingStore = TorchEmbeddingStore
