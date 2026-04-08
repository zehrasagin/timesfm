"""Optional PyTorch Lightning wrapper for embedding-store training."""


from __future__ import annotations

from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

try:
    import lightning.pytorch as pl
except ModuleNotFoundError:
    try:
        import pytorch_lightning as pl
    except ModuleNotFoundError:
        pl = None


LIGHTNING_AVAILABLE = pl is not None
_BaseLightningModule = pl.LightningModule if LIGHTNING_AVAILABLE else nn.Module


class EmbeddingStoreLightningModule(_BaseLightningModule):
    """Lightning adapter around models that implement ``forward_with_embeddings``.

    The underlying graph adapter stays unchanged. This wrapper owns only the
    training step, validation step, optimizer, and scheduler wiring.
    """

    def __init__(
        self,
        model: nn.Module,
        embedding_store,
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-4,
        num_epochs: int = 15,
    ):
        if not LIGHTNING_AVAILABLE:
            raise ImportError(
                "PyTorch Lightning is not installed. Install `lightning` or "
                "`pytorch-lightning` before enabling TRAINING_CONFIG"
                "['use_pytorch_lightning']."
            )

        super().__init__()
        self.model = model
        self.embedding_store = embedding_store
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.num_epochs = num_epochs
        self.save_hyperparameters(ignore=["model", "embedding_store"])

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def _model_uses_temporal_embeddings(self) -> bool:
        return bool(getattr(self.model, "uses_temporal_embeddings", True))

    def _predict_batch(self, batch: Dict) -> torch.Tensor:
        batch_preds: List[torch.Tensor] = []
        for i, position in enumerate(batch["positions"]):
            target_seq = None
            if self._model_uses_temporal_embeddings():
                if self.embedding_store is None:
                    raise ValueError(
                        f"{self.model.__class__.__name__} requires temporal "
                        "embeddings, but no embedding_store was provided."
                    )
                target_seq = self.embedding_store.get(int(position), self.device)
            pred = self.model.forward_with_embeddings(
                target_seq_embeddings=target_seq,
                price_history=batch["price_histories"][i],
            )
            batch_preds.append(pred.squeeze())
        return torch.stack(batch_preds)

    def _shared_step(self, batch: Dict, stage: str) -> torch.Tensor:
        targets = batch["targets"].to(self.device)
        predictions = self._predict_batch(batch)
        loss = F.smooth_l1_loss(predictions, targets)
        self.log(
            f"{stage}_loss",
            loss,
            on_step=(stage == "train"),
            on_epoch=True,
            prog_bar=True,
            batch_size=len(batch["positions"]),
        )
        return loss

    def training_step(self, batch: Dict, batch_idx: int) -> torch.Tensor:
        del batch_idx
        return self._shared_step(batch, "train")

    def validation_step(self, batch: Dict, batch_idx: int) -> torch.Tensor:
        del batch_idx
        return self._shared_step(batch, "val")

    def configure_optimizers(self):
        optimizer = AdamW(
            self.model.get_trainable_params(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=max(self.num_epochs, 1),
            eta_min=1e-6,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
            },
        }
