"""Training and forecasting utilities for Torch embedding-store runs."""

from __future__ import annotations

import os
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from graph_adapter.embedding_dataset import create_embedding_dataloaders
from graph_adapter.embedding_store import TorchEmbeddingStore

from .forecast_config import MODEL_CONFIG, OUTPUT_CONFIG, TRAINING_CONFIG
from .reporting_utils import (
    calculate_all_metrics,
    display_metrics,
    plot_training_curve,
    save_metrics_to_csv,
    visualize_forecast,
)


def precompute_embeddings(
    model_with_extractor: nn.Module,
    all_data: np.ndarray,
    positions: List[int],
    store_path: Optional[str] = None,
) -> TorchEmbeddingStore:
    """Pre-compute and optionally persist TimesFM embeddings."""
    embedding_store = TorchEmbeddingStore(
        embedding_extractor=model_with_extractor.embedding_extractor,
        max_context=MODEL_CONFIG["max_context"],
        target_idx=model_with_extractor.target_idx,
    )

    if store_path and os.path.exists(store_path):
        try:
            embedding_store.load(store_path)
            missing = [p for p in positions if p not in embedding_store]
            if not missing:
                print(f"  All {len(positions)} positions loaded from embedding store!")
                return embedding_store
            print(
                f"  {len(positions) - len(missing)} loaded, "
                f"{len(missing)} to compute..."
            )
            positions = missing
        except ValueError as exc:
            print(f"  Existing embedding store is incompatible: {exc}")
            print("  Rebuilding embeddings...")

    embedding_store.build(all_data, positions, save_path=store_path)
    return embedding_store


def model_uses_temporal_embeddings(model: nn.Module) -> bool:
    """Whether this downstream model needs TimesFM target embeddings."""
    return bool(getattr(model, "uses_temporal_embeddings", True))


def get_model_device(model: nn.Module) -> torch.device:
    """Infer the device from trainable model parameters."""
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")


def predict_with_optional_embeddings(
    model: nn.Module,
    embedding_store: Optional[TorchEmbeddingStore],
    position: int,
    price_history: np.ndarray,
    device: torch.device,
) -> torch.Tensor:
    """Run one model step, using target embeddings only when the model needs them."""
    target_seq = None
    if model_uses_temporal_embeddings(model):
        if embedding_store is None:
            raise ValueError(
                f"{model.__class__.__name__} requires temporal embeddings, "
                "but no embedding_store was provided."
            )
        target_seq = embedding_store.get(int(position), device)

    return model.forward_with_embeddings(
        target_seq_embeddings=target_seq,
        price_history=price_history,
    )


def build_optimizer_param_groups(
    model: nn.Module,
    learning_rate: float,
    weight_decay: float,
) -> List[Dict[str, object]]:
    """Create optimizer param groups while optionally slowing fusion learning."""
    fusion_lr_scale = float(TRAINING_CONFIG.get("fusion_lr_scale", 1.0))
    trainable_params = model.get_trainable_params()

    if (
        fusion_lr_scale >= 1.0
        or not hasattr(model, "fusion_adapter")
        or model.fusion_adapter is None
    ):
        return [
            {
                "params": trainable_params,
                "lr": learning_rate,
                "weight_decay": weight_decay,
            }
        ]

    fusion_param_ids = {
        id(param)
        for param in model.fusion_adapter.parameters()
        if param.requires_grad
    }
    main_params = []
    fusion_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad or "embedding_extractor" in name:
            continue
        if id(param) in fusion_param_ids:
            fusion_params.append(param)
        else:
            main_params.append(param)

    param_groups: List[Dict[str, object]] = []
    if main_params:
        param_groups.append(
            {
                "params": main_params,
                "lr": learning_rate,
                "weight_decay": weight_decay,
            }
        )
    if fusion_params:
        param_groups.append(
            {
                "params": fusion_params,
                "lr": learning_rate * fusion_lr_scale,
                "weight_decay": weight_decay,
            }
        )
    return param_groups


def outputs_to_price_space(
    outputs: torch.Tensor,
    last_prices: torch.Tensor,
) -> torch.Tensor:
    """Convert return/delta outputs back to comparable price space."""
    if MODEL_CONFIG["target_mode"] == "log_return":
        return last_prices * torch.exp(outputs)
    return last_prices + outputs


def compute_price_mape(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    last_prices: torch.Tensor,
) -> torch.Tensor:
    """Price-level MAPE used for checkpoint selection."""
    pred_prices = outputs_to_price_space(predictions, last_prices)
    actual_prices = outputs_to_price_space(targets, last_prices)
    denom = actual_prices.abs().clamp_min(1e-8)
    ape = (pred_prices - actual_prices).abs() / denom
    return ape.mean() * 100.0


def _import_lightning():
    try:
        import lightning.pytorch as pl
    except ModuleNotFoundError:
        try:
            import pytorch_lightning as pl
        except ModuleNotFoundError as exc:
            raise ImportError(
                "PyTorch Lightning is not installed. Install `lightning` or "
                "`pytorch-lightning`, then set "
                "TRAINING_CONFIG['use_pytorch_lightning'] = True."
            ) from exc
    return pl


def _load_lightning_best_weights(
    model: nn.Module,
    lightning_checkpoint_path: str,
) -> None:
    checkpoint = torch.load(
        lightning_checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )
    lightning_state = checkpoint.get("state_dict", {})

    model_state = model.state_dict()
    for key, value in lightning_state.items():
        if not key.startswith("model."):
            continue
        model_key = key.removeprefix("model.")
        if "embedding_extractor" in model_key:
            continue
        if model_key in model_state:
            model_state[model_key] = value
    model.load_state_dict(model_state)


def train_with_embedding_store_lightning(
    model: nn.Module,
    embedding_store: Optional[TorchEmbeddingStore],
    all_data: np.ndarray,
    train_positions: List[int],
    val_positions: List[int],
    target_idx: int,
    model_label: str,
    checkpoint_path: str,
) -> Dict[str, List[float]]:
    """Train with PyTorch Lightning when the optional dependency exists."""
    pl = _import_lightning()
    from graph_adapter.lightning_module import EmbeddingStoreLightningModule

    pl.seed_everything(TRAINING_CONFIG["seed"], workers=True)

    train_loader, val_loader = create_embedding_dataloaders(
        all_data,
        train_positions,
        val_positions,
        target_idx=target_idx,
        batch_size=TRAINING_CONFIG["batch_size"],
        corr_lookback=MODEL_CONFIG["corr_window"],
        target_mode=MODEL_CONFIG["target_mode"],
        seed=TRAINING_CONFIG["seed"],
    )

    monitor_metric = str(TRAINING_CONFIG.get("checkpoint_metric", "val_mape"))
    if monitor_metric not in {"val_loss", "val_mape"}:
        raise ValueError(
            "TRAINING_CONFIG['checkpoint_metric'] must be 'val_loss' or 'val_mape'."
        )

    history = {"train_loss": [], "val_loss": [], "val_mape": []}

    class LossHistoryCallback(pl.Callback):
        def on_train_epoch_end(self, trainer, pl_module) -> None:
            del pl_module
            metric = trainer.callback_metrics.get("train_loss_epoch")
            if metric is not None:
                history["train_loss"].append(float(metric.detach().cpu()))

        def on_validation_epoch_end(self, trainer, pl_module) -> None:
            del pl_module
            if trainer.sanity_checking:
                return
            val_loss = trainer.callback_metrics.get("val_loss")
            if val_loss is not None:
                history["val_loss"].append(float(val_loss.detach().cpu()))
            val_mape = trainer.callback_metrics.get("val_mape")
            if val_mape is not None:
                history["val_mape"].append(float(val_mape.detach().cpu()))

    checkpoint_dir = os.path.dirname(checkpoint_path) or "."
    checkpoint_stem, _ = os.path.splitext(os.path.basename(checkpoint_path))
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename=f"{checkpoint_stem}-lightning",
        monitor=monitor_metric,
        mode="min",
        save_top_k=1,
    )

    lightning_model = EmbeddingStoreLightningModule(
        model=model,
        embedding_store=embedding_store,
        learning_rate=TRAINING_CONFIG["learning_rate"],
        weight_decay=TRAINING_CONFIG["weight_decay"],
        num_epochs=TRAINING_CONFIG["num_epochs"],
        fusion_lr_scale=float(TRAINING_CONFIG.get("fusion_lr_scale", 1.0)),
        target_mode=str(MODEL_CONFIG["target_mode"]),
    )
    source_label = (
        "PYTORCH LIGHTNING + EMBEDDING STORE"
        if model_uses_temporal_embeddings(model)
        else "PYTORCH LIGHTNING + GRAPH FEATURES ONLY"
    )

    print(f"\n{'═' * 60}")
    print(f"TRAINING ({model_label} — {source_label})")
    print(f"  Train: {len(train_positions)} | Val: {len(val_positions)}")
    print(
        f"  Epochs: {TRAINING_CONFIG['num_epochs']} | "
        f"Batch: {TRAINING_CONFIG['batch_size']}"
    )
    print(f"  Checkpoint metric: {monitor_metric}")
    print(f"{'═' * 60}\n")

    trainer = pl.Trainer(
        max_epochs=TRAINING_CONFIG["num_epochs"],
        accelerator="auto",
        devices="auto",
        gradient_clip_val=TRAINING_CONFIG["grad_clip_norm"],
        logger=False,
        enable_model_summary=False,
        deterministic=True,
        callbacks=[LossHistoryCallback(), checkpoint_callback],
    )
    trainer.fit(lightning_model, train_loader, val_loader)

    best_path = checkpoint_callback.best_model_path
    best_checkpoint_score = float("inf")
    best_val_loss = float("inf")
    best_val_mape = float("inf")
    if best_path:
        _load_lightning_best_weights(model, best_path)
        if checkpoint_callback.best_model_score is not None:
            best_checkpoint_score = float(checkpoint_callback.best_model_score.cpu())
        if history["val_loss"]:
            best_val_loss = min(history["val_loss"])
        if history["val_mape"]:
            best_val_mape = min(history["val_mape"])

    torch.save(
        {
            "model_state_dict": {
                k: v for k, v in model.state_dict().items()
                if "embedding_extractor" not in k
            },
            "optimizer_state_dict": {},
            "epoch": max(TRAINING_CONFIG["num_epochs"] - 1, 0),
            "val_loss": best_val_loss,
            "val_mape": best_val_mape,
            "checkpoint_metric": monitor_metric,
            "checkpoint_score": best_checkpoint_score,
        },
        checkpoint_path,
    )
    print(f"  Lightning-compatible checkpoint saved to {checkpoint_path}")
    return history


def train_with_embedding_store(
    model: nn.Module,
    embedding_store: Optional[TorchEmbeddingStore],
    all_data: np.ndarray,
    train_positions: List[int],
    val_positions: List[int],
    target_idx: int,
    model_label: str,
    checkpoint_path: str,
) -> Dict[str, List[float]]:
    """Train from a Torch embedding store without running TimesFM in the loop."""
    if TRAINING_CONFIG.get("use_pytorch_lightning", False):
        return train_with_embedding_store_lightning(
            model,
            embedding_store,
            all_data,
            train_positions,
            val_positions,
            target_idx,
            model_label,
            checkpoint_path,
        )

    device = get_model_device(model)

    train_loader, val_loader = create_embedding_dataloaders(
        all_data,
        train_positions,
        val_positions,
        target_idx=target_idx,
        batch_size=TRAINING_CONFIG["batch_size"],
        corr_lookback=MODEL_CONFIG["corr_window"],
        target_mode=MODEL_CONFIG["target_mode"],
        seed=TRAINING_CONFIG["seed"],
    )

    trainable_params = model.get_trainable_params()
    optimizer = AdamW(
        build_optimizer_param_groups(
            model,
            learning_rate=TRAINING_CONFIG["learning_rate"],
            weight_decay=TRAINING_CONFIG["weight_decay"],
        )
    )
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=TRAINING_CONFIG["num_epochs"],
        eta_min=1e-6,
    )

    history = {"train_loss": [], "val_loss": [], "val_mape": []}
    monitor_metric = str(TRAINING_CONFIG.get("checkpoint_metric", "val_mape"))
    if monitor_metric not in {"val_loss", "val_mape"}:
        raise ValueError(
            "TRAINING_CONFIG['checkpoint_metric'] must be 'val_loss' or 'val_mape'."
        )
    best_checkpoint_score = float("inf")
    best_val_loss = float("inf")
    best_val_mape = float("inf")
    source_label = (
        "TORCH EMBEDDING STORE"
        if model_uses_temporal_embeddings(model)
        else "GRAPH FEATURES ONLY"
    )

    print(f"\n{'═' * 60}")
    print(f"TRAINING ({model_label} — {source_label})")
    print(f"  Train: {len(train_positions)} | Val: {len(val_positions)}")
    print(
        f"  Epochs: {TRAINING_CONFIG['num_epochs']} | "
        f"Batch: {TRAINING_CONFIG['batch_size']}"
    )
    print(f"  Batches/epoch: {len(train_loader)} train + {len(val_loader)} val")
    print(f"  LR: {TRAINING_CONFIG['learning_rate']}")
    print(f"  Checkpoint metric: {monitor_metric}")
    if hasattr(model, "fusion_adapter"):
        print(f"  Fusion LR scale: {TRAINING_CONFIG.get('fusion_lr_scale', 1.0)}")
    print(f"{'═' * 60}\n")

    total_start = time.time()
    for epoch in range(TRAINING_CONFIG["num_epochs"]):
        epoch_start = time.time()
        model.train()
        epoch_losses = []

        pbar = tqdm(
            train_loader,
            desc=f"Epoch {epoch+1}/{TRAINING_CONFIG['num_epochs']}",
            leave=False,
        )

        for batch in pbar:
            optimizer.zero_grad()

            positions = batch["positions"]
            price_histories = batch["price_histories"]
            targets = batch["targets"].to(device)

            batch_preds = []
            for i, position in enumerate(positions):
                pred = predict_with_optional_embeddings(
                    model,
                    embedding_store,
                    position,
                    price_histories[i],
                    device,
                )
                batch_preds.append(pred.squeeze())

            predictions = torch.stack(batch_preds)
            loss = F.smooth_l1_loss(predictions, targets)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                trainable_params,
                TRAINING_CONFIG["grad_clip_norm"],
            )
            optimizer.step()

            epoch_losses.append(loss.item())
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        scheduler.step()
        val_metrics = validate_with_embedding_store(
            model,
            embedding_store,
            val_loader,
            device,
        )
        val_loss = val_metrics["loss"]
        val_mape = val_metrics["mape"]
        avg_train_loss = np.mean(epoch_losses)
        epoch_time = time.time() - epoch_start

        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(val_loss)
        history["val_mape"].append(val_mape)

        print(
            f"  Epoch {epoch+1:3d} | "
            f"Train: {avg_train_loss:.4f} | "
            f"Val: {val_loss:.4f} | "
            f"Val MAPE: {val_mape:.4f}% | "
            f"LR: {scheduler.get_last_lr()[0]:.2e} | "
            f"{epoch_time:.1f}s"
        )

        checkpoint_score = val_mape if monitor_metric == "val_mape" else val_loss
        if checkpoint_score < best_checkpoint_score:
            best_checkpoint_score = checkpoint_score
            best_val_loss = val_loss
            best_val_mape = val_mape
            torch.save(
                {
                    "model_state_dict": {
                        k: v for k, v in model.state_dict().items()
                        if "embedding_extractor" not in k
                    },
                    "optimizer_state_dict": optimizer.state_dict(),
                    "epoch": epoch,
                    "val_loss": float(val_loss),
                    "val_mape": float(val_mape),
                    "checkpoint_metric": monitor_metric,
                    "checkpoint_score": float(checkpoint_score),
                },
                checkpoint_path,
            )

    total_time = time.time() - total_start
    print(f"\n  Training: {total_time:.0f}s ({total_time/60:.1f} min)")
    print(f"  Best validation loss: {best_val_loss:.6f}")
    print(f"  Best validation MAPE: {best_val_mape:.6f}%")
    print(f"  Best checkpoint {monitor_metric}: {best_checkpoint_score:.6f}")
    return history


@torch.no_grad()
def validate_with_embedding_store(
    model: nn.Module,
    embedding_store: Optional[TorchEmbeddingStore],
    val_loader,
    device: torch.device,
) -> Dict[str, float]:
    """Compute validation loss and price-level MAPE for checkpointing."""
    model.eval()
    losses = []
    mapes = []

    for batch in val_loader:
        positions = batch["positions"]
        price_histories = batch["price_histories"]
        targets = batch["targets"].to(device)
        last_prices = batch["last_prices"].to(device)

        batch_preds = []
        for i, position in enumerate(positions):
            pred = predict_with_optional_embeddings(
                model,
                embedding_store,
                position,
                price_histories[i],
                device,
            )
            batch_preds.append(pred.squeeze())

        predictions = torch.stack(batch_preds)
        loss = F.smooth_l1_loss(predictions, targets)
        losses.append(loss.item())
        batch_mape = compute_price_mape(predictions, targets, last_prices)
        mapes.append(float(batch_mape.item()))

    return {
        "loss": np.mean(losses) if losses else float("inf"),
        "mape": np.mean(mapes) if mapes else float("inf"),
    }


@torch.no_grad()
def rolling_forecast_with_embedding_store(
    model: nn.Module,
    all_data: np.ndarray,
    train_size: int,
    test_size: int,
    asset_cols: List[str],
    target_idx: int,
    max_context: int,
    corr_lookback: int,
    embedding_store: Optional[TorchEmbeddingStore] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run rolling forecast using the Torch embedding store when available."""
    model.eval()
    device = get_model_device(model)

    if embedding_store is not None and model_uses_temporal_embeddings(model):
        test_positions = list(range(train_size, train_size + test_size))
        missing = [p for p in test_positions if p not in embedding_store]
        if missing:
            print(f"  Pre-computing {len(missing)} test embeddings...")
            embedding_store.build(all_data, missing)

    predictions = []
    actuals = []
    last_prices = []

    for step in tqdm(range(test_size), desc="Rolling Forecast"):
        current_end = train_size + step
        corr_start = max(0, current_end - corr_lookback)
        price_history = all_data[corr_start:current_end]

        if (
            not model_uses_temporal_embeddings(model)
            or (embedding_store is not None and current_end in embedding_store)
        ):
            pred = predict_with_optional_embeddings(
                model,
                embedding_store,
                current_end,
                price_history,
                device,
            )
        else:
            context_start = max(0, current_end - max_context)
            context_data = all_data[context_start:current_end]
            context_series = [context_data[:, i] for i in range(len(asset_cols))]
            pred = model(
                multi_asset_series=context_series,
                price_history=price_history,
            )

        log_return_hat = pred.squeeze().cpu().item()
        last_price = float(all_data[current_end - 1, target_idx])
        if MODEL_CONFIG["target_mode"] == "log_return":
            price_hat = last_price * np.exp(log_return_hat)
        else:
            price_hat = last_price + log_return_hat

        predictions.append(price_hat)
        actuals.append(float(all_data[current_end, target_idx]))
        last_prices.append(last_price)

    return np.array(predictions), np.array(actuals), np.array(last_prices)


def build_mode_output_path(base_path: str, mode: str) -> str:
    """Add a mode suffix for non-default experiments."""
    if mode == "graph_adapter_v2":
        return base_path

    stem, ext = os.path.splitext(base_path)
    return f"{stem}_{mode}{ext}"


def load_best_checkpoint(model: nn.Module, checkpoint_path: str) -> None:
    """Load the best saved trainable weights into the model."""
    if not os.path.exists(checkpoint_path):
        return

    checkpoint = torch.load(checkpoint_path, weights_only=False)
    model_dict = model.state_dict()
    for k, v in checkpoint["model_state_dict"].items():
        if k in model_dict:
            model_dict[k] = v
    model.load_state_dict(model_dict)
    metric_name = checkpoint.get("checkpoint_metric", "val_loss")
    metric_score = checkpoint.get("checkpoint_score", checkpoint.get("val_loss"))
    print(
        f"  Best checkpoint loaded (epoch {checkpoint['epoch']+1}, "
        f"{metric_name}={metric_score:.6f})"
    )


def run_embedding_store_experiment(
    model: nn.Module,
    mode: str,
    display_name: str,
    embedding_store: Optional[TorchEmbeddingStore],
    all_data: np.ndarray,
    train_positions: List[int],
    val_positions: List[int],
    train_size: int,
    test_size: int,
    asset_cols: List[str],
    target_idx: int,
    target_col: str,
) -> Dict[str, object]:
    """Train and evaluate one experiment end-to-end."""
    checkpoint_path = build_mode_output_path(OUTPUT_CONFIG["model_save_path"], mode)
    metrics_path = build_mode_output_path(OUTPUT_CONFIG["metrics_output_path"], mode)
    visualization_path = build_mode_output_path(
        OUTPUT_CONFIG["visualization_output_path"], mode
    )
    training_curve_path = build_mode_output_path(
        OUTPUT_CONFIG["training_curve_path"], mode
    )

    print(f"\n[5/6] {display_name} eğitiliyor...")
    history = train_with_embedding_store(
        model,
        embedding_store,
        all_data,
        train_positions,
        val_positions,
        target_idx,
        model_label=display_name,
        checkpoint_path=checkpoint_path,
    )
    plot_training_curve(
        history,
        title=f"Training Curve ({display_name})",
        save_path=training_curve_path,
    )
    load_best_checkpoint(model, checkpoint_path)

    print(f"\n[6/6] {display_name} rolling forecast...")
    predictions, actuals, last_prices = rolling_forecast_with_embedding_store(
        model,
        all_data,
        train_size,
        test_size,
        asset_cols,
        target_idx,
        MODEL_CONFIG["max_context"],
        MODEL_CONFIG["corr_window"],
        embedding_store=embedding_store,
    )

    metrics = calculate_all_metrics(actuals, predictions, last_prices=last_prices)
    display_metrics(metrics, title=display_name)
    save_metrics_to_csv(metrics, target_col, mode=mode, output_path=metrics_path)
    visualize_forecast(
        actuals,
        predictions,
        target_col,
        prediction_label=f"{display_name} Predicted",
        title_label=display_name,
        save_path=visualization_path,
    )

    return {
        "mode": mode,
        "display_name": display_name,
        "metrics": metrics,
        "predictions": predictions,
        "actuals": actuals,
        "last_prices": last_prices,
    }
