"""Cached training and forecasting utilities."""

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

from graph_adapter.cached_dataset import create_cached_dataloaders
from graph_adapter.embedding_cache import EmbeddingCache

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
    cache_path: Optional[str] = None,
) -> EmbeddingCache:
    """Pre-compute and optionally persist TimesFM embeddings."""
    cache = EmbeddingCache(
        embedding_extractor=model_with_extractor.embedding_extractor,
        max_context=MODEL_CONFIG["max_context"],
        target_idx=model_with_extractor.target_idx,
    )

    if cache_path and os.path.exists(cache_path):
        cache.load(cache_path)
        missing = [p for p in positions if p not in cache]
        if not missing:
            print(f"  All {len(positions)} positions loaded from cache!")
            return cache
        print(f"  {len(positions) - len(missing)} loaded, {len(missing)} to compute...")
        positions = missing

    cache.build(all_data, positions, save_path=cache_path)
    return cache


def train_cached_model(
    model: nn.Module,
    embedding_cache: EmbeddingCache,
    all_data: np.ndarray,
    train_positions: List[int],
    val_positions: List[int],
    target_idx: int,
    model_label: str,
    checkpoint_path: str,
) -> Dict[str, List[float]]:
    """Train a cached model without running TimesFM in the loop."""
    device = model.embedding_extractor.device

    train_loader, val_loader = create_cached_dataloaders(
        embedding_cache,
        all_data,
        train_positions,
        val_positions,
        target_idx=target_idx,
        batch_size=TRAINING_CONFIG["batch_size"],
        corr_lookback=MODEL_CONFIG["corr_window"],
        target_mode=MODEL_CONFIG["target_mode"],
    )

    trainable_params = model.get_trainable_params()
    optimizer = AdamW(
        trainable_params,
        lr=TRAINING_CONFIG["learning_rate"],
        weight_decay=TRAINING_CONFIG["weight_decay"],
    )
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=TRAINING_CONFIG["num_epochs"],
        eta_min=1e-6,
    )

    history = {"train_loss": [], "val_loss": []}
    best_val_loss = float("inf")

    print(f"\n{'═' * 60}")
    print(f"TRAINING ({model_label} — CACHED, NO TIMESFM FORWARD)")
    print(f"  Train: {len(train_positions)} | Val: {len(val_positions)}")
    print(
        f"  Epochs: {TRAINING_CONFIG['num_epochs']} | "
        f"Batch: {TRAINING_CONFIG['batch_size']}"
    )
    print(f"  Batches/epoch: {len(train_loader)} train + {len(val_loader)} val")
    print(f"  LR: {TRAINING_CONFIG['learning_rate']}")
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
                target_seq = embedding_cache.get(position, device)
                pred = model.forward_cached(
                    target_seq_embeddings=target_seq,
                    price_history=price_histories[i],
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
        val_loss = validate_cached(model, embedding_cache, val_loader, device)
        avg_train_loss = np.mean(epoch_losses)
        epoch_time = time.time() - epoch_start

        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(val_loss)

        print(
            f"  Epoch {epoch+1:3d} | "
            f"Train: {avg_train_loss:.4f} | "
            f"Val: {val_loss:.4f} | "
            f"LR: {scheduler.get_last_lr()[0]:.2e} | "
            f"{epoch_time:.1f}s"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(
                {
                    "model_state_dict": {
                        k: v for k, v in model.state_dict().items()
                        if "embedding_extractor" not in k
                    },
                    "optimizer_state_dict": optimizer.state_dict(),
                    "epoch": epoch,
                    "val_loss": float(val_loss),
                },
                checkpoint_path,
            )

    total_time = time.time() - total_start
    print(f"\n  Training: {total_time:.0f}s ({total_time/60:.1f} min)")
    print(f"  Best validation loss: {best_val_loss:.6f}")
    return history


@torch.no_grad()
def validate_cached(
    model: nn.Module,
    embedding_cache: EmbeddingCache,
    val_loader,
    device: torch.device,
) -> float:
    """Compute validation loss for cached embeddings."""
    model.eval()
    losses = []

    for batch in val_loader:
        positions = batch["positions"]
        price_histories = batch["price_histories"]
        targets = batch["targets"].to(device)

        batch_preds = []
        for i, position in enumerate(positions):
            target_seq = embedding_cache.get(position, device)
            pred = model.forward_cached(
                target_seq_embeddings=target_seq,
                price_history=price_histories[i],
            )
            batch_preds.append(pred.squeeze())

        predictions = torch.stack(batch_preds)
        loss = F.smooth_l1_loss(predictions, targets)
        losses.append(loss.item())

    return np.mean(losses) if losses else float("inf")


@torch.no_grad()
def rolling_forecast_with_cached_model(
    model: nn.Module,
    all_data: np.ndarray,
    train_size: int,
    test_size: int,
    asset_cols: List[str],
    target_idx: int,
    max_context: int,
    corr_lookback: int,
    embedding_cache: Optional[EmbeddingCache] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run rolling forecast using cached embeddings when available."""
    model.eval()
    device = model.embedding_extractor.device

    if embedding_cache is not None:
        test_positions = list(range(train_size, train_size + test_size))
        missing = [p for p in test_positions if p not in embedding_cache]
        if missing:
            print(f"  Pre-computing {len(missing)} test embeddings...")
            embedding_cache.build(all_data, missing)

    predictions = []
    actuals = []
    last_prices = []

    for step in tqdm(range(test_size), desc="Rolling Forecast"):
        current_end = train_size + step
        corr_start = max(0, current_end - corr_lookback)
        price_history = all_data[corr_start:current_end]

        if embedding_cache is not None and current_end in embedding_cache:
            target_seq = embedding_cache.get(current_end, device)
            pred = model.forward_cached(
                target_seq_embeddings=target_seq,
                price_history=price_history,
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
    print(
        f"  Best checkpoint loaded (epoch {checkpoint['epoch']+1}, "
        f"val_loss={checkpoint['val_loss']:.6f})"
    )


def run_cached_experiment(
    model: nn.Module,
    mode: str,
    display_name: str,
    embedding_cache: EmbeddingCache,
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
    history = train_cached_model(
        model,
        embedding_cache,
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
    predictions, actuals, last_prices = rolling_forecast_with_cached_model(
        model,
        all_data,
        train_size,
        test_size,
        asset_cols,
        target_idx,
        MODEL_CONFIG["max_context"],
        MODEL_CONFIG["corr_window"],
        embedding_cache=embedding_cache,
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
