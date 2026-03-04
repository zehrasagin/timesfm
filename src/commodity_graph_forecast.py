"""
TSFM-Graph Adapter — Optimized Commodity Forecast Pipeline
============================================================

Pre-computed embedding cache ile optimize edilmiş versiyon.
TimesFM embedding'leri BİR KEZ hesaplanır, training ~50x hızlanır.

Akış:
  1. Veri yükleme
  2. TimesFM başlatma (frozen)
  3. Graph Adapter başlatma (trainable ~1.4%)
  4. ★ Embedding pre-computation (bir kez, ~10-15 dk)
  5. Training: Cached embeddings üzerinde (~5-10 dk)
  6. Rolling forecast: Graph-enhanced tahmin
  7. Metrik + Visualization

Kullanım:
  python src/commodity_graph_forecast.py
"""

import sys
import os
import time
from typing import Tuple, Dict, List, Optional
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    mean_absolute_percentage_error,
)
import matplotlib.pyplot as plt

import timesfm

# Graph adapter modülleri
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from graph_adapter import TSFMGraphAdapterModel
from graph_adapter.embedding_cache import EmbeddingCache
from graph_adapter.cached_dataset import (
    CachedEmbeddingDataset,
    cached_collate_fn,
    create_cached_dataloaders,
)

# ═══════════════════════════════════════════════════════════════════════════════
# KONFİGÜRASYON
# ═══════════════════════════════════════════════════════════════════════════════

DATA_CONFIG = {
    "csv_path": "src/commodity_features.csv",
    "target_column": "CO1 Comdty",
    "asset_columns": [
        "CO1 Comdty",   # Brent Crude (TARGET)
        "CL1 Comdty",   # WTI Crude
        "GC1 Comdty",   # Gold
        "HG1 Comdty",   # Copper
        "HO1 Comdty",   # Heating Oil
        "NG1 Comdty",   # Natural Gas
        "PA1 Comdty",   # Palladium
        "PL1 Comdty",   # Platinum
        "SI1 Comdty",   # Silver
        "C 1 Comdty",   # Corn
    ],
    "test_split_ratio": 0.10,
}

MODEL_CONFIG = {
    "max_context": 1024,
    "graph_dim": 256,
    "adapter_dim": 256,
    "num_gat_heads": 4,
    "num_gat_layers": 2,
    "num_adapter_heads": 4,
    "dropout": 0.1,
    "corr_window": 60,
    "corr_threshold": 0.3,
    "initial_alpha": 0.7,
}

TRAINING_CONFIG = {
    "num_epochs": 15,             # Sabit epoch sayısı (early stopping yok)
    "batch_size": 64,             # Cache'de olduğu için büyük batch OK
    "learning_rate": 1e-4,
    "weight_decay": 1e-4,
    "warmup_epochs": 5,
    "grad_clip_norm": 1.0,
    "val_ratio": 0.15,            # Training verinin son %15'i validation
    "stride": 1,                  # Sliding window stride (1=her gün)
}

TIMESFM_CONFIG = {
    "max_context": 1024,
    "max_horizon": 1,
    "normalize_inputs": True,
    "use_continuous_quantile_head": True,
    "force_flip_invariance": True,
    "infer_is_positive": True,
    "fix_quantile_crossing": True,
    "return_backcast": True,
}

OUTPUT_CONFIG = {
    "model_save_path": "graph_adapter_checkpoint.pt",
    "cache_save_path": "embedding_cache.npz",
    "metrics_output_path": "forecast_metrics_graph_adapter.csv",
    "visualization_output_path": "forecast_visualization_graph_adapter.png",
    "training_curve_path": "training_curve_graph_adapter.png",
}


# ═══════════════════════════════════════════════════════════════════════════════
# VERİ YÜKLEME
# ═══════════════════════════════════════════════════════════════════════════════

def load_multi_asset_data(
    csv_path: str,
    target_column: str,
    asset_columns: List[str],
) -> Tuple[pd.DataFrame, np.ndarray]:
    """Multi-asset fiyat verisi yükle."""
    df = pd.read_csv(csv_path)
    df_subset = df[asset_columns].dropna().reset_index(drop=True)
    all_data = df_subset.values.astype(np.float32)
    return df_subset, all_data


# ═══════════════════════════════════════════════════════════════════════════════
# MODEL BAŞLATMA
# ═══════════════════════════════════════════════════════════════════════════════

def initialize_timesfm_model() -> timesfm.TimesFM_2p5_200M_torch:
    """TimesFM 2.5 modelini başlat ve compile et."""
    torch.set_float32_matmul_precision("high")

    model = timesfm.TimesFM_2p5_200M_torch.from_pretrained(
        "google/timesfm-2.5-200m-pytorch"
    )

    model.compile(
        timesfm.ForecastConfig(
            max_context=TIMESFM_CONFIG["max_context"],
            max_horizon=TIMESFM_CONFIG["max_horizon"],
            normalize_inputs=TIMESFM_CONFIG["normalize_inputs"],
            use_continuous_quantile_head=TIMESFM_CONFIG["use_continuous_quantile_head"],
            force_flip_invariance=TIMESFM_CONFIG["force_flip_invariance"],
            infer_is_positive=TIMESFM_CONFIG["infer_is_positive"],
            fix_quantile_crossing=TIMESFM_CONFIG["fix_quantile_crossing"],
            return_backcast=TIMESFM_CONFIG["return_backcast"],
        )
    )
    return model


def initialize_graph_adapter(
    timesfm_model: timesfm.TimesFM_2p5_200M_torch,
    asset_names: List[str],
    target_idx: int,
) -> TSFMGraphAdapterModel:
    """Graph Adapter modelini başlat."""
    model = TSFMGraphAdapterModel(
        timesfm_model=timesfm_model,
        asset_names=asset_names,
        target_idx=target_idx,
        max_context=MODEL_CONFIG["max_context"],
        graph_dim=MODEL_CONFIG["graph_dim"],
        adapter_dim=MODEL_CONFIG["adapter_dim"],
        num_gat_heads=MODEL_CONFIG["num_gat_heads"],
        num_gat_layers=MODEL_CONFIG["num_gat_layers"],
        num_adapter_heads=MODEL_CONFIG["num_adapter_heads"],
        dropout=MODEL_CONFIG["dropout"],
        corr_window=MODEL_CONFIG["corr_window"],
        corr_threshold=MODEL_CONFIG["corr_threshold"],
        initial_alpha=MODEL_CONFIG["initial_alpha"],
    )

    # Parametre raporu
    param_info = model.count_parameters()
    print(f"\n{'═' * 60}")
    print(f"TSFM-Graph Adapter Model")
    print(f"{'═' * 60}")
    print(f"  Trainable parameters:  {param_info['trainable']:>12,}")
    print(f"  Frozen parameters:     {param_info['frozen']:>12,}")
    print(f"  Total parameters:      {param_info['total']:>12,}")
    print(f"  Trainable ratio:       {param_info['trainable_pct']:>11.2f}%")
    print(f"  Graph dim:             {MODEL_CONFIG['graph_dim']:>12}")
    print(f"  Adapter dim:           {MODEL_CONFIG['adapter_dim']:>12}")
    print(f"  α (initial):           {MODEL_CONFIG['initial_alpha']:>12.2f}")
    print(f"{'═' * 60}")

    return model


# ═══════════════════════════════════════════════════════════════════════════════
# EMBEDDING PRE-COMPUTATION ★
# ═══════════════════════════════════════════════════════════════════════════════

def compute_training_positions(
    train_size: int,
    context_length: int,
    stride: int = 1,
) -> Tuple[List[int], List[int]]:
    """Training ve validation için window bitiş pozisyonları hesapla.

    Returns:
        train_positions, val_positions
    """
    min_start = context_length
    all_positions = list(range(min_start, train_size, stride))

    val_ratio = TRAINING_CONFIG["val_ratio"]
    split = int(len(all_positions) * (1 - val_ratio))
    train_positions = all_positions[:split]
    val_positions = all_positions[split:]

    return train_positions, val_positions


def precompute_embeddings(
    adapter_model: TSFMGraphAdapterModel,
    all_data: np.ndarray,
    positions: List[int],
    cache_path: Optional[str] = None,
) -> EmbeddingCache:
    """Tüm pozisyonlar için embedding'leri pre-compute et.

    İlk sefer ~10-15 dk sürer. Sonraki çalıştırmalarda disk cache'den yüklenir.
    """
    cache = EmbeddingCache(
        embedding_extractor=adapter_model.embedding_extractor,
        max_context=MODEL_CONFIG["max_context"],
        target_idx=adapter_model.target_idx,
    )

    # Disk cache varsa yükle
    if cache_path and os.path.exists(cache_path):
        cache.load(cache_path)
        missing = [p for p in positions if p not in cache]
        if not missing:
            print(f"  All {len(positions)} positions loaded from cache!")
            return cache
        print(f"  {len(positions) - len(missing)} loaded, "
              f"{len(missing)} to compute...")
        positions = missing

    cache.build(all_data, positions, save_path=cache_path)
    return cache


# ═══════════════════════════════════════════════════════════════════════════════
# TRAINING (OPTIMIZED — CACHED EMBEDDINGS)
# ═══════════════════════════════════════════════════════════════════════════════

def train_graph_adapter(
    adapter_model: TSFMGraphAdapterModel,
    embedding_cache: EmbeddingCache,
    all_data: np.ndarray,
    train_positions: List[int],
    val_positions: List[int],
    target_idx: int,
) -> Dict[str, List[float]]:
    """Graph Adapter'ı cached embedding'ler üzerinde eğit.

    TimesFM ÇALIŞMAZ — sadece graph adapter bileşenleri güncellenir.
    Her batch ~0.01s (vs eski ~1.75s).
    """
    device = adapter_model.embedding_extractor.device

    # DataLoader'lar
    train_loader, val_loader = create_cached_dataloaders(
        embedding_cache, all_data,
        train_positions, val_positions,
        target_idx=target_idx,
        batch_size=TRAINING_CONFIG["batch_size"],
        corr_lookback=MODEL_CONFIG["corr_window"],
    )

    # Sadece trainable parametreleri optimize et
    trainable_params = adapter_model.get_trainable_params()
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

    history = {"train_loss": [], "val_loss": [], "alpha": []}
    best_val_loss = float("inf")

    n_train_batches = len(train_loader)
    n_val_batches = len(val_loader)

    print(f"\n{'═' * 60}")
    print(f"TRAINING (CACHED — NO TIMESFM FORWARD)")
    print(f"  Train: {len(train_positions)} | Val: {len(val_positions)}")
    print(f"  Epochs: {TRAINING_CONFIG['num_epochs']} | "
          f"Batch: {TRAINING_CONFIG['batch_size']}")
    print(f"  Batches/epoch: {n_train_batches} train + {n_val_batches} val")
    print(f"  LR: {TRAINING_CONFIG['learning_rate']}")
    print(f"{'═' * 60}\n")

    total_start = time.time()

    for epoch in range(TRAINING_CONFIG["num_epochs"]):
        epoch_start = time.time()

        # ── Train Epoch ──
        adapter_model.train()
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
            B = len(positions)

            batch_preds = []
            for i in range(B):
                # Cache'den target seq embedding al — TimesFM forward yok!
                target_seq = embedding_cache.get(positions[i], device)

                pred = adapter_model.forward_cached(
                    target_seq_embeddings=target_seq,
                    price_history=price_histories[i],
                )
                batch_preds.append(pred.squeeze())

            predictions = torch.stack(batch_preds)
            loss = F.smooth_l1_loss(predictions, targets)  # Huber on delta

            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                trainable_params, TRAINING_CONFIG["grad_clip_norm"]
            )
            optimizer.step()

            epoch_losses.append(loss.item())
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        scheduler.step()

        # ── Validation ──
        val_loss = validate_cached(
            adapter_model, embedding_cache, val_loader, device
        )

        avg_train_loss = np.mean(epoch_losses)
        alpha_val = adapter_model.graph_structure.alpha.item()
        epoch_time = time.time() - epoch_start

        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(val_loss)
        history["alpha"].append(alpha_val)

        print(
            f"  Epoch {epoch+1:3d} | "
            f"Train: {avg_train_loss:.4f} | "
            f"Val: {val_loss:.4f} | "
            f"α: {alpha_val:.4f} | "
            f"LR: {scheduler.get_last_lr()[0]:.2e} | "
            f"{epoch_time:.1f}s"
        )

        # ── Best checkpoint kaydet ──
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(
                {
                    "model_state_dict": {
                        k: v for k, v in adapter_model.state_dict().items()
                        if "embedding_extractor" not in k
                    },
                    "optimizer_state_dict": optimizer.state_dict(),
                    "epoch": epoch,
                    "val_loss": float(val_loss),
                    "alpha": float(alpha_val),
                },
                OUTPUT_CONFIG["model_save_path"],
            )

    total_time = time.time() - total_start
    print(f"\n  Training: {total_time:.0f}s ({total_time/60:.1f} min)")
    print(f"  Best validation loss: {best_val_loss:.6f}")
    return history


@torch.no_grad()
def validate_cached(
    adapter_model: TSFMGraphAdapterModel,
    embedding_cache: EmbeddingCache,
    val_loader,
    device: torch.device,
) -> float:
    """Cached validation loss hesapla."""
    adapter_model.eval()
    losses = []

    for batch in val_loader:
        positions = batch["positions"]
        price_histories = batch["price_histories"]
        targets = batch["targets"].to(device)
        B = len(positions)

        batch_preds = []
        for i in range(B):
            target_seq = embedding_cache.get(positions[i], device)
            pred = adapter_model.forward_cached(
                target_seq_embeddings=target_seq,
                price_history=price_histories[i],
            )
            batch_preds.append(pred.squeeze())

        predictions = torch.stack(batch_preds)
        loss = F.smooth_l1_loss(predictions, targets)  # Huber on delta
        losses.append(loss.item())

    return np.mean(losses) if losses else float("inf")


# ═══════════════════════════════════════════════════════════════════════════════
# ROLLING FORECAST
# ═══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def graph_enhanced_rolling_forecast(
    adapter_model: TSFMGraphAdapterModel,
    all_data: np.ndarray,
    train_size: int,
    test_size: int,
    asset_cols: List[str],
    target_idx: int,
    max_context: int,
    embedding_cache: Optional[EmbeddingCache] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Graph-enhanced rolling forecast.

    Test pozisyonları cache'de varsa cache'den okur (hızlı),
    yoksa TimesFM'i çalıştırır.
    """
    adapter_model.eval()
    device = adapter_model.embedding_extractor.device

    # Test pozisyonlarını cache'le
    if embedding_cache is not None:
        test_positions = list(range(train_size, train_size + test_size))
        missing = [p for p in test_positions if p not in embedding_cache]
        if missing:
            print(f"  Pre-computing {len(missing)} test embeddings...")
            embedding_cache.build(all_data, missing)

    predictions = []
    actuals = []

    for step in tqdm(range(test_size), desc="Rolling Forecast"):
        current_end = train_size + step

        # Korelasyon için price history
        corr_start = max(0, current_end - max_context - 90)
        price_history = all_data[corr_start:current_end]

        if embedding_cache is not None and current_end in embedding_cache:
            # ★ Cache'den oku — çok hızlı
            target_seq = embedding_cache.get(current_end, device)
            pred = adapter_model.forward_cached(
                target_seq_embeddings=target_seq,
                price_history=price_history,
            )
        else:
            # Full forward (cache yoksa)
            context_start = max(0, current_end - max_context)
            context_data = all_data[context_start:current_end]
            context_series = [
                context_data[:, i] for i in range(len(asset_cols))
            ]
            pred = adapter_model(
                multi_asset_series=context_series,
                price_history=price_history,
            )

        # Model outputs delta_hat → convert to price_hat
        delta_hat = pred.squeeze().cpu().item()
        last_price = float(all_data[current_end - 1, target_idx])
        price_hat = last_price + delta_hat

        predictions.append(price_hat)
        actuals.append(float(all_data[current_end, target_idx]))

    return np.array(predictions), np.array(actuals)


# ═══════════════════════════════════════════════════════════════════════════════
# METRİKLER
# ═══════════════════════════════════════════════════════════════════════════════

def calculate_all_metrics(
    actual: np.ndarray, predicted: np.ndarray
) -> Dict[str, float]:
    """Tüm tahmin metriklerini hesapla."""
    return {
        "MAPE": mean_absolute_percentage_error(actual, predicted) * 100,
        "R2": r2_score(actual, predicted),
        "MAE": mean_absolute_error(actual, predicted),
        "MSE": mean_squared_error(actual, predicted),
        "RMSE": np.sqrt(mean_squared_error(actual, predicted)),
        "Directional_Accuracy": directional_accuracy(actual, predicted),
    }


def directional_accuracy(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Yön doğruluğu: Tahmin yönü doğru mu?"""
    if len(actual) < 2:
        return 0.0
    actual_dir = np.diff(actual) > 0
    pred_dir = np.diff(predicted) > 0
    return np.mean(actual_dir == pred_dir) * 100


def display_metrics(metrics: Dict[str, float]) -> None:
    """Metrikleri formatlı yazdır."""
    print(f"\n{'=' * 27}METRICS{'=' * 26}")
    for name, value in metrics.items():
        unit = "%" if name in ["MAPE", "Directional_Accuracy"] else ""
        print(f"  {name:<25s}: {value:12.4f}{unit}")
    print(f"{'=' * 60}")


def save_metrics_to_csv(
    metrics: Dict[str, float],
    target_column: str,
    alpha: float,
    output_path: str = OUTPUT_CONFIG["metrics_output_path"],
) -> None:
    """Metrikleri CSV'ye kaydet."""
    metrics_data = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "target_column": target_column,
        "mode": "graph_adapter",
        "alpha": alpha,
        **metrics,
    }
    pd.DataFrame([metrics_data]).to_csv(output_path, index=False)
    print(f"  Metrics saved to {output_path}")


# ═══════════════════════════════════════════════════════════════════════════════
# VİZUALİZASYON
# ═══════════════════════════════════════════════════════════════════════════════

def visualize_forecast(
    actuals: np.ndarray,
    predictions: np.ndarray,
    column_name: str,
    save_path: str = OUTPUT_CONFIG["visualization_output_path"],
) -> None:
    """Tahmin vs gerçek grafiği çiz."""
    test_len = len(actuals)
    indices = np.arange(1, test_len + 1)

    plt.figure(figsize=(14, 7))

    plt.plot(
        indices, actuals,
        label="Actual Values", color="#06A77D",
        linewidth=2.5, marker="o", markersize=5,
    )
    plt.plot(
        indices, predictions,
        label="Graph Adapter Predicted", color="#D62828",
        linewidth=2.5, marker="s", markersize=5, linestyle="--",
    )

    plt.xlabel("Forecast Day", fontsize=12, fontweight="bold")
    plt.ylabel(column_name, fontsize=12, fontweight="bold")

    mape = mean_absolute_percentage_error(actuals, predictions) * 100
    plt.title(
        f"{column_name} — TSFM-Graph Adapter Rolling Forecast\n"
        f"MAPE: {mape:.2f}%",
        fontsize=14, fontweight="bold", pad=20,
    )

    plt.legend(loc="best", fontsize=11, framealpha=0.9)
    plt.grid(True, alpha=0.3, linestyle="--")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Visualization saved to {save_path}")


def plot_training_curve(
    history: Dict[str, List[float]],
    save_path: str = OUTPUT_CONFIG["training_curve_path"],
) -> None:
    """Training loss eğrisi çiz."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Loss curve
    ax1.plot(history["train_loss"], label="Train Loss", color="#2196F3", linewidth=2)
    if history["val_loss"]:
        ax1.plot(history["val_loss"], label="Val Loss", color="#F44336", linewidth=2)
    ax1.set_xlabel("Epoch", fontweight="bold")
    ax1.set_ylabel("Huber Loss (delta)", fontweight="bold")
    ax1.set_title("Training Curve", fontweight="bold")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Alpha curve
    if history["alpha"]:
        ax2.plot(history["alpha"], color="#4CAF50", linewidth=2)
        ax2.set_xlabel("Epoch", fontweight="bold")
        ax2.set_ylabel("α value", fontweight="bold")
        ax2.set_title("Hybrid α Evolution\n(Static vs Learned Balance)",
                       fontweight="bold")
        ax2.axhline(y=0.7, color="gray", linestyle="--", alpha=0.5, label="Initial α")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Training curve saved to {save_path}")


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN PIPELINE
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    """Ana pipeline: Pre-compute → Train → Forecast → Evaluate."""

    pipeline_start = time.time()

    print("\n" + "═" * 60)
    print("  TSFM-GRAPH ADAPTER — OPTIMIZED COMMODITY FORECAST")
    print("═" * 60)

    # ── 1. Veri Yükleme ──
    print("\n[1/6] Veri yükleniyor...")
    target_col = DATA_CONFIG["target_column"]
    asset_cols = DATA_CONFIG["asset_columns"]

    df_subset, all_data = load_multi_asset_data(
        DATA_CONFIG["csv_path"], target_col, asset_cols,
    )

    total_len = len(all_data)
    test_size = int(total_len * DATA_CONFIG["test_split_ratio"])
    train_size = total_len - test_size
    target_idx = asset_cols.index(target_col)

    print(f"  Total: {total_len} days | Train: {train_size} | Test: {test_size}")
    print(f"  Target: {target_col} (idx={target_idx})")

    # ── 2. TimesFM Başlatma ──
    print("\n[2/6] TimesFM modeli başlatılıyor...")
    timesfm_model = initialize_timesfm_model()
    print("  TimesFM 2.5 200M ready (frozen)")

    # ── 3. Graph Adapter Başlatma ──
    print("\n[3/6] Graph Adapter başlatılıyor...")
    adapter_model = initialize_graph_adapter(
        timesfm_model, asset_cols, target_idx,
    )

    # ── 4. Embedding Pre-Computation ★ ──
    print("\n[4/6] Embedding'ler pre-compute ediliyor...")
    print("  (İlk sefer ~10-15 dk. Sonraki sefer disk cache'den yüklenir.)")

    train_positions, val_positions = compute_training_positions(
        train_size,
        MODEL_CONFIG["max_context"],
        TRAINING_CONFIG["stride"],
    )
    test_positions = list(range(train_size, train_size + test_size))
    all_positions = sorted(set(train_positions + val_positions + test_positions))

    print(f"  Positions: {len(train_positions)} train + "
          f"{len(val_positions)} val + {len(test_positions)} test = "
          f"{len(all_positions)} unique")

    cache_start = time.time()
    embedding_cache = precompute_embeddings(
        adapter_model, all_data, all_positions,
        cache_path=OUTPUT_CONFIG["cache_save_path"],
    )
    cache_time = time.time() - cache_start
    print(f"  Pre-computation: {cache_time:.0f}s ({cache_time/60:.1f} min)")

    # ── 5. Training (Fast!) ──
    print("\n[5/6] Graph Adapter eğitiliyor (cached, TimesFM YOK)...")
    history = train_graph_adapter(
        adapter_model, embedding_cache, all_data,
        train_positions, val_positions, target_idx,
    )
    plot_training_curve(history)

    # Best checkpoint yükle
    if os.path.exists(OUTPUT_CONFIG["model_save_path"]):
        checkpoint = torch.load(
            OUTPUT_CONFIG["model_save_path"], weights_only=False
        )
        model_dict = adapter_model.state_dict()
        for k, v in checkpoint["model_state_dict"].items():
            if k in model_dict:
                model_dict[k] = v
        adapter_model.load_state_dict(model_dict)
        print(f"  Best checkpoint loaded (epoch {checkpoint['epoch']+1}, "
              f"val_loss={checkpoint['val_loss']:.6f})")

    # ── 6. Rolling Forecast ──
    print("\n[6/6] Graph-enhanced rolling forecast...")
    predictions, actuals = graph_enhanced_rolling_forecast(
        adapter_model, all_data, train_size, test_size,
        asset_cols, target_idx,
        MODEL_CONFIG["max_context"],
        embedding_cache=embedding_cache,
    )

    # ── Metrikler ──
    metrics = calculate_all_metrics(actuals, predictions)
    display_metrics(metrics)

    alpha_final = adapter_model.graph_structure.alpha.item()
    save_metrics_to_csv(metrics, target_col, alpha_final)

    # ── Visualization ──
    visualize_forecast(actuals, predictions, target_col)

    pipeline_time = time.time() - pipeline_start
    print(f"\n{'═' * 60}")
    print(f"  Pipeline tamamlandı!")
    print(f"  Toplam süre: {pipeline_time:.0f}s ({pipeline_time/60:.1f} min)")
    print(f"  Final α = {alpha_final:.4f}")
    print(f"{'═' * 60}\n")


if __name__ == "__main__":
    main()
