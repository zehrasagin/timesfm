"""
TSFM-Graph Adapter V2 — Commodity Forecast Pipeline
===================================================

Orchestration entrypoint:
  1. Load data
  2. Initialize TimesFM and downstream models
  3. Build shared Torch embedding store
  4. Train and evaluate experiments
"""

from __future__ import annotations

import os
import random
import sys
import time
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch

SRC_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if SRC_ROOT not in sys.path:
    sys.path.insert(0, SRC_ROOT)

import timesfm

from graph_adapter import (
    TSFMEmbeddingOnlyModel,
    TSFMGraphAdapterModelV2,
    TSFMGraphOnlyModel,
)
from graph_adapter.experiment.forecast_config import (
    DATA_CONFIG,
    EXPERIMENT_CONFIG,
    MODEL_CONFIG,
    OUTPUT_CONFIG,
    TIMESFM_CONFIG,
    TRAINING_CONFIG,
)
from graph_adapter.experiment.pipeline_utils import (
    model_uses_temporal_embeddings,
    precompute_embeddings,
    run_embedding_store_experiment,
)
from graph_adapter.experiment.reporting_utils import (
    print_experiment_summary,
    visualize_experiment_comparison,
)

def set_global_seed(seed: int | None = None) -> None:
    """Configure reproducibility for local runs."""
    if seed is None:
        seed = int(TRAINING_CONFIG["seed"])

    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if hasattr(torch, "use_deterministic_algorithms"):
        torch.use_deterministic_algorithms(True, warn_only=True)


def experiment_seed(mode: str) -> int:
    """Stable per-experiment seed so toggling baselines does not affect init."""
    offsets = {
        "embedding_only": 0,
        "graph_only": 1,
        "graph_adapter_v2": 2,
    }
    return int(TRAINING_CONFIG["seed"]) + offsets.get(mode, 0)


def load_multi_asset_data(
    csv_path: str,
    asset_columns: List[str],
) -> Tuple[pd.DataFrame, np.ndarray]:
    """Load multi-asset price data."""
    df = pd.read_csv(csv_path)
    df_subset = df[asset_columns].dropna().reset_index(drop=True)
    all_data = df_subset.values.astype(np.float32)
    return df_subset, all_data


def print_model_summary(
    title: str,
    param_info: Dict[str, float],
    extra_lines: Dict[str, object],
) -> None:
    """Print a consistent model summary block."""
    print(f"\n{'═' * 60}")
    print(title)
    print(f"{'═' * 60}")
    print(f"  Trainable parameters:  {param_info['trainable']:>12,}")
    print(f"  Frozen parameters:     {param_info['frozen']:>12,}")
    print(f"  Total parameters:      {param_info['total']:>12,}")
    print(f"  Trainable ratio:       {param_info['trainable_pct']:>11.2f}%")
    for key, value in extra_lines.items():
        print(f"  {key:<22s}: {value:>12}")
    print(f"{'═' * 60}")


def initialize_timesfm_model() -> timesfm.TimesFM_2p5_200M_torch:
    """Initialize and compile the frozen TimesFM backbone."""
    torch.set_float32_matmul_precision("high")

    model = timesfm.TimesFM_2p5_200M_torch.from_pretrained(
        "google/timesfm-2.5-200m-pytorch"
    )
    model.compile(
        timesfm.ForecastConfig(
            max_context=TIMESFM_CONFIG["max_context"],
            max_horizon=TIMESFM_CONFIG["max_horizon"],
            normalize_inputs=TIMESFM_CONFIG["normalize_inputs"],
            use_continuous_quantile_head=TIMESFM_CONFIG[
                "use_continuous_quantile_head"
            ],
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
) -> TSFMGraphAdapterModelV2:
    """Initialize the graph-enhanced downstream model."""
    model = TSFMGraphAdapterModelV2(
        timesfm_model=timesfm_model,
        asset_names=asset_names,
        target_idx=target_idx,
        max_context=MODEL_CONFIG["max_context"],
        embed_dim=1280,
        graph_dim=MODEL_CONFIG["graph_dim"],
        num_gat_heads=MODEL_CONFIG["num_gat_heads"],
        num_gat_layers=MODEL_CONFIG["num_gat_layers"],
        dropout=MODEL_CONFIG["dropout"],
        corr_window=MODEL_CONFIG["corr_window"],
        use_absolute_corr=MODEL_CONFIG["use_absolute_corr"],
        add_self_loops=MODEL_CONFIG["add_self_loops"],
    )
    print_model_summary(
        title="TSFM-Graph Adapter Model V2",
        param_info=model.count_parameters(),
        extra_lines={
            "Graph dim": MODEL_CONFIG["graph_dim"],
            "Graph edges": "Full weighted",
            "Abs corr": MODEL_CONFIG["use_absolute_corr"],
            "Self loops": MODEL_CONFIG["add_self_loops"],
            "Target mode": MODEL_CONFIG["target_mode"],
        },
    )
    return model


def initialize_embedding_only_model(
    timesfm_model: timesfm.TimesFM_2p5_200M_torch,
    target_idx: int,
) -> TSFMEmbeddingOnlyModel:
    """Initialize the embedding-only baseline."""
    model = TSFMEmbeddingOnlyModel(
        timesfm_model=timesfm_model,
        target_idx=target_idx,
        max_context=MODEL_CONFIG["max_context"],
        embed_dim=1280,
        hidden_dim=MODEL_CONFIG["graph_dim"],
        dropout=MODEL_CONFIG["dropout"],
    )
    print_model_summary(
        title="TSFM Embedding-Only Baseline",
        param_info=model.count_parameters(),
        extra_lines={
            "Hidden dim": MODEL_CONFIG["graph_dim"],
            "Target mode": MODEL_CONFIG["target_mode"],
        },
    )
    return model


def initialize_graph_only_model(
    asset_names: List[str],
    target_idx: int,
) -> TSFMGraphOnlyModel:
    """Initialize the graph-only baseline."""
    model = TSFMGraphOnlyModel(
        asset_names=asset_names,
        target_idx=target_idx,
        max_context=MODEL_CONFIG["max_context"],
        embed_dim=1280,
        graph_dim=MODEL_CONFIG["graph_dim"],
        num_gat_heads=MODEL_CONFIG["num_gat_heads"],
        num_gat_layers=MODEL_CONFIG["num_gat_layers"],
        dropout=MODEL_CONFIG["dropout"],
        corr_window=MODEL_CONFIG["corr_window"],
        use_absolute_corr=MODEL_CONFIG["use_absolute_corr"],
        add_self_loops=MODEL_CONFIG["add_self_loops"],
    )
    print_model_summary(
        title="Graph-Only Baseline",
        param_info=model.count_parameters(),
        extra_lines={
            "Graph dim": MODEL_CONFIG["graph_dim"],
            "Graph edges": "Full weighted",
            "Abs corr": MODEL_CONFIG["use_absolute_corr"],
            "Self loops": MODEL_CONFIG["add_self_loops"],
            "Target mode": MODEL_CONFIG["target_mode"],
        },
    )
    return model


def build_experiments(
    timesfm_model: timesfm.TimesFM_2p5_200M_torch,
    asset_cols: List[str],
    target_idx: int,
) -> List[Dict[str, object]]:
    """Create all enabled experiment variants."""
    experiments: List[Dict[str, object]] = []

    if EXPERIMENT_CONFIG["run_embedding_only"]:
        mode = "embedding_only"
        set_global_seed(experiment_seed(mode))
        print("\n  -> Embedding-only baseline hazırlanıyor...")
        experiments.append(
            {
                "mode": mode,
                "display_name": "TSFM Embedding-Only",
                "model": initialize_embedding_only_model(timesfm_model, target_idx),
            }
        )

    if EXPERIMENT_CONFIG["run_graph_only"]:
        mode = "graph_only"
        set_global_seed(experiment_seed(mode))
        print("\n  -> Graph-only baseline hazırlanıyor...")
        experiments.append(
            {
                "mode": mode,
                "display_name": "Graph-Only",
                "model": initialize_graph_only_model(asset_cols, target_idx),
            }
        )

    if EXPERIMENT_CONFIG["run_graph_adapter"]:
        mode = "graph_adapter_v2"
        set_global_seed(experiment_seed(mode))
        print("\n  -> Total graph adapter hazırlanıyor...")
        experiments.append(
            {
                "mode": mode,
                "display_name": "Total TSFM-Graph Adapter",
                "model": initialize_graph_adapter(
                    timesfm_model,
                    asset_cols,
                    target_idx,
                ),
            }
        )

    return experiments


def compute_training_positions(
    train_size: int,
    context_length: int,
    stride: int = 1,
) -> Tuple[List[int], List[int]]:
    """Compute sliding-window train and validation positions."""
    min_start = context_length
    all_positions = list(range(min_start, train_size, stride))
    split = int(len(all_positions) * (1 - TRAINING_CONFIG["val_ratio"]))
    train_positions = all_positions[:split]
    val_positions = all_positions[split:]
    return train_positions, val_positions


def main() -> None:
    """Run the end-to-end forecasting pipeline."""
    set_global_seed()
    pipeline_start = time.time()

    print("\n" + "═" * 60)
    print("  TSFM-GRAPH ADAPTER V2 — OPTIMIZED COMMODITY FORECAST")
    print("═" * 60)

    print("\n[1/6] Veri yükleniyor...")
    target_col = DATA_CONFIG["target_column"]
    asset_cols = DATA_CONFIG["asset_columns"]
    _, all_data = load_multi_asset_data(DATA_CONFIG["csv_path"], asset_cols)

    total_len = len(all_data)
    test_size = int(total_len * DATA_CONFIG["test_split_ratio"])
    train_size = total_len - test_size
    target_idx = asset_cols.index(target_col)

    print(f"  Total: {total_len} days | Train: {train_size} | Test: {test_size}")
    print(f"  Target: {target_col} (idx={target_idx})")

    print("\n[2/6] TimesFM modeli başlatılıyor...")
    timesfm_model = initialize_timesfm_model()
    print("  TimesFM 2.5 200M ready (frozen)")

    print("\n[3/6] Deney modelleri başlatılıyor...")
    experiments = build_experiments(timesfm_model, asset_cols, target_idx)
    if not experiments:
        raise ValueError("En az bir experiment aktif olmalı.")

    print("\n[4/6] Position split ve embedding store hazırlanıyor...")
    train_positions, val_positions = compute_training_positions(
        train_size,
        MODEL_CONFIG["max_context"],
        TRAINING_CONFIG["stride"],
    )
    test_positions = list(range(train_size, train_size + test_size))
    all_positions = sorted(set(train_positions + val_positions + test_positions))

    print(
        f"  Positions: {len(train_positions)} train + "
        f"{len(val_positions)} val + {len(test_positions)} test = "
        f"{len(all_positions)} unique"
    )

    embedding_store = None
    embedding_source = next(
        (
            experiment["model"]
            for experiment in experiments
            if model_uses_temporal_embeddings(experiment["model"])
        ),
        None,
    )
    if embedding_source is not None:
        print("  Embedding'ler pre-compute ediliyor...")
        print("  (İlk sefer ~10-15 dk. Sonraki sefer Torch .pt dosyasından yüklenir.)")
        set_global_seed(int(TRAINING_CONFIG["seed"]))
        store_start = time.time()
        embedding_store = precompute_embeddings(
            embedding_source,
            all_data,
            all_positions,
            store_path=OUTPUT_CONFIG["embedding_store_path"],
        )
        store_time = time.time() - store_start
        print(f"  Pre-computation: {store_time:.0f}s ({store_time/60:.1f} min)")
    else:
        print("  Embedding pre-compute atlandı (aktif model yok).")

    results = []
    for experiment in experiments:
        set_global_seed(experiment_seed(str(experiment["mode"])))
        results.append(
            run_embedding_store_experiment(
                model=experiment["model"],
                mode=experiment["mode"],
                display_name=experiment["display_name"],
                embedding_store=embedding_store,
                all_data=all_data,
                train_positions=train_positions,
                val_positions=val_positions,
                train_size=train_size,
                test_size=test_size,
                asset_cols=asset_cols,
                target_idx=target_idx,
                target_col=target_col,
            )
        )

    print_experiment_summary(results)
    visualize_experiment_comparison(
        results,
        target_col,
        save_path=OUTPUT_CONFIG["comparison_visualization_output_path"],
    )

    pipeline_time = time.time() - pipeline_start
    print(f"\n{'═' * 60}")
    print("  Pipeline V2 tamamlandı!")
    print(f"  Toplam süre: {pipeline_time:.0f}s ({pipeline_time/60:.1f} min)")
    print(f"  Target mode: {MODEL_CONFIG['target_mode']}")
    print(f"  Seed: {TRAINING_CONFIG['seed']}")
    print(f"  Experiments: {len(results)}")
    print(f"{'═' * 60}\n")


if __name__ == "__main__":
    main()
