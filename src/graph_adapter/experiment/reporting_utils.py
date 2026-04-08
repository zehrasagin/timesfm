"""Metrics, CSV logging, and visualization helpers."""

from __future__ import annotations

from datetime import datetime
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    r2_score,
)

from .forecast_config import MODEL_CONFIG, OUTPUT_CONFIG


def calculate_all_metrics(
    actual: np.ndarray,
    predicted: np.ndarray,
    last_prices: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """Calculate all forecast metrics."""
    return {
        "MAPE": mean_absolute_percentage_error(actual, predicted) * 100,
        "R2": r2_score(actual, predicted),
        "MAE": mean_absolute_error(actual, predicted),
        "MSE": mean_squared_error(actual, predicted),
        "RMSE": np.sqrt(mean_squared_error(actual, predicted)),
        "Directional_Accuracy": directional_accuracy(
            actual, predicted, last_prices=last_prices
        ),
    }


def directional_accuracy(
    actual: np.ndarray,
    predicted: np.ndarray,
    last_prices: Optional[np.ndarray] = None,
) -> float:
    """Direction accuracy for per-step forecasts."""
    if len(actual) == 0:
        return 0.0

    if last_prices is None:
        if len(actual) < 2:
            return 0.0
        actual_dir = np.diff(actual) > 0
        pred_dir = np.diff(predicted) > 0
        return np.mean(actual_dir == pred_dir) * 100

    actual_dir = (actual - last_prices) > 0
    pred_dir = (predicted - last_prices) > 0
    return np.mean(actual_dir == pred_dir) * 100


def display_metrics(metrics: Dict[str, float], title: Optional[str] = None) -> None:
    """Print formatted metrics."""
    if title:
        print(f"\n{title}")
    print(f"\n{'=' * 27}METRICS{'=' * 26}")
    for name, value in metrics.items():
        unit = "%" if name in ["MAPE", "Directional_Accuracy"] else ""
        print(f"  {name:<25s}: {value:12.4f}{unit}")
    print(f"{'=' * 60}")


def save_metrics_to_csv(
    metrics: Dict[str, float],
    target_column: str,
    mode: str,
    output_path: str = OUTPUT_CONFIG["metrics_output_path"],
) -> None:
    """Save metrics to CSV."""
    metrics_data = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "target_column": target_column,
        "mode": mode,
        "target_mode": MODEL_CONFIG["target_mode"],
        **metrics,
    }
    pd.DataFrame([metrics_data]).to_csv(output_path, index=False)
    print(f"  Metrics saved to {output_path}")


def visualize_forecast(
    actuals: np.ndarray,
    predictions: np.ndarray,
    column_name: str,
    prediction_label: str = "Predicted",
    title_label: str = "Model",
    save_path: str = OUTPUT_CONFIG["visualization_output_path"],
) -> None:
    """Draw actual vs predicted forecast chart."""
    test_len = len(actuals)
    indices = np.arange(1, test_len + 1)

    plt.figure(figsize=(14, 7))
    plt.plot(
        indices,
        actuals,
        label="Actual Values",
        color="#06A77D",
        linewidth=2.5,
        marker="o",
        markersize=5,
    )
    plt.plot(
        indices,
        predictions,
        label=prediction_label,
        color="#D62828",
        linewidth=2.5,
        marker="s",
        markersize=5,
        linestyle="--",
    )

    plt.xlabel("Forecast Day", fontsize=12, fontweight="bold")
    plt.ylabel(column_name, fontsize=12, fontweight="bold")

    mape = mean_absolute_percentage_error(actuals, predictions) * 100
    plt.title(
        f"{column_name} — {title_label} Rolling Forecast\n"
        f"MAPE: {mape:.2f}%",
        fontsize=14,
        fontweight="bold",
        pad=20,
    )
    plt.legend(loc="best", fontsize=11, framealpha=0.9)
    plt.grid(True, alpha=0.3, linestyle="--")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Visualization saved to {save_path}")


def plot_training_curve(
    history: Dict[str, List[float]],
    title: str = "Training Curve",
    save_path: str = OUTPUT_CONFIG["training_curve_path"],
) -> None:
    """Draw training and validation loss curve."""
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(history["train_loss"], label="Train Loss", color="#2196F3", linewidth=2)
    if history["val_loss"]:
        ax.plot(history["val_loss"], label="Val Loss", color="#F44336", linewidth=2)
    ax.set_xlabel("Epoch", fontweight="bold")
    ax.set_ylabel(f"Huber Loss ({MODEL_CONFIG['target_mode']})", fontweight="bold")
    ax.set_title(title, fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Training curve saved to {save_path}")


def print_experiment_summary(results: List[Dict[str, object]]) -> None:
    """Print a short comparison summary across experiments."""
    if len(results) < 2:
        return

    print(f"\n{'=' * 22}EXPERIMENT SUMMARY{'=' * 20}")
    for result in results:
        metrics = result["metrics"]
        print(
            f"  {result['display_name']:<24s} | "
            f"MAPE: {metrics['MAPE']:>8.3f}% | "
            f"MAE: {metrics['MAE']:>10.4f} | "
            f"RMSE: {metrics['RMSE']:>10.4f} | "
            f"DirAcc: {metrics['Directional_Accuracy']:>7.2f}%"
        )
    print(f"{'=' * 60}")
