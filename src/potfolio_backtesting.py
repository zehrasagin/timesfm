"""Portfolio backtesting utilities for graph-adapter forecasting experiments.

The graph experiment compares three model variants:
- Temporal-Only
- Graph-Only
- Total Fusion

This module consumes single-asset rolling forecasts from that pipeline and
evaluates them against two sanity baselines:
- buy-and-hold
- always-flat
"""

from __future__ import annotations

import os
import re
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

try:
    import vectorbt as vbt
except ModuleNotFoundError:
    vbt = None

if vbt is not None:
    vbt.settings.returns["year_freq"] = "252 days"
    vbt.settings.array_wrapper["freq"] = "d"

sns.set_style("darkgrid")

# Local fallbacks for standalone use. In the graph experiment pipeline,
# active portfolio settings should come from forecast_config.py.
DEFAULT_INITIAL_CASH = 100000.0
DEFAULT_TRANSACTION_COST = 0.001
DEFAULT_EXECUTION_DELAY_STEPS = 0


def _to_single_asset_frame(
    values: np.ndarray,
    index: pd.Index,
    column_name: str,
) -> pd.DataFrame:
    """Convert a 1D forecast-related array into a single-column DataFrame."""
    return pd.DataFrame(
        {column_name: np.asarray(values, dtype=float)},
        index=index,
    )


def _ensure_series(value: pd.Series | pd.DataFrame, name: str) -> pd.Series:
    """Accept only 1D series-like outputs from vectorbt accessors."""
    if isinstance(value, pd.Series):
        return value
    if isinstance(value, pd.DataFrame) and value.shape[1] == 1:
        return value.iloc[:, 0]
    shape = getattr(value, "shape", None)
    raise ValueError(
        f"{name} must be a Series or single-column DataFrame, got shape={shape}."
    )


def _sign_series(values: pd.Series) -> pd.Series:
    """Return {-1, 0, 1} directional labels for a return series."""
    signs = pd.Series(0, index=values.index, dtype=int)
    signs[values > 0] = 1
    signs[values < 0] = -1
    return signs


def apply_signal_policy(
    signals_df: pd.DataFrame,
    signal_policy: str = "normal",
) -> pd.DataFrame:
    """Transform raw long/short/flat signals with a simple policy."""
    policy = signal_policy.strip().lower()
    signals = signals_df.astype(int).copy()

    if policy == "normal":
        return signals
    if policy == "reverse":
        return -signals
    if policy == "long_flat":
        signals[signals < 0] = 0
        return signals
    if policy == "short_flat":
        signals[signals > 0] = 0
        return signals

    raise ValueError(
        "signal_policy must be one of: "
        "'normal', 'reverse', 'long_flat', 'short_flat'."
    )


def generate_trading_signals(
    predicted_returns: pd.DataFrame,
    threshold: float = 0.0,
    long_threshold: Optional[float] = None,
    short_threshold: Optional[float] = None,
    signal_policy: str = "normal",
) -> pd.DataFrame:
    """Map predicted returns to long/short/flat signals."""
    if long_threshold is None:
        long_threshold = threshold
    if short_threshold is None:
        short_threshold = threshold

    signals = pd.DataFrame(
        0,
        index=predicted_returns.index,
        columns=predicted_returns.columns,
        dtype=int,
    )
    signals[predicted_returns > long_threshold] = 1
    signals[predicted_returns < -short_threshold] = -1
    return apply_signal_policy(signals, signal_policy=signal_policy)


def calculate_weights_from_signals(signals_df: pd.DataFrame) -> pd.DataFrame:
    """Normalize signals to equal gross exposure per row."""
    abs_sum = signals_df.abs().sum(axis=1)
    weights = signals_df.div(abs_sum, axis=0).fillna(0.0)
    return weights.astype(float)


def calculate_confidence_scaled_weights(
    predicted_returns: pd.DataFrame,
    signals_df: pd.DataFrame,
    confidence_window: int = 252,
    confidence_quantile: float = 0.75,
    max_abs_weight: float = 1.0,
) -> pd.DataFrame:
    """Scale active signals by forecast magnitude using only past forecasts."""
    if confidence_window <= 0:
        raise ValueError("confidence_window must be positive")
    if not 0 < confidence_quantile <= 1:
        raise ValueError("confidence_quantile must be in (0, 1]")
    if max_abs_weight <= 0:
        raise ValueError("max_abs_weight must be positive")

    rolling_scale = (
        predicted_returns.abs()
        .rolling(confidence_window, min_periods=max(5, min(20, confidence_window)))
        .quantile(confidence_quantile)
        .shift(1)
    )
    fallback_scale = predicted_returns.abs().expanding(min_periods=2).quantile(
        confidence_quantile
    )
    rolling_scale = rolling_scale.fillna(fallback_scale).replace(0.0, np.nan)
    raw_strength = predicted_returns.abs().div(rolling_scale).fillna(0.0)
    strength = raw_strength.clip(lower=0.0, upper=max_abs_weight)
    return (signals_df.astype(float) * strength).astype(float)


def build_decision_weights(
    predicted_returns: pd.DataFrame,
    signals_df: pd.DataFrame,
    weighting_scheme: str = "signal",
    confidence_window: int = 252,
    confidence_quantile: float = 0.75,
    max_abs_weight: float = 1.0,
) -> pd.DataFrame:
    """Build target weights from signals with a selectable sizing scheme."""
    scheme = weighting_scheme.strip().lower()
    if scheme == "signal":
        return calculate_weights_from_signals(signals_df)
    if scheme == "confidence":
        return calculate_confidence_scaled_weights(
            predicted_returns=predicted_returns,
            signals_df=signals_df,
            confidence_window=confidence_window,
            confidence_quantile=confidence_quantile,
            max_abs_weight=max_abs_weight,
        )
    raise ValueError("weighting_scheme must be one of: 'signal', 'confidence'.")


def apply_holding_period(
    target_weights: pd.DataFrame,
    holding_period_steps: int = 1,
) -> pd.DataFrame:
    """Average recent target weights to reduce one-day signal churn."""
    if holding_period_steps <= 1:
        return target_weights.astype(float)
    return (
        target_weights.rolling(holding_period_steps, min_periods=1)
        .mean()
        .clip(lower=-1.0, upper=1.0)
        .astype(float)
    )


def apply_volatility_target(
    target_weights: pd.DataFrame,
    actual_returns: pd.DataFrame,
    volatility_target: Optional[float] = None,
    volatility_window: int = 20,
    max_leverage: float = 1.0,
) -> pd.DataFrame:
    """Risk-scale weights with realized volatility known before each target row."""
    if volatility_target is None:
        return target_weights.astype(float)
    if volatility_target <= 0:
        raise ValueError("volatility_target must be positive when provided")
    if volatility_window <= 1:
        raise ValueError("volatility_window must be greater than 1")
    if max_leverage <= 0:
        raise ValueError("max_leverage must be positive")

    realized_vol = actual_returns.rolling(
        volatility_window,
        min_periods=max(5, min(10, volatility_window)),
    ).std().shift(1)
    scale = volatility_target / realized_vol.replace(0.0, np.nan)
    scale = scale.clip(lower=0.0, upper=max_leverage).fillna(1.0)
    return (target_weights * scale).clip(lower=-max_leverage, upper=max_leverage)


def apply_execution_delay(
    target_weights: pd.DataFrame,
    delay_steps: int = 1,
) -> pd.DataFrame:
    """Delay target weights by full bars to avoid same-bar close execution."""
    if delay_steps < 0:
        raise ValueError("delay_steps must be non-negative")

    delayed = target_weights.shift(delay_steps).fillna(0.0)
    return delayed.astype(float)


def materialize_target_weights(weights_df: pd.DataFrame) -> pd.DataFrame:
    """Convert sparse target order rows into continuous holdings."""
    return weights_df.ffill().fillna(0.0).astype(float)


def build_buy_and_hold_orders(
    template_index: pd.Index,
    columns: List[str],
    execution_delay_steps: int = 1,
    selected_assets: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Build one entry order and keep holding afterwards."""
    orders = pd.DataFrame(np.nan, index=template_index, columns=columns, dtype=float)
    if len(template_index) == 0:
        return orders.fillna(0.0)

    selected = list(columns if selected_assets is None else selected_assets)
    selected = [asset for asset in selected if asset in columns]
    if not selected:
        raise ValueError("selected_assets must include at least one valid column")

    entry_loc = execution_delay_steps
    if entry_loc >= len(template_index):
        return orders.fillna(0.0)

    orders.iloc[:entry_loc] = 0.0
    entry_weights = pd.Series(0.0, index=columns, dtype=float)
    entry_weights[selected] = 1.0 / len(selected)
    orders.iloc[entry_loc] = entry_weights.values
    return orders


def build_flat_orders(
    template_index: pd.Index,
    columns: List[str],
) -> pd.DataFrame:
    """Cash-only baseline."""
    return pd.DataFrame(0.0, index=template_index, columns=columns, dtype=float)


def build_timing_summary(
    forecast_df: pd.DataFrame,
    execution_delay_steps: int,
) -> Dict[str, object]:
    """Summarize forecast and execution timing for auditability."""
    summary: Dict[str, object] = {
        "execution_delay_steps": int(execution_delay_steps),
        "forecast_count": int(len(forecast_df)),
        "first_forecast_target_date": None,
        "last_forecast_target_date": None,
        "first_executable_date": None,
    }
    if forecast_df.empty:
        return summary

    summary["first_forecast_target_date"] = forecast_df.index[0]
    summary["last_forecast_target_date"] = forecast_df.index[-1]
    if len(forecast_df.index) > execution_delay_steps:
        summary["first_executable_date"] = forecast_df.index[execution_delay_steps]
    return summary


def display_timing_summary(timing: Dict[str, object]) -> None:
    """Print timing guardrails so the close-only execution policy stays explicit."""
    print("\nTiming Guardrails")
    print("  - Each forecast for day T+1 is produced using data only through day T.")
    delay_steps = int(timing["execution_delay_steps"])
    if delay_steps == 0:
        print(
            "  - Forecast-return backtest applies the decision to the forecast "
            "target return row."
        )
    else:
        print(
            f"  - Orders are delayed by {delay_steps} full close bar(s) before "
            "execution."
        )

    if timing["forecast_count"] == 0:
        print("  - No forecast rows were produced.")
        return

    first_target = pd.Timestamp(timing["first_forecast_target_date"]).date()
    last_target = pd.Timestamp(timing["last_forecast_target_date"]).date()
    print(f"  - Forecast target window: {first_target} -> {last_target}")

    first_exec = timing.get("first_executable_date")
    if first_exec is None:
        print("  - Test window is too short for a delayed execution step.")
    else:
        print(f"  - First executable date: {pd.Timestamp(first_exec).date()}")


def run_backtest(
    price_df: pd.DataFrame,
    weights_df: pd.DataFrame,
    init_cash: float = DEFAULT_INITIAL_CASH,
    transaction_cost: Optional[float] = None,
) -> vbt.Portfolio:
    """Run a vectorbt backtest with target-percent orders."""
    if vbt is None:
        raise ModuleNotFoundError(
            "vectorbt is required for run_backtest(...). Use the "
            "'forecast_return' backtest engine or install vectorbt."
        )
    if transaction_cost is None:
        transaction_cost = float(DEFAULT_TRANSACTION_COST)

    common_dates = price_df.index.intersection(weights_df.index)
    price_aligned = price_df.loc[common_dates]
    weights_aligned = weights_df.loc[common_dates]

    return vbt.Portfolio.from_orders(
        close=price_aligned,
        size=weights_aligned,
        size_type="targetpercent",
        init_cash=init_cash,
        freq="D",
        cash_sharing=True,
        call_seq="auto",
        fees=transaction_cost,
    )


def calculate_return_portfolio_metrics(
    returns: pd.Series,
    value: pd.Series,
    weights_df: Optional[pd.DataFrame] = None,
    turnover: Optional[pd.Series] = None,
) -> Dict[str, float]:
    """Compute portfolio metrics from an explicit return series."""
    returns = returns.dropna()
    value = value.loc[returns.index]
    ann_factor = 252.0
    if returns.empty:
        return {
            "Total Return (%)": 0.0,
            "Average Daily Return (%)": 0.0,
            "Average Daily Volatility (%)": 0.0,
            "Annualized Return (%)": 0.0,
            "Annualized Volatility (%)": 0.0,
            "Sharpe Ratio": 0.0,
            "Sortino Ratio": 0.0,
            "Max Drawdown (%)": 0.0,
            "Calmar Ratio": 0.0,
            "Win Rate (%)": 0.0,
            "Number of Trading Days": 0,
        }

    cumulative_returns = (1 + returns).cumprod()
    total_return = float((cumulative_returns.iloc[-1] - 1) * 100)
    avg_daily_return = float(returns.mean() * 100)
    avg_daily_volatility = float(returns.std() * 100)
    annualized_return = float(returns.mean() * ann_factor * 100)
    annualized_volatility = float(returns.std() * np.sqrt(ann_factor) * 100)

    if returns.std() > 0:
        sharpe_ratio = float(
            (returns.mean() * ann_factor) / (returns.std() * np.sqrt(ann_factor))
        )
    else:
        sharpe_ratio = 0.0

    downside = returns[returns < 0]
    if not downside.empty and downside.std() > 0:
        sortino_ratio = float(
            (returns.mean() * ann_factor) / (downside.std() * np.sqrt(ann_factor))
        )
    else:
        sortino_ratio = 0.0

    rolling_max = cumulative_returns.expanding().max()
    drawdowns = (cumulative_returns - rolling_max) / rolling_max
    max_drawdown = float(drawdowns.min() * 100)
    calmar_ratio = (
        float(annualized_return / abs(max_drawdown))
        if max_drawdown != 0
        else 0.0
    )
    win_rate = float((returns > 0).mean() * 100) if len(returns) else 0.0

    metrics: Dict[str, float] = {
        "Total Return (%)": total_return,
        "Average Daily Return (%)": avg_daily_return,
        "Average Daily Volatility (%)": avg_daily_volatility,
        "Annualized Return (%)": annualized_return,
        "Annualized Volatility (%)": annualized_volatility,
        "Sharpe Ratio": sharpe_ratio,
        "Sortino Ratio": sortino_ratio,
        "Max Drawdown (%)": max_drawdown,
        "Calmar Ratio": calmar_ratio,
        "Win Rate (%)": win_rate,
        "Number of Trading Days": int(len(returns)),
    }

    if weights_df is not None:
        weights_aligned = weights_df.loc[returns.index]
        gross_exposure = weights_aligned.abs().sum(axis=1)
        metrics["Average Gross Exposure (%)"] = float(gross_exposure.mean() * 100)
        metrics["Max Gross Exposure (%)"] = float(gross_exposure.max() * 100)

    if turnover is not None:
        turnover_aligned = turnover.loc[returns.index]
        metrics["Average Daily Turnover"] = float(turnover_aligned.mean())
        metrics["Total Turnover"] = float(turnover_aligned.sum())

    return metrics


def run_forecast_return_backtest(
    actual_returns: pd.DataFrame,
    weights_df: pd.DataFrame,
    init_cash: float = DEFAULT_INITIAL_CASH,
    transaction_cost: Optional[float] = None,
    drop_zero_return_days: bool = True,
) -> Dict[str, object]:
    """Backtest weights directly against the forecast target return rows."""
    if transaction_cost is None:
        transaction_cost = float(DEFAULT_TRANSACTION_COST)

    holdings = materialize_target_weights(weights_df)
    common_dates = actual_returns.index.intersection(holdings.index)
    returns_aligned = actual_returns.loc[common_dates]
    weights_aligned = holdings.loc[common_dates]

    if drop_zero_return_days:
        active_price_mask = returns_aligned.abs().sum(axis=1) > 0
        returns_aligned = returns_aligned.loc[active_price_mask]
        weights_aligned = weights_aligned.loc[active_price_mask]

    turnover = weights_aligned.diff().abs().fillna(weights_aligned.abs()).sum(axis=1)
    gross_returns = (weights_aligned * returns_aligned).sum(axis=1)
    net_returns = gross_returns - float(transaction_cost) * turnover
    net_returns.name = "portfolio_return"
    value = init_cash * (1 + net_returns).cumprod()
    value.name = "portfolio_value"

    metrics = calculate_return_portfolio_metrics(
        returns=net_returns,
        value=value,
        weights_df=weights_aligned,
        turnover=turnover,
    )
    return {
        "returns": net_returns,
        "value": value,
        "weights": weights_aligned,
        "turnover": turnover,
        "metrics": metrics,
    }


def _calculate_trade_win_rate(pf: vbt.Portfolio) -> float:
    """Return average trade win rate across columns when available."""
    try:
        win_rate = pf.trades.win_rate()
    except Exception:
        return 0.0

    if hasattr(win_rate, "mean"):
        return float(win_rate.mean() * 100)
    return float(win_rate * 100)


def calculate_portfolio_metrics(pf: vbt.Portfolio) -> Dict[str, float]:
    """Compute portfolio metrics with zero risk-free-rate assumption."""
    returns = _ensure_series(pf.returns(), "pf.returns()").dropna()
    ann_factor = float(returns.vbt.returns().ann_factor)

    value = _ensure_series(pf.value(), "pf.value()")
    total_return = float((value.iloc[-1] / value.iloc[0] - 1) * 100)
    avg_daily_return = float(returns.mean() * 100)
    avg_daily_volatility = float(returns.std() * 100)
    annualized_return = float(returns.mean() * ann_factor * 100)
    annualized_volatility = float(returns.std() * np.sqrt(ann_factor) * 100)

    if returns.std() > 0:
        sharpe_ratio = float(
            (returns.mean() * ann_factor) / (returns.std() * np.sqrt(ann_factor))
        )
    else:
        sharpe_ratio = 0.0

    cumulative_returns = (1 + returns).cumprod()
    rolling_max = cumulative_returns.expanding().max()
    drawdowns = (cumulative_returns - rolling_max) / rolling_max
    max_drawdown = float(drawdowns.min() * 100)
    calmar_ratio = (
        float(annualized_return / abs(max_drawdown))
        if max_drawdown != 0
        else 0.0
    )

    return {
        "Total Return (%)": total_return,
        "Average Daily Return (%)": avg_daily_return,
        "Average Daily Volatility (%)": avg_daily_volatility,
        "Annualized Return (%)": annualized_return,
        "Annualized Volatility (%)": annualized_volatility,
        "Sharpe Ratio": sharpe_ratio,
        "Max Drawdown (%)": max_drawdown,
        "Calmar Ratio": calmar_ratio,
        "Win Rate (%)": _calculate_trade_win_rate(pf),
        "Number of Trading Days": int(len(returns)),
    }


def calculate_signal_diagnostics(
    predicted_returns: pd.DataFrame,
    actual_returns: pd.DataFrame,
    signals_df: pd.DataFrame,
) -> Dict[str, float]:
    """Summarize signal activity and active directional accuracy."""
    predicted_series = _ensure_series(predicted_returns, "predicted_returns")
    actual_series = _ensure_series(actual_returns, "actual_returns")
    signal_series = _ensure_series(signals_df, "signals_df").astype(int)

    active_mask = signal_series != 0
    actual_sign = _sign_series(actual_series)
    predicted_sign = _sign_series(predicted_series)
    valid_overall_mask = actual_sign != 0
    valid_active_mask = active_mask & valid_overall_mask
    previous_signal = signal_series.shift(1).fillna(0).astype(int)

    overall_directional_accuracy = float("nan")
    if valid_overall_mask.any():
        overall_directional_accuracy = float(
            (
                predicted_sign[valid_overall_mask]
                == actual_sign[valid_overall_mask]
            ).mean()
            * 100
        )

    active_directional_accuracy = float("nan")
    if valid_active_mask.any():
        active_directional_accuracy = float(
            (signal_series[valid_active_mask] == actual_sign[valid_active_mask]).mean()
            * 100
        )

    active_actual_returns = actual_series[active_mask]

    return {
        "Long Count": int((signal_series == 1).sum()),
        "Short Count": int((signal_series == -1).sum()),
        "Flat Count": int((signal_series == 0).sum()),
        "Active Signal Count": int(active_mask.sum()),
        "Active Signal Ratio (%)": float(active_mask.mean() * 100),
        "Signal Change Count": int(signal_series.ne(previous_signal).sum()),
        "Overall Directional Accuracy (%)": overall_directional_accuracy,
        "Active Directional Accuracy (%)": active_directional_accuracy,
        "Average Predicted Return (%)": float(predicted_series.mean() * 100),
        "Average Absolute Predicted Return (%)": float(
            predicted_series.abs().mean() * 100
        ),
        "Average Actual Return When Active (%)": float(
            active_actual_returns.mean() * 100
        )
        if not active_actual_returns.empty
        else float("nan"),
    }


def display_signal_diagnostics(
    signal_diagnostics: Dict[str, float],
    signal_policy: str,
    long_threshold: float,
    short_threshold: float,
) -> None:
    """Print signal activity so threshold and trade density are visible."""
    print("\nSignal Diagnostics")
    print(
        f"  - Signal policy: {signal_policy} | "
        f"long_threshold={long_threshold:.4f} | "
        f"short_threshold={short_threshold:.4f}"
    )
    print(
        "  - Counts: "
        f"Long={signal_diagnostics['Long Count']} | "
        f"Short={signal_diagnostics['Short Count']} | "
        f"Flat={signal_diagnostics['Flat Count']} | "
        f"Active={signal_diagnostics['Active Signal Count']} "
        f"({signal_diagnostics['Active Signal Ratio (%)']:.2f}%)"
    )
    print(
        "  - Direction: "
        f"Overall DA={signal_diagnostics['Overall Directional Accuracy (%)']:.2f}% | "
        f"Active DA={signal_diagnostics['Active Directional Accuracy (%)']:.2f}%"
    )
    print(
        "  - Predicted return stats: "
        f"mean={signal_diagnostics['Average Predicted Return (%)']:.4f}% | "
        f"|mean|={signal_diagnostics['Average Absolute Predicted Return (%)']:.4f}% | "
        f"changes={signal_diagnostics['Signal Change Count']}"
    )


def display_strategy_comparison(metrics_rows: List[Dict[str, object]]) -> None:
    """Print a compact comparison table across forecast and baseline strategies."""
    if not metrics_rows:
        return

    metrics_df = pd.DataFrame(metrics_rows).sort_values(
        by="Total Return (%)",
        ascending=False,
        kind="stable",
    )

    print(f"\n{'=' * 24} STRATEGY COMPARISON {'=' * 24}")
    for row in metrics_df.to_dict(orient="records"):
        print(
            f"{row['strategy']:<24s} | "
            f"Total Return: {row['Total Return (%)']:>9.4f}% | "
            f"Sharpe: {row['Sharpe Ratio']:>7.4f} | "
            f"MaxDD: {row['Max Drawdown (%)']:>9.4f}%"
        )
    print(f"{'=' * 70}")


def plot_portfolio_performance(
    pf: vbt.Portfolio,
    save_path: str = "portfolio_performance.png",
) -> None:
    """Visualize portfolio value, cumulative return and drawdown."""
    fig, axes = plt.subplots(3, 1, figsize=(14, 12))
    value = _ensure_series(pf.value(), "pf.value()")
    returns = _ensure_series(pf.returns(), "pf.returns()")

    ax1 = axes[0]
    value.plot(ax=ax1, color="#2E86AB", linewidth=2)
    ax1.set_title("Portfolio Value Over Time", fontsize=14, fontweight="bold")
    ax1.set_ylabel("Portfolio Value ($)", fontsize=12)
    ax1.grid(True, alpha=0.3)

    ax2 = axes[1]
    cumulative_returns = (1 + returns).cumprod() - 1
    cumulative_returns.plot(ax=ax2, color="#06A77D", linewidth=2)
    ax2.set_title("Cumulative Returns", fontsize=14, fontweight="bold")
    ax2.set_ylabel("Cumulative Return", fontsize=12)
    ax2.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    ax2.grid(True, alpha=0.3)

    ax3 = axes[2]
    cum_returns = (1 + returns).cumprod()
    rolling_max = cum_returns.expanding().max()
    drawdowns = (cum_returns - rolling_max) / rolling_max
    drawdowns.plot(ax=ax3, color="#D62828", linewidth=2)
    ax3.fill_between(
        drawdowns.index,
        drawdowns.to_numpy(),
        0,
        alpha=0.3,
        color="#D62828",
    )
    ax3.set_title("Drawdown", fontsize=14, fontweight="bold")
    ax3.set_ylabel("Drawdown", fontsize=12)
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"\nPortfolio performance chart saved to {save_path}")


def plot_return_portfolio_performance(
    value: pd.Series,
    returns: pd.Series,
    save_path: str = "portfolio_performance.png",
) -> None:
    """Visualize portfolio value, cumulative return and drawdown from returns."""
    fig, axes = plt.subplots(3, 1, figsize=(14, 12))

    ax1 = axes[0]
    value.plot(ax=ax1, color="#2E86AB", linewidth=2)
    ax1.set_title("Portfolio Value Over Time", fontsize=14, fontweight="bold")
    ax1.set_ylabel("Portfolio Value ($)", fontsize=12)
    ax1.grid(True, alpha=0.3)

    ax2 = axes[1]
    cumulative_returns = (1 + returns).cumprod() - 1
    cumulative_returns.plot(ax=ax2, color="#06A77D", linewidth=2)
    ax2.set_title("Cumulative Returns", fontsize=14, fontweight="bold")
    ax2.set_ylabel("Cumulative Return", fontsize=12)
    ax2.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    ax2.grid(True, alpha=0.3)

    ax3 = axes[2]
    cum_returns = (1 + returns).cumprod()
    rolling_max = cum_returns.expanding().max()
    drawdowns = (cum_returns - rolling_max) / rolling_max
    drawdowns.plot(ax=ax3, color="#D62828", linewidth=2)
    ax3.fill_between(
        drawdowns.index,
        drawdowns.to_numpy(),
        0,
        alpha=0.3,
        color="#D62828",
    )
    ax3.set_title("Drawdown", fontsize=14, fontweight="bold")
    ax3.set_ylabel("Drawdown", fontsize=12)
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"\nPortfolio performance chart saved to {save_path}")


def plot_asset_allocation(
    weights_df: pd.DataFrame,
    save_path: str = "asset_allocation.png",
) -> None:
    """Visualize executed asset weights over time."""
    fig, ax = plt.subplots(figsize=(14, 6))
    weights_df.plot.area(ax=ax, alpha=0.7, cmap="tab10", stacked=False)
    ax.set_title("Asset Allocation Over Time", fontsize=14, fontweight="bold")
    ax.set_ylabel("Weight", fontsize=12)
    ax.set_xlabel("Date", fontsize=12)
    ax.legend(loc="center left", bbox_to_anchor=(1, 0.5), fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Asset allocation chart saved to {save_path}")


def save_results(
    metrics_rows: List[Dict[str, object]],
    forecast_df: pd.DataFrame,
    predicted_returns: pd.DataFrame,
    signals_df: pd.DataFrame,
    order_schedules: Dict[str, pd.DataFrame],
    timing: Dict[str, object],
    output_dir: str = "backtest_results",
    decision_weights: Optional[pd.DataFrame] = None,
    extra_frames: Optional[Dict[str, pd.DataFrame]] = None,
    signal_diagnostics: Optional[Dict[str, float]] = None,
) -> None:
    """Persist backtest outputs for a single architecture run."""
    os.makedirs(output_dir, exist_ok=True)

    pd.DataFrame(metrics_rows).to_csv(
        os.path.join(output_dir, "portfolio_metrics.csv"),
        index=False,
    )
    forecast_df.to_csv(os.path.join(output_dir, "forecasts.csv"))
    predicted_returns.to_csv(os.path.join(output_dir, "predicted_returns.csv"))
    signals_df.to_csv(os.path.join(output_dir, "signals.csv"))

    if decision_weights is not None:
        decision_weights.to_csv(os.path.join(output_dir, "decision_weights.csv"))

    if signal_diagnostics is not None:
        pd.DataFrame([signal_diagnostics]).to_csv(
            os.path.join(output_dir, "signal_diagnostics.csv"),
            index=False,
        )

    for strategy_name, orders_df in order_schedules.items():
        safe_name = re.sub(r"[^A-Za-z0-9]+", "_", strategy_name).strip("_").lower()
        orders_df.to_csv(os.path.join(output_dir, f"weights_{safe_name}.csv"))

    if "forecast_strategy" in order_schedules:
        order_schedules["forecast_strategy"].to_csv(os.path.join(output_dir, "weights.csv"))

    if extra_frames is not None:
        for file_stem, frame in extra_frames.items():
            frame.to_csv(os.path.join(output_dir, f"{file_stem}.csv"))

    pd.DataFrame([timing]).to_csv(
        os.path.join(output_dir, "backtest_timing.csv"),
        index=False,
    )
    print(f"\nResults saved to {output_dir}/")


def run_single_asset_backtest_from_forecasts(
    predicted_prices: np.ndarray,
    actual_prices: np.ndarray,
    last_prices: np.ndarray,
    target_dates: pd.Index,
    target_name: str,
    signal_threshold: float = 0.001,
    long_threshold: Optional[float] = None,
    short_threshold: Optional[float] = None,
    signal_policy: str = "normal",
    output_dir: str = "backtest_results_single_asset",
    initial_cash: float = DEFAULT_INITIAL_CASH,
    execution_delay_steps: Optional[int] = None,
    transaction_cost: Optional[float] = None,
    run_delay0_diagnostic: bool = True,
    diagnostic_delay_steps: int = 0,
    run_fee0_diagnostic: bool = True,
    diagnostic_fee0_transaction_cost: float = 0.0,
    backtest_engine: str = "forecast_return",
    drop_zero_return_days: bool = True,
    weighting_scheme: str = "signal",
    confidence_window: int = 252,
    confidence_quantile: float = 0.75,
    max_abs_weight: float = 1.0,
    holding_period_steps: int = 1,
    volatility_target: Optional[float] = None,
    volatility_window: int = 20,
    max_leverage: float = 1.0,
    strategy_label: str = "Forecast Strategy",
) -> Dict[str, object]:
    """Run a leakage-safe single-asset backtest from pre-computed forecasts.

    Inputs follow the graph-adapter rolling forecast convention:
    - `predicted_prices[t]` forecasts the close at target date T+1
    - `actual_prices[t]` is the realized close at target date T+1
    - `last_prices[t]` is the last observed close at decision date T
    """
    engine = backtest_engine.strip().lower()
    if engine not in {"forecast_return", "vectorbt_orders"}:
        raise ValueError(
            "backtest_engine must be one of: 'forecast_return', 'vectorbt_orders'."
        )

    if execution_delay_steps is None:
        execution_delay_steps = int(DEFAULT_EXECUTION_DELAY_STEPS)
    if transaction_cost is None:
        transaction_cost = float(DEFAULT_TRANSACTION_COST)
    if long_threshold is None:
        long_threshold = signal_threshold
    if short_threshold is None:
        short_threshold = signal_threshold

    if not (
        len(predicted_prices)
        == len(actual_prices)
        == len(last_prices)
        == len(target_dates)
    ):
        raise ValueError(
            "predicted_prices, actual_prices, last_prices and target_dates "
            "must have the same length."
        )

    os.makedirs(output_dir, exist_ok=True)

    target_index = pd.Index(pd.to_datetime(target_dates), name="date")
    asset_cols = [target_name]

    forecast_df = _to_single_asset_frame(predicted_prices, target_index, target_name)
    actual_df = _to_single_asset_frame(actual_prices, target_index, target_name)
    last_price_df = _to_single_asset_frame(last_prices, target_index, target_name)
    predicted_returns = (forecast_df - last_price_df) / last_price_df
    actual_returns = (actual_df - last_price_df) / last_price_df

    timing = build_timing_summary(forecast_df, execution_delay_steps)
    print(f"\n{strategy_label} — Single-Asset Portfolio Backtest")
    display_timing_summary(timing)

    signals = generate_trading_signals(
        predicted_returns,
        threshold=signal_threshold,
        long_threshold=long_threshold,
        short_threshold=short_threshold,
        signal_policy=signal_policy,
    )
    signal_diagnostics = calculate_signal_diagnostics(
        predicted_returns=predicted_returns,
        actual_returns=actual_returns,
        signals_df=signals,
    )
    display_signal_diagnostics(
        signal_diagnostics,
        signal_policy=signal_policy,
        long_threshold=float(long_threshold),
        short_threshold=float(short_threshold),
    )
    print(
        "  - Weighting: "
        f"engine={engine} | scheme={weighting_scheme} | "
        f"hold={holding_period_steps} | vol_target={volatility_target}"
    )

    decision_weights = build_decision_weights(
        predicted_returns=predicted_returns,
        signals_df=signals,
        weighting_scheme=weighting_scheme,
        confidence_window=confidence_window,
        confidence_quantile=confidence_quantile,
        max_abs_weight=max_abs_weight,
    )
    decision_weights = apply_holding_period(
        decision_weights,
        holding_period_steps=holding_period_steps,
    )
    decision_weights = apply_volatility_target(
        decision_weights,
        actual_returns=actual_returns,
        volatility_target=volatility_target,
        volatility_window=volatility_window,
        max_leverage=max_leverage,
    )
    executed_weights = apply_execution_delay(
        decision_weights,
        delay_steps=execution_delay_steps,
    )

    strategy_specs: List[Dict[str, object]] = [
        {
            "strategy": "forecast_strategy",
            "role": "main_forecast",
            "orders": executed_weights,
            "execution_delay_steps": execution_delay_steps,
            "transaction_cost": float(transaction_cost),
            "uses_signal_diagnostics": True,
        }
    ]

    if run_delay0_diagnostic and diagnostic_delay_steps != execution_delay_steps:
        strategy_specs.append(
            {
                "strategy": f"forecast_delay{diagnostic_delay_steps}_diag",
                "role": f"diagnostic_delay{diagnostic_delay_steps}",
                "orders": apply_execution_delay(
                    decision_weights,
                    delay_steps=diagnostic_delay_steps,
                ),
                "execution_delay_steps": diagnostic_delay_steps,
                "transaction_cost": float(transaction_cost),
                "uses_signal_diagnostics": True,
            }
        )

    if run_fee0_diagnostic and diagnostic_fee0_transaction_cost != transaction_cost:
        strategy_specs.append(
            {
                "strategy": "forecast_fee0_diag",
                "role": "diagnostic_fee0",
                "orders": executed_weights,
                "execution_delay_steps": execution_delay_steps,
                "transaction_cost": float(diagnostic_fee0_transaction_cost),
                "uses_signal_diagnostics": True,
            }
        )

    strategy_specs.extend(
        [
            {
                "strategy": "buy_hold_target",
                "role": "baseline_buy_hold",
                "orders": build_buy_and_hold_orders(
                    template_index=target_index,
                    columns=asset_cols,
                    execution_delay_steps=execution_delay_steps,
                ),
                "execution_delay_steps": execution_delay_steps,
                "transaction_cost": float(transaction_cost),
                "uses_signal_diagnostics": False,
            },
            {
                "strategy": "always_flat",
                "role": "baseline_flat",
                "orders": build_flat_orders(target_index, asset_cols),
                "execution_delay_steps": execution_delay_steps,
                "transaction_cost": float(transaction_cost),
                "uses_signal_diagnostics": False,
            },
        ]
    )

    strategy_orders = {}
    for spec in strategy_specs:
        strategy_name = str(spec["strategy"])
        orders_df = spec["orders"]
        if engine == "forecast_return":
            orders_df = materialize_target_weights(orders_df)
        strategy_orders[strategy_name] = orders_df

    portfolios: Dict[str, object] = {}
    return_backtests: Dict[str, Dict[str, object]] = {}
    metrics_rows: List[Dict[str, object]] = []
    for spec in strategy_specs:
        strategy_name = str(spec["strategy"])
        if engine == "forecast_return":
            backtest = run_forecast_return_backtest(
                actual_returns=actual_returns,
                weights_df=spec["orders"],
                init_cash=float(initial_cash),
                transaction_cost=float(spec["transaction_cost"]),
                drop_zero_return_days=drop_zero_return_days,
            )
            metrics = backtest["metrics"]
            return_backtests[strategy_name] = backtest
        else:
            portfolio = run_backtest(
                actual_df,
                spec["orders"],
                init_cash=float(initial_cash),
                transaction_cost=float(spec["transaction_cost"]),
            )
            metrics = calculate_portfolio_metrics(portfolio)
            portfolios[strategy_name] = portfolio

        metrics_row: Dict[str, object] = {
            "strategy": strategy_name,
            "Strategy Role": spec["role"],
            "Initial Cash": float(initial_cash),
            "Backtest Engine": engine,
            "Weighting Scheme": weighting_scheme
            if bool(spec["uses_signal_diagnostics"])
            else "",
            "Holding Period Steps": int(holding_period_steps)
            if bool(spec["uses_signal_diagnostics"])
            else 1,
            "Volatility Target": float(volatility_target)
            if bool(spec["uses_signal_diagnostics"])
            and volatility_target is not None
            else float("nan"),
            "Signal Policy": signal_policy
            if bool(spec["uses_signal_diagnostics"])
            else "",
            "Long Threshold": float(long_threshold)
            if bool(spec["uses_signal_diagnostics"])
            else float("nan"),
            "Short Threshold": float(short_threshold)
            if bool(spec["uses_signal_diagnostics"])
            else float("nan"),
            "Execution Delay Steps": int(spec["execution_delay_steps"]),
            "Transaction Cost": float(spec["transaction_cost"]),
            **metrics,
        }
        if bool(spec["uses_signal_diagnostics"]):
            metrics_row.update(signal_diagnostics)
        metrics_rows.append(metrics_row)

    forecast_metrics = next(
        row for row in metrics_rows if row["strategy"] == "forecast_strategy"
    )

    display_strategy_comparison(metrics_rows)
    if engine == "forecast_return":
        forecast_backtest = return_backtests["forecast_strategy"]
        plot_return_portfolio_performance(
            forecast_backtest["value"],
            forecast_backtest["returns"],
            os.path.join(output_dir, "portfolio_performance.png"),
        )
    else:
        plot_portfolio_performance(
            portfolios["forecast_strategy"],
            os.path.join(output_dir, "portfolio_performance.png"),
        )
    plot_asset_allocation(
        executed_weights,
        os.path.join(output_dir, "asset_allocation_executed.png"),
    )

    extra_frames = {
        "actual_prices": actual_df,
        "actual_returns": actual_returns,
        "last_observed_prices": last_price_df,
    }
    if engine == "forecast_return":
        forecast_backtest = return_backtests["forecast_strategy"]
        extra_frames.update(
            {
                "portfolio_returns": forecast_backtest["returns"].to_frame(),
                "portfolio_value": forecast_backtest["value"].to_frame(),
                "portfolio_turnover": forecast_backtest["turnover"].to_frame(
                    "turnover"
                ),
            }
        )

    save_results(
        metrics_rows=metrics_rows,
        forecast_df=forecast_df,
        predicted_returns=predicted_returns,
        signals_df=signals,
        order_schedules=strategy_orders,
        timing=timing,
        output_dir=output_dir,
        decision_weights=decision_weights,
        extra_frames=extra_frames,
        signal_diagnostics=signal_diagnostics,
    )

    return {
        "strategy_label": strategy_label,
        "target_name": target_name,
        "backtest_engine": engine,
        "metrics": metrics_rows,
        "forecast_metrics": forecast_metrics,
        "predictions": forecast_df,
        "actual_prices": actual_df,
        "last_prices": last_price_df,
        "actual_returns": actual_returns,
        "signals": signals,
        "signal_diagnostics": signal_diagnostics,
        "decision_weights": decision_weights,
        "executed_weights": executed_weights,
        "return_backtests": return_backtests,
        "timing": timing,
    }
