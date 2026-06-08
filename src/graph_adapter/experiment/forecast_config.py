"""Project configuration for TSFM graph forecasting experiments."""

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
    "num_gat_heads": 4,
    "num_gat_layers": 2,
    "dropout": 0.1,
    "corr_window": 60,
    "use_absolute_corr": False,
    "add_self_loops": True,
    "target_mode": "log_return",  # "log_return" or "delta"
}

TRAINING_CONFIG = {
    "seed": 42,
    "num_epochs": 15,
    "batch_size": 64,
    "learning_rate": 1e-4,
    "fusion_lr_scale": 0.25,
    "weight_decay": 1e-4,
    "grad_clip_norm": 1.0,
    "val_ratio": 0.15,
    "stride": 1,
    "checkpoint_metric": "val_mape",  # "val_mape" or "val_loss"
    "use_pytorch_lightning": False,
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

EXPERIMENT_CONFIG = {
    "run_embedding_only": True,
    "run_graph_only": True,
    "run_graph_adapter": True,
    "corr_window_sweep": [60],
    "use_absolute_corr_sweep": [True],
}

PORTFOLIO_BACKTEST_CONFIG = {
    "enabled": True,
    "initial_cash": 100000,
    "backtest_engine": "forecast_return",  # "forecast_return" or "vectorbt_orders"
    "drop_zero_return_days": True,
    "signal_threshold": 0.003,
    "long_threshold": 0.003,
    "short_threshold": 0.003,
    "default_signal_policy": "normal",
    "signal_policy_by_mode": {
        "embedding_only": "normal",
        "graph_only": "reverse",
        "graph_adapter_v2": "reverse",
    },
    "weighting_scheme": "signal",  # "signal" or "confidence"
    "weighting_scheme_by_mode": {
        "graph_adapter_v2": "confidence",
    },
    "confidence_window": 252,
    "confidence_quantile": 0.75,
    "max_abs_weight": 1.0,
    "holding_period_steps": 1,
    "holding_period_steps_by_mode": {
        "graph_adapter_v2": 3,
    },
    "volatility_target": None,
    "volatility_target_by_mode": {
        "graph_adapter_v2": 0.01,
    },
    "volatility_window": 20,
    "max_leverage": 1.0,
    "transaction_cost": 0.001,
    "execution_delay_steps": 0,
    "run_delay0_diagnostic": True,
    "diagnostic_delay_steps": 1,
    "run_fee0_diagnostic": True,
    "diagnostic_fee0_transaction_cost": 0.0,
}

OUTPUT_CONFIG = {
    "model_save_path": "graph_adapter_checkpoint.pt",
    "embedding_store_path": "embeddings.pt",
    "metrics_output_path": "forecast_metrics_graph_adapter.csv",
    "visualization_output_path": "forecast_visualization_graph_adapter.png",
    "comparison_visualization_output_path": "forecast_comparison_graph_adapter.png",
    "training_curve_path": "training_curve_graph_adapter.png",
    "portfolio_output_dir": "portfolio_backtest",
}
