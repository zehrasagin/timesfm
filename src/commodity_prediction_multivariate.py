from typing import Tuple, Dict, List, Optional
from datetime import datetime
import re
import numpy as np
import pandas as pd
import torch
import timesfm
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import (
  mean_absolute_error,
  mean_squared_error,
  r2_score,
  mean_absolute_percentage_error,
)

# CL=F target seçtim, test split ratio timegpt ile aynı 
DATA_CONFIGURATION = {
  "csv_path": "src/commodity_prices.csv",
  "target_column": "CL=F",  
  "test_split_ratio": 0.10,  # 90-10 train-test split
}

FORECAST_MODE = {
  "use_multivariate": True,
  "covariate_columns": None,
  "covariate_columns_regex": r"=F$",  
}

FORECAST_CONFIG_OPTIONS = {
  "max_context": 1024, 
  "max_horizon": 1,
  "normalize_inputs": True,
  "use_continuous_quantile_head": True, 
  "force_flip_invariance": True, 
  "infer_is_positive": True,
  "fix_quantile_crossing": True,
  "return_backcast": True,  
}

OUTPUT_CONFIGURATION = {
  "metrics_output_path": "forecast_metrics_multivariate.csv",
  "visualization_output_path": "forecast_visualization_multivariate.png",
}


# regex kısmını burada açıyoruz
#  "covariate_columns": None böyle iken ["BZ=F", "GC=F", "HG=F", "HO=F", "NG=F", "PA=F", "PL=F", "SI=F", "ZC=F"]
# bu şekilde oluyor elle yazmak yerine
def select_columns_by_regex(
  csv_path: str, regex_pattern: str, exclude_column: Optional[str] = None
) -> List[str]:
  df = pd.read_csv(csv_path)
  columns = df.columns.tolist()

  compiled_pattern = re.compile(regex_pattern)
  selected_columns = [col for col in columns if compiled_pattern.search(col)]

  if exclude_column and exclude_column in selected_columns:
    selected_columns.remove(exclude_column)

  return selected_columns


def resolve_covariate_columns(
  csv_path: str,
  covariate_columns: Optional[List[str]] = None,
  covariate_columns_regex: Optional[str] = None,
  target_column: Optional[str] = None,
) -> List[str]:
  if covariate_columns_regex:
    return select_columns_by_regex(
      csv_path, covariate_columns_regex, exclude_column=target_column
    )
  elif covariate_columns:
    return [col for col in covariate_columns if col != target_column]
  else:
    raise ValueError(
      "Either covariate_columns or covariate_columns_regex must be provided"
    )


def load_multivariate_data(
  csv_path: str, target_column: str, covariate_columns: List[str]
) -> Tuple[np.ndarray, np.ndarray]:
  df = pd.read_csv(csv_path)

  all_columns = [target_column] + covariate_columns
  df_subset = df[all_columns].dropna()

  target = df_subset[target_column].values
  covariates = df_subset[covariate_columns].values

  return target, covariates


def split_train_test(
  time_series: np.ndarray, test_size: int
) -> Tuple[np.ndarray, np.ndarray]:
  assert len(time_series) > test_size, "Time series too short for split"

  train_data = time_series[:-test_size]
  test_data = time_series[-test_size:]

  return train_data, test_data


def calculate_all_metrics(
  actual: np.ndarray, predicted: np.ndarray, last_train_value: float
) -> Dict[str, float]:
  return {
    "MAPE": mean_absolute_percentage_error(actual, predicted) * 100,
    "R2": r2_score(actual, predicted),
    "MAE": mean_absolute_error(actual, predicted),
    "MSE": mean_squared_error(actual, predicted),
    "RMSE": np.sqrt(mean_squared_error(actual, predicted)),
  }


def save_metrics_to_csv(
  metrics: Dict[str, float],
  csv_path: str,
  column_name: str,
  forecast_horizon: int,
  mode: str = "multivariate",
  covariate_columns: Optional[List[str]] = None,
  output_path: str = OUTPUT_CONFIGURATION["metrics_output_path"],
) -> None:
  timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

  metrics_data = {
    "timestamp": timestamp,
    "target_column": column_name,
    "mode": mode,
    "forecast_horizon": forecast_horizon,
    "data_source": csv_path,
    **metrics,
  }

  if covariate_columns:
    metrics_data["covariates"] = ", ".join(covariate_columns)
  else:
    metrics_data["covariates"] = "None"

  metrics_df = pd.DataFrame([metrics_data])

  metrics_df.to_csv(output_path, index=False)


def initialize_timesfm_model() -> timesfm.TimesFM_2p5_200M_torch:
  torch.set_float32_matmul_precision("high")

  model = timesfm.TimesFM_2p5_200M_torch.from_pretrained(
    "google/timesfm-2.5-200m-pytorch"
  )

  model.compile(
    timesfm.ForecastConfig(
      max_context=FORECAST_CONFIG_OPTIONS["max_context"],
      max_horizon=FORECAST_CONFIG_OPTIONS["max_horizon"],
      normalize_inputs=FORECAST_CONFIG_OPTIONS["normalize_inputs"],
      use_continuous_quantile_head=FORECAST_CONFIG_OPTIONS[
        "use_continuous_quantile_head"
      ],
      force_flip_invariance=FORECAST_CONFIG_OPTIONS["force_flip_invariance"],
      infer_is_positive=FORECAST_CONFIG_OPTIONS["infer_is_positive"],
      fix_quantile_crossing=FORECAST_CONFIG_OPTIONS["fix_quantile_crossing"],
      return_backcast=FORECAST_CONFIG_OPTIONS["return_backcast"],
    )
  )

  return model

# rolling forecast kısmı 
def forecast_multivariate_rolling(
  model: timesfm.TimesFM_2p5_200M_torch,
  train_target: np.ndarray,
  train_covariates: np.ndarray,
  test_target: np.ndarray,
  test_covariates: np.ndarray,
  horizon: int,
  covariate_columns: List[str],
) -> Tuple[np.ndarray, np.ndarray]:
  num_predictions = min(horizon, len(test_target))

  point_forecasts = []
  quantile_forecasts = []

  current_history = train_target.copy()
  current_covariates = train_covariates.copy()

  for step in tqdm(
    range(num_predictions), desc="Multivariate Rolling Forecast Progress"
  ):
    if len(current_history) > FORECAST_CONFIG_OPTIONS["max_context"]:
      input_history = current_history[-FORECAST_CONFIG_OPTIONS["max_context"] :]
      input_covariates = current_covariates[-FORECAST_CONFIG_OPTIONS["max_context"] :]
    else:
      input_history = current_history
      input_covariates = current_covariates

    dynamic_numerical_covariates = {}

    for idx, col_name in enumerate(covariate_columns):
      hist_cov = input_covariates[:, idx]
      future_cov = test_covariates[step, idx]

      full_seq = np.concatenate([hist_cov, [future_cov]])

      dynamic_numerical_covariates[col_name] = [full_seq.tolist()]

    point_fc, quantile_fc = model.forecast_with_covariates(
      inputs=[input_history],
      dynamic_numerical_covariates=dynamic_numerical_covariates,
      xreg_mode="xreg + timesfm", #xreg(external) ile covariate etkisi ve timesfm ile zaman serisi etkisi birlikte kullandım.
      normalize_xreg_target_per_input=True,
      force_on_cpu=True,
    )

    point_forecasts.append(point_fc[0][0])
    quantile_forecasts.append(quantile_fc[0][0, :])

    current_history = np.append(current_history, test_target[step])
    current_covariates = np.vstack([current_covariates, test_covariates[step]])

  return np.array(point_forecasts), np.array(quantile_forecasts)


def extract_prediction_intervals(
  quantile_forecast: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
  mean_forecast = quantile_forecast[:, 0]
  median_forecast = quantile_forecast[:, 5]
  lower_bound_10 = quantile_forecast[:, 1]
  upper_bound_90 = quantile_forecast[:, 9]

  return mean_forecast, median_forecast, lower_bound_10, upper_bound_90


def display_metrics(metrics: Dict[str, float]) -> None:
  print(f"\n{'=' * 27}METRICS{'=' * 26}")

  for metric_name, metric_value in metrics.items():
    if metric_name in ["MAPE"] or metric_name.startswith("Directional_Accuracy"):
      unit = "%"
    else:
      unit = ""
    print(f"{metric_name:<25s}: {metric_value:12.4f}{unit}")

  print(f"{'=' * 60}")


def visualize_forecast(
  train_data: np.ndarray,
  test_data: np.ndarray,
  predicted: np.ndarray,
  lower_bound: np.ndarray,
  upper_bound: np.ndarray,
  column_name: str,
  save_path: str = OUTPUT_CONFIGURATION["visualization_output_path"],
) -> None:
  test_len = len(test_data)
  forecast_indices = np.arange(1, test_len + 1)

  plt.figure(figsize=(14, 7))

  plt.plot(
    forecast_indices,
    test_data,
    label="Actual Values",
    color="#06A77D",
    linewidth=2.5,
    marker="o",
    markersize=7,
  )

  plt.plot(
    forecast_indices,
    predicted,
    label="Predicted Values",
    color="#D62828",
    linewidth=2.5,
    marker="s",
    markersize=7,
    linestyle="--",
  )

  plt.fill_between(
    forecast_indices,
    lower_bound,
    upper_bound,
    alpha=0.25,
    color="#F77F00",
    label="80% Prediction Interval",
  )

  plt.xlabel("Forecast Day", fontsize=12, fontweight="bold")
  plt.ylabel(column_name, fontsize=12, fontweight="bold")
  plt.title(
    f"{column_name} Multivariate Rolling Forecast - Actual vs Predicted\
    \nMAPE: {mean_absolute_percentage_error(test_data, predicted) * 100:.2f}%",
    fontsize=14,
    fontweight="bold",
    pad=20,
  )

  plt.legend(loc="best", fontsize=11, framealpha=0.9, edgecolor="#333333")

  plt.grid(True, alpha=0.3, linestyle="--", linewidth=0.5)
  plt.tight_layout()

  plt.savefig(save_path, dpi=300, bbox_inches="tight")
  plt.close()
  print(f"\n{'=' * 24}VISUALIZATION{'=' * 23}")
  print(f"Forecast visualization saved to {save_path}")
  print(f"{'=' * 60}")


def run_multivariate_forecast(
  csv_path: str = DATA_CONFIGURATION["csv_path"],
  target_column: str = DATA_CONFIGURATION["target_column"],
  covariate_columns: Optional[List[str]] = FORECAST_MODE["covariate_columns"],
  test_split_ratio: float = DATA_CONFIGURATION["test_split_ratio"],
  covariate_columns_regex: Optional[str] = FORECAST_MODE["covariate_columns_regex"],
) -> None:
  covariate_columns = resolve_covariate_columns(
    csv_path,
    covariate_columns=covariate_columns,
    covariate_columns_regex=covariate_columns_regex,
    target_column=target_column,
  )
  print(f"\n{'=' * 24}MULTIVARIATE{'=' * 24}")
  print(f"Target column: {target_column}")
  print(f"Selected covariate columns: {covariate_columns}")
  print(f"{'=' * 60}")

  target_data, covariate_data = load_multivariate_data(
    csv_path, target_column, covariate_columns
  )

  # bugünün covariate'larıyla yarını tahmin ediyoruz
  covariate_data = covariate_data[:-1]
  target_data = target_data[1:]
  
  forecast_horizon = int(len(target_data) * test_split_ratio)
  print(f"Data size: {len(target_data)} | Train: {len(target_data) - forecast_horizon} ({(1-test_split_ratio)*100:.1f}%) | Test: {forecast_horizon} ({test_split_ratio*100:.1f}%)")

  train_target, test_target = split_train_test(target_data, forecast_horizon)
  train_covariates, test_covariates = split_train_test(covariate_data, forecast_horizon)

  model = initialize_timesfm_model()

  point_forecast, quantile_forecast = forecast_multivariate_rolling(
    model,
    train_target,
    train_covariates,
    test_target,
    test_covariates,
    forecast_horizon,
    covariate_columns,
  )

  mean_fc, median_fc, lower_fc, upper_fc = extract_prediction_intervals(
    quantile_forecast
  )

  metrics = calculate_all_metrics(test_target, point_forecast, train_target[-1])
  display_metrics(metrics)
  save_metrics_to_csv(
    metrics,
    csv_path,
    target_column,
    forecast_horizon,
    mode="multivariate",
    covariate_columns=covariate_columns,
  )
  visualize_forecast(
    train_target, test_target, point_forecast, lower_fc, upper_fc, target_column
  )


def main() -> None:
  run_multivariate_forecast()


if __name__ == "__main__":
  main()
