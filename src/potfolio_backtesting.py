"""
Portfolio Backtesting with TimesFM Forecasts
- Sharpe Ratio
- Maximum Drawdown
- Average Return
- Average Standard Deviation
"""

import numpy as np
import pandas as pd
import vectorbt as vbt
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple, Dict, List, Optional
from datetime import datetime
import re
from tqdm import tqdm
import torch
import timesfm
from sklearn.metrics import mean_absolute_percentage_error

# VectorBT ayarları
vbt.settings.returns['year_freq'] = '252 days'
vbt.settings.array_wrapper['freq'] = 'd'

sns.set_style('darkgrid')

# ==================== KONFİGÜRASYON ====================
DATA_CONFIG = {
    "csv_path": "src/commodity_features.csv",
    "test_split_ratio": 0.10,  # Son %10'u test için kullan
}

FORECAST_CONFIG = {
    "max_context": 1024,
    "max_horizon": 1,
    "normalize_inputs": True,
    "use_continuous_quantile_head": True,
    "force_flip_invariance": True,
    "infer_is_positive": True,
    "fix_quantile_crossing": True,
    "return_backcast": True,
}

PORTFOLIO_CONFIG = {
    "initial_cash": 100000,  # Başlangıç sermayesi
    "transaction_cost": 0.001,  # %0.1 işlem maliyeti
}


# ==================== VERİ YÜKLEME ====================
def load_commodity_data(csv_path: str) -> pd.DataFrame:
    """Commodity fiyat verilerini yükle"""
    df = pd.read_csv(csv_path, index_col='date', parse_dates=['date'])
    df = df.dropna()
    return df


def select_columns_by_regex(
    df: pd.DataFrame, regex_pattern: str = r" Comdty$"
) -> List[str]:
    """Regex pattern ile kolon seç"""
    compiled_pattern = re.compile(regex_pattern)
    selected_columns = [col for col in df.columns if compiled_pattern.search(col)]
    return selected_columns


# ==================== TAHMİN FONKSİYONLARI ====================
def initialize_timesfm_model() -> timesfm.TimesFM_2p5_200M_torch:
    """TimesFM modelini başlat"""
    torch.set_float32_matmul_precision("high")
    
    model = timesfm.TimesFM_2p5_200M_torch.from_pretrained(
        "google/timesfm-2.5-200m-pytorch"
    )
    
    model.compile(
        timesfm.ForecastConfig(
            max_context=FORECAST_CONFIG["max_context"],
            max_horizon=FORECAST_CONFIG["max_horizon"],
            normalize_inputs=FORECAST_CONFIG["normalize_inputs"],
            use_continuous_quantile_head=FORECAST_CONFIG["use_continuous_quantile_head"],
            force_flip_invariance=FORECAST_CONFIG["force_flip_invariance"],
            infer_is_positive=FORECAST_CONFIG["infer_is_positive"],
            fix_quantile_crossing=FORECAST_CONFIG["fix_quantile_crossing"],
            return_backcast=FORECAST_CONFIG["return_backcast"],
        )
    )
    
    return model


def generate_forecasts_for_all_assets(
    model: timesfm.TimesFM_2p5_200M_torch,
    price_df: pd.DataFrame,
    test_size: int,
) -> pd.DataFrame:
    """
    Tüm varlıklar için rolling forecast yap.
    
    Rolling forecast mantığı:
    - T gününe kadar olan veri ile T+1 gününü tahmin et
    - forecast_df.index = T+1 (tahmin edilen hedef gün)
    - forecast_df.values = T+1 günü için tahmin edilen fiyat
    - Her tahmin sonrası T+1'in gerçek değerini geçmişe ekle (rolling)
    """
    assets = price_df.columns.tolist()
    forecast_results = []
    
    # Her asset için rolling forecast başlat: train periyodu ile başla
    asset_histories = {asset: price_df[asset].iloc[:len(price_df)-test_size].values.copy() for asset in assets}
    test_dates = price_df.index[-test_size:]  # T+1 tarihleri (tahmin hedefleri)
    
    pbar = tqdm(range(test_size), desc="Generating Forecasts", ncols=80, dynamic_ncols=True, leave=False, bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]')
    for step in pbar:
        target_date = test_dates[step]  # Tahmin hedef günü (T+1)
        forecasts_for_day = {"date": target_date}
        
        for asset in assets:
            # T gününe kadar olan veriyi kullan (asset_histories[asset])
            input_data = asset_histories[asset]
            if len(input_data) > FORECAST_CONFIG["max_context"]:
                input_data = input_data[-FORECAST_CONFIG["max_context"]:]
            
            # T+1 günü için tahmin yap
            point_fc, _ = model.forecast(
                horizon=1,
                inputs=[input_data],
            )
            forecasts_for_day[asset] = point_fc[0][0]  # T+1 tahmini
            
            # Rolling: T+1'in gerçek değerini geçmişe ekle (bir sonraki iterasyon için)
            true_next = price_df[asset].iloc[len(asset_histories[asset])]
            asset_histories[asset] = np.append(asset_histories[asset], true_next)
        
        forecast_results.append(forecasts_for_day)
    forecast_df = pd.DataFrame(forecast_results)
    forecast_df.set_index("date", inplace=True)
    return forecast_df


def calculate_forecast_returns(
    forecast_df: pd.DataFrame, 
    actual_prices: pd.DataFrame
) -> pd.DataFrame:
    """
    Tahmin edilen getirileri hesapla.
    
    forecast_df.index = T+1 (tahmin hedef günü)
    forecast_df.values = T+1 için tahmin edilen fiyat
    
    Doğru hizalama:
    - T gününde T+1'i tahmin ediyoruz
    - predicted_return = (forecast_T+1 - actual_T) / actual_T
    - Payda T gününün kapanış fiyatı olmalı (shift ile)
    """
    # T günü fiyatları: forecast_df.index'ten bir gün önce
    # actual_prices.shift(-1) ile forecast_df.index'teki her satır için
    # bir önceki günün fiyatını alıyoruz
    current_prices = actual_prices.shift(1).loc[forecast_df.index]
    
    # Tahmin edilen getiri: (forecast_T+1 - actual_T) / actual_T
    predicted_returns = (forecast_df - current_prices) / current_prices
    
    # İlk satırda NaN olabilir (shift nedeniyle), dropna ile temizle
    predicted_returns = predicted_returns.dropna()
    
    return predicted_returns


def generate_trading_signals(
    predicted_returns: pd.DataFrame,
    threshold: float = 0.0
) -> pd.DataFrame:
    """
    Tahmin edilen getirilere göre trading sinyalleri oluştur.
    - predicted_return > threshold: LONG (1)
    - predicted_return < -threshold: SHORT (-1)
    - else: NO POSITION (0)
    """
    signals = pd.DataFrame(index=predicted_returns.index, columns=predicted_returns.columns)
    
    signals[predicted_returns > threshold] = 1   # Long
    signals[predicted_returns < -threshold] = -1  # Short
    signals = signals.fillna(0)  # No position
    
    return signals.astype(int)


def calculate_weights_from_signals(signals_df: pd.DataFrame) -> pd.DataFrame:
    """
    Sinyallerden portföy ağırlıklarını hesapla.
    Eşit ağırlıklı dağılım (equal weight allocation)
    """
    abs_sum = signals_df.abs().sum(axis=1)
    weights = signals_df.div(abs_sum, axis=0).fillna(0)
    return weights


# ==================== BACKTEST FONKSİYONLARI ====================
def run_backtest(
    price_df: pd.DataFrame,
    weights_df: pd.DataFrame,
    init_cash: float = PORTFOLIO_CONFIG["initial_cash"],
) -> vbt.Portfolio:
    """
    VectorBT ile backtest çalıştır
    
    ÖNEMLİ: weights_df zaten shifted olarak gelir (SEÇENEK A)
    - weights_df.index = T+2 (işlemin gerçekleşeceği gün)
    - price_df.loc[T+2] = T+2 günü kapanış fiyatı ile işlem yapılır
    - Bu sayede T günü tahmini → T+1 signal → T+2 execution (look-ahead bias YOK)
    """
    
    # Fiyat ve ağırlık verilerini hizala
    common_dates = price_df.index.intersection(weights_df.index)
    price_aligned = price_df.loc[common_dates]
    weights_aligned = weights_df.loc[common_dates]
    
    # Kullanıcının isteği üzerine shift işlemi kaldırıldı ve parametreler güncellendi
    pf = vbt.Portfolio.from_orders(
        close=price_aligned,
        size=weights_aligned,
        size_type='targetpercent',
        init_cash=init_cash,
        freq="D",
        cash_sharing=True,
        call_seq='auto',
        fees=PORTFOLIO_CONFIG["transaction_cost"],
    )
    
    return pf


def calculate_portfolio_metrics(pf: vbt.Portfolio) -> Dict[str, float]:
    """Portföy performans metriklerini hesapla"""
    
    returns = pf.returns()
    ann_factor = returns.vbt.returns().ann_factor
    
    # Metrikler
    total_return = (pf.value().iloc[-1] / pf.value().iloc[0] - 1) * 100
    
    # Günlük metrikler
    avg_daily_return = returns.mean() * 100
    avg_daily_volatility = returns.std() * 100
    
    # Yıllıklaştırılmış metrikler
    annualized_return = returns.mean() * ann_factor * 100
    annualized_volatility = returns.std() * np.sqrt(ann_factor) * 100
    sharpe_ratio = (returns.mean() * ann_factor) / (returns.std() * np.sqrt(ann_factor)) if returns.std() > 0 else 0
    
    # Maximum Drawdown
    cumulative_returns = (1 + returns).cumprod()
    rolling_max = cumulative_returns.expanding().max()
    drawdowns = (cumulative_returns - rolling_max) / rolling_max
    max_drawdown = drawdowns.min() * 100
    
    # Calmar Ratio
    calmar_ratio = annualized_return / abs(max_drawdown) if max_drawdown != 0 else 0
    
    # Win Rate
    # positive_returns = (returns > 0).sum()
    # total_trades = (returns != 0).sum()
    # win_rate = (positive_returns / total_trades * 100) if total_trades > 0 else 0
    
    # Trade Win Rate (Average across assets)
    try:
        win_rate = pf.trades.win_rate().mean() * 100
    except:
        win_rate = 0.0
    
    metrics = {
        "Total Return (%)": total_return,
        "Average Daily Return (%)": avg_daily_return,
        "Average Daily Volatility (%)": avg_daily_volatility,
        "Annualized Return (%)": annualized_return,
        "Annualized Volatility (%)": annualized_volatility,
        "Sharpe Ratio": sharpe_ratio,
        "Max Drawdown (%)": max_drawdown,
        "Calmar Ratio": calmar_ratio,
        "Win Rate (%)": win_rate,
        "Number of Trading Days": len(returns),
    }
    
    return metrics


def display_metrics(metrics: Dict[str, float]) -> None:
    """Metrikleri yazdır"""
    print(f"\n{'=' * 25} PORTFOLIO METRICS {'=' * 25}")
    
    for metric_name, metric_value in metrics.items():
        if isinstance(metric_value, float):
            print(f"{metric_name:<30s}: {metric_value:>12.4f}")
        else:
            print(f"{metric_name:<30s}: {metric_value:>12}")
    
    print(f"{'=' * 70}")


def plot_portfolio_performance(pf: vbt.Portfolio, save_path: str = "portfolio_performance.png") -> None:
    """Portföy performansını görselleştir"""
    
    fig, axes = plt.subplots(3, 1, figsize=(14, 12))
    
    # 1. Portfolio Value
    ax1 = axes[0]
    pf.value().plot(ax=ax1, color='#2E86AB', linewidth=2)
    ax1.set_title('Portfolio Value Over Time', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Portfolio Value ($)', fontsize=12)
    ax1.grid(True, alpha=0.3)
    
    # 2. Cumulative Returns
    ax2 = axes[1]
    cumulative_returns = (1 + pf.returns()).cumprod() - 1
    cumulative_returns.plot(ax=ax2, color='#06A77D', linewidth=2)
    ax2.set_title('Cumulative Returns', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Cumulative Return', fontsize=12)
    ax2.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax2.grid(True, alpha=0.3)
    
    # 3. Drawdown
    ax3 = axes[2]
    returns = pf.returns()
    cum_returns = (1 + returns).cumprod()
    rolling_max = cum_returns.expanding().max()
    drawdowns = (cum_returns - rolling_max) / rolling_max
    drawdowns.plot(ax=ax3, color='#D62828', linewidth=2)
    ax3.fill_between(drawdowns.index, drawdowns.values, 0, alpha=0.3, color='#D62828')
    ax3.set_title('Drawdown', fontsize=14, fontweight='bold')
    ax3.set_ylabel('Drawdown', fontsize=12)
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\nPortfolio performance chart saved to {save_path}")


def plot_asset_allocation(weights_df: pd.DataFrame, save_path: str = "asset_allocation.png") -> None:
    """Varlık dağılımını görselleştir"""
    
    fig, ax = plt.subplots(figsize=(14, 6))
    
    # stacked=False is required because weights can be negative (short positions)
    weights_df.plot.area(ax=ax, alpha=0.7, cmap='tab10', stacked=False)
    ax.set_title('Asset Allocation Over Time', fontsize=14, fontweight='bold')
    ax.set_ylabel('Weight', fontsize=12)
    ax.set_xlabel('Date', fontsize=12)
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=8)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Asset allocation chart saved to {save_path}")


# ==================== DEMO FONKSİYONU ====================

"""
def run_demo_backtest():
    print("\n" + "=" * 70)
    print("DEMO PORTFOLIO BACKTESTING")
    print("=" * 70)
    
    # Demo veriler
    dates = pd.date_range(start='2024-01-01', periods=100, freq='D')
    np.random.seed(42)
    
    price_df = pd.DataFrame({
        "Asset_A": 100 * (1 + np.random.randn(100).cumsum() * 0.02),
        "Asset_B": 50 * (1 + np.random.randn(100).cumsum() * 0.015),
        "Asset_C": 75 * (1 + np.random.randn(100).cumsum() * 0.025),
    }, index=dates)
    
    # Basit momentum sinyalleri (demo için)
    returns_5d = price_df.pct_change(5)
    signals = pd.DataFrame(0, index=price_df.index, columns=price_df.columns)
    signals[returns_5d > 0.02] = 1
    signals[returns_5d < -0.02] = -1
    
    weights = calculate_weights_from_signals(signals)
    
    # Backtest
    pf = run_backtest(price_df, weights)
    
    # Metrikler
    metrics = calculate_portfolio_metrics(pf)
    display_metrics(metrics)
    
    # Görselleştirme
    plot_portfolio_performance(pf, "demo_portfolio_performance.png")
    
    return pf, metrics
    """

  


# ==================== ANA FONKSİYON ====================
def run_full_backtest_with_forecasts(
    csv_path: str = DATA_CONFIG["csv_path"],
    test_split_ratio: float = DATA_CONFIG["test_split_ratio"],
    signal_threshold: float = 0.001,  # %0.1 threshold
    limit_test_size: Optional[int] = None,
):
    """
    TimesFM tahminleri ile tam backtest çalıştır
    """
    print("\n" + "=" * 70)
    print("PORTFOLIO BACKTESTING WITH TIMESFM FORECASTS")
    print("=" * 70)
    
    # 1. Veri yükle
    print("\n[1/5] Loading commodity data...")
    price_df = load_commodity_data(csv_path)
    assets = select_columns_by_regex(price_df)
    price_df = price_df[assets]
    print(f"Loaded {len(price_df)} days of data for {len(assets)} assets")
    print(f"Assets: {assets}")
    
    # 2. Test boyutunu hesapla
    test_size = int(len(price_df) * test_split_ratio)
    if limit_test_size is not None:
        test_size = min(test_size, limit_test_size)
        print(f"Limiting test size to {test_size} days for testing.")
        
    print(f"\nTrain: {len(price_df) - test_size} days | Test: {test_size} days")
    
    # 3. Model yükle ve tahminler üret
    print("\n[2/5] Initializing TimesFM model...")
    model = initialize_timesfm_model()
    
    print("\n[3/5] Generating forecasts for all assets...")
    forecast_df = generate_forecasts_for_all_assets(model, price_df, test_size)
    
    # 4. Trading sinyalleri oluştur
    print("\n[4/5] Generating trading signals...")
    predicted_returns = calculate_forecast_returns(forecast_df, price_df)
    signals = generate_trading_signals(predicted_returns, threshold=signal_threshold)
    weights = calculate_weights_from_signals(signals)
    
    print(f"Signal distribution:")
    for asset in assets:
        long_count = (signals[asset] == 1).sum()
        short_count = (signals[asset] == -1).sum()
        neutral_count = (signals[asset] == 0).sum()
        print(f"  {asset}: Long={long_count}, Short={short_count}, Neutral={neutral_count}")
    
    # 4b. Execution shift: SEÇENEK A - Close-only veride en temiz akış
    # ═══════════════════════════════════════════════════════════════
    # Mantık:
    #   T günü    → T+1 tahmini yap (forecast_df.index = T+1)
    #   T+1 close → Tahmini değerlendir, signal/weight üret (weights.index = T+1)
    #   T+2 close → İşlemi gerçekleştir (weights_exec.index = T+2)
    #
    # Bu sayede:
    #   - T+1 tahmini T günü bilgisi ile yapılır ✓
    #   - T+1 kapanışında signal üretilir (T+1 close biliniyor) ✓
    #   - T+2 kapanışında işlem yapılır (gerçekçi execution) ✓
    #   - SIFIR look-ahead bias ✓
    # ═══════════════════════════════════════════════════════════════
    weights_exec = weights.shift(1).fillna(0)
    
    print("\n📊 Execution Logic (SEÇENEK A - Close-only temiz akış):")
    print("   Day T      → Forecast T+1 (model tahmin eder)")
    print("   Day T+1    → Evaluate & Signal (close'da değerlendir)")
    print("   Day T+2    → Execute trades (close'da işlem yap)")
    print("   ✓ Tamamen look-ahead bias'tan korunmuş")
    
    # 5. Backtest çalıştır
    print("\n[5/5] Running backtest...")
    pf = run_backtest(price_df, weights_exec)
    
    # Sonuçlar
    metrics = calculate_portfolio_metrics(pf)
    display_metrics(metrics)
    
    # Görselleştirme
    plot_portfolio_performance(pf, "portfolio_performance_forecast.png")
    plot_asset_allocation(weights, "asset_allocation_forecast.png")
    
    # Sonuçları kaydet
    save_results(metrics, forecast_df, signals, weights_exec)
    
    return pf, metrics, forecast_df, signals, weights_exec


def save_results(
    metrics: Dict,
    forecast_df: pd.DataFrame,
    signals_df: pd.DataFrame,
    weights_df: pd.DataFrame,
    output_dir: str = "backtest_results"
):
    """Sonuçları CSV dosyalarına kaydet"""
    import os
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Metrikleri kaydet
    metrics_df = pd.DataFrame([metrics])
    metrics_df.to_csv(f"{output_dir}/portfolio_metrics.csv", index=False)
    
    # Tahminleri kaydet
    forecast_df.to_csv(f"{output_dir}/forecasts.csv")
    
    # Sinyalleri kaydet
    signals_df.to_csv(f"{output_dir}/signals.csv")
    
    # Ağırlıkları kaydet
    weights_df.to_csv(f"{output_dir}/weights.csv")
    
    print(f"\nResults saved to {output_dir}/")


# ==================== ÖRNEK KULLANIM ====================
def example_with_sample_data():
    """
    Örnek verilerle kullanım (kullanıcının istediği format)
    """
    print("\n" + "=" * 70)
    print("SAMPLE DATA PORTFOLIO BACKTESTING (User's Example Style)")
    print("=" * 70)
    
    # Kullanıcının örnek verileri
    # dates = pd.read_csv('commodity_features.csv', index_col='date', parse_dates=['date']).index[:10]
    # Dosya yoksa manuel tarih oluştur
    dates = pd.date_range(start='2010-01-04', periods=10, freq='B')
    
    assets = ["A", "B", "C"]
    
    price_df = pd.DataFrame({
        "A":     np.asarray([1, 2, 4,  12, 6,   1,   2, 4, 6, 3]),
        "B": 4 * np.asarray([1, 3, 12, 3,  1.5, 6,   3, 1, 4, 1]),
        "C": 2 * np.asarray([5, 1, 4,  2,  6,   1.5, 3, 9, 7, 14])
    }, index=dates)
    
    decisions = np.array([
        [1, 0, 0],  # 1000
        [1, 0, 0],  # 2000
        [1, 0, 0],  # 4000
        [-1, 1, 1], # 12000
        [-1, 1, 1], # 6000 + 2000 + 12000 = 20000
        [0, 1, 0],  # (20000*(2-1/6) + 20000*4 + 20000/4)/3 = 40555.555555555555
        [0, 0, 0],  # 20277.777778
        [0, 0, 0],  # 20277.777778
        [-1, 0, 0], # 20277.777778
        [0, 0, 0]   # 20277.777778*1.5 = 30416.666666999998
    ])
    decisions_df = pd.DataFrame(decisions, index=dates, columns=assets)
    weights = decisions_df.div(decisions_df.abs().sum(axis=1), axis=0).fillna(0)
    
    # Backtest
    pf = vbt.Portfolio.from_orders(
        close=price_df,
        size=weights,
        size_type='targetpercent', 
        init_cash=1000,
        freq="D",
        cash_sharing=True,
        call_seq='auto'
    )
    
    print("Prices:")
    print(price_df, "\n")
    
    print("Decisions (1 = allocated long, 0 = not allocated):")
    print(decisions_df, "\n")
    
    print("Portfolio Stats:")
    full_stats = pf.stats()
    ann_factor = pf.returns().vbt.returns().ann_factor
    print(f"Ann Factor:                         {ann_factor}")
    print(f"Total Return [%]:                   {full_stats['Total Return [%]']:.3f}%")
    print(f"Annualized Expected Return [%]:     {(pf.returns().mean() * ann_factor):.3f}%")
    print(f"Annualized Expected Volatility [%]: {pf.returns().std() * (ann_factor ** .5):.3f}%")
    print(f"Sharpe Ratio:                       {full_stats['Sharpe Ratio']:.3f}")
    print(f"Sharpe Ratio:                       {((pf.returns().mean() * ann_factor)/(pf.returns().std() * (ann_factor ** .5))):.3f}")
    print(f"Max Drawdown [%]:                   {full_stats['Max Drawdown [%]']:.3f}%")
    
    print('\nValues', pf.value())
    print('Returns', pf.returns())
    
    # Görselleştirme
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    
    # Portfolio Value
    pf.value().plot(ax=axes[0], color='#2E86AB', linewidth=2, marker='o')
    axes[0].set_title('Portfolio Value Over Time', fontsize=14, fontweight='bold')
    axes[0].set_ylabel('Value ($)')
    axes[0].grid(True, alpha=0.3)
    
    # Returns
    pf.returns().plot(ax=axes[1], color='#06A77D', linewidth=2, marker='s')
    axes[1].axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    axes[1].set_title('Daily Returns', fontsize=14, fontweight='bold')
    axes[1].set_ylabel('Return')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('sample_portfolio_backtest.png', dpi=300, bbox_inches='tight')
    # plt.show() # Otomatik çalıştırmada show() bloklayabilir
    
    return pf


# ==================== MAIN ====================
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "full":
            run_full_backtest_with_forecasts()
        elif sys.argv[1] == "test":
            run_full_backtest_with_forecasts(limit_test_size=5)
    else:
        # Varsayılan: Örnek veri ile çalıştır
        print("Usage:")
        print("  python potfolio_backtesting.py full    - Run full backtest with TimesFM forecasts")
        print("  python potfolio_backtesting.py test    - Run short backtest with TimesFM forecasts (5 days)")
        print("\nRunning full backtest...")
        run_full_backtest_with_forecasts()
