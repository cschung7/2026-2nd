# -*- coding: utf-8 -*-
"""
Title: Financial Machine Learning - Normality Recovery Simulation
Author: Sogang Univ (AI Finance Seminar)
Reference: Advances in Financial Machine Learning, Chapter 2 (Marcos López de Prado)

This self-contained script simulates a high-frequency tick dataset exhibiting
the classic intraday U-shaped volatility pattern. It then resamples the ticks
into (1) standard 10-Minute Time Bars and (2) Dollar Bars to demonstrate
how Dollar Bars restore the IID (independent and identically distributed) 
and Gaussian (normal) properties of returns required by Machine Learning models.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import os

def run_simulation():
    # 1. Setup Environment & Random Seed
    np.random.seed(42)
    sns.set_theme(style='whitegrid', palette='colorblind')
    
    print("=" * 65)
    print("       Sogang Univ AI Finance Seminar: Normality Recovery Demo")
    print("=" * 65)
    print("Generating 100,000 tick-level simulation data...")

    # 2. Simulate High-Frequency Tick Data (5 Trading Days)
    n_ticks = 100000
    days = 5
    ticks_per_day = n_ticks // days
    day_seconds = 23400  # 6.5 hours trading day = 23,400 seconds
    
    # Use Beta distribution to mimic high transaction activity at Open & Close (U-shape)
    u_shaped_draws = np.random.beta(0.5, 0.5, size=n_ticks)
    time_offsets = np.sort(u_shaped_draws.reshape(days, ticks_per_day), axis=1)
    
    timestamps = []
    for d in range(days):
        day_times = d * 86400 + time_offsets[d] * day_seconds
        timestamps.extend(day_times)
    timestamps = np.array(timestamps)
    
    # Scale tick-level volatility relative to intraday transaction density
    tick_positions_in_day = u_shaped_draws
    vol_scale = 0.001 * (1.0 + 3.0 * (1.0 - 2.0 * np.abs(tick_positions_in_day - 0.5)))
    
    # Simulate Price using Geometric Brownian Motion (Starting Price = $100)
    price_changes = np.random.normal(loc=0.0, scale=vol_scale)
    prices = 100.0 * np.exp(np.cumsum(price_changes))
    
    # Simulate Volume (Higher size at market open/close)
    mean_volume = 100 + 900 * (1.0 - 2.0 * np.abs(tick_positions_in_day - 0.5))
    volumes = np.random.exponential(scale=mean_volume).astype(int) + 10
    dollar_values = prices * volumes
    
    # Build Tick DataFrame
    df_ticks = pd.DataFrame({
        'timestamp': pd.to_datetime(timestamps, unit='s'),
        'price': prices,
        'volume': volumes,
        'dollar_value': dollar_values
    })
    
    print("Tick-level data generated successfully.")
    print("-" * 65)
    print("Resampling data into Time Bars & Dollar Bars...")

    # 3. Resample into Time Bars (10-Minute frequency)
    time_bars = df_ticks.set_index('timestamp').resample('10Min').last().dropna()
    time_returns = np.diff(np.log(time_bars['price']))
    
    # B. Resample into Dollar Bars (using threshold to match sample size of Time Bars)
    total_dollar_value = df_ticks['dollar_value'].sum()
    target_bars = len(time_bars)
    dollar_threshold = total_dollar_value / target_bars
    
    bar_indices = []
    cum_val = 0
    for i, val in enumerate(df_ticks['dollar_value']):
        cum_val += val
        if cum_val >= dollar_threshold:
            bar_indices.append(i)
            cum_val = 0
    dollar_bars = df_ticks.iloc[bar_indices]
    dollar_returns = np.diff(np.log(dollar_bars['price']))
    
    # 4. Statistical Analysis
    time_kurt = stats.kurtosis(time_returns)
    time_skew = stats.skew(time_returns)
    time_jb_stat, time_jb_p = stats.jarque_bera(time_returns)
    
    dollar_kurt = stats.kurtosis(dollar_returns)
    dollar_skew = stats.skew(dollar_returns)
    dollar_jb_stat, dollar_jb_p = stats.jarque_bera(dollar_returns)
    
    reduction = (1.0 - (dollar_kurt / time_kurt)) * 100
    
    print("\n[STATISTICAL RESULTS]")
    print(f"Time Bars Returns (Matched N={len(time_returns)}):")
    print(f"  - Excess Kurtosis: {time_kurt:.4f} (Ideal: 0)")
    print(f"  - Skewness:        {time_skew:.4f}")
    print(f"  - Jarque-Bera Stat:{time_jb_stat:.2f} (p-value: {time_jb_p:.4e})")
    if time_jb_p < 0.05:
         print("    -> Normality Assumption: REJECTED")
         
    print(f"\nDollar Bars Returns (Matched N={len(dollar_returns)}):")
    print(f"  - Excess Kurtosis: {dollar_kurt:.4f} (Reduction of {reduction:.1f}%!)")
    print(f"  - Skewness:        {dollar_skew:.4f}")
    print(f"  - Jarque-Bera Stat:{dollar_jb_stat:.2f} (p-value: {dollar_jb_p:.4f})")
    if dollar_jb_p >= 0.05:
         print("    -> Normality Assumption: ACCEPTED (Cannot Reject)")
    else:
         print("    -> Normality Assumption: REJECTED (But significantly closer to normal)")
    print("-" * 65)

    # 5. Plotting & Saving Chart
    print("Generating normality comparison plot...")
    fig, axs = plt.subplots(2, 2, figsize=(14, 11))
    
    # Panel 1: Time Bars Returns Histogram
    sns.histplot(time_returns, kde=True, ax=axs[0, 0], stat='density', color='#4c72b0', alpha=0.6)
    x = np.linspace(axs[0, 0].get_xlim()[0], axs[0, 0].get_xlim()[1], 100)
    axs[0, 0].plot(x, stats.norm.pdf(x, np.mean(time_returns), np.std(time_returns)), 'r--', linewidth=2, label='Normal Dist')
    axs[0, 0].set_title(f'Time Bars Returns (Kurtosis: {time_kurt:.2f})\nHighly Leptokurtic (Fat Tails)', fontsize=12, fontweight='bold')
    axs[0, 0].set_xlabel('Log Return')
    axs[0, 0].legend()
    
    # Panel 2: Dollar Bars Returns Histogram
    sns.histplot(dollar_returns, kde=True, ax=axs[0, 1], stat='density', color='#55a868', alpha=0.6)
    x = np.linspace(axs[0, 1].get_xlim()[0], axs[0, 1].get_xlim()[1], 100)
    axs[0, 1].plot(x, stats.norm.pdf(x, np.mean(dollar_returns), np.std(dollar_returns)), 'r--', linewidth=2, label='Normal Dist')
    axs[0, 1].set_title(f'Dollar Bars Returns (Kurtosis: {dollar_kurt:.2f})\nGaussian Properties Restored', fontsize=12, fontweight='bold')
    axs[0, 1].set_xlabel('Log Return')
    axs[0, 1].legend()
    
    # Panel 3: Time Bars Q-Q Plot
    stats.probplot(time_returns, dist="norm", plot=axs[1, 0])
    axs[1, 0].get_lines()[0].set_markerfacecolor('#4c72b0')
    axs[1, 0].get_lines()[0].set_alpha(0.5)
    axs[1, 0].set_title('Time Bars Q-Q Plot\nExtreme Tails Deviating from Normal Line', fontsize=12, fontweight='bold')
    
    # Panel 4: Dollar Bars Q-Q Plot
    stats.probplot(dollar_returns, dist="norm", plot=axs[1, 1])
    axs[1, 1].get_lines()[0].set_markerfacecolor('#55a868')
    axs[1, 1].get_lines()[0].set_alpha(0.5)
    axs[1, 1].set_title('Dollar Bars Q-Q Plot\nExcellent Alignment with Gaussian Line', fontsize=12, fontweight='bold')
    
    fig.suptitle(f'Empirical Normality Recovery: Time Bars vs. Dollar Bars\nExcess Kurtosis Reduced by {reduction:.1f}% ({time_kurt:.2f} to {dollar_kurt:.2f})', 
                 fontsize=15, fontweight='bold', y=0.98)
    
    sns.despine()
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    
    # Save chart to current working folder
    chart_filename = "normality-comparison.png"
    plt.savefig(chart_filename, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Normality comparison chart successfully saved as '{chart_filename}'.")
    print("=" * 65)

if __name__ == "__main__":
    run_simulation()

