"""Time series analysis visualizations."""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def create_timeseries_plot(ts_df: pd.DataFrame, time_col: str, value_col: str, 
                          is_indexed: bool = False) -> None:
    """Create time series line plot with trend."""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    values = ts_df[value_col].values
    x_vals = ts_df[time_col] if not is_indexed else ts_df[time_col].values
    
    ax.plot(x_vals, values, marker='o', linewidth=2, markersize=4, 
            color='#667eea', label='Observed Values')
    
    z = np.polyfit(range(len(values)), values, 1)
    p = np.poly1d(z)
    ax.plot(x_vals, p(range(len(values))), "r--", linewidth=2, 
            label=f'Trend Line (slope: {z[0]:.2f})', alpha=0.7)
    
    ax.axhline(np.mean(values), color='green', linestyle=':', 
              linewidth=2, label=f'Mean: {np.mean(values):.2f}', alpha=0.7)
    
    ax.set_ylabel(value_col, fontsize=12)
    ax.set_xlabel("Date" if not is_indexed else "Time Index", fontsize=12)
    ax.set_title(f"Time Series: {value_col}", fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    if not is_indexed:
        plt.xticks(rotation=45)
    
    plt.tight_layout()
    return fig


def create_moving_average_plot(ts_df: pd.DataFrame, time_col: str, value_col: str, 
                              window_size: int, is_indexed: bool = False) -> None:
    """Create moving average plot."""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    values = ts_df[value_col].values
    x_vals = ts_df[time_col] if not is_indexed else ts_df[time_col].values
    
    ma = pd.Series(values).rolling(window=window_size).mean()
    
    ax.plot(x_vals, values, marker='o', linewidth=1.5, markersize=4, 
            color='#667eea', label='Original', alpha=0.5)
    ax.plot(x_vals, ma, linewidth=3, color='#e74c3c', 
            label=f'{window_size}-period Moving Average')
    
    ax.set_ylabel(value_col, fontsize=12)
    ax.set_xlabel("Date" if not is_indexed else "Time Index", fontsize=12)
    ax.set_title(f"Moving Average Analysis", fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    if not is_indexed:
        plt.xticks(rotation=45)
    
    plt.tight_layout()
    return fig


def create_changes_plot(diff: np.ndarray, avg_change: float) -> None:
    """Create period-to-period changes bar plot."""
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.bar(range(len(diff)), diff, color=['#e74c3c' if d < 0 else '#2ecc71' for d in diff], alpha=0.7)
    ax.axhline(0, color='black', linewidth=1)
    ax.axhline(avg_change, color='blue', linestyle='--', linewidth=2, 
              label=f'Avg Change: {avg_change:.2f}')
    ax.set_xlabel("Period", fontsize=12)
    ax.set_ylabel("Change", fontsize=12)
    ax.set_title("Period-to-Period Changes", fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    return fig


def create_distribution_histogram(data: np.ndarray) -> None:
    """Create distribution histogram for time series values."""
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.hist(data, bins=min(20, len(data)//2), edgecolor='black', 
           alpha=0.7, color='#667eea')
    ax.axvline(np.mean(data), color='red', linestyle='--', 
              linewidth=2, label=f'Mean: {np.mean(data):.2f}')
    ax.axvline(np.median(data), color='green', linestyle='--', 
              linewidth=2, label=f'Median: {np.median(data):.2f}')
    ax.set_xlabel("Value", fontsize=12)
    ax.set_ylabel("Frequency", fontsize=12)
    ax.set_title("Value Distribution", fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    return fig


def create_distribution_boxplot(data: np.ndarray) -> None:
    """Create distribution box plot for time series values."""
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.boxplot(data, vert=False, patch_artist=True,
              boxprops=dict(facecolor='#667eea', alpha=0.7),
              medianprops=dict(color='red', linewidth=2))
    ax.set_xlabel("Value", fontsize=12)
    ax.set_title("Box Plot", fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')
    
    return fig
