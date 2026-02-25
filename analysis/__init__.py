"""Univariate analysis module."""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats


def create_histogram_polygon(data: np.ndarray, chart_style: str = "default") -> None:
    """Create histogram with frequency polygon."""
    plt.style.use(chart_style)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    bins = np.arange(np.floor(data.min()), np.ceil(data.max()) + 1, 1)
    counts, bin_edges, patches = ax.hist(data, bins=bins, edgecolor='black', 
                                         alpha=0.7, color='#667eea')
    
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    ax.plot(bin_centers, counts, marker='o', linewidth=2, 
           markersize=8, label='Frequency Polygon', color='#764ba2')
    
    quartiles = np.percentile(data, [25, 50, 75])
    ax.axvline(quartiles[0], color='green', linestyle='--', 
              linewidth=2, label=f'Q1: {quartiles[0]:.2f}', alpha=0.7)
    ax.axvline(quartiles[1], color='red', linestyle='--', 
              linewidth=2, label=f'Median: {quartiles[1]:.2f}', alpha=0.7)
    ax.axvline(quartiles[2], color='orange', linestyle='--', 
              linewidth=2, label=f'Q3: {quartiles[2]:.2f}', alpha=0.7)
    
    ax.set_title("Histogram with Frequency Polygon", fontsize=14, fontweight='bold')
    ax.set_xlabel("Value", fontsize=12)
    ax.set_ylabel("Frequency", fontsize=12)
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    
    return fig


def create_boxplot(data: np.ndarray, show_outliers: bool = True, 
                   outliers: np.ndarray = None) -> None:
    """Create box plot."""
    fig, ax = plt.subplots(figsize=(10, 6))
    bp = ax.boxplot(data, vert=False, patch_artist=True,
                   boxprops=dict(facecolor='#667eea', alpha=0.7),
                   medianprops=dict(color='red', linewidth=2),
                   whiskerprops=dict(linewidth=1.5),
                   capprops=dict(linewidth=1.5))
    
    if show_outliers and outliers is not None and len(outliers) > 0:
        for outlier in outliers:
            ax.plot(outlier, 1, 'ro', markersize=10, label='Outlier')
    
    ax.set_title("Box Plot", fontsize=14, fontweight='bold')
    ax.set_xlabel("Value", fontsize=12)
    ax.grid(True, alpha=0.3, axis='x')
    
    return fig


def create_distribution_plot(data: np.ndarray) -> None:
    """Create data distribution scatter plot with mean and std dev."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    mean = np.mean(data)
    std_dev = np.std(data, ddof=1)
    
    ax.scatter(range(len(data)), data, alpha=0.6, s=100, 
              color='#667eea', edgecolor='black', linewidth=1)
    
    ax.axhline(mean, color='red', linestyle='-', linewidth=2, 
              label=f'Mean: {mean:.2f}')
    
    ax.axhline(mean + std_dev, color='orange', linestyle='--', 
              linewidth=1.5, alpha=0.7, label=f'+1 SD: {mean+std_dev:.2f}')
    ax.axhline(mean - std_dev, color='orange', linestyle='--', 
              linewidth=1.5, alpha=0.7, label=f'-1 SD: {mean-std_dev:.2f}')
    
    ax.set_title("Data Distribution", fontsize=14, fontweight='bold')
    ax.set_xlabel("Observation Index", fontsize=12)
    ax.set_ylabel("Value", fontsize=12)
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    
    return fig
