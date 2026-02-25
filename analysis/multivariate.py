"""Multivariate analysis visualizations."""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats


def create_correlation_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate and return correlation matrix."""
    return df.corr()


def create_regression_plot(x: np.ndarray, y: np.ndarray, col_x: str, col_y: str) -> tuple:
    """Create regression plot and calculate regression statistics."""
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    ax.scatter(x, y, alpha=0.6, s=50, color='#667eea', edgecolor='black', linewidth=0.5)
    
    x_line = np.linspace(x.min(), x.max(), 100)
    y_line = slope * x_line + intercept
    ax.plot(x_line, y_line, 'r-', linewidth=2, label=f'y = {slope:.2f}x + {intercept:.2f}')
    
    ax.set_xlabel(col_x, fontsize=12)
    ax.set_ylabel(col_y, fontsize=12)
    ax.set_title(f"Regression: {col_y} vs {col_x}\nR² = {r_value**2:.4f}", 
               fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    return fig, {
        'slope': slope,
        'intercept': intercept,
        'r_value': r_value,
        'r_squared': r_value**2,
        'p_value': p_value,
        'std_err': std_err
    }


def create_pairplot(df: pd.DataFrame) -> None:
    """Create pairwise scatter plots and histograms."""
    num_vars = len(df.columns)
    fig, axes = plt.subplots(num_vars, num_vars, figsize=(12, 12))
    
    for i in range(num_vars):
        for j in range(num_vars):
            ax = axes[i, j]
            
            if i == j:
                ax.hist(df.iloc[:, i], bins=15, color='#667eea', alpha=0.7, edgecolor='black')
                ax.set_ylabel('Frequency')
            else:
                ax.scatter(df.iloc[:, j], df.iloc[:, i], 
                         alpha=0.5, s=20, color='#667eea')
            
            if i == num_vars - 1:
                ax.set_xlabel(df.columns[j], fontsize=9)
            if j == 0:
                ax.set_ylabel(df.columns[i], fontsize=9)
            
            ax.tick_params(labelsize=7)
    
    plt.tight_layout()
    return fig
