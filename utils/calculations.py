"""Core statistical calculations."""
import numpy as np
from scipy import stats


def calculate_basic_stats(data: np.ndarray) -> dict:
    """Calculate basic statistical measures."""
    mean = np.mean(data)
    median = np.median(data)
    mode_result = stats.mode(data, keepdims=True)
    mode = mode_result.mode.item()
    std_dev = np.std(data, ddof=1)
    cv = std_dev / mean * 100 if mean != 0 else 0
    skewness = stats.skew(data)
    kurtosis = stats.kurtosis(data)
    
    return {
        'mean': mean,
        'median': median,
        'mode': mode,
        'std_dev': std_dev,
        'cv': cv,
        'skewness': skewness,
        'kurtosis': kurtosis
    }


def calculate_quartiles(data: np.ndarray) -> dict:
    """Calculate quartiles and IQR."""
    quartiles = np.percentile(data, [25, 50, 75])
    iqr = quartiles[2] - quartiles[0]
    
    return {
        'q1': quartiles[0],
        'q2': quartiles[1],
        'q3': quartiles[2],
        'iqr': iqr
    }


def calculate_range_stats(data: np.ndarray) -> dict:
    """Calculate min, max and range."""
    min_val = np.min(data)
    max_val = np.max(data)
    range_val = max_val - min_val
    
    return {
        'min': min_val,
        'max': max_val,
        'range': range_val
    }


def detect_outliers(data: np.ndarray, iqr: float, q1: float, q3: float) -> np.ndarray:
    """Detect outliers using IQR method."""
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    outliers = data[(data < lower_bound) | (data > upper_bound)]
    return outliers


def calculate_all_stats(data: np.ndarray) -> dict:
    """Calculate all statistics in one call."""
    basic = calculate_basic_stats(data)
    quartiles = calculate_quartiles(data)
    ranges = calculate_range_stats(data)
    outliers = detect_outliers(data, quartiles['iqr'], quartiles['q1'], quartiles['q3'])
    
    return {
        **basic,
        **quartiles,
        **ranges,
        'outliers': outliers,
        'outlier_count': len(outliers)
    }
