"""Data processing and loading utilities."""
import pandas as pd
import numpy as np


def load_csv(uploaded_file) -> pd.DataFrame:
    """Load and return CSV file as DataFrame."""
    return pd.read_csv(uploaded_file)


def get_numeric_columns(df: pd.DataFrame) -> list:
    """Get list of numeric columns from DataFrame."""
    return df.select_dtypes(include=[np.number]).columns.tolist()


def process_single_column(df: pd.DataFrame, column: str) -> str:
    """Process single column for single variable analysis."""
    numeric_data = pd.to_numeric(df[column], errors='coerce').dropna()
    return " ".join(map(str, numeric_data.values))


def process_multivariate_data(df: pd.DataFrame, selected_cols: list) -> pd.DataFrame:
    """Process multiple columns for multivariate analysis."""
    return df[selected_cols].dropna()


def process_timeseries_data(df: pd.DataFrame, date_column: str = None, value_column: str = None) -> pd.DataFrame:
    """Process data for time series analysis."""
    if date_column:
        ts_df = df[[date_column, value_column]].copy()
        ts_df[date_column] = pd.to_datetime(ts_df[date_column])
        ts_df = ts_df.sort_values(date_column)
        ts_df = ts_df.dropna()
    else:
        ts_df = df[[value_column]].copy()
        ts_df = ts_df.dropna()
        ts_df['Time_Index'] = range(len(ts_df))
    
    return ts_df


def detect_date_columns(df: pd.DataFrame) -> list:
    """Detect columns that can be parsed as dates."""
    date_cols = []
    for col in df.columns:
        try:
            pd.to_datetime(df[col])
            date_cols.append(col)
        except:
            pass
    return date_cols
