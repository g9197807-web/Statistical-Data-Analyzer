"""Data validation functions."""
import numpy as np


def parse_input_data(raw_input: str) -> np.ndarray:
    """
    Parse user input string to numpy array.
    
    Args:
        raw_input: Input string with space or comma separated numbers
        
    Returns:
        numpy array of floats
        
    Raises:
        ValueError: If input cannot be parsed
    """
    data = [float(x.replace(',', '.')) for x in raw_input.replace(',', ' ').split()]
    return np.array(data)


def validate_data_length(data: np.ndarray, min_length: int = 2) -> bool:
    """Check if data has minimum required length."""
    return len(data) >= min_length
