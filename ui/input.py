"""UI components for data input."""
import streamlit as st
import numpy as np


def render_quick_examples():
    """Render quick example buttons and return selected data or None."""
    st.markdown("#### Quick Examples")
    
    if st.button("Student Scores"):
        st.session_state.data_input = "78 82 85 90 88 92 75 80 95 87"
        st.rerun()
    
    if st.button("Salaries (k)"):
        st.session_state.data_input = "45 52 48 60 55 50 58 62 47 53"
        st.rerun()
    
    if st.button("Random Data"):
        st.session_state.data_input = " ".join(map(str, np.random.randint(1, 100, 20)))
        st.rerun()


def render_text_input() -> str:
    """Render text area for data input and return raw input."""
    raw_input = st.text_area(
        "Enter your data",
        value=st.session_state.data_input,
        placeholder="Example: 5 7 8 10 12 15 or 5, 7, 8, 10, 12, 15",
        height=100,
        help="You can use spaces or commas to separate numbers"
    )
    return raw_input


def render_csv_upload_section():
    """Render CSV upload section and return uploaded file."""
    st.markdown("#### Or Upload CSV File")
    col_upload1, col_upload2 = st.columns([2, 1])
    
    with col_upload1:
        uploaded_file = st.file_uploader("Choose a CSV file", type=['csv'])
    
    return uploaded_file


def initialize_session_state():
    """Initialize session state variables."""
    if 'data_input' not in st.session_state:
        st.session_state.data_input = ""
    if 'last_uploaded_file' not in st.session_state:
        st.session_state.last_uploaded_file = None
    if 'multivariate_data' not in st.session_state:
        st.session_state.multivariate_data = None
    if 'timeseries_data' not in st.session_state:
        st.session_state.timeseries_data = None
