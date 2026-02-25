"""Chart styling utilities."""
import streamlit as st
import pandas as pd


def style_correlation_matrix(corr_matrix: pd.DataFrame) -> None:
    """Display styled correlation matrix."""
    def style_correlation(val):
        """Color coding for correlation values"""
        if abs(val) == 1.0:
            return 'background-color: #d4d4d4; font-weight: bold; color: #000000'
        elif abs(val) >= 0.7:
            color = '#e74c3c' if val > 0 else '#3498db'
            return f'background-color: {color}; color: white; font-weight: bold'
        elif abs(val) >= 0.4:
            color = '#f39c12' if val > 0 else '#5dade2'
            return f'background-color: {color}; color: white'
        else:
            color = '#fef5e7' if val > 0 else '#ebf5fb'
            return f'background-color: {color}; color: #2c3e50'
    
    styled_corr = corr_matrix.style.applymap(style_correlation)\
                                   .format("{:.3f}")\
                                   .set_properties(**{
                                       'text-align': 'center',
                                       'font-size': '14px',
                                       'border': '1px solid #ddd',
                                       'padding': '8px'
                                   })\
                                   .set_table_styles([
                                       {'selector': 'th', 'props': [
                                           ('background-color', '#667eea'),
                                           ('color', 'white'),
                                           ('font-weight', 'bold'),
                                           ('text-align', 'center'),
                                           ('padding', '10px'),
                                           ('border', '1px solid #ddd')
                                       ]},
                                       {'selector': 'tr:hover', 'props': [
                                           ('background-color', '#f5f5f5')
                                       ]}
                                   ])
    
    st.dataframe(styled_corr, use_container_width=True)


def show_correlation_legend():
    """Display correlation strength guide."""
    st.markdown("""
    <div style='background-color: #f8f9fa; padding: 15px; border-radius: 8px; margin-top: 10px;'>
        <b>📖 Correlation Strength Guide:</b><br>
        <span style='background-color: #e74c3c; color: white; padding: 3px 8px; border-radius: 3px; margin-right: 10px;'>|r| ≥ 0.7</span> Strong correlation<br>
        <span style='background-color: #f39c12; color: white; padding: 3px 8px; border-radius: 3px; margin-right: 10px;'>0.4 ≤ |r| < 0.7</span> Moderate correlation<br>
        <span style='background-color: #fef5e7; color: #2c3e50; padding: 3px 8px; border-radius: 3px; margin-right: 10px;'>|r| < 0.4</span> Weak correlation
    </div>
    """, unsafe_allow_html=True)
