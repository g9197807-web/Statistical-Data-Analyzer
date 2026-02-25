import os
from dotenv import load_dotenv
import streamlit as st
import numpy as np
import pandas as pd

from ui.sidebar import render_sidebar, get_ai_model_map
from ui.input import render_quick_examples, render_text_input, render_csv_upload_section, initialize_session_state
from ui.charts import style_correlation_matrix, show_correlation_legend
from utils.validators import parse_input_data, validate_data_length
from utils.calculations import calculate_all_stats
from utils.data_processing import (
    load_csv, get_numeric_columns, process_single_column,
    process_multivariate_data, process_timeseries_data, detect_date_columns
)
from analysis.univariate import create_histogram_polygon, create_boxplot, create_distribution_plot
from analysis.multivariate import create_correlation_matrix, create_regression_plot, create_pairplot
from analysis.timeseries import (
    create_timeseries_plot, create_moving_average_plot, create_changes_plot,
    create_distribution_histogram, create_distribution_boxplot
)
from ai_service import get_ai_analysis, format_stats_for_ai

load_dotenv()

st.set_page_config(
    page_title="Statistical Data Analyzer",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
    <style>
    .main-header {
        font-size: 3rem; 
        font-weight: 700; 
        color: #1f1f1f;
        margin-bottom: 1rem;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown('<h1 class="main-header">Statistical Data Analyzer</h1>', unsafe_allow_html=True)
st.markdown("### Advanced statistical analysis with beautiful visualizations")

initialize_session_state()

sidebar_config = render_sidebar()
chart_style = sidebar_config['chart_style']
show_outliers = sidebar_config['show_outliers']
use_ai = sidebar_config['use_ai']
ai_provider = sidebar_config['ai_provider']
groq_key = sidebar_config['groq_key']

col1, col2 = st.columns([3, 1])

with col2:
    render_quick_examples()

with col1:
    raw_input = render_text_input()

uploaded_file = render_csv_upload_section()

if uploaded_file is None and st.session_state.last_uploaded_file is not None:
    st.session_state.data_input = ""
    st.session_state.last_uploaded_file = None
    st.session_state.multivariate_data = None
    st.session_state.timeseries_data = None
    st.rerun()

if uploaded_file is not None:
    st.session_state.last_uploaded_file = uploaded_file.name
    
    try:
        df = load_csv(uploaded_file)
        numeric_cols = get_numeric_columns(df)
        
        if len(numeric_cols) == 0:
            st.error("No numeric columns found in CSV")
        else:
            col_prev1, col_prev2 = st.columns([2, 1])
            
            with col_prev1:
                st.markdown("**Preview:**")
                st.dataframe(df.head(5), use_container_width=True)
            
            with col_prev2:
                st.markdown("**Analysis Type:**")
                analysis_mode = st.radio(
                    "mode",
                    ["Single Variable", "Correlation & Regression", "Time Series"],
                    label_visibility="collapsed"
                )
                
                if analysis_mode == "Single Variable":
                    st.markdown("**Select Column:**")
                    selected_column = st.selectbox("Column:", numeric_cols, label_visibility="collapsed")
                    
                    if st.button("Load Data", use_container_width=True, type="primary"):
                        numeric_data = process_single_column(df, selected_column)
                        st.session_state.data_input = numeric_data
                        st.session_state.multivariate_data = None
                        st.session_state.timeseries_data = None
                        st.success(f"Loaded {len(st.session_state.data_input.split())} values!")
                        st.rerun()
                
                elif analysis_mode == "Time Series":
                    st.markdown("**Select Data:**")
                    
                    date_cols = detect_date_columns(df)
                    
                    if date_cols:
                        date_column = st.selectbox("Date column:", date_cols, label_visibility="collapsed")
                    else:
                        st.info("No date column detected. Will use row index as time.")
                        date_column = None
                    
                    value_column = st.selectbox("Value column:", numeric_cols, label_visibility="collapsed", key="ts_value")
                    
                    if st.button("Load Time Series", use_container_width=True, type="primary"):
                        ts_df = process_timeseries_data(df, date_column, value_column)
                        st.session_state.timeseries_data = ts_df
                        st.session_state.data_input = ""
                        st.session_state.multivariate_data = None
                        st.success(f"Loaded time series with {len(ts_df)} observations!")
                        st.rerun()
                
                else:
                    if len(numeric_cols) < 2:
                        st.warning("Need at least 2 numeric columns for correlation/regression")
                    else:
                        st.markdown("**Select Columns:**")
                        selected_cols = st.multiselect(
                            "columns",
                            numeric_cols,
                            default=numeric_cols[:min(2, len(numeric_cols))],
                            label_visibility="collapsed"
                        )
                        
                        if len(selected_cols) >= 2 and st.button("Load Data", use_container_width=True, type="primary"):
                            st.session_state.multivariate_data = process_multivariate_data(df, selected_cols)
                            st.session_state.data_input = ""
                            st.session_state.timeseries_data = None
                            st.success(f"Loaded {len(selected_cols)} variables!")
                            st.rerun()
            
    except Exception as e:
        st.error(f"Error reading CSV: {str(e)}")

if raw_input.strip() == "" and st.session_state.multivariate_data is None and st.session_state.timeseries_data is None:
    st.info("Please enter data above and click **Analyze** to begin")

elif st.session_state.timeseries_data is not None:
    if st.button("Analyze Time Series", type="primary", use_container_width=True):
        ts_df = st.session_state.timeseries_data
        
        if 'Time_Index' in ts_df.columns:
            time_col = 'Time_Index'
            value_col = [col for col in ts_df.columns if col != 'Time_Index'][0]
            is_indexed = True
        else:
            time_col = ts_df.columns[0]
            value_col = ts_df.columns[1]
            is_indexed = False
        
        values = ts_df[value_col].values
        
        st.markdown("---")
        st.subheader("Time Series Analysis")
        
        st.write(f"**Variable:** {value_col}")
        st.write(f"**Observations:** {len(values)}")
        st.write(f"**Time period:** {ts_df[time_col].iloc[0]} to {ts_df[time_col].iloc[-1]}")
        
        st.markdown("---")
        st.subheader("Descriptive Statistics")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Mean", f"{np.mean(values):.2f}")
            st.metric("Median", f"{np.median(values):.2f}")
        
        with col2:
            st.metric("Std Dev", f"{np.std(values, ddof=1):.2f}")
            st.metric("Min", f"{np.min(values):.2f}")
        
        with col3:
            st.metric("Max", f"{np.max(values):.2f}")
            st.metric("Range", f"{np.max(values) - np.min(values):.2f}")
        
        with col4:
            st.metric("Trend", "↗️" if values[-1] > values[0] else "↘️")
            change_pct = ((values[-1] - values[0]) / values[0]) * 100 if values[0] != 0 else 0
            st.metric("Change", f"{change_pct:+.1f}%")
        
        st.markdown("---")
        st.subheader("Time Series Visualization")
        fig = create_timeseries_plot(ts_df, time_col, value_col, is_indexed)
        st.pyplot(fig)
        
        st.markdown("---")
        st.subheader("Moving Averages")
        
        col1, col2 = st.columns(2)
        
        with col1:
            window_size = st.slider("Moving Average Window", min_value=2, max_value=min(20, len(values)//2), value=min(5, len(values)//2))
        
        fig = create_moving_average_plot(ts_df, time_col, value_col, window_size, is_indexed)
        st.pyplot(fig)
        
        if len(values) >= 10:
            st.markdown("---")
            st.subheader("Change Analysis")
            
            diff = np.diff(values)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Period-to-Period Changes:**")
                changes_df = pd.DataFrame({
                    'Period': range(1, len(diff) + 1),
                    'Change': diff,
                    'Change %': [(d/values[i])*100 if values[i] != 0 else 0 for i, d in enumerate(diff)]
                })
                st.dataframe(changes_df.tail(10), hide_index=True, use_container_width=True)
                
                avg_change = np.mean(diff)
                st.metric("Average Change", f"{avg_change:.2f}")
                st.metric("Volatility (Std)", f"{np.std(diff):.2f}")
            
            with col2:
                fig = create_changes_plot(diff, avg_change)
                st.pyplot(fig)
        
        st.markdown("---")
        st.subheader("Distribution Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig = create_distribution_histogram(values)
            st.pyplot(fig)
        
        with col2:
            fig = create_distribution_boxplot(values)
            st.pyplot(fig)
        
        st.markdown("---")
        st.subheader("Key Insights")
        
        col1, col2 = st.columns(2)
        
        with col1:
            z = np.polyfit(range(len(values)), values, 1)
            st.markdown("**Trend Analysis:**")
            if z[0] > 0:
                st.write(f"**Upward trend** - Values are increasing over time (slope: {z[0]:.3f})")
            elif z[0] < 0:
                st.write(f"**Downward trend** - Values are decreasing over time (slope: {z[0]:.3f})")
            else:
                st.write("**No clear trend** - Values are relatively stable")
            
            diff = np.diff(values)
            volatility = np.std(diff) if len(diff) > 0 else 0
            mean_val = np.mean(values)
            cv = (np.std(values) / mean_val * 100) if mean_val != 0 else 0
            
            st.markdown("**Volatility:**")
            if cv < 10:
                st.write(f"**Low volatility** - Data is stable (CV: {cv:.1f}%)")
            elif cv < 25:
                st.write(f"**Moderate volatility** - Some fluctuation (CV: {cv:.1f}%)")
            else:
                st.write(f"**High volatility** - Significant fluctuation (CV: {cv:.1f}%)")
        
        with col2:
            st.markdown("**Statistical Summary:**")
            summary_stats = pd.DataFrame({
                'Metric': ['Count', 'Mean', 'Std Dev', 'Min', 'Q1', 'Median', 'Q3', 'Max'],
                'Value': [
                    f"{len(values)}",
                    f"{np.mean(values):.2f}",
                    f"{np.std(values, ddof=1):.2f}",
                    f"{np.min(values):.2f}",
                    f"{np.percentile(values, 25):.2f}",
                    f"{np.median(values):.2f}",
                    f"{np.percentile(values, 75):.2f}",
                    f"{np.max(values):.2f}"
                ]
            })
            st.dataframe(summary_stats, hide_index=True, use_container_width=True)
        
        st.markdown("---")
        csv_ts = ts_df.to_csv(index=False)
        st.download_button(
            label="Download Time Series Data (CSV)",
            data=csv_ts,
            file_name="timeseries_data.csv",
            mime="text/csv",
            use_container_width=True
        )

elif st.session_state.multivariate_data is not None:
    if st.button("Analyze Correlation & Regression", type="primary", use_container_width=True):
        df_multi = st.session_state.multivariate_data
        
        st.markdown("---")
        st.subheader("Multi-Variable Analysis")
        
        st.write(f"**Variables:** {', '.join(df_multi.columns)}")
        st.write(f"**Observations:** {len(df_multi)}")
        
        st.markdown("---")
        st.subheader("Correlation Matrix")
        
        corr_matrix = create_correlation_matrix(df_multi)
        style_correlation_matrix(corr_matrix)
        show_correlation_legend()
        
        if len(df_multi.columns) == 2:
            st.markdown("---")
            st.subheader("Linear Regression Analysis")
            
            col_x = df_multi.columns[0]
            col_y = df_multi.columns[1]
            
            x = df_multi[col_x].values
            y = df_multi[col_y].values
            
            fig, reg_stats_dict = create_regression_plot(x, y, col_x, col_y)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Regression Statistics:**")
                reg_stats = pd.DataFrame({
                    'Metric': ['Slope (β₁)', 'Intercept (β₀)', 'R-squared (R²)', 'Correlation (r)', 'P-value', 'Std Error'],
                    'Value': [f'{reg_stats_dict["slope"]:.4f}', f'{reg_stats_dict["intercept"]:.4f}', 
                             f'{reg_stats_dict["r_squared"]:.4f}', f'{reg_stats_dict["r_value"]:.4f}', 
                             f'{reg_stats_dict["p_value"]:.4e}', f'{reg_stats_dict["std_err"]:.4f}']
                })
                st.dataframe(reg_stats, hide_index=True, use_container_width=True)
                
                st.markdown("**Regression Equation:**")
                st.latex(f"y = {reg_stats_dict['slope']:.4f}x + {reg_stats_dict['intercept']:.4f}")
                
                st.markdown("**Interpretation:**")
                if abs(reg_stats_dict['r_value']) > 0.7:
                    st.write("Strong correlation")
                elif abs(reg_stats_dict['r_value']) > 0.4:
                    st.write("Moderate correlation")
                else:
                    st.write("Weak correlation")
                
                if reg_stats_dict['p_value'] < 0.05:
                    st.write("Statistically significant (p < 0.05)")
                else:
                    st.write("Not statistically significant (p ≥ 0.05)")
            
            with col2:
                st.pyplot(fig)
        
        elif len(df_multi.columns) > 2:
            st.markdown("---")
            st.subheader("Pairwise Scatter Plots")
            
            fig = create_pairplot(df_multi)
            st.pyplot(fig)
            
            st.markdown("**Pairwise Correlations:**")
            pairs = []
            for i in range(len(df_multi.columns)):
                for j in range(i+1, len(df_multi.columns)):
                    corr_val = corr_matrix.iloc[i, j]
                    pairs.append({
                        'Variable 1': df_multi.columns[i],
                        'Variable 2': df_multi.columns[j],
                        'Correlation': f'{corr_val:.4f}',
                        'Strength': 'Strong' if abs(corr_val) > 0.7 else 'Moderate' if abs(corr_val) > 0.4 else 'Weak'
                    })
            
            pairs_df = pd.DataFrame(pairs)
            st.dataframe(pairs_df, hide_index=True, use_container_width=True)
        
        st.markdown("---")
        csv_corr = corr_matrix.to_csv()
        st.download_button(
            label="Download Correlation Matrix (CSV)",
            data=csv_corr,
            file_name="correlation_matrix.csv",
            mime="text/csv",
            use_container_width=True
        )

else:
    if st.button("Analyze", type="primary", use_container_width=True):
        try:
            data = parse_input_data(raw_input)
            
            if not validate_data_length(data):
                st.error("Please enter at least 2 numbers")
            else:
                all_stats = calculate_all_stats(data)
                
                st.markdown("---")
                st.subheader("Key Metrics")
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Mean (Average)", f"{all_stats['mean']:.2f}")
                    st.metric("Median", f"{all_stats['q2']:.2f}")
                
                with col2:
                    st.metric("Mode", f"{all_stats['mode']:.2f}")
                    st.metric("Range", f"{all_stats['range']:.2f}")
                
                with col3:
                    st.metric("Std Deviation", f"{all_stats['std_dev']:.2f}")
                    st.metric("CV (%)", f"{all_stats['cv']:.2f}%")
                
                with col4:
                    st.metric("Min Value", f"{all_stats['min']:.2f}")
                    st.metric("Max Value", f"{all_stats['max']:.2f}")
                
                st.markdown("---")
                st.subheader("Detailed Statistics")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**Distribution Measures**")
                    stats_df = pd.DataFrame({
                        'Statistic': ['Skewness', 'Kurtosis', 'Q1 (25%)', 'Q2 (50%)', 'Q3 (75%)', 'IQR'],
                        'Value': [f"{all_stats['skewness']:.3f}", f"{all_stats['kurtosis']:.3f}", 
                                 f"{all_stats['q1']:.2f}", f"{all_stats['q2']:.2f}", 
                                 f"{all_stats['q3']:.2f}", f"{all_stats['iqr']:.2f}"]
                    })
                    st.dataframe(stats_df, hide_index=True, use_container_width=True)
                
                with col2:
                    if show_outliers and all_stats['outlier_count'] > 0:
                        st.markdown("**Outliers Detected**")
                        st.write(f"Found {all_stats['outlier_count']} outlier(s): {', '.join(map(lambda x: f'{x:.2f}', all_stats['outliers']))}")
                
                st.markdown("---")
                st.subheader("Visualizations")
                
                tab1, tab2, tab3 = st.tabs(["Histogram & Polygon", "Box Plot", "Distribution"])
                
                with tab1:
                    fig = create_histogram_polygon(data, chart_style)
                    st.pyplot(fig)
                
                with tab2:
                    fig = create_boxplot(data, show_outliers, all_stats['outliers'])
                    st.pyplot(fig)
                
                with tab3:
                    fig = create_distribution_plot(data)
                    st.pyplot(fig)
                
                st.markdown("---")
                st.subheader("Data Summary")
                
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    st.write(f"**Total observations:** {len(data)}")
                    st.write(f"**Data range:** [{all_stats['min']:.2f}, {all_stats['max']:.2f}]")
                    
                    st.markdown("**Interpretation:**")
                    if abs(all_stats['skewness']) < 0.5:
                        st.write("Distribution is approximately symmetric")
                    elif all_stats['skewness'] > 0.5:
                        st.write("Distribution is positively skewed (right tail)")
                    else:
                        st.write("Distribution is negatively skewed (left tail)")
                    
                    if all_stats['cv'] < 15:
                        st.write("Low variability in data")
                    elif all_stats['cv'] < 30:
                        st.write("Moderate variability in data")
                    else:
                        st.write("High variability in data")
                
                with col2:
                    csv = pd.DataFrame({'Value': data}).to_csv(index=False)
                    st.download_button(
                        label="Download Data (CSV)",
                        data=csv,
                        file_name="statistical_data.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
                
                if use_ai and groq_key:
                    st.markdown("---")
                    st.subheader("AI-Powered Analysis")
                    
                    model_map = get_ai_model_map()
                    selected_model = model_map[ai_provider]
                    
                    with st.spinner(f"{ai_provider.split(' - ')[1]} is analyzing your data..."):
                        stats_summary = format_stats_for_ai(all_stats, len(data))
                        
                        try:
                            ai_text = get_ai_analysis(stats_summary, groq_key, selected_model)
                            st.success("AI Analysis Complete!")
                            st.markdown(ai_text)
                            
                            st.markdown("")
                            col1, col2, col3 = st.columns([1, 1, 4])
                            with col1:
                                st.button("Helpful", key="helpful")
                            with col2:
                                st.button("Not helpful", key="not_helpful")
                        
                        except Exception as e:
                            st.error(f"AI Analysis Error: {str(e)}")
                            st.info("Check your .env file and API key validity")
                
                elif use_ai and not groq_key:
                    st.info("Please set GROQ_API_KEY in your .env file to enable AI analysis")
                
        except Exception as e:
            st.error(f"Error: {str(e)}")
            st.info("Please check your input format and try again")
