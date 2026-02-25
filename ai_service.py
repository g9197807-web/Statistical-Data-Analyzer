"""AI service for Groq API integration."""
import requests


def get_ai_analysis(stats_summary: str, api_key: str, model: str) -> str:
    """
    Get AI analysis from Groq API.
    
    Args:
        stats_summary: Statistics summary text to analyze
        api_key: Groq API key
        model: Model identifier (e.g., 'llama-3.3-70b-versatile')
        
    Returns:
        AI-generated analysis text
        
    Raises:
        requests.exceptions.RequestException: If API request fails
    """
    response = requests.post(
        "https://api.groq.com/openai/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        },
        json={
            "model": model,
            "messages": [
                {
                    "role": "system",
                    "content": "You are an expert statistician and data analyst. Provide detailed, report-ready analysis in English. Structure your response with clear sections: 1) Distribution Characteristics, 2) Skewness & Kurtosis Analysis, 3) Data Quality Assessment, 4) Key Insights, 5) Recommendations. Use professional language suitable for academic or business reports. Reference specific statistical values in your interpretation."
                },
                {
                    "role": "user",
                    "content": stats_summary
                }
            ],
            "temperature": 0.7,
            "max_tokens": 2500
        },
        timeout=30
    )
    
    if response.status_code == 200:
        return response.json()["choices"][0]["message"]["content"]
    else:
        raise requests.exceptions.RequestException(
            f"API Error: {response.status_code} - {response.text}"
        )


def format_stats_for_ai(stats_dict: dict, data_length: int) -> str:
    """
    Format statistics dictionary for AI analysis.
    
    Args:
        stats_dict: Dictionary with calculated statistics
        data_length: Number of observations
        
    Returns:
        Formatted statistics string for AI prompt
    """
    outliers_text = ""
    if stats_dict['outlier_count'] > 0:
        outlier_vals = stats_dict['outliers'][:5]
        outliers_text = ', '.join([f'{x:.2f}' for x in outlier_vals])
        if stats_dict['outlier_count'] > 5:
            outliers_text += '...'
        outliers_text = f"({outliers_text})"
    else:
        outliers_text = "(none)"
    
    return f"""
I have a dataset and already calculated descriptive statistics:

DATASET STATISTICS:
- Sample size: {data_length} observations
- Mode: {stats_dict['mode']:.2f}
- Median: {stats_dict['q2']:.2f}
- Mean: {stats_dict['mean']:.2f}
- Quartiles: Q1={stats_dict['q1']:.2f}, Q2={stats_dict['q2']:.2f}, Q3={stats_dict['q3']:.2f}
- Interquartile Range (IQR): {stats_dict['iqr']:.2f}
- Range: [{stats_dict['min']:.2f}, {stats_dict['max']:.2f}]
- Standard Deviation: {stats_dict['std_dev']:.2f}
- Coefficient of Variation: {stats_dict['cv']:.2f}%
- Skewness: {stats_dict['skewness']:.3f}
- Kurtosis: {stats_dict['kurtosis']:.3f}
- Outliers detected: {stats_dict['outlier_count']} {outliers_text}

Please create a detailed **data analysis** based on these statistics. Include:

1. **Distribution Characteristics**: Analyze the center (mean, median, mode), spread (standard deviation, range), and shape of the distribution

2. **Skewness & Kurtosis Interpretation**: Explain what the skewness value tells us about the distribution's asymmetry and what the kurtosis indicates about the tails (heavy/light)

3. **Data Quality & Outliers**: Comment on the presence of outliers and what they might indicate. Assess data reliability based on CV

4. **Key Insights**: What meaningful patterns or conclusions can be drawn from this data? What does the relationship between mean, median, and mode tell us?

5. **Practical Recommendations**: Based on this analysis, what actions or further investigations would you recommend?

Format the response as a **clear, report-ready analysis** with proper sections and bullet points where appropriate. Be specific and reference the actual statistical values in your interpretation.
"""
