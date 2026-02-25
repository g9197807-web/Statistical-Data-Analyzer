"""UI components for sidebar configuration."""
import streamlit as st
import os


def render_sidebar():
    """Render and return sidebar configuration options."""
    with st.sidebar:
        st.header("Settings")
        chart_style = st.selectbox("Chart Style", ["default", "seaborn", "ggplot", "bmh"])
        show_outliers = st.checkbox("Highlight Outliers", value=True)
        
        st.divider()
        st.header("AI Analysis")
        use_ai = st.checkbox("Enable AI Analysis", value=True)
        
        ai_provider = None
        groq_key = None
        
        if use_ai:
            ai_provider = st.selectbox("Choose AI Model", 
                ["Groq - Llama 3.3 70B (Fast)", 
                 "Groq - Mixtral 8x7B", 
                 "Groq - Llama 3.1 8B (Speed)"])
            
            groq_key = os.getenv("GROQ_API_KEY")
            if not groq_key:
                st.warning("GROQ_API_KEY not found in environment variables")
                st.info("Add to .env file: GROQ_API_KEY=your_key_here")
            else:
                st.success("API Key loaded")
        
        st.divider()
        st.markdown("### How to use")
        st.markdown("""
        1. Enter your data (space or comma separated)
        2. Click **Analyze** button
        3. Explore statistics and visualizations
        4. Get AI insights automatically
        """)
        
        st.divider()
        st.markdown("### Setup API Key")
        st.code("""
# Create .env file:
GROQ_API_KEY=your_api_key_here

# Get free key at:
# https://console.groq.com
        """, language="bash")
    
    return {
        'chart_style': chart_style,
        'show_outliers': show_outliers,
        'use_ai': use_ai,
        'ai_provider': ai_provider,
        'groq_key': groq_key
    }


def get_ai_model_map() -> dict:
    """Get mapping of AI provider names to model identifiers."""
    return {
        "Groq - Llama 3.3 70B (Fast)": "llama-3.3-70b-versatile",
        "Groq - Mixtral 8x7B": "mixtral-8x7b-32768",
        "Groq - Llama 3.1 8B (Speed)": "llama-3.1-8b-instant"
    }
