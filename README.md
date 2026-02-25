# Statistical-Data-Analyzer

Advanced statistical analysis tool with beautiful visualizations, powered by Streamlit.

## Project Structure

```
Statistical-Data-Analyzer/
├── app.py                    # Main Streamlit application entry point
├── ai_service.py             # Groq API integration for AI analysis
├── requirements.txt          # Python dependencies
├── README.md
│
├── utils/                    # Utility modules for data processing
│   ├── __init__.py
│   ├── validators.py         # Input validation functions
│   ├── calculations.py       # Core statistical calculations
│   └── data_processing.py    # CSV loading and data transformations
│
├── analysis/                 # Analysis modules
│   ├── __init__.py
│   ├── univariate.py         # Single variable analysis visualizations
│   ├── multivariate.py       # Multi-variable correlation & regression analysis
│   └── timeseries.py         # Time series analysis and visualization
│
└── ui/                       # UI component modules
    ├── __init__.py
    ├── sidebar.py            # Sidebar configuration and settings
    ├── input.py              # Data input components
    └── charts.py             # Chart styling utilities
```

## Module Organization

### Core (`app.py`)

- Main Streamlit application
- Orchestrates all modules
- Handles session state and page configuration

### Utils (`utils/`)

- `validators.py`: Parse and validate user input data
- `calculations.py`: Calculate statistical measures (mean, std, quartiles, outliers, etc.)
- `data_processing.py`: Load CSV files, detect columns, process data for different analysis modes

### Analysis (`analysis/`)

- `univariate.py`: Histogram, box plots, distribution plots for single variable analysis
- `multivariate.py`: Correlation matrices, regression plots, pairwise scatter plots
- `timeseries.py`: Time series plots, moving averages, trend analysis, change analysis

### UI (`ui/`)

- `sidebar.py`: Sidebar settings, AI model selection, API configuration
- `input.py`: Data input text area, quick example buttons, CSV upload, session state initialization
- `charts.py`: Styled correlation matrix display, legend rendering

### AI Service (`ai_service.py`)

- Groq API integration for AI-powered statistical analysis
- Generates detailed analysis reports based on calculated statistics

## Features

- **Single Variable Analysis**: Descriptive statistics, distribution analysis, outlier detection
- **Multi-Variable Analysis**: Correlation matrices, linear regression (2 variables), pairwise plots
- **Time Series Analysis**: Trend detection, moving averages, volatility analysis, change analysis
- **AI-Powered Insights**: Automatic analysis generation using Groq LLM
- **Data Visualization**: Multiple chart types with customizable styling
- **CSV Support**: Upload and analyze CSV files with automatic column detection

## Setup

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Create `.env` file with API key:

```bash
GROQ_API_KEY=your_api_key_here
```

3. Run the application:

```bash
streamlit run app.py
```

## Usage

### Single Variable Analysis

1. Enter data (space or comma separated)
2. Click **Analyze**
3. View statistics, visualizations, and AI insights

### Multi-Variable Analysis

1. Upload CSV file
2. Select **Correlation & Regression** mode
3. Choose 2+ numeric columns
4. Click **Load Data** then **Analyze**

### Time Series Analysis

1. Upload CSV file
2. Select **Time Series** mode
3. Choose date and value columns
4. Click **Load Time Series** then **Analyze**
