import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import time
import os
import io
import json
import base64
import uuid
import requests
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import warnings
import traceback
from typing import Dict, List, Tuple, Optional, Union, Any

# Suppress warnings
warnings.filterwarnings('ignore')

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Product Profitability Analysis",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- THEME COLORS - Professional palette ---
PRIMARY_COLOR = "#0096C7"         # Blue - primary actions, headers
SECONDARY_COLOR = "#48CAE4"       # Light blue - secondary elements
TERTIARY_COLOR = "#90E0EF"        # Lighter blue - tertiary elements
BACKGROUND_COLOR = "#F8F9FA"      # Light grey - backgrounds
CARD_BACKGROUND = "#FFFFFF"       # White - card backgrounds
TEXT_PRIMARY = "#212529"          # Dark grey - primary text
TEXT_SECONDARY = "#6C757D"        # Medium grey - secondary text
TEXT_MUTED = "#ADB5BD"            # Light grey - muted text
SUCCESS_COLOR = "#40916C"         # Green - success metrics
WARNING_COLOR = "#E9C46A"         # Amber - warning metrics
DANGER_COLOR = "#E76F51"          # Red - danger/error metrics
BORDER_COLOR = "#DEE2E6"          # Light grey - borders

# --- CUSTOM STYLING ---
st.markdown(f"""
<style>
    /* Main theme colors and base styles */
    :root {{
        --primary: {PRIMARY_COLOR};
        --secondary: {SECONDARY_COLOR};
        --tertiary: {TERTIARY_COLOR};
        --background: {BACKGROUND_COLOR};
        --card-bg: {CARD_BACKGROUND};
        --text-primary: {TEXT_PRIMARY};
        --text-secondary: {TEXT_SECONDARY};
        --text-muted: {TEXT_MUTED};
        --success: {SUCCESS_COLOR};
        --warning: {WARNING_COLOR};
        --danger: {DANGER_COLOR};
        --border: {BORDER_COLOR};
    }}
    
    /* Global styles */
    .stApp {{
        background-color: var(--background);
    }}
    
    /* Typography */
    h1, h2, h3, h4, h5, h6 {{
        color: var(--text-primary);
        font-weight: 600;
    }}
    
    p, li, label, .stMarkdown {{
        color: var(--text-primary);
    }}
    
    a {{
        color: var(--primary);
        text-decoration: none;
    }}
    
    a:hover {{
        text-decoration: underline;
    }}
    
    /* Buttons */
    .stButton>button {{
        background-color: var(--primary);
        color: white;
        border: none;
        border-radius: 4px;
        padding: 0.5rem 1rem;
        font-weight: 500;
        transition: all 0.2s ease;
    }}
    
    .stButton>button:hover {{
        background-color: #007BA7;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        transform: translateY(-1px);
    }}
    
    .secondary-button > button {{
        background-color: #F8F9FA;
        color: var(--text-primary);
        border: 1px solid var(--border);
    }}
    
    .secondary-button > button:hover {{
        background-color: #E9ECEF;
    }}
    
    /* Form elements */
    .stCheckbox label p {{
        color: var(--text-primary);
    }}
    
    .stTextInput>div>div>input, 
    .stNumberInput>div>div>input,
    .stSelectbox>div>div>div,
    .stMultiselect>div>div>div {{
        border-radius: 4px;
        border-color: var(--border);
    }}
    
    .stTextInput>div>div>input:focus, 
    .stNumberInput>div>div>input:focus {{
        border-color: var(--primary);
        box-shadow: 0 0 0 1px var(--tertiary);
    }}
    
    /* Headers */
    .main-header {{
        font-size: 1.8rem;
        font-weight: 600;
        color: var(--text-primary);
        margin-bottom: 1rem;
        border-bottom: 2px solid var(--primary);
        padding-bottom: 0.5rem;
    }}
    
    .sub-header {{
        font-size: 1.4rem;
        font-weight: 600;
        color: var(--text-primary);
        margin-bottom: 0.8rem;
        border-left: 3px solid var(--primary);
        padding-left: 0.5rem;
    }}
    
    .section-header {{
        font-size: 1.2rem;
        font-weight: 600;
        color: var(--text-primary);
        margin: 1rem 0 0.5rem 0;
    }}
    
    /* Cards */
    .card {{
        background-color: var(--card-bg);
        border-radius: 8px;
        padding: 1.5rem;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.05);
        margin-bottom: 1.5rem;
        border-top: 3px solid var(--primary);
    }}
    
    .metric-card {{
        background-color: var(--card-bg);
        border-radius: 8px;
        padding: 1.25rem;
        box-shadow: 0 2px 6px rgba(0, 0, 0, 0.04);
        margin-bottom: 1rem;
        border-left: 3px solid var(--primary);
        transition: transform 0.2s, box-shadow 0.2s;
    }}
    
    .metric-card:hover {{
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.08);
    }}
    
    /* Metrics */
    .metric-label {{
        font-size: 0.9rem;
        font-weight: 500;
        color: var(--text-secondary);
        margin-bottom: 0.25rem;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }}
    
    .metric-value {{
        font-size: 1.5rem;
        font-weight: 600;
        color: var(--text-primary);
    }}
    
    .metric-comparison {{
        font-size: 0.85rem;
        font-weight: 500;
        margin-top: 0.25rem;
    }}
    
    .metric-positive {{
        color: var(--success);
    }}
    
    .metric-negative {{
        color: var(--danger);
    }}
    
    /* Recommendations */
    .recommendation-high {{
        background-color: rgba(64, 145, 108, 0.1);
        color: var(--success);
        padding: 0.75rem;
        border-radius: 4px;
        font-weight: 600;
        margin-top: 0.5rem;
        border-left: 4px solid var(--success);
    }}
    
    .recommendation-medium {{
        background-color: rgba(233, 196, 106, 0.1);
        color: #A27B11;
        padding: 0.75rem;
        border-radius: 4px;
        font-weight: 600;
        margin-top: 0.5rem;
        border-left: 4px solid var(--warning);
    }}
    
    .recommendation-low {{
        background-color: rgba(231, 111, 81, 0.1);
        color: #B23C18;
        padding: 0.75rem;
        border-radius: 4px;
        font-weight: 600;
        margin-top: 0.5rem;
        border-left: 4px solid var(--danger);
    }}
    
    /* Mode Toggle */
    .mode-toggle {{
        background-color: var(--card-bg);
        border-radius: 8px;
        padding: 0.75rem;
        box-shadow: 0 1px 4px rgba(0, 0, 0, 0.04);
        margin-bottom: 1rem;
        display: flex;
        justify-content: center;
        border: 1px solid var(--border);
    }}
    
    .basic-mode-btn,
    .advanced-mode-btn {{
        padding: 0.5rem 1rem;
        border-radius: 4px;
        cursor: pointer;
        font-weight: 500;
        text-align: center;
        transition: all 0.2s;
        flex: 1;
    }}
    
    .mode-active {{
        background-color: var(--primary);
        color: white;
    }}
    
    .mode-inactive {{
        background-color: var(--card-bg);
        color: var(--text-secondary);
    }}
    
    /* Form sections */
    .form-section {{
        background-color: var(--card-bg);
        border-radius: 8px;
        padding: 1.5rem;
        margin-bottom: 1.5rem;
        border: 1px solid var(--border);
    }}
    
    .form-section-header {{
        font-size: 1.1rem;
        font-weight: 600;
        color: var(--text-primary);
        margin-bottom: 1rem;
        padding-bottom: 0.5rem;
        border-bottom: 1px solid var(--border);
    }}
    
    /* Sidebar */
    section[data-testid="stSidebar"] {{
        background-color: #F8FAFC;
    }}
    
    section[data-testid="stSidebar"] > div {{
        padding-top: 2rem;
        padding-left: 1rem;
        padding-right: 1rem;
    }}
    
    section[data-testid="stSidebar"] .sidebar-title {{
        font-size: 1.25rem;
        font-weight: 600;
        color: var(--text-primary);
        margin-bottom: 1.5rem;
        text-align: center;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid var(--primary);
    }}
    
    /* Charts and Visualizations */
    .chart-container {{
        background-color: var(--card-bg);
        border-radius: 8px;
        padding: 1.25rem;
        box-shadow: 0 2px 6px rgba(0, 0, 0, 0.05);
        margin-bottom: 1.5rem;
    }}
    
    /* Tabs styling */
    .stTabs [data-baseweb="tab-list"] {{
        gap: 2px;
    }}

    .stTabs [data-baseweb="tab"] {{
        height: 50px;
        white-space: pre-wrap;
        background-color: #f8f9fa;
        border-radius: 4px 4px 0 0;
        gap: 1px;
        padding-top: 10px;
        padding-bottom: 10px;
    }}

    .stTabs [aria-selected="true"] {{
        background-color: {PRIMARY_COLOR};
        color: white;
    }}
    
    /* Field requirements */
    .required-field::after {{
        content: " *";
        color: var(--danger);
        font-weight: bold;
    }}
    
    /* Help tooltips */
    .tooltip {{
        position: relative;
        display: inline-block;
        margin-left: 5px;
        cursor: help;
    }}
    
    .tooltip .tooltiptext {{
        visibility: hidden;
        width: 200px;
        background-color: var(--text-primary);
        color: var(--card-bg);
        text-align: center;
        border-radius: 4px;
        padding: 5px;
        position: absolute;
        z-index: 1;
        bottom: 125%;
        left: 50%;
        margin-left: -100px;
        opacity: 0;
        transition: opacity 0.3s;
        font-size: 0.8rem;
    }}
    
    .tooltip:hover .tooltiptext {{
        visibility: visible;
        opacity: 1;
    }}
    
    /* Status indicators */
    .status-indicator {{
        display: inline-block;
        width: 10px;
        height: 10px;
        border-radius: 50%;
        margin-right: 5px;
    }}
    
    .status-success {{
        background-color: var(--success);
    }}
    
    .status-warning {{
        background-color: var(--warning);
    }}
    
    .status-danger {{
        background-color: var(--danger);
    }}
    
    /* Alerts and notifications */
    .alert {{
        padding: 0.75rem 1rem;
        border-radius: 4px;
        margin-bottom: 1rem;
    }}
    
    .alert-info {{
        background-color: rgba(0, 150, 199, 0.1);
        border-left: 4px solid var(--primary);
    }}
    
    .alert-success {{
        background-color: rgba(64, 145, 108, 0.1);
        border-left: 4px solid var(--success);
    }}
    
    .alert-warning {{
        background-color: rgba(233, 196, 106, 0.1);
        border-left: 4px solid var(--warning);
    }}
    
    .alert-danger {{
        background-color: rgba(231, 111, 81, 0.1);
        border-left: 4px solid var(--danger);
    }}
    
    /* File upload */
    [data-testid="stFileUploader"] {{
        background-color: var(--card-bg);
        padding: 1rem;
        border-radius: 8px;
        border: 1px dashed var(--border);
    }}
    
    [data-testid="stFileUploader"] div:first-child {{
        background-color: var(--background);
    }}
    
    /* Results badge */
    .results-badge {{
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 16px;
        font-size: 0.85rem;
        font-weight: 600;
        margin-right: 0.5rem;
    }}
    
    .results-badge-success {{
        background-color: rgba(64, 145, 108, 0.1);
        color: var(--success);
    }}
    
    .results-badge-warning {{
        background-color: rgba(233, 196, 106, 0.1);
        color: #A27B11;
    }}
    
    .results-badge-danger {{
        background-color: rgba(231, 111, 81, 0.1);
        color: #B23C18;
    }}
    
    /* Help page styles */
    .formula-box {{
        background-color: rgba(0, 150, 199, 0.05);
        border-radius: 4px;
        padding: 1rem;
        margin: 0.5rem 0 1rem 0;
        border-left: 3px solid var(--primary);
        font-family: monospace;
    }}
    
    .help-section {{
        margin-bottom: 2rem;
    }}
    
    .help-title {{
        font-size: 1.2rem;
        font-weight: 600;
        color: var(--text-primary);
        margin: 1rem 0 0.5rem 0;
        padding-bottom: 0.25rem;
        border-bottom: 1px solid var(--border);
    }}
    
    /* Tariff Tool Styling */
    .info-box {{
        background-color: #E3F2FD;
        padding: 15px;
        border-radius: 5px;
        border-left: 5px solid #1E88E5;
        margin-bottom: 15px;
    }}
    
    .result-box {{
        background-color: #E8F5E9;
        padding: 15px;
        border-radius: 5px;
        border-left: 5px solid #43A047;
        margin: 15px 0;
    }}
    
    .warning-box {{
        background-color: #FFF3E0;
        padding: 15px;
        border-radius: 5px;
        border-left: 5px solid #FF9800;
        margin-bottom: 15px;
    }}
    
    .positive-value {{
        color: #4CAF50;
        font-weight: bold;
    }}
    
    .negative-value {{
        color: #F44336;
        font-weight: bold;
    }}
    
    .section-divider {{
        height: 3px;
        background-color: #f0f2f6;
        margin: 20px 0;
    }}
</style>
""", unsafe_allow_html=True)

#------------------------------------
# SESSION STATE INITIALIZATION
#------------------------------------
if 'quality_analysis_results' not in st.session_state:
    st.session_state.quality_analysis_results = None
    
if 'analysis_submitted' not in st.session_state:
    st.session_state.analysis_submitted = False
    
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
    
if 'batch_analysis_results' not in st.session_state:
    st.session_state.batch_analysis_results = []
    
if 'current_page' not in st.session_state:
    st.session_state.current_page = "analysis"
    
if 'view_mode' not in st.session_state:
    st.session_state.view_mode = "basic"  # "basic" or "advanced"
    
if 'analysis_results_expanded' not in st.session_state:
    st.session_state.analysis_results_expanded = {}

if 'tariff_calculations' not in st.session_state:
    st.session_state.tariff_calculations = []
    
if 'monte_carlo_scenario' not in st.session_state:
    st.session_state.monte_carlo_scenario = None
    
if 'compare_list' not in st.session_state:
    st.session_state.compare_list = []

#------------------------------------
# UTILITY FUNCTIONS
#------------------------------------
def app_header():
    """Displays a branded header"""
    st.markdown(f"""
    <div style="background-color:{PRIMARY_COLOR}; padding:1rem; border-radius:8px; display:flex; align-items:center; margin-bottom:1.5rem;">
        <div style="font-size:1.5rem; font-weight:600; color:white; margin-right:1rem;">PRODUCT PROFITABILITY ANALYZER</div>
        <div style="color:white; font-weight:500;">Comprehensive Medical Device Analysis Tool</div>
    </div>
    """, unsafe_allow_html=True)

def format_currency(value: float) -> str:
    """Format a value as currency with $ symbol"""
    if value is None:
        return "N/A"
    return f"${value:,.2f}"

def format_percentage(value: float) -> str:
    """Format a value as percentage with % symbol"""
    if value is None:
        return "N/A"
    return f"{value:.1f}%"

def format_number(value: float) -> str:
    """Format a value as number with commas"""
    if value is None:
        return "N/A"
    return f"{value:,.2f}"

def toggle_mode():
    """Toggle between basic and advanced view modes"""
    if st.session_state.view_mode == "basic":
        st.session_state.view_mode = "advanced"
    else:
        st.session_state.view_mode = "basic"

def display_mode_toggle():
    """Display the mode toggle UI element"""
    cols = st.columns([1, 1])
    with cols[0]:
        if st.button("Basic Mode", key="basic_mode_btn", use_container_width=True):
            st.session_state.view_mode = "basic"
            st.rerun()
    with cols[1]:
        if st.button("Advanced Mode", key="advanced_mode_btn", use_container_width=True):
            st.session_state.view_mode = "advanced"
            st.rerun()

def navigate_to_page(page_name):
    """Navigate to a specific page in the application"""
    st.session_state.current_page = page_name
    st.rerun()

def reset_analysis():
    """Reset the analysis and return to the input form"""
    st.session_state.quality_analysis_results = None
    st.session_state.analysis_submitted = False
    st.session_state.chat_history = []
    st.rerun()

def export_as_pdf(results):
    """Generate a PDF report of the analysis results"""
    buffer = BytesIO()
    
    with PdfPages(buffer) as pdf:
        # Create a figure for the report
        fig, ax = plt.subplots(figsize=(8.5, 11))
        ax.axis('off')
        
        # Add report title
        ax.text(0.5, 0.98, "Product Profitability Analysis Report", fontsize=16, fontweight='bold', ha='center')
        ax.text(0.5, 0.96, f"Generated on {datetime.now().strftime('%Y-%m-%d %H:%M')}", fontsize=10, ha='center')
        
        # Add product info
        ax.text(0.1, 0.92, f"Product: {results['sku']}", fontsize=12, fontweight='bold')
        ax.text(0.1, 0.90, f"Type: {results['product_type']}", fontsize=10)
        
        # Add key metrics
        ax.text(0.1, 0.86, "Key Metrics", fontsize=12, fontweight='bold')
        ax.text(0.1, 0.84, f"Return Rate (30d): {format_percentage(results['current_metrics']['return_rate_30d'])}", fontsize=10)
        ax.text(0.1, 0.82, f"Monthly Return Cost: {format_currency(results['financial_impact']['monthly_return_cost'])}", fontsize=10)
        ax.text(0.1, 0.80, f"Estimated Monthly Savings: {format_currency(results['financial_impact']['estimated_monthly_savings'])}", fontsize=10)
        ax.text(0.1, 0.78, f"Payback Period: {results['financial_impact']['payback_months']:.1f} months", fontsize=10)
        ax.text(0.1, 0.76, f"3-Year ROI: {format_percentage(results['financial_impact']['roi_3yr'])}", fontsize=10)
        
        # Add recommendation
        ax.text(0.1, 0.72, "Recommendation:", fontsize=12, fontweight='bold')
        ax.text(0.1, 0.70, results['recommendation'], fontsize=10)
        
        # Add issue description
        ax.text(0.1, 0.66, "Issue Description:", fontsize=12, fontweight='bold')
        
        # Wrap the text to fit the page
        import textwrap
        wrapped_desc = textwrap.fill(results['issue_description'], width=80)
        y_pos = 0.64
        for line in wrapped_desc.split('\n'):
            ax.text(0.1, y_pos, line, fontsize=9)
            y_pos -= 0.02
        
        # Add ROI graph
        if y_pos > 0.3:  # Only add graph if there's enough space
            # Create ROI data
            periods = ['Initial', 'Year 1', 'Year 2', 'Year 3']
            initial_investment = -results['financial_impact']['fix_cost_upfront']
            per_unit_cost_total = results['financial_impact']['fix_cost_per_unit'] * results['financial_impact']['projected_sales_36m'] / 3
            annual_savings = results['financial_impact']['annual_savings']
            
            year1_net = annual_savings - per_unit_cost_total
            year2_net = year1_net
            year3_net = year1_net
            
            cumulative = [initial_investment, 
                         initial_investment + year1_net, 
                         initial_investment + year1_net + year2_net,
                         initial_investment + year1_net + year2_net + year3_net]
            
            # Plot the ROI graph at the bottom of the page
            roi_ax = fig.add_axes([0.1, 0.1, 0.8, 0.4])
            roi_ax.bar(periods, [initial_investment, year1_net, year2_net, year3_net], color=['#E76F51', '#40916C', '#40916C', '#40916C'])
            roi_ax.plot(periods, cumulative, 'o-', color='#0096C7', linewidth=2)
            roi_ax.axhline(y=0, color='#ADB5BD', linestyle='-', linewidth=0.5)
            roi_ax.set_title('ROI Analysis Over 3 Years', fontsize=12)
            roi_ax.set_ylabel('Amount ($)', fontsize=10)
            
            # Add value labels
            for i, v in enumerate([initial_investment, year1_net, year2_net, year3_net]):
                roi_ax.text(i, v + (0.1 * v if v > 0 else -0.1 * abs(v)), 
                         f"${abs(v):,.0f}", 
                         ha='center', 
                         fontsize=8,
                         color='#212529')
        
        pdf.savefig(fig)
        plt.close()
    
    # Reset buffer position to the beginning
    buffer.seek(0)
    return buffer

def safe_divide(a: float, b: float, default: float = 0.0) -> float:
    """Safely divide two numbers, returning a default if divisor is zero."""
    try:
        if b == 0:
            return default
        return a / b
    except:
        return default

#------------------------------------
# AI ASSISTANT FUNCTIONS
#------------------------------------
def call_openai_api(messages, model="gpt-4o", temperature=0.7, max_tokens=1024):
    """
    Call OpenAI API with the provided messages
    """
    try:
        # Try to get the API key from Streamlit secrets
        api_key = st.secrets.get("openai_api_key", None)
        
        if not api_key:
            return "AI assistant not available. API key not configured in secrets."
        
        import requests
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        }
        
        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens
        }
        
        response = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers=headers,
            json=payload,
            timeout=30
        )
        
        if response.status_code == 200:
            return response.json()["choices"][0]["message"]["content"]
        else:
            return f"Error from API: {response.status_code} - {response.text}"
    
    except Exception as e:
        st.error(f"Error calling OpenAI API: {str(e)}")
        return f"Error: {str(e)}"

#------------------------------------
# QUALITY ANALYSIS FUNCTIONS
#------------------------------------
def analyze_quality_issue(
    sku: str,
    product_type: str,  # B2B, B2C, or Both
    sales_30d: float,
    returns_30d: float,
    issue_description: str,
    current_unit_cost: float,
    fix_cost_upfront: float,
    fix_cost_per_unit: float,
    sales_price: float,
    expected_reduction: float,  # Expected return rate reduction percentage
    solution_confidence: float,  # Confidence in solution percentage
    new_sales_price: Optional[float] = None,
    asin: Optional[str] = None,
    ncx_rate: Optional[float] = None,
    sales_365d: Optional[float] = None,
    returns_365d: Optional[float] = None,
    star_rating: Optional[float] = None,
    total_reviews: Optional[int] = None,
    fba_fee: Optional[float] = None,
    risk_level: str = "Medium",
    regulatory_impact: str = "None"
) -> Dict[str, Any]:
    """
    Enhanced quality issue analysis with additional fields for sales price, 
    expected reduction, and solution confidence.
    """
    try:
        # Calculate basic metrics with error handling
        return_rate_30d = (returns_30d / sales_30d) * 100 if sales_30d > 0 else 0
        
        # Include 365-day data if available
        return_rate_365d = None
        if sales_365d is not None and returns_365d is not None and sales_365d > 0:
            return_rate_365d = (returns_365d / sales_365d) * 100
        
        # Calculate current financial metrics
        monthly_return_cost = returns_30d * current_unit_cost
        monthly_revenue = sales_30d * sales_price
        monthly_profit = monthly_revenue - (sales_30d * current_unit_cost)
        profit_margin = (monthly_profit / monthly_revenue) * 100 if monthly_revenue > 0 else 0
        
        # Use the expected reduction provided by the user
        reduction_factor = expected_reduction / 100.0
        
        # Adjust reduction based on confidence
        adjusted_reduction = reduction_factor * (solution_confidence / 100.0)
        
        # Estimate reduced returns
        reduced_returns_30d = returns_30d * (1 - adjusted_reduction)
        returns_saved_30d = returns_30d - reduced_returns_30d
        
        # Calculate financial impact
        estimated_monthly_savings = returns_saved_30d * current_unit_cost
        
        # Calculate new revenue if price changes
        if new_sales_price is not None and new_sales_price > 0:
            new_monthly_revenue = sales_30d * new_sales_price
            revenue_change = new_monthly_revenue - monthly_revenue
        else:
            new_sales_price = sales_price
            new_monthly_revenue = monthly_revenue
            revenue_change = 0
        
        # Calculate new unit cost with fix
        new_unit_cost = current_unit_cost + fix_cost_per_unit
        
        # Calculate new profit
        new_monthly_profit = new_monthly_revenue - (sales_30d * new_unit_cost)
        new_profit_margin = (new_monthly_profit / new_monthly_revenue) * 100 if new_monthly_revenue > 0 else 0
        
        # Annual projections
        annual_return_cost = monthly_return_cost * 12
        annual_savings = estimated_monthly_savings * 12
        annual_revenue_change = revenue_change * 12
        
        # Simple payback period (months) with error handling
        total_monthly_benefit = estimated_monthly_savings + revenue_change
        if total_monthly_benefit > 0:
            payback_months = fix_cost_upfront / total_monthly_benefit
        else:
            payback_months = float('inf')
        
        # Calculate 3-year ROI
        projected_sales_36m = sales_30d * 36  # 36 months projection
        total_investment = fix_cost_upfront + (fix_cost_per_unit * projected_sales_36m)
        total_savings = (annual_savings + annual_revenue_change) * 3
        
        if total_investment > 0:
            roi_3yr = ((total_savings - total_investment) / total_investment) * 100
        else:
            roi_3yr = float('inf') if total_savings > 0 else 0
        
        # Determine recommendation based on metrics, risk level, and regulatory impact
        if regulatory_impact == "Significant":
            recommendation = "Highly Recommended - Regulatory compliance required"
            recommendation_class = "recommendation-high"
        elif risk_level == "High" and payback_months < 12:
            recommendation = "Highly Recommended - High risk issue with positive ROI"
            recommendation_class = "recommendation-high"
        elif payback_months < 3:
            recommendation = "Highly Recommended - Quick ROI expected"
            recommendation_class = "recommendation-high"
        elif payback_months < 6:
            recommendation = "Recommended - Good medium-term ROI"
            recommendation_class = "recommendation-medium"
        elif payback_months < 12:
            recommendation = "Consider - Long-term benefits may outweigh costs"
            recommendation_class = "recommendation-medium"
        elif risk_level == "High":
            recommendation = "Consider despite ROI - High risk issue"
            recommendation_class = "recommendation-medium"
        else:
            recommendation = "Not Recommended - Poor financial return"
            recommendation_class = "recommendation-low"
        
        # Adjust recommendation based on B2B/B2C and review metrics
        brand_impact = None
        if product_type in ["B2C", "Both"] and star_rating is not None:
            if star_rating < 3.5:
                brand_impact = "Significant - Low star rating indicates potential brand damage"
                # Adjust recommendation if star rating is concerning
                if recommendation_class == "recommendation-low" and risk_level != "Low":
                    recommendation = "Recommended despite ROI - Brand protection needed"
                    recommendation_class = "recommendation-medium"
        
        # For B2B products, return rate is more important
        if product_type in ["B2B", "Both"] and return_rate_30d > 10:
            if recommendation_class == "recommendation-low":
                recommendation = "Consider despite ROI - High return rate for B2B product"
                recommendation_class = "recommendation-medium"
                if not brand_impact:
                    brand_impact = "Moderate - High return rate for B2B product may affect customer relationships"
        
        # Calculate customer impact metrics
        customer_impact_metrics = {}
        if total_reviews is not None and star_rating is not None:
            # Calculate potential rating improvement
            potential_rating_improvement = min(5.0, star_rating + (star_rating < 4.0) * 0.5)
            customer_impact_metrics["potential_rating"] = potential_rating_improvement
            
            # Calculate potential review impact
            negative_reviews_ratio = max(0, min(1, (5 - star_rating) / 4))
            potential_review_improvement = int(total_reviews * negative_reviews_ratio * adjusted_reduction * 0.5)
            customer_impact_metrics["potential_review_improvement"] = potential_review_improvement
        
        # Add medical impact data
        medical_device_impact = {
            "risk_level": risk_level,
            "regulatory_impact": regulatory_impact,
            "patient_safety_impact": "High" if risk_level == "High" else ("Medium" if risk_level == "Medium" else "Low"),
        }
        
        # Build the complete results dictionary
        results = {
            "sku": sku,
            "asin": asin,
            "product_type": product_type,
            "current_metrics": {
                "return_rate_30d": return_rate_30d,
                "return_rate_365d": return_rate_365d,
                "ncx_rate": ncx_rate,
                "star_rating": star_rating,
                "total_reviews": total_reviews,
                "sales_price": sales_price,
                "monthly_revenue": monthly_revenue,
                "profit_margin": profit_margin
            },
            "solution_metrics": {
                "expected_reduction": expected_reduction,
                "solution_confidence": solution_confidence,
                "adjusted_reduction": adjusted_reduction * 100,  # Convert to percentage
                "new_sales_price": new_sales_price
            },
            "financial_impact": {
                "monthly_return_cost": monthly_return_cost,
                "annual_return_cost": annual_return_cost,
                "returns_saved_30d": returns_saved_30d,
                "reduced_returns_30d": reduced_returns_30d,
                "estimated_monthly_savings": estimated_monthly_savings,
                "revenue_change": revenue_change,
                "total_monthly_benefit": total_monthly_benefit,
                "annual_savings": annual_savings,
                "annual_revenue_change": annual_revenue_change,
                "payback_months": payback_months,
                "roi_3yr": roi_3yr,
                "fix_cost_upfront": fix_cost_upfront,
                "fix_cost_per_unit": fix_cost_per_unit,
                "current_unit_cost": current_unit_cost,
                "new_unit_cost": new_unit_cost,
                "fba_fee": fba_fee,
                "projected_sales_36m": projected_sales_36m,
                "total_investment": total_investment,
                "total_savings": total_savings,
                "new_profit_margin": new_profit_margin,
                "profit_margin_change": new_profit_margin - profit_margin
            },
            "recommendation": recommendation,
            "recommendation_class": recommendation_class,
            "brand_impact": brand_impact,
            "issue_description": issue_description,
            "customer_impact_metrics": customer_impact_metrics,
            "medical_device_impact": medical_device_impact
        }
        
        return results
    
    except Exception as e:
        st.error(f"Error in quality analysis: {str(e)}")
        # Return minimal results with error
        return {
            "sku": sku,
            "error": str(e),
            "recommendation": "Analysis Error - Please check inputs",
            "recommendation_class": "recommendation-low"
        }

def display_quality_issue_results(results, expanded=True):
    """Display the quality issue analysis results in a visually appealing way"""
    st.markdown(f'<div class="card">', unsafe_allow_html=True)
    
    # Results header with summary metrics
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        st.markdown(f'<div class="sub-header">Analysis Results: {results["sku"]}</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div style="text-align: right;">
            <span class="results-badge results-badge-{results['recommendation_class'].split('-')[1]}">
                ROI: {format_percentage(results['financial_impact']['roi_3yr'])}
            </span>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div style="text-align: right;">
            <span class="results-badge results-badge-{results['recommendation_class'].split('-')[1]}">
                Payback: {results['financial_impact']['payback_months']:.1f} months
            </span>
        </div>
        """, unsafe_allow_html=True)
    
    # Check for error in results
    if "error" in results:
        st.error(f"Analysis Error: {results['error']}")
        st.info("Please check your inputs and try again.")
        st.markdown('</div>', unsafe_allow_html=True)
        return
    
    # If not expanded, show just summary and recommendation
    if not expanded:
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown('<div class="metric-label">Return Rate</div>', unsafe_allow_html=True)
            return_rate = results["current_metrics"]["return_rate_30d"]
            color = DANGER_COLOR if return_rate > 10 else (WARNING_COLOR if return_rate > 5 else SUCCESS_COLOR)
            st.markdown(f'<div class="metric-value" style="color:{color}">{return_rate:.2f}%</div>', unsafe_allow_html=True)
            
            st.markdown('<div class="metric-label">Monthly Return Cost</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="metric-value">{format_currency(results["financial_impact"]["monthly_return_cost"])}</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="metric-label">Est. Monthly Savings</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="metric-value" style="color:{SUCCESS_COLOR}">{format_currency(results["financial_impact"]["estimated_monthly_savings"])}</div>', unsafe_allow_html=True)
            
            if results["financial_impact"]["revenue_change"] > 0:
                st.markdown('<div class="metric-label">Monthly Revenue Change</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="metric-value" style="color:{SUCCESS_COLOR}">{format_currency(results["financial_impact"]["revenue_change"])}</div>', unsafe_allow_html=True)
        
        # Recommendation
        st.markdown(f'<div class="{results["recommendation_class"]}">{results["recommendation"]}</div>', unsafe_allow_html=True)
        
        # Add button to expand
        if st.button("Show Details", key=f"expand_{results['sku']}"):
            return True
        
        st.markdown('</div>', unsafe_allow_html=True)
        return False
    
    # Tabs for different aspects of the analysis
    tabs = st.tabs(["Overview", "Financial Impact", "Quality Metrics", "Solution Details"])
    
    with tabs[0]:  # Overview tab
        # Current metrics in a 3-column layout
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.markdown('<div class="metric-label">Return Rate (30d)</div>', unsafe_allow_html=True)
            
            # Color-code return rate based on severity
            return_rate = results["current_metrics"]["return_rate_30d"]
            if return_rate > 10:
                color = DANGER_COLOR
            elif return_rate > 5:
                color = WARNING_COLOR
            else:
                color = SUCCESS_COLOR
                
            st.markdown(f'<div class="metric-value" style="color:{color}">{return_rate:.2f}%</div>', unsafe_allow_html=True)
            
            if results["current_metrics"]["return_rate_365d"] is not None:
                st.markdown('<div class="metric-label">Return Rate (365d)</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="metric-value">{results["current_metrics"]["return_rate_365d"]:.2f}%</div>', unsafe_allow_html=True)
            
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.markdown('<div class="metric-label">Monthly Return Cost</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="metric-value">{format_currency(results["financial_impact"]["monthly_return_cost"])}</div>', unsafe_allow_html=True)
            
            st.markdown('<div class="metric-label">Annual Return Cost</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="metric-value">{format_currency(results["financial_impact"]["annual_return_cost"])}</div>', unsafe_allow_html=True)
            
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col3:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.markdown('<div class="metric-label">Est. Monthly Savings</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="metric-value" style="color:{SUCCESS_COLOR}">{format_currency(results["financial_impact"]["estimated_monthly_savings"])}</div>', unsafe_allow_html=True)
            
            if "adjusted_reduction" in results["solution_metrics"]:
                st.markdown(f'<div style="font-size:0.9rem;color:{TEXT_SECONDARY}">({results["solution_metrics"]["adjusted_reduction"]:.0f}% adjusted reduction)</div>', unsafe_allow_html=True)
            
            if results["financial_impact"]["revenue_change"] != 0:
                revenue_change = results["financial_impact"]["revenue_change"]
                revenue_color = SUCCESS_COLOR if revenue_change > 0 else DANGER_COLOR
                st.markdown('<div class="metric-label">Monthly Revenue Change</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="metric-value" style="color:{revenue_color}">{format_currency(revenue_change)}</div>', unsafe_allow_html=True)
            
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Risk and regulatory info
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.markdown('<div class="metric-label">Medical Risk Level</div>', unsafe_allow_html=True)
            risk_level = results["medical_device_impact"]["risk_level"]
            risk_color = {"Low": SUCCESS_COLOR, "Medium": WARNING_COLOR, "High": DANGER_COLOR}.get(risk_level, WARNING_COLOR)
            st.markdown(f'<div class="metric-value" style="color:{risk_color}">{risk_level}</div>', unsafe_allow_html=True)
            
            if results["medical_device_impact"]["regulatory_impact"] != "None":
                st.markdown('<div class="metric-label">Regulatory Impact</div>', unsafe_allow_html=True)
                reg_impact = results["medical_device_impact"]["regulatory_impact"]
                reg_color = DANGER_COLOR if reg_impact == "Significant" else WARNING_COLOR
                st.markdown(f'<div class="metric-value" style="color:{reg_color}">{reg_impact}</div>', unsafe_allow_html=True)
            
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.markdown('<div class="metric-label">Payback Period</div>', unsafe_allow_html=True)
            if results["financial_impact"]["payback_months"] == float('inf'):
                payback_text = "N/A"
                payback_color = DANGER_COLOR
            else:
                payback_months = results['financial_impact']['payback_months']
                payback_text = f"{payback_months:.1f} months"
                
                if payback_months < 3:
                    payback_color = SUCCESS_COLOR
                elif payback_months < 6:
                    payback_color = SECONDARY_COLOR
                elif payback_months < 12:
                    payback_color = WARNING_COLOR
                else:
                    payback_color = DANGER_COLOR
                    
            st.markdown(f'<div class="metric-value" style="color:{payback_color}">{payback_text}</div>', unsafe_allow_html=True)
            
            st.markdown('<div class="metric-label">3-Year ROI</div>', unsafe_allow_html=True)
            roi = results["financial_impact"]["roi_3yr"]
            
            if roi == float('inf'):
                roi_text = "∞"
                roi_color = SUCCESS_COLOR
            else:
                roi_text = f"{roi:.1f}%"
                roi_color = SUCCESS_COLOR if roi > 0 else DANGER_COLOR
                
            st.markdown(f'<div class="metric-value" style="color:{roi_color}">{roi_text}</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Recommendation box
        st.markdown('<div class="metric-label" style="margin-top:10px;">Recommendation</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="{results["recommendation_class"]}">{results["recommendation"]}</div>', unsafe_allow_html=True)
        
        # Brand impact if available
        if results["brand_impact"]:
            st.markdown('<div class="metric-label" style="margin-top:10px;">Brand Impact Assessment</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="metric-value">{results["brand_impact"]}</div>', unsafe_allow_html=True)
        
        # Issue description
        st.markdown('<div class="metric-label" style="margin-top:15px;">Issue Description</div>', unsafe_allow_html=True)
        st.markdown(f'<div style="padding: 10px; background-color:{BACKGROUND_COLOR}; border-radius:5px;">{results["issue_description"]}</div>', unsafe_allow_html=True)
    
    with tabs[1]:  # Financial Impact tab
        # Financial metrics in a comparison view
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown('<div class="section-header">Current Financial State</div>', unsafe_allow_html=True)
            
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.markdown('<div class="metric-label">Monthly Revenue</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="metric-value">{format_currency(results["current_metrics"]["monthly_revenue"])}</div>', unsafe_allow_html=True)
            
            st.markdown('<div class="metric-label">Unit Cost</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="metric-value">{format_currency(results["financial_impact"]["current_unit_cost"])}</div>', unsafe_allow_html=True)
            
            st.markdown('<div class="metric-label">Profit Margin</div>', unsafe_allow_html=True)
            margin = results["current_metrics"]["profit_margin"]
            margin_color = SUCCESS_COLOR if margin > 20 else (WARNING_COLOR if margin > 10 else DANGER_COLOR)
            st.markdown(f'<div class="metric-value" style="color:{margin_color}">{format_percentage(margin)}</div>', unsafe_allow_html=True)
            
            st.markdown('<div class="metric-label">Monthly Return Cost</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="metric-value" style="color:{DANGER_COLOR}">{format_currency(results["financial_impact"]["monthly_return_cost"])}</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="section-header">Projected Financial State After Fix</div>', unsafe_allow_html=True)
            
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            if results["solution_metrics"]["new_sales_price"] != results["current_metrics"]["sales_price"]:
                new_revenue = results["current_metrics"]["monthly_revenue"] + results["financial_impact"]["revenue_change"]
                st.markdown('<div class="metric-label">New Monthly Revenue</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="metric-value">{format_currency(new_revenue)}</div>', unsafe_allow_html=True)
                
                revenue_change = results["financial_impact"]["revenue_change"]
                change_color = SUCCESS_COLOR if revenue_change > 0 else DANGER_COLOR
                change_sign = "+" if revenue_change > 0 else ""
                st.markdown(f'<div class="metric-comparison" style="color:{change_color}">{change_sign}{format_currency(revenue_change)}</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="metric-label">Monthly Revenue</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="metric-value">{format_currency(results["current_metrics"]["monthly_revenue"])}</div>', unsafe_allow_html=True)
                st.markdown('<div class="metric-comparison">(unchanged)</div>', unsafe_allow_html=True)
            
            st.markdown('<div class="metric-label">New Unit Cost</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="metric-value">{format_currency(results["financial_impact"]["new_unit_cost"])}</div>', unsafe_allow_html=True)
            
            cost_increase = results["financial_impact"]["fix_cost_per_unit"]
            if cost_increase > 0:
                st.markdown(f'<div class="metric-comparison" style="color:{DANGER_COLOR}">+{format_currency(cost_increase)}</div>', unsafe_allow_html=True)
            
            st.markdown('<div class="metric-label">New Profit Margin</div>', unsafe_allow_html=True)
            new_margin = results["financial_impact"]["new_profit_margin"]
            margin_color = SUCCESS_COLOR if new_margin > 20 else (WARNING_COLOR if new_margin > 10 else DANGER_COLOR)
            st.markdown(f'<div class="metric-value" style="color:{margin_color}">{format_percentage(new_margin)}</div>', unsafe_allow_html=True)
            
            margin_change = results["financial_impact"]["profit_margin_change"]
            change_color = SUCCESS_COLOR if margin_change > 0 else DANGER_COLOR
            change_sign = "+" if margin_change > 0 else ""
            st.markdown(f'<div class="metric-comparison" style="color:{change_color}">{change_sign}{format_percentage(margin_change)}</div>', unsafe_allow_html=True)
            
            st.markdown('<div class="metric-label">Reduced Monthly Return Cost</div>', unsafe_allow_html=True)
            reduced_return_cost = results["financial_impact"]["reduced_returns_30d"] * results["financial_impact"]["current_unit_cost"]
            st.markdown(f'<div class="metric-value" style="color:{DANGER_COLOR}">{format_currency(reduced_return_cost)}</div>', unsafe_allow_html=True)
            
            savings = results["financial_impact"]["monthly_return_cost"] - reduced_return_cost
            st.markdown(f'<div class="metric-comparison" style="color:{SUCCESS_COLOR}">-{format_currency(savings)} ({format_percentage(results["solution_metrics"]["adjusted_reduction"])} reduction)</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Overall financial impact summary
        st.markdown('<div class="section-header">Investment & Returns</div>', unsafe_allow_html=True)
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.markdown('<div class="metric-label">Initial Investment</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="metric-value" style="color:{DANGER_COLOR}">{format_currency(results["financial_impact"]["fix_cost_upfront"])}</div>', unsafe_allow_html=True)
            
            if results["financial_impact"]["fix_cost_per_unit"] > 0:
                st.markdown('<div class="metric-label">Additional Cost Per Unit</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="metric-value" style="color:{WARNING_COLOR}">{format_currency(results["financial_impact"]["fix_cost_per_unit"])}</div>', unsafe_allow_html=True)
            
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.markdown('<div class="metric-label">Total Monthly Benefit</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="metric-value" style="color:{SUCCESS_COLOR}">{format_currency(results["financial_impact"]["total_monthly_benefit"])}</div>', unsafe_allow_html=True)
            
            if results["financial_impact"]["revenue_change"] != 0:
                revenue_note = f"Includes {format_currency(results['financial_impact']['revenue_change'])} revenue change"
                st.markdown(f'<div style="font-size:0.9rem;color:{TEXT_SECONDARY}">{revenue_note}</div>', unsafe_allow_html=True)
            
            st.markdown('<div class="metric-label">Annual Savings</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="metric-value" style="color:{SUCCESS_COLOR}">{format_currency(results["financial_impact"]["annual_savings"] + results["financial_impact"]["annual_revenue_change"])}</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col3:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.markdown('<div class="metric-label">Payback Period</div>', unsafe_allow_html=True)
            
            if results["financial_impact"]["payback_months"] == float('inf'):
                payback_text = "N/A"
                payback_color = DANGER_COLOR
            else:
                payback_months = results['financial_impact']['payback_months']
                payback_text = f"{payback_months:.1f} months"
                
                if payback_months < 3:
                    payback_color = SUCCESS_COLOR
                elif payback_months < 6:
                    payback_color = SECONDARY_COLOR
                elif payback_months < 12:
                    payback_color = WARNING_COLOR
                else:
                    payback_color = DANGER_COLOR
                    
            st.markdown(f'<div class="metric-value" style="color:{payback_color}">{payback_text}</div>', unsafe_allow_html=True)
            
            st.markdown('<div class="metric-label">3-Year ROI</div>', unsafe_allow_html=True)
            roi = results["financial_impact"]["roi_3yr"]
            
            if roi == float('inf'):
                roi_text = "∞"
                roi_color = SUCCESS_COLOR
            else:
                roi_text = f"{roi:.1f}%"
                roi_color = SUCCESS_COLOR if roi > 0 else DANGER_COLOR
                
            st.markdown(f'<div class="metric-value" style="color:{roi_color}">{roi_text}</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        # ROI chart with improved design
        st.markdown('<div style="margin-top:20px;" class="chart-container">', unsafe_allow_html=True)
        fig = go.Figure()
        
        # Initial investment
        fig.add_trace(go.Bar(
            x=['Initial Investment'],
            y=[-results["financial_impact"]["fix_cost_upfront"]],
            name='Initial Investment',
            marker_color=DANGER_COLOR
        ))
        
        # Per-unit cost increase (total over 3 years)
        per_unit_cost_total = results["financial_impact"]["fix_cost_per_unit"] * results["financial_impact"]["projected_sales_36m"]
        
        if per_unit_cost_total > 0:
            fig.add_trace(go.Bar(
                x=['Additional Unit Costs (3 years)'],
                y=[-per_unit_cost_total],
                name='Additional Unit Costs',
                marker_color=WARNING_COLOR
            ))
        
        # Revenue change component
        annual_revenue_change = results["financial_impact"]["annual_revenue_change"]
        
        # 1-year, 2-year, 3-year benefits (savings + revenue)
        annual_benefit = results["financial_impact"]["annual_savings"] + annual_revenue_change
        
        fig.add_trace(go.Bar(
            x=['Year 1', 'Year 2', 'Year 3'],
            y=[annual_benefit, annual_benefit, annual_benefit],
            name='Annual Benefit',
            marker_color=SUCCESS_COLOR
        ))
        
        # Split between savings and revenue if applicable
        if annual_revenue_change != 0:
            # Create a stacked bar chart for the benefits
            fig = go.Figure()
            
            # Initial investment (negative value)
            fig.add_trace(go.Bar(
                x=['Initial Investment'],
                y=[-results["financial_impact"]["fix_cost_upfront"]],
                name='Initial Investment',
                marker_color=DANGER_COLOR
            ))
            
            # Per-unit cost increase (negative value)
            if per_unit_cost_total > 0:
                fig.add_trace(go.Bar(
                    x=['Additional Unit Costs (3 years)'],
                    y=[-per_unit_cost_total],
                    name='Additional Unit Costs',
                    marker_color=WARNING_COLOR
                ))
            
            # Return savings component for each year
            fig.add_trace(go.Bar(
                x=['Year 1', 'Year 2', 'Year 3'],
                y=[results["financial_impact"]["annual_savings"], 
                   results["financial_impact"]["annual_savings"], 
                   results["financial_impact"]["annual_savings"]],
                name='Return Savings',
                marker_color=SECONDARY_COLOR
            ))
            
            # Revenue change component for each year (if applicable)
            if annual_revenue_change != 0:
                fig.add_trace(go.Bar(
                    x=['Year 1', 'Year 2', 'Year 3'],
                    y=[annual_revenue_change, annual_revenue_change, annual_revenue_change],
                    name='Revenue Change',
                    marker_color=SUCCESS_COLOR if annual_revenue_change > 0 else DANGER_COLOR
                ))
        
        # Cumulative net benefit line
        year1_net = -results["financial_impact"]["fix_cost_upfront"] - (per_unit_cost_total/3) + annual_benefit
        year2_net = year1_net - (per_unit_cost_total/3) + annual_benefit
        year3_net = year2_net - (per_unit_cost_total/3) + annual_benefit
        
        fig.add_trace(go.Scatter(
            x=['Initial', 'Year 1', 'Year 2', 'Year 3'],
            y=[-results["financial_impact"]["fix_cost_upfront"], year1_net, year2_net, year3_net],
            name='Cumulative Net Benefit',
            line=dict(color=PRIMARY_COLOR, width=3, dash='solid'),
            mode='lines+markers'
        ))
        
        fig.update_layout(
            title='ROI Analysis Over 3 Years',
            xaxis_title='Timeline',
            yaxis_title='Amount ($)',
            barmode='stack',
            height=400,
            margin=dict(l=20, r=20, t=40, b=20),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            template="plotly_white",
            plot_bgcolor='rgba(0,0,0,0)',
        )
        
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with tabs[2]:  # Quality Metrics tab
        # Quality metrics visualization
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown('<div class="section-header">Returns Analysis</div>', unsafe_allow_html=True)
            
            # Create returns funnel chart
            returns_fig = go.Figure()
            
            # Current returns
            returns_fig.add_trace(go.Bar(
                x=['Current Returns'],
                y=[returns_30d := results["returns_30d"] if "returns_30d" in results else results["financial_impact"]["monthly_return_cost"] / results["financial_impact"]["current_unit_cost"]],
                name='Current Returns',
                marker_color=DANGER_COLOR,
                width=0.4
            ))
            
            # Expected reduction
            returns_fig.add_trace(go.Bar(
                x=['Expected Returns (After Fix)'],
                y=[reduced_returns := results["financial_impact"]["reduced_returns_30d"]],
                name='Expected Returns',
                marker_color=SECONDARY_COLOR,
                width=0.4
            ))
            
            # Calculate reduction percentage
            reduction_pct = ((returns_30d - reduced_returns) / returns_30d) * 100 if returns_30d > 0 else 0
            
            # Add value labels
            returns_fig.add_annotation(
                x='Current Returns',
                y=returns_30d,
                text=f"{returns_30d:.0f}",
                showarrow=False,
                yshift=10,
                font=dict(color=TEXT_PRIMARY)
            )
            
            returns_fig.add_annotation(
                x='Expected Returns (After Fix)',
                y=reduced_returns,
                text=f"{reduced_returns:.0f} ({reduction_pct:.0f}% reduction)",
                showarrow=False,
                yshift=10,
                font=dict(color=TEXT_PRIMARY)
            )
            
            # Connect the bars with an arrow
            returns_fig.add_shape(
                type="line",
                x0=0, y0=returns_30d,
                x1=1, y1=reduced_returns,
                line=dict(color="black", width=2, dash="dot"),
                xref="x", yref="y"
            )
            
            returns_fig.update_layout(
                title='Return Rate Reduction',
                xaxis_title='',
                yaxis_title='Number of Returns (30 Days)',
                showlegend=False,
                height=300,
                margin=dict(l=20, r=20, t=40, b=20),
                template="plotly_white",
                plot_bgcolor='rgba(0,0,0,0)',
            )
            
            st.plotly_chart(returns_fig, use_container_width=True)
        
        with col2:
            st.markdown('<div class="section-header">Customer Metrics</div>', unsafe_allow_html=True)
            
            if results["current_metrics"]["star_rating"] is not None:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.markdown('<div class="metric-label">Current Star Rating</div>', unsafe_allow_html=True)
                
                # Color-code star rating
                star_rating = results["current_metrics"]["star_rating"]
                if star_rating >= 4.0:
                    rating_color = SUCCESS_COLOR
                elif star_rating >= 3.0:
                    rating_color = WARNING_COLOR
                else:
                    rating_color = DANGER_COLOR
                    
                st.markdown(f'<div class="metric-value" style="color:{rating_color}">{star_rating:.1f}★</div>', unsafe_allow_html=True)
                
                if results["current_metrics"]["total_reviews"]:
                    st.markdown(f'<div style="font-size:0.9rem;color:{TEXT_SECONDARY}">({results["current_metrics"]["total_reviews"]} reviews)</div>', unsafe_allow_html=True)
                
                # Potential improved rating
                if "potential_rating" in results["customer_impact_metrics"]:
                    potential_rating = results["customer_impact_metrics"]["potential_rating"]
                    st.markdown('<div class="metric-label">Potential Improved Rating</div>', unsafe_allow_html=True)
                    
                    potential_color = SUCCESS_COLOR if potential_rating >= 4.0 else (WARNING_COLOR if potential_rating >= 3.0 else DANGER_COLOR)
                    st.markdown(f'<div class="metric-value" style="color:{potential_color}">{potential_rating:.1f}★</div>', unsafe_allow_html=True)
                    
                    rating_increase = potential_rating - star_rating
                    if rating_increase > 0:
                        st.markdown(f'<div class="metric-comparison metric-positive">+{rating_increase:.1f} stars</div>', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
            
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.markdown('<div class="metric-label">Medical Risk Level</div>', unsafe_allow_html=True)
            risk_level = results["medical_device_impact"]["risk_level"]
            risk_color = DANGER_COLOR if risk_level == "High" else (WARNING_COLOR if risk_level == "Medium" else SUCCESS_COLOR)
            st.markdown(f'<div class="metric-value" style="color:{risk_color}">{risk_level}</div>', unsafe_allow_html=True)
            
            st.markdown('<div class="metric-label">Regulatory Impact</div>', unsafe_allow_html=True)
            reg_impact = results["medical_device_impact"]["regulatory_impact"]
            reg_color = DANGER_COLOR if reg_impact == "Significant" else (WARNING_COLOR if reg_impact == "Possible" else TEXT_SECONDARY)
            st.markdown(f'<div class="metric-value" style="color:{reg_color}">{reg_impact}</div>', unsafe_allow_html=True)
            
            st.markdown('<div class="metric-label">Patient Safety Impact</div>', unsafe_allow_html=True)
            safety_impact = results["medical_device_impact"]["patient_safety_impact"]
            safety_color = DANGER_COLOR if safety_impact == "High" else (WARNING_COLOR if safety_impact == "Medium" else SUCCESS_COLOR)
            st.markdown(f'<div class="metric-value" style="color:{safety_color}">{safety_impact}</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        # NCX rate if available
        if results["current_metrics"]["ncx_rate"] is not None:
            st.markdown('<div class="section-header">Negative Customer Experience Analysis</div>', unsafe_allow_html=True)
            
            ncx_rate = results["current_metrics"]["ncx_rate"]
            
            # Calculate potential improved NCX rate
            potential_ncx_rate = ncx_rate * (1 - (results["solution_metrics"]["adjusted_reduction"] / 100))
            
            # Create NCX gauge charts
            col1, col2 = st.columns(2)
            
            with col1:
                current_ncx_fig = go.Figure(go.Indicator(
                    mode = "gauge+number",
                    value = ncx_rate,
                    number = {"suffix": "%", "font": {"size": 24}},
                    gauge = {
                        "axis": {"range": [None, 30], "tickwidth": 1, "tickcolor": TEXT_SECONDARY},
                        "bar": {"color": DANGER_COLOR},
                        "bgcolor": "white",
                        "borderwidth": 2,
                        "bordercolor": TEXT_SECONDARY,
                        "steps": [
                            {"range": [0, 5], "color": "rgba(64, 145, 108, 0.3)"},
                            {"range": [5, 10], "color": "rgba(233, 196, 106, 0.3)"},
                            {"range": [10, 30], "color": "rgba(231, 111, 81, 0.3)"}
                        ],
                        "threshold": {
                            "line": {"color": DANGER_COLOR, "width": 4},
                            "thickness": 0.75,
                            "value": ncx_rate
                        }
                    },
                    title = {"text": "Current NCX Rate"}
                ))
                
                current_ncx_fig.update_layout(
                    height=250,
                    margin=dict(l=20, r=20, t=30, b=20),
                )
                
                st.plotly_chart(current_ncx_fig, use_container_width=True)
            
            with col2:
                potential_ncx_fig = go.Figure(go.Indicator(
                    mode = "gauge+number+delta",
                    value = potential_ncx_rate,
                    number = {"suffix": "%", "font": {"size": 24}},
                    delta = {"reference": ncx_rate, "decreasing": {"color": SUCCESS_COLOR}},
                    gauge = {
                        "axis": {"range": [None, 30], "tickwidth": 1, "tickcolor": TEXT_SECONDARY},
                        "bar": {"color": SECONDARY_COLOR},
                        "bgcolor": "white",
                        "borderwidth": 2,
                        "bordercolor": TEXT_SECONDARY,
                        "steps": [
                            {"range": [0, 5], "color": "rgba(64, 145, 108, 0.3)"},
                            {"range": [5, 10], "color": "rgba(233, 196, 106, 0.3)"},
                            {"range": [10, 30], "color": "rgba(231, 111, 81, 0.3)"}
                        ],
                        "threshold": {
                            "line": {"color": SECONDARY_COLOR, "width": 4},
                            "thickness": 0.75,
                            "value": potential_ncx_rate
                        }
                    },
                    title = {"text": "Projected NCX Rate After Fix"}
                ))
                
                potential_ncx_fig.update_layout(
                    height=250,
                    margin=dict(l=20, r=20, t=30, b=20),
                )
                
                st.plotly_chart(potential_ncx_fig, use_container_width=True)
    
    with tabs[3]:  # Solution Details tab
        # Display solution details and confidence
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.markdown('<div class="metric-label">Expected Return Rate Reduction</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="metric-value">{format_percentage(results["solution_metrics"]["expected_reduction"])}</div>', unsafe_allow_html=True)
            
            st.markdown('<div class="metric-label">Solution Confidence</div>', unsafe_allow_html=True)
            confidence = results["solution_metrics"]["solution_confidence"]
            confidence_color = SUCCESS_COLOR if confidence >= 80 else (WARNING_COLOR if confidence >= 50 else DANGER_COLOR)
            st.markdown(f'<div class="metric-value" style="color:{confidence_color}">{format_percentage(confidence)}</div>', unsafe_allow_html=True)
            
            st.markdown('<div class="metric-label">Adjusted Return Rate Reduction</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="metric-value">{format_percentage(results["solution_metrics"]["adjusted_reduction"])}</div>', unsafe_allow_html=True)
            st.markdown(f'<div style="font-size:0.9rem;color:{TEXT_SECONDARY}">(Expected reduction × Confidence)</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.markdown('<div class="metric-label">Initial Fix Cost</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="metric-value">{format_currency(results["financial_impact"]["fix_cost_upfront"])}</div>', unsafe_allow_html=True)
            
            st.markdown('<div class="metric-label">Additional Cost Per Unit</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="metric-value">{format_currency(results["financial_impact"]["fix_cost_per_unit"])}</div>', unsafe_allow_html=True)
            
            if results["solution_metrics"]["new_sales_price"] != results["current_metrics"]["sales_price"]:
                st.markdown('<div class="metric-label">Original Sales Price</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="metric-value">{format_currency(results["current_metrics"]["sales_price"])}</div>', unsafe_allow_html=True)
                
                st.markdown('<div class="metric-label">New Sales Price</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="metric-value">{format_currency(results["solution_metrics"]["new_sales_price"])}</div>', unsafe_allow_html=True)
                
                price_change = results["solution_metrics"]["new_sales_price"] - results["current_metrics"]["sales_price"]
                price_pct = (price_change / results["current_metrics"]["sales_price"]) * 100
                change_color = SUCCESS_COLOR if price_change > 0 else DANGER_COLOR
                change_sign = "+" if price_change > 0 else ""
                st.markdown(f'<div class="metric-comparison" style="color:{change_color}">{change_sign}{format_currency(price_change)} ({change_sign}{price_pct:.1f}%)</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Confidence visualization
        st.markdown('<div class="section-header">Solution Confidence Analysis</div>', unsafe_allow_html=True)
        
        # Create confidence gauge
        confidence_fig = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = results["solution_metrics"]["solution_confidence"],
            number = {"suffix": "%", "font": {"size": 30}},
            gauge = {
                "axis": {"range": [None, 100], "tickwidth": 1, "tickcolor": TEXT_SECONDARY},
                "bar": {"color": 
                       (SUCCESS_COLOR if confidence >= 80 else 
                        (WARNING_COLOR if confidence >= 50 else DANGER_COLOR))},
                "bgcolor": "white",
                "borderwidth": 2,
                "bordercolor": TEXT_SECONDARY,
                "steps": [
                    {"range": [0, 30], "color": "rgba(231, 111, 81, 0.2)"},
                    {"range": [30, 70], "color": "rgba(233, 196, 106, 0.2)"},
                    {"range": [70, 100], "color": "rgba(64, 145, 108, 0.2)"}
                ],
                "threshold": {
                    "line": {"color": TEXT_PRIMARY, "width": 4},
                    "thickness": 0.75,
                    "value": results["solution_metrics"]["solution_confidence"]
                }
            },
            title = {"text": "Solution Confidence Level", "font": {"size": 16}}
        ))
        
        confidence_fig.update_layout(
            height=300,
            margin=dict(l=20, r=20, t=40, b=20),
        )
        
        col1, col2, col3 = st.columns([1, 3, 1])
        with col2:
            st.plotly_chart(confidence_fig, use_container_width=True)
        
        # Reduction visualization
        st.markdown('<div class="section-header">Return Rate Reduction Impact</div>', unsafe_allow_html=True)
        
        # Create waterfall chart to show the reduction journey
        waterfall_fig = go.Figure(go.Waterfall(
            name = "Return Rate Analysis",
            orientation = "v",
            measure = ["absolute", "relative", "relative", "total"],
            x = ["Current Return Rate", "Expected Reduction", "Confidence Adjustment", "Adjusted Reduction"],
            textposition = "outside",
            text = [
                f"{results['current_metrics']['return_rate_30d']:.2f}%",
                f"-{results['solution_metrics']['expected_reduction']:.1f}%",
                f"+{(results['solution_metrics']['expected_reduction'] - results['solution_metrics']['adjusted_reduction']):.1f}%",
                f"{results['current_metrics']['return_rate_30d'] - results['solution_metrics']['adjusted_reduction']:.2f}%"
            ],
            y = [
                results["current_metrics"]["return_rate_30d"], 
                -results["solution_metrics"]["expected_reduction"],
                (results["solution_metrics"]["expected_reduction"] - results["solution_metrics"]["adjusted_reduction"]),
                0  # Total is calculated automatically
            ],
            connector = {"line":{"color": TEXT_SECONDARY}},
            decreasing = {"marker":{"color": SUCCESS_COLOR}},
            increasing = {"marker":{"color": DANGER_COLOR}},
            totals = {"marker":{"color": SECONDARY_COLOR}}
        ))
        
        waterfall_fig.update_layout(
            title = "Return Rate Reduction Analysis",
            showlegend = False,
            height = 400,
            margin=dict(l=20, r=20, t=40, b=20),
        )
        
        st.plotly_chart(waterfall_fig, use_container_width=True)
        
    # Add collapse button
    if st.button("Hide Details", key=f"collapse_{results['sku']}"):
        return False
    
    st.markdown('</div>', unsafe_allow_html=True)
    return True

def chat_with_ai(results, issue_description, chat_history=None):
    """Initialize or continue chat with AI about quality issues"""
    if chat_history is None:
        chat_history = []
            
        # Generate initial AI analysis using direct API call
        system_prompt = f"""
        You are a Quality Management expert for medical devices at Vive Health. You analyze quality issues, provide insights on cost-benefit analyses, and suggest solutions.
        
        Product details:
        - SKU: {results["sku"]}
        - Type: {results["product_type"]}
        - Issue: {issue_description}
        
        Metrics:
        - Return Rate (30 days): {results["current_metrics"]["return_rate_30d"]:.2f}%
        - Monthly Return Cost: ${results["financial_impact"]["monthly_return_cost"]:.2f}
        - Estimated Savings: ${results["financial_impact"]["estimated_monthly_savings"]:.2f}/month
        - Payback Period: {results["financial_impact"]["payback_months"]:.1f} months
        - Medical Risk Level: {results["medical_device_impact"]["risk_level"]}
        - Regulatory Impact: {results["medical_device_impact"]["regulatory_impact"]}
        
        Recommendation: {results["recommendation"]}
        
        As a Quality Management expert for medical devices, you should provide specific, actionable insights about this quality issue.
        Focus on:
        1. Root cause analysis of the issue
        2. Practical solutions to fix the quality problem
        3. Implementation recommendations considering FDA/regulatory requirements for medical devices
        4. Risk assessment for the proposed solution
        5. Quality management systems (QMS) considerations
        
        Keep your analysis concise, practical, and specific to medical devices and FDA/regulatory requirements.
        """
        
        initial_message = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": "Based on the product information and quality metrics, provide your initial analysis of the issue and suggested next steps. Be specific about potential fixes, quality improvements, and regulatory considerations."}
        ]
        
        initial_analysis = call_openai_api(initial_message)
        
        # Add initial AI message
        chat_history.append({
            "role": "assistant",
            "content": initial_analysis
        })
    
    return chat_history

def import_batch_data(uploaded_file):
    """Process uploaded CSV or Excel file for batch analysis"""
    try:
        file_name = uploaded_file.name
        file_extension = file_name.split('.')[-1].lower()
        
        if file_extension == 'csv':
            df = pd.read_csv(uploaded_file)
        elif file_extension in ['xlsx', 'xls']:
            df = pd.read_excel(uploaded_file)
        else:
            st.error(f"Unsupported file format: {file_extension}. Please upload CSV or Excel file.")
            return None
        
        # Check for required columns
        required_columns = ['sku', 'product_type', 'sales_30d', 'returns_30d', 
                          'issue_description', 'current_unit_cost', 'fix_cost_upfront', 
                          'fix_cost_per_unit', 'sales_price', 'expected_reduction', 
                          'solution_confidence']
        
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            st.error(f"Missing required columns: {', '.join(missing_columns)}")
            return None
        
        return df
    
    except Exception as e:
        st.error(f"Error processing file: {str(e)}")
        return None

def batch_analyze_quality_issues(df):
    """Run quality analysis on batch data"""
    results = []
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    total_rows = len(df)
    
    for i, row in df.iterrows():
        try:
            # Update progress
            progress = int((i / total_rows) * 100)
            progress_bar.progress(progress)
            status_text.text(f"Analyzing {i+1}/{total_rows}: {row['sku']}")
            
            # Extract values with defaults for optional fields
            analysis_result = analyze_quality_issue(
                sku=row['sku'],
                product_type=row['product_type'],
                sales_30d=row['sales_30d'],
                returns_30d=row['returns_30d'],
                issue_description=row['issue_description'],
                current_unit_cost=row['current_unit_cost'],
                fix_cost_upfront=row['fix_cost_upfront'],
                fix_cost_per_unit=row['fix_cost_per_unit'],
                sales_price=row['sales_price'],
                expected_reduction=row['expected_reduction'],
                solution_confidence=row['solution_confidence'],
                new_sales_price=row.get('new_sales_price', None),
                asin=row.get('asin', None),
                ncx_rate=row.get('ncx_rate', None),
                sales_365d=row.get('sales_365d', None),
                returns_365d=row.get('returns_365d', None),
                star_rating=row.get('star_rating', None),
                total_reviews=row.get('total_reviews', None),
                fba_fee=row.get('fba_fee', None),
                risk_level=row.get('risk_level', 'Medium'),
                regulatory_impact=row.get('regulatory_impact', 'None')
            )
            
            results.append(analysis_result)
        
        except Exception as e:
            # Add error result
            results.append({
                "sku": row.get('sku', f"Row {i+1}"),
                "error": str(e),
                "recommendation": "Analysis Error",
                "recommendation_class": "recommendation-low"
            })
    
    # Complete progress
    progress_bar.progress(100)
    status_text.text(f"Analysis complete: {total_rows} products analyzed")
    
    return results

def display_help_page():
    """Display the help page with formula explanations"""
    st.markdown('<div class="main-header">Product Profitability Formula Guide</div>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="alert alert-info">
        This guide explains all the formulas and calculations used in the Product Profitability Analysis tool.
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown('<div class="help-section">', unsafe_allow_html=True)
    st.markdown('<div class="help-title">Basic Metrics</div>', unsafe_allow_html=True)
    
    st.markdown("""
    <p>These fundamental metrics form the basis of our quality analysis:</p>
    
    <p><strong>Return Rate</strong>: The percentage of units returned relative to total sales in a given period.</p>
    <div class="formula-box">Return Rate (%) = (Returns / Sales) × 100</div>
    
    <p><strong>Monthly Return Cost</strong>: The direct financial impact of returns based on unit cost.</p>
    <div class="formula-box">Monthly Return Cost ($) = Returns × Unit Cost</div>
    
    <p><strong>Monthly Revenue</strong>: Total revenue from product sales in the period.</p>
    <div class="formula-box">Monthly Revenue ($) = Sales × Sales Price</div>
    
    <p><strong>Profit Margin</strong>: The percentage of profit relative to revenue.</p>
    <div class="formula-box">Profit Margin (%) = ((Sales × Sales Price) - (Sales × Unit Cost)) / (Sales × Sales Price) × 100</div>
    """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="help-section">', unsafe_allow_html=True)
    st.markdown('<div class="help-title">Solution Impact Calculations</div>', unsafe_allow_html=True)
    
    st.markdown("""
    <p>These calculations estimate the impact of implementing a quality fix:</p>
    
    <p><strong>Adjusted Reduction</strong>: Expected return rate reduction adjusted by confidence level.</p>
    <div class="formula-box">Adjusted Reduction (%) = Expected Reduction (%) × (Solution Confidence (%) / 100)</div>
    
    <p><strong>Reduced Returns</strong>: Estimated number of returns after implementing the fix.</p>
    <div class="formula-box">Reduced Returns = Current Returns × (1 - (Adjusted Reduction / 100))</div>
    
    <p><strong>Returns Saved</strong>: Estimated number of returns prevented by the fix.</p>
    <div class="formula-box">Returns Saved = Current Returns - Reduced Returns</div>
    
    <p><strong>Estimated Monthly Savings</strong>: Financial impact of reduced returns.</p>
    <div class="formula-box">Monthly Savings ($) = Returns Saved × Current Unit Cost</div>
    """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="help-section">', unsafe_allow_html=True)
    st.markdown('<div class="help-title">Price Change Impact</div>', unsafe_allow_html=True)
    
    st.markdown("""
    <p>If the product price changes after the fix, these calculations apply:</p>
    
    <p><strong>New Monthly Revenue</strong>: Revenue after implementing sales price change.</p>
    <div class="formula-box">New Monthly Revenue ($) = Sales × New Sales Price</div>
    
    <p><strong>Revenue Change</strong>: The difference in monthly revenue due to price change.</p>
    <div class="formula-box">Revenue Change ($) = New Monthly Revenue - Current Monthly Revenue</div>
    
    <p><strong>New Unit Cost</strong>: Unit cost after adding the per-unit fix cost.</p>
    <div class="formula-box">New Unit Cost ($) = Current Unit Cost + Fix Cost Per Unit</div>
    
    <p><strong>New Profit Margin</strong>: Profit margin after implementing the fix and price changes.</p>
    <div class="formula-box">New Profit Margin (%) = ((Sales × New Sales Price) - (Sales × New Unit Cost)) / (Sales × New Sales Price) × 100</div>
    """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="help-section">', unsafe_allow_html=True)
    st.markdown('<div class="help-title">ROI Calculations</div>', unsafe_allow_html=True)
    
    st.markdown("""
    <p>These calculations determine the financial return on investment:</p>
    
    <p><strong>Total Monthly Benefit</strong>: Combined impact of return savings and revenue changes.</p>
    <div class="formula-box">Total Monthly Benefit ($) = Monthly Savings + Revenue Change</div>
    
    <p><strong>Payback Period</strong>: Time required to recover the initial investment.</p>
    <div class="formula-box">Payback Period (months) = Fix Cost Upfront / Total Monthly Benefit</div>
    
    <p><strong>Annual Projections</strong>: Yearly projections of financial impact.</p>
    <div class="formula-box">Annual Return Cost ($) = Monthly Return Cost × 12
Annual Savings ($) = Monthly Savings × 12
Annual Revenue Change ($) = Monthly Revenue Change × 12</div>
    
    <p><strong>3-Year ROI</strong>: Return on investment over a 3-year period.</p>
    <div class="formula-box">Projected Sales (36 months) = Monthly Sales × 36
Total Investment ($) = Fix Cost Upfront + (Fix Cost Per Unit × Projected Sales)
Total Savings ($) = (Annual Savings + Annual Revenue Change) × 3
ROI (%) = ((Total Savings - Total Investment) / Total Investment) × 100</div>
    """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="help-section">', unsafe_allow_html=True)
    st.markdown('<div class="help-title">Tariff & Import Cost Calculations</div>', unsafe_allow_html=True)
    
    st.markdown("""
    <p>These calculations determine the impact of tariffs and import costs:</p>
    
    <p><strong>Tariff Amount</strong>: The duty applied to the manufacturing cost of imported goods.</p>
    <div class="formula-box">Tariff Amount ($) = Manufacturing Cost ($) × (Tariff Rate (%) / 100)</div>
    
    <p><strong>Per-Unit Import Costs</strong>: Additional import costs distributed across units.</p>
    <div class="formula-box">Per-Unit Import Cost ($) = Total Import Costs ($) / Units per Shipment</div>
    
    <p><strong>Landed Cost</strong>: Total cost per unit including manufacturing, tariffs, and import costs.</p>
    <div class="formula-box">Landed Cost ($) = Manufacturing Cost + Tariff Amount + Shipping/Unit + Storage/Unit + Customs/Unit + Broker/Unit + Other/Unit</div>
    
    <p><strong>Profit After Import</strong>: Profit per unit after all import costs.</p>
    <div class="formula-box">Profit ($) = Retail Price - Landed Cost</div>
    
    <p><strong>Profit Margin After Import</strong>: Margin percentage considering all import costs.</p>
    <div class="formula-box">Margin (%) = (Profit / Retail Price) × 100</div>
    """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="help-section">', unsafe_allow_html=True)
    st.markdown('<div class="help-title">Marketing ROI Calculations</div>', unsafe_allow_html=True)
    
    st.markdown("""
    <p>These calculations help evaluate the impact of marketing spend changes:</p>
    
    <p><strong>Sales Volume Change</strong>: Expected change in sales volume after price/marketing changes.</p>
    <div class="formula-box">New Sales Volume = Current Sales × (1 + Estimated Sales Change (as decimal))</div>
    
    <p><strong>Marketing ROI</strong>: Return on investment for increased marketing spend.</p>
    <div class="formula-box">Marketing ROI (%) = (Net Profit Change / Ad Spend Change) × 100</div>
    
    <p><strong>Break-even Additional Units</strong>: Number of additional units needed to break even on increased ad spend.</p>
    <div class="formula-box">Breakeven Additional Units = Ad Spend Change / Unit Profit</div>
    
    <p><strong>Monthly Profit Impact</strong>: Change in monthly profit from marketing and price changes.</p>
    <div class="formula-box">Monthly Profit Change = New Profit After Ads - Current Profit After Ads</div>
    """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Navigation back to main page
    if st.button("Return to Analysis Tool"):
        navigate_to_page("analysis")

#------------------------------------
# TARIFF ANALYSIS FUNCTIONS
#------------------------------------
def calculate_landed_cost(msrp, cost_to_produce, tariff_rate, shipping_cost=0, storage_cost=0, customs_fee=0, 
                         broker_fee=0, other_costs=0, units_per_shipment=1):
    """Calculate the landed cost and profitability with given tariff rate"""
    
    # Calculate per unit costs
    if units_per_shipment <= 0:
        units_per_shipment = 1
    
    shipping_per_unit = shipping_cost / units_per_shipment
    storage_per_unit = storage_cost / units_per_shipment
    customs_per_unit = customs_fee / units_per_shipment
    broker_per_unit = broker_fee / units_per_shipment
    other_per_unit = other_costs / units_per_shipment
    
    # Calculate tariff amount
    tariff_amount = cost_to_produce * (tariff_rate / 100)
    
    # Calculate total landed cost per unit
    landed_cost = cost_to_produce + tariff_amount + shipping_per_unit + storage_per_unit + customs_per_unit + broker_per_unit + other_per_unit
    
    # Calculate profit and margin
    profit = msrp - landed_cost
    margin_percentage = (profit / msrp) * 100 if msrp > 0 else 0
    
    # Calculate minimum profitable MSRP
    min_profitable_msrp = landed_cost * 1.01  # Minimum 1% profit margin
    
    # Breakeven price
    breakeven_price = landed_cost
    
    return {
        "landed_cost": landed_cost,
        "tariff_amount": tariff_amount,
        "profit": profit,
        "margin_percentage": margin_percentage,
        "min_profitable_msrp": min_profitable_msrp,
        "breakeven_price": breakeven_price,
        "cost_breakdown": {
            "production": cost_to_produce,
            "tariff": tariff_amount,
            "shipping": shipping_per_unit,
            "storage": storage_per_unit,
            "customs": customs_per_unit,
            "broker": broker_per_unit,
            "other": other_per_unit
        }
    }

def generate_tariff_scenarios(base_msrp, cost_to_produce, min_tariff=0, max_tariff=100, steps=10, 
                             shipping_cost=0, storage_cost=0, customs_fee=0, broker_fee=0, 
                             other_costs=0, units_per_shipment=1):
    """Generate scenarios for different tariff rates"""
    
    tariff_rates = np.linspace(min_tariff, max_tariff, steps)
    scenarios = []
    
    for rate in tariff_rates:
        result = calculate_landed_cost(
            base_msrp, cost_to_produce, rate, shipping_cost, storage_cost, 
            customs_fee, broker_fee, other_costs, units_per_shipment
        )
        
        scenarios.append({
            "tariff_rate": rate,
            "landed_cost": result["landed_cost"],
            "profit": result["profit"],
            "margin": result["margin_percentage"],
            "breakeven_price": result["breakeven_price"]
        })
    
    return pd.DataFrame(scenarios)

def generate_price_scenarios(tariff_rate, cost_to_produce, min_price_factor=0.8, max_price_factor=2.0, steps=10,
                            shipping_cost=0, storage_cost=0, customs_fee=0, broker_fee=0, 
                            other_costs=0, units_per_shipment=1):
    """Generate scenarios for different price points at a fixed tariff rate"""
    
    # Calculate base landed cost without MSRP
    base_result = calculate_landed_cost(
        100, cost_to_produce, tariff_rate, shipping_cost, storage_cost, 
        customs_fee, broker_fee, other_costs, units_per_shipment
    )
    base_landed_cost = base_result["landed_cost"]
    
    # Generate price range from min_price_factor to max_price_factor of landed cost
    min_price = base_landed_cost * min_price_factor
    max_price = base_landed_cost * max_price_factor
    
    price_points = np.linspace(min_price, max_price, steps)
    scenarios = []
    
    for price in price_points:
        result = calculate_landed_cost(
            price, cost_to_produce, tariff_rate, shipping_cost, storage_cost, 
            customs_fee, broker_fee, other_costs, units_per_shipment
        )
        
        scenarios.append({
            "msrp": price,
            "profit": result["profit"],
            "margin": result["margin_percentage"],
            "landed_cost": result["landed_cost"]
        })
    
    return pd.DataFrame(scenarios)

def calculate_ad_roi(current_price, proposed_price, current_ad_spend, proposed_ad_spend, 
                   current_sales_qty, estimated_sales_change, current_return_rate=0, 
                   expected_return_rate=None, cost_to_produce=0, tariff_rate=0):
    """
    Calculate ROI for advertising spend change
    
    Args:
        current_price: Current sales price per unit
        proposed_price: Proposed new sales price per unit
        current_ad_spend: Current monthly ad spend
        proposed_ad_spend: Proposed monthly ad spend
        current_sales_qty: Current monthly sales quantity
        estimated_sales_change: Expected percentage change in sales volume (decimal: 0.1 = 10% increase)
        current_return_rate: Current product return rate (decimal)
        expected_return_rate: Expected product return rate after changes (decimal)
        cost_to_produce: Cost to produce each unit
        tariff_rate: Import tariff rate percentage
    
    Returns:
        Dict with ROI metrics
    """
    # Set expected return rate to current if not provided
    if expected_return_rate is None:
        expected_return_rate = current_return_rate
    
    # Calculate current metrics
    current_sales_dollars = current_sales_qty * current_price
    current_returns_qty = current_sales_qty * current_return_rate
    current_net_sales_qty = current_sales_qty - current_returns_qty
    current_net_sales_dollars = current_net_sales_qty * current_price
    
    # Calculate tariff cost per unit
    tariff_cost_per_unit = cost_to_produce * (tariff_rate / 100) if cost_to_produce > 0 else 0
    
    # Calculate current profit (excluding ad spend)
    unit_profit = current_price - cost_to_produce - tariff_cost_per_unit
    current_profit_before_ads = unit_profit * current_net_sales_qty
    current_profit_after_ads = current_profit_before_ads - current_ad_spend
    
    # Calculate estimated new sales
    new_sales_qty = current_sales_qty * (1 + estimated_sales_change)
    new_returns_qty = new_sales_qty * expected_return_rate
    new_net_sales_qty = new_sales_qty - new_returns_qty
    new_unit_profit = proposed_price - cost_to_produce - tariff_cost_per_unit
    
    # Calculate new sales dollars
    new_sales_dollars = new_sales_qty * proposed_price
    new_net_sales_dollars = new_net_sales_qty * proposed_price
    
    # Calculate new profit
    new_profit_before_ads = new_unit_profit * new_net_sales_qty
    new_profit_after_ads = new_profit_before_ads - proposed_ad_spend
    
    # Calculate changes
    sales_qty_change = new_sales_qty - current_sales_qty
    sales_dollars_change = new_sales_dollars - current_sales_dollars
    net_profit_change = new_profit_after_ads - current_profit_after_ads
    ad_spend_change = proposed_ad_spend - current_ad_spend
    
    # Calculate ROI metrics
    if ad_spend_change > 0:
        roi_percentage = (net_profit_change / ad_spend_change) * 100 if ad_spend_change != 0 else 0
    else:
        roi_percentage = 0  # If ad spend decreased, traditional ROI doesn't apply
    
    # Calculate breakeven metrics for ad spend
    if new_unit_profit > 0:
        breakeven_additional_units = ad_spend_change / new_unit_profit if ad_spend_change > 0 else 0
        breakeven_sales_change = breakeven_additional_units / current_sales_qty if current_sales_qty > 0 else 0
    else:
        breakeven_additional_units = float('inf')
        breakeven_sales_change = float('inf')
    
    return {
        # Current metrics
        "current_price": current_price,
        "current_ad_spend": current_ad_spend,
        "current_sales_qty": current_sales_qty,
        "current_sales_dollars": current_sales_dollars,
        "current_return_rate": current_return_rate,
        "current_returns_qty": current_returns_qty,
        "current_net_sales_qty": current_net_sales_qty,
        "current_net_sales_dollars": current_net_sales_dollars,
        "current_profit_before_ads": current_profit_before_ads,
        "current_profit_after_ads": current_profit_after_ads,
        
        # New metrics
        "proposed_price": proposed_price,
        "proposed_ad_spend": proposed_ad_spend,
        "estimated_sales_change": estimated_sales_change,
        "expected_return_rate": expected_return_rate,
        "new_sales_qty": new_sales_qty,
        "new_sales_dollars": new_sales_dollars,
        "new_returns_qty": new_returns_qty,
        "new_net_sales_qty": new_net_sales_qty,
        "new_net_sales_dollars": new_net_sales_dollars,
        "new_profit_before_ads": new_profit_before_ads,
        "new_profit_after_ads": new_profit_after_ads,
        
        # Changes
        "sales_qty_change": sales_qty_change,
        "sales_dollars_change": sales_dollars_change,
        "ad_spend_change": ad_spend_change,
        "net_profit_change": net_profit_change,
        
        # ROI metrics
        "roi_percentage": roi_percentage,
        "breakeven_additional_units": breakeven_additional_units,
        "breakeven_sales_change": breakeven_sales_change,
        
        # Unit economics
        "unit_profit": unit_profit,
        "new_unit_profit": new_unit_profit
    }

def generate_ad_spend_scenarios(current_price, proposed_price, current_ad_spend, 
                             current_sales_qty, estimated_sales_change_per_ad_dollar,
                             min_spend_factor=0.5, max_spend_factor=3.0, steps=10,
                             current_return_rate=0, expected_return_rate=None,
                             cost_to_produce=0, tariff_rate=0):
    """Generate scenarios for different ad spend levels"""
    
    # Generate range of ad spend values
    min_spend = current_ad_spend * min_spend_factor
    max_spend = current_ad_spend * max_spend_factor
    ad_spend_points = np.linspace(min_spend, max_spend, steps)
    
    scenarios = []
    
    for ad_spend in ad_spend_points:
        # Estimate sales change based on ad spend change
        ad_spend_change_factor = (ad_spend - current_ad_spend) / current_ad_spend if current_ad_spend > 0 else 0
        estimated_sales_change = ad_spend_change_factor * estimated_sales_change_per_ad_dollar
        
        # Calculate ROI
        result = calculate_ad_roi(
            current_price, proposed_price, current_ad_spend, ad_spend, 
            current_sales_qty, estimated_sales_change, current_return_rate, 
            expected_return_rate, cost_to_produce, tariff_rate
        )
        
        scenarios.append({
            "ad_spend": ad_spend,
            "sales_qty": result["new_sales_qty"],
            "sales_dollars": result["new_sales_dollars"],
            "profit": result["new_profit_after_ads"],
            "profit_change": result["net_profit_change"],
            "roi": result["roi_percentage"]
        })
    
    return pd.DataFrame(scenarios)

#------------------------------------
# MONTE CARLO SIMULATION FUNCTIONS
#------------------------------------
def run_monte_carlo_simulation(
    scenario, 
    num_simulations=1000, 
    param_variations=None):
    """
    Run Monte Carlo simulation for a quality issue analysis scenario
    
    Args:
        scenario: Base scenario data dictionary
        num_simulations: Number of simulation runs
        param_variations: Dict with parameters and their variation ranges as percentages
                e.g. {'reduction_rate': 15, 'additional_cost_per_item': 10}
                means reduction_rate can vary by ±15%, additional cost by ±10%
                
    Returns:
        Tuple of (results_dataframe, message)
    """
    if not scenario:
        return None, "Scenario not found"
    
    # Set default parameter variations if not provided
    if not param_variations:
        param_variations = {
            'expected_reduction': 20,        # ±20% variation
            'fix_cost_per_unit': 15,         # ±15% variation
            'fix_cost_upfront': 10,          # ±10% variation
            'sales_30d': 5,                  # ±5% variation
            'returns_30d': 10,               # ±10% variation
            'sales_price': 5                 # ±5% variation
        }
    
    # Initialize results dataframe
    results = pd.DataFrame(columns=[
        'simulation_id', 'expected_reduction', 'fix_cost_per_unit', 'fix_cost_upfront',
        'sales_30d', 'returns_30d', 'sales_price', 'roi_3yr', 'payback_months', 
        'net_benefit', 'annual_savings', 'annual_additional_costs'
    ])
    
    # Run simulations
    for i in range(num_simulations):
        try:
            # Create variations of parameters
            sim_expected_reduction = max(0, scenario['solution_metrics']['expected_reduction'] * np.random.uniform(
                1 - param_variations['expected_reduction']/100,
                1 + param_variations['expected_reduction']/100
            ))
            
            sim_fix_cost_per_unit = max(0, scenario['financial_impact']['fix_cost_per_unit'] * np.random.uniform(
                1 - param_variations['fix_cost_per_unit']/100,
                1 + param_variations['fix_cost_per_unit']/100
            ))
            
            sim_fix_cost_upfront = max(0, scenario['financial_impact']['fix_cost_upfront'] * np.random.uniform(
                1 - param_variations['fix_cost_upfront']/100,
                1 + param_variations['fix_cost_upfront']/100
            ))
            
            sim_sales_30d = max(1, scenario['sales_30d'] if 'sales_30d' in scenario else 
                               scenario['financial_impact']['monthly_return_cost'] / 
                               (scenario['current_metrics']['return_rate_30d'] / 100 * 
                                scenario['financial_impact']['current_unit_cost']))
            sim_sales_30d *= np.random.uniform(
                1 - param_variations['sales_30d']/100,
                1 + param_variations['sales_30d']/100
            )
            
            sim_returns_30d = scenario['returns_30d'] if 'returns_30d' in scenario else scenario['financial_impact']['monthly_return_cost'] / scenario['financial_impact']['current_unit_cost']
            sim_returns_30d = min(sim_sales_30d, max(0, sim_returns_30d * np.random.uniform(
                1 - param_variations['returns_30d']/100,
                1 + param_variations['returns_30d']/100
            )))
            
            sim_sales_price = max(scenario['financial_impact']['current_unit_cost'], scenario['current_metrics']['sales_price'] * np.random.uniform(
                1 - param_variations['sales_price']/100,
                1 + param_variations['sales_price']/100
            ))
            
            # Run quality analysis with simulated values
            sim_result = analyze_quality_issue(
                sku=scenario['sku'],
                product_type=scenario['product_type'],
                sales_30d=sim_sales_30d,
                returns_30d=sim_returns_30d,
                issue_description=scenario['issue_description'],
                current_unit_cost=scenario['financial_impact']['current_unit_cost'],
                fix_cost_upfront=sim_fix_cost_upfront,
                fix_cost_per_unit=sim_fix_cost_per_unit,
                sales_price=sim_sales_price,
                expected_reduction=sim_expected_reduction,
                solution_confidence=scenario['solution_metrics']['solution_confidence'],
                risk_level=scenario['medical_device_impact']['risk_level'],
                regulatory_impact=scenario['medical_device_impact']['regulatory_impact']
            )
            
            # Extract key metrics from simulation result
            results.loc[i] = [
                i, 
                sim_expected_reduction,
                sim_fix_cost_per_unit,
                sim_fix_cost_upfront,
                sim_sales_30d,
                sim_returns_30d,
                sim_sales_price,
                sim_result['financial_impact']['roi_3yr'],
                sim_result['financial_impact']['payback_months'],
                sim_result['financial_impact']['annual_savings'] * 3 - sim_result['financial_impact']['total_investment'],
                sim_result['financial_impact']['annual_savings'],
                sim_fix_cost_per_unit * sim_sales_30d * 12  # Annual additional costs
            ]
            
        except Exception as e:
            # Log error and continue with next simulation
            print(f"Error in simulation {i}: {str(e)}")
            continue
    
    return results, "Simulation completed successfully"

#------------------------------------
# UI COMPONENTS
#------------------------------------

def display_landed_cost_calculator():
    """Display the landed cost calculator UI"""
    st.markdown("<h2 class='sub-header'>Import Cost Calculator</h2>", unsafe_allow_html=True)
    
    # Create two columns for basic product info
    col1, col2 = st.columns(2)
    
    with col1:
        product_name = st.text_input(
            "Product Name", 
            value="",
            help="Enter the full product name or description. Example: 'Premium Knee Brace'"
        )
        
        sku = st.text_input(
            "Product SKU", 
            value="",
            help="Enter the Stock Keeping Unit (unique identifier). Example: 'KB-2025-BLK'"
        )
        
        msrp = st.number_input(
            "MSRP / Retail Price ($)", 
            min_value=0.01, 
            value=100.00, 
            step=0.01,
            help="Manufacturer's Suggested Retail Price - the price at which you plan to sell the product. Enter in USD."
        )
    
    with col2:
        cost_to_produce = st.number_input(
            "Manufacturing Cost per Unit ($)", 
            min_value=0.01, 
            value=50.00, 
            step=0.01,
            help="The direct cost to produce or acquire each unit before any import costs or tariffs. Enter in USD."
        )
        
        tariff_rate = st.slider(
            "Tariff Rate (%)", 
            min_value=0, 
            max_value=100, 
            value=25, 
            step=1,
            help="Import duty rate as a percentage of the manufacturing cost. Varies by product category and country of origin."
        )
        
        currency = st.selectbox(
            "Currency", 
            options=["USD", "EUR", "GBP", "CAD", "AUD"], 
            index=0,
            help="Select the currency for all monetary values. Calculations will be performed in the selected currency."
        )
    
    # Optional import costs section with expander
    with st.expander("Additional Import Costs (Optional)", expanded=False):
        st.info("These costs will be distributed across the number of units in your shipment. All fields are optional.")
        
        col3, col4 = st.columns(2)
        
        with col3:
            shipping_cost = st.number_input(
                "Shipping Cost per Shipment ($)", 
                min_value=0.0, 
                value=1000.0, 
                step=10.0,
                help="Total cost to ship an entire container or shipment. Example: ocean freight, air freight, insurance."
            )
            
            storage_cost = st.number_input(
                "Storage/Warehousing Cost ($)", 
                min_value=0.0, 
                value=0.0, 
                step=10.0,
                help="Costs for warehousing or storage associated with this shipment. Example: receiving fees, monthly storage fees."
            )
            
            customs_fee = st.number_input(
                "Customs Processing Fee ($)", 
                min_value=0.0, 
                value=250.0, 
                step=10.0,
                help="Fees charged by customs for processing your shipment. Example: MPF (Merchandise Processing Fee), HMF (Harbor Maintenance Fee)."
            )
        
        with col4:
            broker_fee = st.number_input(
                "Customs Broker Fee ($)", 
                min_value=0.0, 
                value=150.0, 
                step=10.0,
                help="Fees paid to customs brokers for handling import documentation. Example: entry preparation, ISF filing."
            )
            
            other_costs = st.number_input(
                "Other Import Costs ($)", 
                min_value=0.0, 
                value=0.0, 
                step=10.0,
                help="Any other import-related costs not covered by the categories above. Example: compliance testing, labeling, inspection fees."
            )
            
            units_per_shipment = st.number_input(
                "Units per Shipment", 
                min_value=1, 
                value=1000, 
                step=10,
                help="The total number of product units in this shipment. Used to calculate per-unit costs from total shipment costs."
            )
    
    # Calculate button
    calculate_button = st.button(
        "Calculate Import Costs",
        help="Click to calculate landed cost, profit margin, and breakeven price based on the inputs above."
    )
    
    if calculate_button:
        with st.spinner("Calculating..."):
            # Perform calculation
            result = calculate_landed_cost(
                msrp, cost_to_produce, tariff_rate, shipping_cost, storage_cost,
                customs_fee, broker_fee, other_costs, units_per_shipment
            )
            
            # Add calculation to history
            calculation_entry = {
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "product": product_name if product_name else "Unnamed Product",
                "sku": sku if sku else "No SKU",
                "msrp": msrp,
                "cost": cost_to_produce,
                "tariff_rate": tariff_rate,
                "landed_cost": result['landed_cost'],
                "profit": result['profit'],
                "margin": result['margin_percentage']
            }
            
            if 'tariff_calculations' not in st.session_state:
                st.session_state.tariff_calculations = []
            
            st.session_state.tariff_calculations.append(calculation_entry)
            
            # Display results
            st.markdown("<h2 class='sub-header'>Calculation Results</h2>", unsafe_allow_html=True)
            
            # Create metrics layout
            col7, col8, col9, col10 = st.columns(4)
            
            with col7:
                st.markdown(f"""
                <div class='metric-card'>
                    <p class='metric-label'>Landed Cost</p>
                    <p class='metric-value'>${result['landed_cost']:.2f}</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col8:
                st.markdown(f"""
                <div class='metric-card'>
                    <p class='metric-label'>Tariff Amount</p>
                    <p class='metric-value'>${result['tariff_amount']:.2f}</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col9:
                profit_color = SUCCESS_COLOR if result['profit'] > 0 else DANGER_COLOR
                st.markdown(f"""
                <div class='metric-card'>
                    <p class='metric-label'>Profit per Unit</p>
                    <p class='metric-value' style='color: {profit_color}'>${result['profit']:.2f}</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col10:
                margin_color = SUCCESS_COLOR if result['margin_percentage'] > 15 else (WARNING_COLOR if result['margin_percentage'] > 0 else DANGER_COLOR)
                st.markdown(f"""
                <div class='metric-card'>
                    <p class='metric-label'>Profit Margin</p>
                    <p class='metric-value' style='color: {margin_color}'>{result['margin_percentage']:.1f}%</p>
                </div>
                """, unsafe_allow_html=True)
            
            # Display breakeven and profitability
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("<h2 class='sub-header'>Calculation Results</h2>", unsafe_allow_html=True)
            
            # Create metrics layout
            col11, col12 = st.columns(2)
            
            with col11:
                st.markdown(f"""
                <div class='result-box'>
                    <h3>Breakeven Price: ${result['breakeven_price']:.2f}</h3>
                    <p>At this selling price, you will neither make a profit nor incur a loss after covering all costs including tariffs.</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col12:
                # Display minimum profitable MSRP
                st.markdown(f"""
                <div class='result-box'>
                    <h3>Minimum Profitable MSRP: ${result['min_profitable_msrp']:.2f}</h3>
                    <p>This is the minimum price needed to maintain at least a 1% profit margin after all costs and tariffs.</p>
                </div>
                """, unsafe_allow_html=True)
            
            # Create cost breakdown
            st.markdown("<h3 class='sub-header'>Cost Breakdown</h3>", unsafe_allow_html=True)
            
            # Create a pie chart for cost breakdown
            cost_data = {
                'Component': ['Production', 'Tariff', 'Shipping', 'Storage', 'Customs', 'Broker', 'Other'],
                'Amount': [
                    result['cost_breakdown']['production'],
                    result['cost_breakdown']['tariff'],
                    result['cost_breakdown']['shipping'],
                    result['cost_breakdown']['storage'],
                    result['cost_breakdown']['customs'],
                    result['cost_breakdown']['broker'],
                    result['cost_breakdown']['other']
                ]
            }
            
            cost_df = pd.DataFrame(cost_data)
            
            # Filter out zero values
            cost_df = cost_df[cost_df['Amount'] > 0]
            
            # Create pie chart
            fig = px.pie(
                cost_df, 
                values='Amount', 
                names='Component',
                title='Cost Component Breakdown per Unit',
                color_discrete_sequence=px.colors.sequential.Blues_r,
                hole=0.4,
            )
            
            fig.update_layout(
                legend=dict(orientation="h", yanchor="bottom", y=-0.3, xanchor="center", x=0.5),
                margin=dict(l=20, r=20, t=40, b=20),
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Create a waterfall chart to show how costs build up to landed cost
            components = ['Production Cost', 'Tariff', 'Shipping', 'Storage', 'Customs', 'Broker', 'Other', 'Final Landed Cost']
            measures = ['absolute'] + ['relative'] * 6 + ['total']
            
            waterfall_values = [
                result['cost_breakdown']['production'],
                result['cost_breakdown']['tariff'],
                result['cost_breakdown']['shipping'],
                result['cost_breakdown']['storage'],
                result['cost_breakdown']['customs'],
                result['cost_breakdown']['broker'],
                result['cost_breakdown']['other'],
                0  # This will be calculated as the total
            ]
            
            waterfall_fig = go.Figure(go.Waterfall(
                name="Cost Breakdown",
                orientation="v",
                measure=measures,
                x=components,
                textposition="outside",
                y=waterfall_values,
                connector={"line": {"color": TEXT_SECONDARY}},
                increasing={"marker": {"color": SECONDARY_COLOR}},
                totals={"marker": {"color": PRIMARY_COLOR}}
            ))
            
            waterfall_fig.update_layout(
                title="Build-up to Landed Cost",
                showlegend=False,
                height=400,
                margin=dict(l=20, r=20, t=40, b=20),
            )
            
            st.plotly_chart(waterfall_fig, use_container_width=True)

# Create scenario analysis section
if st.checkbox("Run Tariff Rate Scenario Analysis", key="tariff_scenario_check"):
    st.markdown("<h3 class='sub-header'>Tariff Rate Scenario Analysis</h3>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        min_tariff = st.number_input("Minimum Tariff Rate (%)", min_value=0, max_value=100, value=0)
    
    with col2:
        max_tariff = st.number_input("Maximum Tariff Rate (%)", min_value=0, max_value=100, value=50)
    
    with col3:
        tariff_steps = st.number_input("Number of Steps", min_value=2, max_value=20, value=10)
    
    if st.button("Generate Tariff Scenarios"):
        with st.spinner("Analyzing tariff scenarios..."):
            # Generate scenarios
            scenarios_df = generate_tariff_scenarios(
                msrp, cost_to_produce, min_tariff, max_tariff, tariff_steps,
                shipping_cost, storage_cost, customs_fee, broker_fee,
                other_costs, units_per_shipment
            )
#create plot
fig = make_subplots(specs=[[{"secondary_y": True}]])
                        
                        # Add landed cost line
                        fig.add_trace(
                            go.Scatter(
                                x=scenarios_df['tariff_rate'],
                                y=scenarios_df['landed_cost'],
                                name="Landed Cost",
                                line=dict(color=PRIMARY_COLOR, width=3)
                            )
                        )
                        
                        # Add profit line on secondary axis
                        fig.add_trace(
                            go.Scatter(
                                x=scenarios_df['tariff_rate'],
                                y=scenarios_df['profit'],
                                name="Profit per Unit",
                                line=dict(color=SUCCESS_COLOR, width=3, dash='dot')
                            ),
                            secondary_y=True
                        )
                        
                        # Add margin line on secondary axis
                        fig.add_trace(
                            go.Scatter(
                                x=scenarios_df['tariff_rate'],
                                y=scenarios_df['margin'],
                                name="Margin %",
                                line=dict(color=WARNING_COLOR, width=3, dash='dot')
                            ),
                            secondary_y=True
                        )
                        
                        # Add breakeven line
                        fig.add_trace(
                            go.Scatter(
                                x=scenarios_df['tariff_rate'],
                                y=scenarios_df['breakeven_price'],
                                name="Breakeven Price",
                                line=dict(color=DANGER_COLOR, width=2, dash='dash')
                            )
                        )
                        
                        # Mark the current tariff rate
                        fig.add_vline(
                            x=tariff_rate,
                            line_dash="solid",
                            line_color=TEXT_SECONDARY,
                            annotation_text=f"Current Rate ({tariff_rate}%)",
                            annotation_position="top right"
                        )
                        
                        # Mark zero profit line
                        fig.add_hline(
                            y=0,
                            line_dash="dash",
                            line_color=DANGER_COLOR,
                            secondary_y=True,
                            annotation_text="Break-even",
                            annotation_position="bottom right"
                        )
                        
                        # Update layout
                        fig.update_layout(
                            title_text="Impact of Tariff Rates on Cost, Price, and Profitability",
                            xaxis_title="Tariff Rate (%)",
                            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                            margin=dict(l=20, r=20, t=40, b=20),
                            height=500
                        )
                        
                        # Update y-axes
                        fig.update_yaxes(title_text="Cost & Price ($)", secondary_y=False)
                        fig.update_yaxes(title_text="Profit ($) & Margin (%)", secondary_y=True)
                        
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Display data in table
                        with st.expander("View Scenario Data Table", expanded=False):
                            st.write("Tariff Rate Impact Data:")
                            formatted_df = scenarios_df.copy()
                            formatted_df['tariff_rate'] = formatted_df['tariff_rate'].apply(lambda x: f"{x:.1f}%")
                            formatted_df['landed_cost'] = formatted_df['landed_cost'].apply(lambda x: f"${x:.2f}")
                            formatted_df['profit'] = formatted_df['profit'].apply(lambda x: f"${x:.2f}")
                            formatted_df['margin'] = formatted_df['margin'].apply(lambda x: f"{x:.1f}%")
                            formatted_df['breakeven_price'] = formatted_df['breakeven_price'].apply(lambda x: f"${x:.2f}")
                            
                            st.dataframe(formatted_df, use_container_width=True)
                            
                            # Download link for CSV
                            csv = scenarios_df.to_csv(index=False)
                            b64 = base64.b64encode(csv.encode()).decode()
                            href = f'<a href="data:file/csv;base64,{b64}" download="tariff_scenarios.csv" class="download-link">Download Scenario Data as CSV</a>'
                            st.markdown(href, unsafe_allow_html=True)
            
            # Create price scenario analysis section
            if st.checkbox("Run Price Scenario Analysis", key="price_scenario_check"):
                st.markdown("<h3 class='sub-header'>Price Scenario Analysis</h3>", unsafe_allow_html=True)
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    min_price_factor = st.slider("Minimum Price Factor", min_value=0.5, max_value=1.0, value=0.8, step=0.05,
                                               help="Minimum price as a factor of landed cost (e.g., 0.8 = 80% of landed cost)")
                
                with col2:
                    max_price_factor = st.slider("Maximum Price Factor", min_value=1.0, max_value=3.0, value=2.0, step=0.1,
                                               help="Maximum price as a factor of landed cost (e.g., 2.0 = 200% of landed cost)")
                
                with col3:
                    price_steps = st.number_input("Number of Price Points", min_value=2, max_value=20, value=10)
                
                if st.button("Generate Price Scenarios"):
                    with st.spinner("Analyzing price scenarios..."):
                        # Generate scenarios
                        price_scenarios_df = generate_price_scenarios(
                            tariff_rate, cost_to_produce, min_price_factor, max_price_factor, price_steps,
                            shipping_cost, storage_cost, customs_fee, broker_fee,
                            other_costs, units_per_shipment
                        )
                        
                        # Create plot
                        price_fig = px.line(
                            price_scenarios_df,
                            x="msrp",
                            y=["profit", "margin"],
                            labels={"msrp": "Selling Price ($)", "value": "Value", "variable": "Metric"},
                            title="Profit and Margin at Different Price Points",
                            color_discrete_map={"profit": SUCCESS_COLOR, "margin": WARNING_COLOR}
                        )
                        
                        # Add vertical line for current MSRP
                        price_fig.add_vline(
                            x=msrp,
                            line_dash="solid",
                            line_color=TEXT_SECONDARY,
                            annotation_text=f"Current MSRP (${msrp:.2f})",
                            annotation_position="top right"
                        )
                        
                        # Add horizontal line for zero profit
                        price_fig.add_hline(
                            y=0,
                            line_dash="dash",
                            line_color=DANGER_COLOR,
                            annotation_text="Break-even",
                            annotation_position="bottom right"
                        )
                        
                        # Add landed cost vertical line
                        price_fig.add_vline(
                            x=result['landed_cost'],
                            line_dash="dash",
                            line_color=DANGER_COLOR,
                            annotation_text=f"Landed Cost (${result['landed_cost']:.2f})",
                            annotation_position="bottom left"
                        )
                        
                        # Update layout
                        price_fig.update_layout(
                            xaxis_title="Selling Price ($)",
                            yaxis_title="Value",
                            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                            margin=dict(l=20, r=20, t=40, b=20),
                            height=500
                        )
                        
                        st.plotly_chart(price_fig, use_container_width=True)
                        
                        # Create profit margin heat map
                        profit_margin_df = price_scenarios_df.copy()
                        
                        # Add profit margin categories
                        def categorize_margin(margin):
                            if margin <= 0:
                                return "Loss"
                            elif margin < 15:
                                return "Low"
                            elif margin < 30:
                                return "Medium"
                            else:
                                return "High"
                        
                        profit_margin_df['margin_category'] = profit_margin_df['margin'].apply(categorize_margin)
                        
                        # Create color map
                        color_map = {
                            "Loss": DANGER_COLOR,
                            "Low": WARNING_COLOR,
                            "Medium": SECONDARY_COLOR,
                            "High": SUCCESS_COLOR
                        }
                        
                        # Create margin category chart
                        margin_fig = px.bar(
                            profit_margin_df,
                            x="msrp",
                            y="profit",
                            color="margin_category",
                            labels={"msrp": "Selling Price ($)", "profit": "Profit per Unit ($)", "margin_category": "Margin Category"},
                            title="Profit by Selling Price and Margin Category",
                            color_discrete_map=color_map
                        )
                        
                        # Add current MSRP line
                        margin_fig.add_vline(
                            x=msrp,
                            line_dash="solid",
                            line_color=TEXT_SECONDARY,
                            annotation_text=f"Current MSRP (${msrp:.2f})",
                            annotation_position="top right"
                        )
                        
                        # Update layout
                        margin_fig.update_layout(
                            xaxis_title="Selling Price ($)",
                            yaxis_title="Profit per Unit ($)",
                            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                            margin=dict(l=20, r=20, t=40, b=20),
                            height=400
                        )
                        
                        st.plotly_chart(margin_fig, use_container_width=True)
                        
                        # Display data in table
                        with st.expander("View Price Scenario Data Table", expanded=False):
                            st.write("Price Impact Data:")
                            formatted_price_df = price_scenarios_df.copy()
                            formatted_price_df['msrp'] = formatted_price_df['msrp'].apply(lambda x: f"${x:.2f}")
                            formatted_price_df['profit'] = formatted_price_df['profit'].apply(lambda x: f"${x:.2f}")
                            formatted_price_df['margin'] = formatted_price_df['margin'].apply(lambda x: f"{x:.1f}%")
                            formatted_price_df['landed_cost'] = formatted_price_df['landed_cost'].apply(lambda x: f"${x:.2f}")
                            
                            st.dataframe(formatted_price_df, use_container_width=True)
                            
                            # Download link for CSV
                            price_csv = price_scenarios_df.to_csv(index=False)
                            price_b64 = base64.b64encode(price_csv.encode()).decode()
                            price_href = f'<a href="data:file/csv;base64,{price_b64}" download="price_scenarios.csv" class="download-link">Download Price Scenario Data as CSV</a>'
                            st.markdown(price_href, unsafe_allow_html=True)

def calculate_ad_roi_ui():
    """Display the advertising ROI calculator UI"""
    st.markdown("<h2 class='sub-header'>Marketing ROI Calculator</h2>", unsafe_allow_html=True)
    
    # Create two columns for basic inputs
    col1, col2 = st.columns(2)
    
    with col1:
        current_price = st.number_input(
            "Current Sales Price ($)", 
            min_value=0.01, 
            value=100.00, 
            step=0.01,
            help="Current selling price per unit"
        )
        
        proposed_price = st.number_input(
            "Proposed Sales Price ($)", 
            min_value=0.0, 
            value=0.0, 
            step=0.01,
            help="Proposed new selling price per unit (leave at 0 to keep current price)"
        )
        
        if proposed_price <= 0:
            proposed_price = current_price
        
        current_ad_spend = st.number_input(
            "Current Monthly Ad Spend ($)", 
            min_value=0.0, 
            value=1000.0, 
            step=100.0,
            help="Current monthly advertising spend"
        )
        
        proposed_ad_spend = st.number_input(
            "Proposed Monthly Ad Spend ($)", 
            min_value=0.0, 
            value=2000.0, 
            step=100.0,
            help="Proposed monthly advertising spend"
        )
    
    with col2:
        current_sales_qty = st.number_input(
            "Current Monthly Sales Quantity", 
            min_value=1, 
            value=100, 
            step=1,
            help="Number of units sold per month at current ad spend"
        )
        
        estimated_sales_change = st.slider(
            "Estimated Sales Volume Change (%)", 
            min_value=-50.0, 
            max_value=200.0, 
            value=20.0, 
            step=5.0,
            help="Estimated percentage change in sales volume after ad spend change"
        ) / 100.0  # Convert to decimal
        
        current_return_rate = st.slider(
            "Current Return Rate (%)", 
            min_value=0.0, 
            max_value=50.0, 
            value=5.0, 
            step=1.0,
            help="Current product return rate percentage"
        ) / 100.0  # Convert to decimal
        
        expected_return_rate = st.slider(
            "Expected Return Rate After Changes (%)", 
            min_value=0.0, 
            max_value=50.0, 
            value=5.0, 
            step=1.0,
            help="Expected return rate after implementing price and marketing changes"
        ) / 100.0  # Convert to decimal
    
    # Optional production cost inputs
    with st.expander("Production Cost Details (Optional)", expanded=False):
        col3, col4 = st.columns(2)
        
        with col3:
            cost_to_produce = st.number_input(
                "Cost to Produce per Unit ($)", 
                min_value=0.0, 
                value=50.0, 
                step=1.0,
                help="Cost to produce or acquire each unit before any other costs"
            )
            
            tariff_rate = st.slider(
                "Import Tariff Rate (%)", 
                min_value=0.0, 
                max_value=100.0, 
                value=25.0, 
                step=1.0,
                help="Import duty rate as a percentage of the production cost, if applicable"
            )
    
    # Calculate button
    calculate_button = st.button(
        "Calculate Marketing ROI",
        help="Click to calculate ROI and financial impact of ad spend and price changes"
    )
    
    if calculate_button:
        with st.spinner("Calculating..."):
            # Perform calculation
            result = calculate_ad_roi(
                current_price=current_price,
                proposed_price=proposed_price,
                current_ad_spend=current_ad_spend,
                proposed_ad_spend=proposed_ad_spend,
                current_sales_qty=current_sales_qty,
                estimated_sales_change=estimated_sales_change,
                current_return_rate=current_return_rate,
                expected_return_rate=expected_return_rate,
                cost_to_produce=cost_to_produce,
                tariff_rate=tariff_rate
            )
            
            # Display results
            st.markdown("<h2 class='sub-header'>Marketing ROI Analysis Results</h2>", unsafe_allow_html=True)
            
            # Key metrics
            col5, col6, col7, col8 = st.columns(4)
            
            with col5:
                roi_color = SUCCESS_COLOR if result['roi_percentage'] > 0 else DANGER_COLOR
                st.markdown(f"""
                <div class='metric-card'>
                    <p class='metric-label'>ROI on Ad Spend Change</p>
                    <p class='metric-value' style='color:{roi_color}'>{result['roi_percentage']:.1f}%</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col6:
                profit_change_color = SUCCESS_COLOR if result['net_profit_change'] > 0 else DANGER_COLOR
                st.markdown(f"""
                <div class='metric-card'>
                    <p class='metric-label'>Monthly Profit Change</p>
                    <p class='metric-value' style='color:{profit_change_color}'>${result['net_profit_change']:.2f}</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col7:
                sales_color = SUCCESS_COLOR if result['sales_qty_change'] > 0 else DANGER_COLOR
                st.markdown(f"""
                <div class='metric-card'>
                    <p class='metric-label'>Sales Volume Change</p>
                    <p class='metric-value' style='color:{sales_color}'>{result['sales_qty_change']:.0f} units</p>
                    <p style='font-size:0.8rem;'>({estimated_sales_change * 100:.1f}%)</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col8:
                st.markdown(f"""
                <div class='metric-card'>
                    <p class='metric-label'>Breakeven Additional Units</p>
                    <p class='metric-value'>{result['breakeven_additional_units']:.1f}</p>
                    <p style='font-size:0.8rem;'>({result['breakeven_sales_change'] * 100:.1f}% increase)</p>
                </div>
                """, unsafe_allow_html=True)
            
            # Comparison table
            st.markdown("<h3>Current vs. Proposed Scenario</h3>", unsafe_allow_html=True)
            
            comparison_data = {
                'Metric': [
                    'Monthly Ad Spend',
                    'Sales Price',
                    'Sales Quantity',
                    'Sales Revenue',
                    'Return Rate',
                    'Returns Quantity',
                    'Net Sales Quantity',
                    'Net Sales Revenue',
                    'Profit Before Ads',
                    'Profit After Ads'
                ],
                'Current': [
                    f"${result['current_ad_spend']:.2f}",
                    f"${result['current_price']:.2f}",
                    f"{result['current_sales_qty']:.0f}",
                    f"${result['current_sales_dollars']:.2f}",
                    f"{result['current_return_rate'] * 100:.1f}%",
                    f"{result['current_returns_qty']:.1f}",
                    f"{result['current_net_sales_qty']:.1f}",
                    f"${result['current_net_sales_dollars']:.2f}",
                    f"${result['current_profit_before_ads']:.2f}",
                    f"${result['current_profit_after_ads']:.2f}"
                ],
                'Proposed': [
                    f"${result['proposed_ad_spend']:.2f}",
                    f"${result['proposed_price']:.2f}",
                    f"{result['new_sales_qty']:.0f}",
                    f"${result['new_sales_dollars']:.2f}",
                    f"{result['expected_return_rate'] * 100:.1f}%",
                    f"{result['new_returns_qty']:.1f}",
                    f"{result['new_net_sales_qty']:.1f}",
                    f"${result['new_net_sales_dollars']:.2f}",
                    f"${result['new_profit_before_ads']:.2f}",
                    f"${result['new_profit_after_ads']:.2f}"
                ],
                'Change': [
                    f"${result['ad_spend_change']:.2f}",
                    f"${proposed_price - current_price:.2f}",
                    f"{result['sales_qty_change']:.0f}",
                    f"${result['sales_dollars_change']:.2f}",
                    f"{(result['expected_return_rate'] - result['current_return_rate']) * 100:.1f}%",
                    f"{result['new_returns_qty'] - result['current_returns_qty']:.1f}",
                    f"{result['new_net_sales_qty'] - result['current_net_sales_qty']:.1f}",
                    f"${result['new_net_sales_dollars'] - result['current_net_sales_dollars']:.2f}",
                    f"${result['new_profit_before_ads'] - result['current_profit_before_ads']:.2f}",
                    f"${result['net_profit_change']:.2f}"
                ]
            }
            
            comparison_df = pd.DataFrame(comparison_data)
            st.dataframe(comparison_df, use_container_width=True, hide_index=True)
            
            # Create visualization
            st.markdown("<h3>Visual Analysis</h3>", unsafe_allow_html=True)
            
            # Create bar chart comparison
            fig = go.Figure()
            
            # Add current scenario
            fig.add_trace(go.Bar(
                name='Current Scenario',
                x=['Ad Spend', 'Net Sales', 'Profit After Ads'],
                y=[
                    result['current_ad_spend'],
                    result['current_net_sales_dollars'],
                    result['current_profit_after_ads']
                ],
                marker_color=SECONDARY_COLOR
            ))
            
            # Add proposed scenario
            fig.add_trace(go.Bar(
                name='Proposed Scenario',
                x=['Ad Spend', 'Net Sales', 'Profit After Ads'],
                y=[
                    result['proposed_ad_spend'],
                    result['new_net_sales_dollars'],
                    result['new_profit_after_ads']
                ],
                marker_color=PRIMARY_COLOR
            ))
            
            # Update layout
            fig.update_layout(
                title='Financial Comparison: Current vs. Proposed',
                barmode='group',
                yaxis_title='Amount ($)',
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="center",
                    x=0.5
                ),
                height=400,
                margin=dict(l=20, r=20, t=40, b=20)
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Create ROI visualization
            roi_fig = go.Figure()
            
            # Add waterfall chart
            roi_fig.add_trace(go.Waterfall(
                name='ROI Analysis',
                orientation='v',
                measure=['absolute', 'relative', 'relative', 'total'],
                x=['Current Profit', 'Ad Spend Change', 'Revenue Change', 'New Profit'],
                textposition='outside',
                text=[
                    f"${result['current_profit_after_ads']:.2f}",
                    f"${result['ad_spend_change']:.2f}",
                    f"${result['new_profit_before_ads'] - result['current_profit_before_ads']:.2f}",
                    f"${result['new_profit_after_ads']:.2f}"
                ],
                y=[
                    result['current_profit_after_ads'],
                    result['ad_spend_change'],
                    result['new_profit_before_ads'] - result['current_profit_before_ads'],
                    0
                ],
                connector={'line': {'color': TEXT_SECONDARY}},
                decreasing={'marker': {'color': DANGER_COLOR}},
                increasing={'marker': {'color': SUCCESS_COLOR}},
                totals={'marker': {'color': PRIMARY_COLOR}}
            ))
            
            # Update layout
            roi_fig.update_layout(
                title='Profit Waterfall: Impact of Ad Spend and Revenue Changes',
                yaxis_title='Amount ($)',
                showlegend=False,
                height=400,
                margin=dict(l=20, r=20, t=40, b=20)
            )
            
            st.plotly_chart(roi_fig, use_container_width=True)
            
            # Run ad spend sensitivity analysis
            if st.checkbox("Run Ad Spend Sensitivity Analysis", key="ad_sensitivity"):
                st.markdown("<h3 class='sub-header'>Ad Spend Sensitivity Analysis</h3>", unsafe_allow_html=True)
                
                col9, col10, col11 = st.columns(3)
                
                with col9:
                    min_spend_factor = st.slider(
                        "Minimum Ad Spend Factor", 
                        min_value=0.1, 
                        max_value=1.0, 
                        value=0.5, 
                        step=0.1,
                        help="Minimum ad spend as a factor of current spend (e.g., 0.5 = 50% of current spend)"
                    )
                
                with col10:
                    max_spend_factor = st.slider(
                        "Maximum Ad Spend Factor", 
                        min_value=1.0, 
                        max_value=5.0, 
                        value=3.0, 
                        step=0.25,
                        help="Maximum ad spend as a factor of current spend (e.g., 3.0 = 300% of current spend)"
                    )
                
                with col11:
                    spend_steps = st.number_input(
                        "Number of Spend Points", 
                        min_value=5, 
                        max_value=20, 
                        value=10
                    )
                    
                    sales_change_per_ad_dollar = st.number_input(
                        "Est. Sales % Change per Ad Spend % Change", 
                        min_value=0.0, 
                        max_value=2.0, 
                        value=0.8, 
                        step=0.1,
                        help="Estimated sales percentage change for each percentage of ad spend change (elasticity)"
                    )
                
                if st.button("Generate Ad Spend Scenarios"):
                    with st.spinner("Analyzing ad spend scenarios..."):
                        # Generate scenarios
                        ad_scenarios_df = generate_ad_spend_scenarios(
                            current_price=current_price,
                            proposed_price=proposed_price,
                            current_ad_spend=current_ad_spend,
                            current_sales_qty=current_sales_qty,
                            estimated_sales_change_per_ad_dollar=sales_change_per_ad_dollar,
                            min_spend_factor=min_spend_factor,
                            max_spend_factor=max_spend_factor,
                            steps=spend_steps,
                            current_return_rate=current_return_rate,
                            expected_return_rate=expected_return_rate,
                            cost_to_produce=cost_to_produce,
                            tariff_rate=tariff_rate
                        )
                        
                        # Create line chart
                        ad_fig = make_subplots(specs=[[{"secondary_y": True}]])
                        
                        # Add profit line
                        ad_fig.add_trace(
                            go.Scatter(
                                x=ad_scenarios_df['ad_spend'],
                                y=ad_scenarios_df['profit'],
                                name="Profit After Ads",
                                line=dict(color=SUCCESS_COLOR, width=3)
                            ),
                            secondary_y=False
                        )
                        
                        # Add ROI line on secondary axis
                        ad_fig.add_trace(
                            go.Scatter(
                                x=ad_scenarios_df['ad_spend'],
                                y=ad_scenarios_df['roi'],
                                name="ROI %",
                                line=dict(color=PRIMARY_COLOR, width=3, dash='dot')
                            ),
                            secondary_y=True
                        )
                        
                        # Mark the current ad spend
                        ad_fig.add_vline(
                            x=current_ad_spend,
                            line_dash="dash",
                            line_color=TEXT_SECONDARY,
                            annotation_text=f"Current (${current_ad_spend})",
                            annotation_position="top right"
                        )
                        
                        # Mark the proposed ad spend
                        ad_fig.add_vline(
                            x=proposed_ad_spend,
                            line_dash="solid",
                            line_color=TEXT_PRIMARY,
                            annotation_text=f"Proposed (${proposed_ad_spend})",
                            annotation_position="top left"
                        )
                        
                        # Update layout
                        ad_fig.update_layout(
                            title_text="Impact of Ad Spend on Profit and ROI",
                            xaxis_title="Monthly Ad Spend ($)",
                            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                            margin=dict(l=20, r=20, t=40, b=20),
                            height=500
                        )
                        
                        # Update y-axes
                        ad_fig.update_yaxes(title_text="Monthly Profit ($)", secondary_y=False)
                        ad_fig.update_yaxes(title_text="ROI (%)", secondary_y=True)
                        
                        st.plotly_chart(ad_fig, use_container_width=True)
                        
                        # Add scatter plot of sales vs. ad spend
                        sales_fig = px.scatter(
                            ad_scenarios_df,
                            x="ad_spend",
                            y="sales_qty",
                            size="profit",
                            color="profit_change",
                            hover_name="ad_spend",
                            labels={
                                "ad_spend": "Monthly Ad Spend ($)",
                                "sales_qty": "Monthly Sales Volume",
                                "profit": "Monthly Profit ($)",
                                "profit_change": "Profit Change vs. Current ($)"
                            },
                            title="Sales Volume vs. Ad Spend with Profit Indicators",
                            color_continuous_scale=px.colors.sequential.Blugrn,
                        )
                        
                        # Mark the current ad spend
                        sales_fig.add_vline(
                            x=current_ad_spend,
                            line_dash="dash",
                            line_color=TEXT_SECONDARY,
                            annotation_text=f"Current (${current_ad_spend})",
                            annotation_position="top right"
                        )
                        
                        # Mark the proposed ad spend
                        sales_fig.add_vline(
                            x=proposed_ad_spend,
                            line_dash="solid",
                            line_color=TEXT_PRIMARY,
                            annotation_text=f"Proposed (${proposed_ad_spend})",
                            annotation_position="top left"
                        )
                        
                        # Update layout
                        sales_fig.update_layout(
                            margin=dict(l=20, r=20, t=40, b=20),
                            height=500
                        )
                        
                        st.plotly_chart(sales_fig, use_container_width=True)
                        
                        # Display data in table
                        with st.expander("View Ad Spend Scenario Data Table", expanded=False):
                            st.write("Ad Spend Impact Data:")
                            formatted_ad_df = ad_scenarios_df.copy()
                            formatted_ad_df['ad_spend'] = formatted_ad_df['ad_spend'].apply(lambda x: f"${x:.2f}")
                            formatted_ad_df['sales_qty'] = formatted_ad_df['sales_qty'].apply(lambda x: f"{x:.0f}")
                            formatted_ad_df['sales_dollars'] = formatted_ad_df['sales_dollars'].apply(lambda x: f"${x:.2f}")
                            formatted_ad_df['profit'] = formatted_ad_df['profit'].apply(lambda x: f"${x:.2f}")
                            formatted_ad_df['profit_change'] = formatted_ad_df['profit_change'].apply(lambda x: f"${x:.2f}")
                            formatted_ad_df['roi'] = formatted_ad_df['roi'].apply(lambda x: f"{x:.1f}%")
                            
                            st.dataframe(formatted_ad_df, use_container_width=True)
                            
                            # Download link for CSV
                            ad_csv = ad_scenarios_df.to_csv(index=False)
                            ad_b64 = base64.b64encode(ad_csv.encode()).decode()
                            ad_href = f'<a href="data:file/csv;base64,{ad_b64}" download="ad_spend_scenarios.csv" class="download-link">Download Ad Spend Scenario Data as CSV</a>'
                            st.markdown(ad_href, unsafe_allow_html=True)

def run_monte_carlo_simulation_ui():
    """Display UI for Monte Carlo simulation of quality issue analysis"""
    st.markdown("<h2 class='sub-header'>Monte Carlo Risk Simulation</h2>", unsafe_allow_html=True)
    
    # Instructions
    st.markdown("""
    <div class="info-box">
        Run Monte Carlo simulations to model uncertainty in quality issue fixes. 
        This allows you to better understand the range of possible outcomes and the probability 
        of achieving your ROI targets.
    </div>
    """, unsafe_allow_html=True)
    
    # Check if there are quality analysis results available
    if st.session_state.quality_analysis_results:
        # Use the current analysis results
        current_scenario = st.session_state.quality_analysis_results
        
        # Display scenario info
        st.markdown(f"""
        <div class="form-section">
            <div class="form-section-header">Base Scenario: {current_scenario['sku']}</div>
            <p>This simulation will use your current analysis as the base scenario.</p>
            <ul>
                <li><strong>Product:</strong> {current_scenario['sku']}</li>
                <li><strong>Return Rate:</strong> {current_scenario['current_metrics']['return_rate_30d']:.2f}%</li>
                <li><strong>Expected Reduction:</strong> {current_scenario['solution_metrics']['expected_reduction']:.1f}%</li>
                <li><strong>Solution Confidence:</strong> {current_scenario['solution_metrics']['solution_confidence']:.1f}%</li>
                <li><strong>Fix Cost Upfront:</strong> ${current_scenario['financial_impact']['fix_cost_upfront']:.2f}</li>
                <li><strong>Additional Cost Per Unit:</strong> ${current_scenario['financial_impact']['fix_cost_per_unit']:.2f}</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        # Simulation parameters
        st.markdown('<div class="form-section-header">Simulation Parameters</div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            num_simulations = st.slider(
                "Number of Simulations", 
                min_value=100, 
                max_value=10000, 
                value=1000, 
                step=100,
                help="Higher number of simulations gives more accurate results but takes longer to process"
            )
        
        with col2:
            confidence_level = st.slider(
                "Confidence Level for Results (%)", 
                min_value=50, 
                max_value=99, 
                value=90, 
                step=1,
                help="Confidence level for calculating result ranges"
            )
        
        # Parameter variation settings
        st.markdown('<div class="form-section-header">Parameter Variation Ranges</div>', unsafe_allow_html=True)
        st.markdown("""
        <p>Set the percentage by which each parameter can vary in the simulation. For example, 
        a 20% variation for Expected Reduction means the actual reduction could be up to 20% higher or lower 
        than your estimate.</p>
        """, unsafe_allow_html=True)
        
        col3, col4, col5 = st.columns(3)
        
        with col3:
            expected_reduction_var = st.slider(
                "Expected Reduction Variation (%)", 
                min_value=0, 
                max_value=50, 
                value=20, 
                step=5,
                help="How much the expected return rate reduction can vary in the simulation"
            )
            
            fix_cost_upfront_var = st.slider(
                "Fix Cost Upfront Variation (%)", 
                min_value=0, 
                max_value=50, 
                value=10, 
                step=5,
                help="How much the upfront fix cost can vary in the simulation"
            )
        
        with col4:
            fix_cost_per_unit_var = st.slider(
                "Cost Per Unit Variation (%)", 
                min_value=0, 
                max_value=50, 
                value=15, 
                step=5,
                help="How much the additional cost per unit can vary in the simulation"
            )
            
            sales_30d_var = st.slider(
                "Sales Volume Variation (%)", 
                min_value=0, 
                max_value=30, 
                value=5, 
                step=5,
                help="How much the monthly sales volume can vary in the simulation"
            )
        
        with col5:
            returns_30d_var = st.slider(
                "Returns Variation (%)", 
                min_value=0, 
                max_value=30, 
                value=10, 
                step=5,
                help="How much the monthly returns can vary in the simulation"
            )
            
            sales_price_var = st.slider(
                "Sales Price Variation (%)", 
                min_value=0, 
                max_value=30, 
                value=5, 
                step=5,
                help="How much the sales price can vary in the simulation"
            )
        
        # Create variation parameters dictionary
        param_variations = {
            'expected_reduction': expected_reduction_var,
            'fix_cost_per_unit': fix_cost_per_unit_var,
            'fix_cost_upfront': fix_cost_upfront_var,
            'sales_30d': sales_30d_var,
            'returns_30d': returns_30d_var,
            'sales_price': sales_price_var
        }
        
        # Run simulation button
        if st.button("Run Monte Carlo Simulation"):
            with st.spinner(f"Running {num_simulations} simulations..."):
                # Run the simulation
                simulation_results, message = run_monte_carlo_simulation(
                    scenario=current_scenario,
                    num_simulations=num_simulations,
                    param_variations=param_variations
                )
                
                if simulation_results is not None:
                    # Store results in session state
                    st.session_state.monte_carlo_scenario = {
                        'base_scenario': current_scenario,
                        'results': simulation_results,
                        'num_simulations': num_simulations,
                        'confidence_level': confidence_level,
                        'param_variations': param_variations
                    }
                    
                    # Display results
                    st.success(f"Simulation complete: {num_simulations} iterations processed successfully.")
                    st.rerun()
                else:
                    st.error(f"Simulation failed: {message}")
    
    # Display simulation results if available
    if 'monte_carlo_scenario' in st.session_state and st.session_state.monte_carlo_scenario:
        sim_data = st.session_state.monte_carlo_scenario
        results_df = sim_data['results']
        
        st.markdown("<h2 class='sub-header'>Monte Carlo Simulation Results</h2>", unsafe_allow_html=True)
        
        # Calculate key metrics
        roi_mean = results_df['roi_3yr'].mean()
        roi_median = results_df['roi_3yr'].median()
        roi_std = results_df['roi_3yr'].std()
        payback_mean = results_df['payback_months'].mean()
        payback_median = results_df['payback_months'].median()
        
        # Calculate probability of positive ROI
        positive_roi_prob = (results_df['roi_3yr'] > 0).mean() * 100
        
        # Calculate confidence intervals
        alpha = (100 - sim_data['confidence_level']) / 100
        roi_lower = results_df['roi_3yr'].quantile(alpha/2)
        roi_upper = results_df['roi_3yr'].quantile(1-alpha/2)
        payback_lower = results_df['payback_months'].quantile(alpha/2)
        payback_upper = results_df['payback_months'].quantile(1-alpha/2)
        
        # Display summary metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(f"""
            <div class='metric-card'>
                <p class='metric-label'>Mean ROI (3-Year)</p>
                <p class='metric-value' style='color:{SUCCESS_COLOR if roi_mean > 0 else DANGER_COLOR}'>{roi_mean:.1f}%</p>
                <p style='font-size:0.85rem;'>Median: {roi_median:.1f}%</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            roi_ci_color = SUCCESS_COLOR if roi_lower > 0 else (WARNING_COLOR if roi_upper > 0 else DANGER_COLOR)
            st.markdown(f"""
            <div class='metric-card'>
                <p class='metric-label'>{sim_data['confidence_level']}% Confidence Interval (ROI)</p>
                <p class='metric-value' style='color:{roi_ci_color}'>{roi_lower:.1f}% to {roi_upper:.1f}%</p>
                <p style='font-size:0.85rem;'>Standard Deviation: {roi_std:.1f}%</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div class='metric-card'>
                <p class='metric-label'>Mean Payback Period</p>
                <p class='metric-value'>{payback_mean:.1f} months</p>
                <p style='font-size:0.85rem;'>Median: {payback_median:.1f} months</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            st.markdown(f"""
            <div class='metric-card'>
                <p class='metric-label'>Probability of Positive ROI</p>
                <p class='metric-value' style='color:{SUCCESS_COLOR if positive_roi_prob > 80 else (WARNING_COLOR if positive_roi_prob > 50 else DANGER_COLOR)}'>{positive_roi_prob:.1f}%</p>
                <p style='font-size:0.85rem;'>Based on {sim_data['num_simulations']} simulations</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Create ROI distribution histogram
        roi_hist_fig = px.histogram(
            results_df, 
            x='roi_3yr',
            marginal='box',
            nbins=30,
            title='Distribution of 3-Year ROI Results',
            labels={'roi_3yr': 'ROI (%)'},
            color_discrete_sequence=[PRIMARY_COLOR]
        )
        
        # Add vertical line at 0% ROI
        roi_hist_fig.add_vline(
            x=0, 
            line_dash="dash", 
            line_color=DANGER_COLOR,
            annotation_text="Break-even",
            annotation_position="top right"
        )
        
        # Add vertical line at base scenario ROI
        roi_hist_fig.add_vline(
            x=sim_data['base_scenario']['financial_impact']['roi_3yr'], 
            line_dash="solid", 
            line_color=TEXT_PRIMARY,
            annotation_text="Base Scenario",
            annotation_position="top left"
        )
        
        # Update layout
        roi_hist_fig.update_layout(
            xaxis_title="3-Year ROI (%)",
            yaxis_title="Number of Simulations",
            showlegend=False,
            margin=dict(l=20, r=20, t=40, b=20),
            height=400
        )
        
        st.plotly_chart(roi_hist_fig, use_container_width=True)
        
        # Create payback period distribution
        filtered_payback = results_df[results_df['payback_months'] < 100]  # Filter out extreme values for better visualization
        
        payback_hist_fig = px.histogram(
            filtered_payback, 
            x='payback_months',
            marginal='box',
            nbins=30,
            title='Distribution of Payback Period Results',
            labels={'payback_months': 'Payback Period (months)'},
            color_discrete_sequence=[SECONDARY_COLOR]
        )
        
        # Add vertical line at 12 months payback
        payback_hist_fig.add_vline(
            x=12, 
            line_dash="dash", 
            line_color=WARNING_COLOR,
            annotation_text="1 Year",
            annotation_position="top right"
        )
        
        # Add vertical line at base scenario payback
        payback_hist_fig.add_vline(
            x=sim_data['base_scenario']['financial_impact']['payback_months'], 
            line_dash="solid", 
            line_color=TEXT_PRIMARY,
            annotation_text="Base Scenario",
            annotation_position="top left"
        )
        
        # Update layout
        payback_hist_fig.update_layout(
            xaxis_title="Payback Period (months)",
            yaxis_title="Number of Simulations",
            showlegend=False,
            margin=dict(l=20, r=20, t=40, b=20),
            height=400
        )
        
        st.plotly_chart(payback_hist_fig, use_container_width=True)
        
        # Create scatterplot of key relationships
        scatter_fig = px.scatter(
            results_df,
            x='expected_reduction',
            y='roi_3yr',
            color='payback_months',
            size='net_benefit',
            hover_data=['fix_cost_upfront', 'fix_cost_per_unit', 'annual_savings'],
            labels={
                'expected_reduction': 'Expected Reduction (%)',
                'roi_3yr': 'ROI (%)',
                'payback_months': 'Payback Period (months)',
                'net_benefit': 'Net Benefit ($)',
                'fix_cost_upfront': 'Upfront Cost ($)',
                'fix_cost_per_unit': 'Cost per Unit ($)',
                'annual_savings': 'Annual Savings ($)'
            },
            title='Relationship Between Reduction Rate, ROI, and Payback Period',
            color_continuous_scale=px.colors.sequential.Viridis_r
        )
        
        # Update layout
        scatter_fig.update_layout(
            xaxis_title="Expected Return Rate Reduction (%)",
            yaxis_title="3-Year ROI (%)",
            margin=dict(l=20, r=20, t=40, b=20),
            height=500
        )
        
        st.plotly_chart(scatter_fig, use_container_width=True)
        
        # Create heatmap of parameter sensitivities
        st.markdown("<h3>Parameter Sensitivity Analysis</h3>", unsafe_allow_html=True)
        
        # Calculate correlations
        correlation_columns = ['expected_reduction', 'fix_cost_per_unit', 'fix_cost_upfront', 
                              'sales_30d', 'returns_30d', 'sales_price', 'roi_3yr', 'payback_months']
        correlation_df = results_df[correlation_columns].corr()
        
        # Extract correlations with ROI and payback
        roi_correlations = correlation_df['roi_3yr'].drop(['roi_3yr', 'payback_months'])
        payback_correlations = correlation_df['payback_months'].drop(['roi_3yr', 'payback_months'])
        
        # Combine into a new dataframe
        sensitivity_df = pd.DataFrame({
            'Parameter': roi_correlations.index,
            'ROI Impact': roi_correlations.values,
            'Payback Impact': payback_correlations.values
        })
        
        # Sort by absolute ROI impact
        sensitivity_df = sensitivity_df.reindex(sensitivity_df['ROI Impact'].abs().sort_values(ascending=False).index)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Create bar chart for ROI sensitivity
            roi_sens_fig = px.bar(
                sensitivity_df,
                y='Parameter',
                x='ROI Impact',
                orientation='h',
                title='Parameter Sensitivity Impact on ROI',
                color='ROI Impact',
                color_continuous_scale=px.colors.sequential.Blues,
                labels={'ROI Impact': 'Correlation with ROI', 'Parameter': ''}
            )
            
            # Update layout
            roi_sens_fig.update_layout(
                xaxis_title="Correlation Coefficient",
                margin=dict(l=20, r=20, t=40, b=20),
                height=350
            )
            
            st.plotly_chart(roi_sens_fig, use_container_width=True)
        
        with col2:
            # Create bar chart for payback sensitivity
            payback_sens_fig = px.bar(
                sensitivity_df,
                y='Parameter',
                x='Payback Impact',
                orientation='h',
                title='Parameter Sensitivity Impact on Payback Period',
                color='Payback Impact',
                color_continuous_scale=px.colors.sequential.Blues_r,
                labels={'Payback Impact': 'Correlation with Payback', 'Parameter': ''}
            )
            
            # Update layout
            payback_sens_fig.update_layout(
                xaxis_title="Correlation Coefficient",
                margin=dict(l=20, r=20, t=40, b=20),
                height=350
            )
            
            st.plotly_chart(payback_sens_fig, use_container_width=True)
        
        # Create heatmap of correlation matrix
        heatmap_fig = px.imshow(
            correlation_df,
            text_auto='.2f',
            aspect="auto",
            color_continuous_scale=px.colors.diverging.RdBu_r,
            title='Parameter Correlation Matrix'
        )
        
        # Update layout
        heatmap_fig.update_layout(
            margin=dict(l=20, r=20, t=40, b=20),
            height=500
        )
        
        with st.expander("View Detailed Correlation Matrix", expanded=False):
            st.plotly_chart(heatmap_fig, use_container_width=True)
        
        # Display data table with summary statistics
        with st.expander("View Simulation Statistics", expanded=False):
            # Calculate summary statistics
            summary_stats = results_df.describe().transpose()
            
            # Add confidence intervals
            summary_stats[f'{sim_data["confidence_level"]}% Lower'] = results_df.quantile(alpha/2)
            summary_stats[f'{sim_data["confidence_level"]}% Upper'] = results_df.quantile(1-alpha/2)
            
            # Round and display
            st.dataframe(summary_stats.round(2), use_container_width=True)
            
            # Download link for CSV
            csv = results_df.to_csv(index=False)
            b64 = base64.b64encode(csv.encode()).decode()
            href = f'<a href="data:file/csv;base64,{b64}" download="monte_carlo_results.csv" class="download-link">Download Full Simulation Results as CSV</a>'
            st.markdown(href, unsafe_allow_html=True)
    
    elif not st.session_state.quality_analysis_results:
        st.warning("Please run a quality analysis first before using the Monte Carlo Simulation tool.")

def display_analysis_page():
    """Display the single product analysis page"""
    # Display header
    app_header()
    st.markdown('<div class="main-header">Medical Device Quality ROI Analysis</div>', unsafe_allow_html=True)
    
    # Display mode toggle
    display_mode_toggle()
    
    # Navigation buttons
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Batch Analysis Import", key="batch_button", use_container_width=True, help="Upload a CSV or Excel file to analyze multiple products at once"):
            navigate_to_page("batch_analysis")
    with col2:
        if st.button("Formula Guide", key="help_button", use_container_width=True, help="View detailed explanations of formulas used in the analysis"):
            navigate_to_page("help")
    
    # Check if analysis results exist
    if not st.session_state.analysis_submitted:
        # New Analysis button
        if st.button("New Analysis", key="new_analysis"):
            st.session_state.quality_analysis_results = None
            st.session_state.analysis_submitted = False
            st.session_state.chat_history = []
            
        with st.form("quality_issue_form"):
            st.markdown('<div class="form-section-header">Product & Issue Details</div>', unsafe_allow_html=True)
            
            # Basic product information
            col1, col2, col3 = st.columns(3)
            
            with col1:
                # Required inputs
                sku = st.text_input(
                    "SKU", 
                    help="Product SKU (Stock Keeping Unit)"
                )
                st.markdown('<span class="required-field"></span>', unsafe_allow_html=True)
                
                product_type = st.selectbox(
                    "Product Type", 
                    ["B2C", "B2B", "Both"],
                    help="Distribution channel for this product"
                )
                st.markdown('<span class="required-field"></span>', unsafe_allow_html=True)
            
            with col2:
                sales_30d = st.number_input(
                    "Units Sold (30 Days)", 
                    min_value=0.0,
                    help="Number of units sold in the last 30 days"
                )
                st.markdown('<span class="required-field"></span>', unsafe_allow_html=True)
                
                returns_30d = st.number_input(
                    "Units Returned (30 Days)", 
                    min_value=0.0,
                    help="Number of units returned in the last 30 days"
                )
                st.markdown('<span class="required-field"></span>', unsafe_allow_html=True)
            
            with col3:
                risk_level = st.select_slider(
                    "Medical Risk Level",
                    options=["Low", "Medium", "High"],
                    value="Medium",
                    help="Risk level assessment for this medical device issue"
                )
                st.markdown('<span class="required-field"></span>', unsafe_allow_html=True)
                
                regulatory_impact = st.selectbox(
                    "Regulatory Impact",
                    ["None", "Possible", "Significant"],
                    help="Potential regulatory impact of this issue"
                )
                st.markdown('<span class="required-field"></span>', unsafe_allow_html=True)
            
            # Description of issue
            issue_description = st.text_area(
                "Description of Quality Issue",
                height=100,
                help="Detailed description of the quality problem, including failure modes and customer impact"
            )
            st.markdown('<span class="required-field"></span>', unsafe_allow_html=True)
            
            # Financial Information
            st.markdown('<div class="form-section-header">Financial Information</div>', unsafe_allow_html=True)
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                current_unit_cost = st.number_input(
                    "Current Unit Cost", 
                    min_value=0.0,
                    help="Current per-unit cost to produce/acquire (landed cost)"
                )
                st.markdown('<span class="required-field"></span>', unsafe_allow_html=True)
                
                sales_price = st.number_input(
                    "Current Sales Price", 
                    min_value=0.0,
                    help="Current selling price per unit"
                )
                st.markdown('<span class="required-field"></span>', unsafe_allow_html=True)
            
            with col2:
                fix_cost_upfront = st.number_input(
                    "Fix Cost Upfront", 
                    min_value=0.0,
                    help="One-time cost to implement the quality fix (engineering, design changes, tooling, etc.)"
                )
                st.markdown('<span class="required-field"></span>', unsafe_allow_html=True)
                
                fix_cost_per_unit = st.number_input(
                    "Additional Cost Per Unit", 
                    min_value=0.0,
                    help="Additional cost per unit after implementing the fix (extra material, labor, etc.)"
                )
                st.markdown('<span class="required-field"></span>', unsafe_allow_html=True)
            
            with col3:
                new_sales_price = st.number_input(
                    "New Sales Price (if changing)", 
                    min_value=0.0,
                    help="New selling price per unit after the fix (if staying the same, leave at 0)"
                )
                
                # Only show in advanced mode
                if st.session_state.view_mode == "advanced":
                    fba_fee = st.number_input(
                        "Fulfillment/Shipping Fee", 
                        min_value=0.0,
                        help="Fulfillment fee per unit, if applicable"
                    )
                else:
                    fba_fee = 0.0
            
            # Solution expectations
            st.markdown('<div class="form-section-header">Solution Expectations</div>', unsafe_allow_html=True)
            
            col1, col2 = st.columns(2)
            
            with col1:
                expected_reduction = st.slider(
                    "Expected Return Rate Reduction (%)", 
                    min_value=0.0, 
                    max_value=100.0,
                    value=50.0,
                    help="Expected percentage reduction in return rate after implementing the fix"
                )
                st.markdown('<span class="required-field"></span>', unsafe_allow_html=True)
            
            with col2:
                solution_confidence = st.slider(
                    "Solution Confidence (%)", 
                    min_value=0.0, 
                    max_value=100.0,
                    value=75.0,
                    help="Confidence level that the solution will achieve the expected reduction"
                )
                st.markdown('<span class="required-field"></span>', unsafe_allow_html=True)
            
            # Expandable section for optional metrics (only in advanced mode)
            if st.session_state.view_mode == "advanced":
                with st.expander("Additional Metrics (Optional)"):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        asin = st.text_input(
                            "Alternative Product ID", 
                            help="Alternative product identification number"
                        )
                        
                        ncx_rate = st.number_input(
                            "Negative Customer Experience Rate (%)", 
                            min_value=0.0, 
                            max_value=100.0,
                            help="Composite of bad reviews, returns, and customer complaints divided by total sales"
                        )
                        
                        sales_365d = st.number_input(
                            "Units Sold (365 Days)", 
                            min_value=0.0,
                            help="Number of units sold in the last 365 days (for long-term trends)"
                        )
                        
                        returns_365d = st.number_input(
                            "Units Returned (365 Days)", 
                            min_value=0.0,
                            help="Number of units returned in the last 365 days (for long-term trends)"
                        )
                    
                    with col2:
                        star_rating = st.number_input(
                            "Current Star Rating", 
                            min_value=1.0, 
                            max_value=5.0,
                            value=4.0,
                            help="Current average star rating (if applicable)"
                        )
                        
                        total_reviews = st.number_input(
                            "Total Reviews", 
                            min_value=0,
                            help="Total number of reviews (if applicable)"
                        )
            else:
                # Set default values for optional fields
                asin = None
                ncx_rate = None
                sales_365d = None
                returns_365d = None
                star_rating = None
                total_reviews = None
            
            # Note about required fields
            st.markdown('<p><small>* Required fields</small></p>', unsafe_allow_html=True)
            
            # Form submission
            submit_button = st.form_submit_button("Analyze Quality Issue")
            
            if submit_button:
                # Validate required fields
                if not all([sku, issue_description]):
                    st.error("Please fill in all required fields marked with *")
                elif sales_30d <= 0:
                    st.error("Units Sold (30 Days) must be greater than zero")
                elif current_unit_cost <= 0:
                    st.error("Current Unit Cost must be greater than zero")
                elif sales_price <= 0:
                    st.error("Current Sales Price must be greater than zero")
                else:
                    with st.spinner("Analyzing quality issue..."):
                        # Set new_sales_price to current sales_price if not specified
                        if new_sales_price <= 0:
                            new_sales_price = sales_price
                        
                        # Perform analysis
                        results = analyze_quality_issue(
                            sku=sku,
                            product_type=product_type,
                            sales_30d=sales_30d,
                            returns_30d=returns_30d,
                            issue_description=issue_description,
                            current_unit_cost=current_unit_cost,
                            fix_cost_upfront=fix_cost_upfront,
                            fix_cost_per_unit=fix_cost_per_unit,
                            sales_price=sales_price,
                            expected_reduction=expected_reduction,
                            solution_confidence=solution_confidence,
                            new_sales_price=new_sales_price,
                            asin=asin if asin else None,
                            ncx_rate=ncx_rate if ncx_rate > 0 else None,
                            sales_365d=sales_365d if sales_365d > 0 else None,
                            returns_365d=returns_365d if returns_365d > 0 else None,
                            star_rating=star_rating if star_rating > 0 else None,
                            total_reviews=total_reviews if total_reviews > 0 else None,
                            fba_fee=fba_fee if fba_fee > 0 else None,
                            risk_level=risk_level,
                            regulatory_impact=regulatory_impact
                        )
                        
                        # Store results in session state
                        st.session_state.quality_analysis_results = results
                        st.session_state.analysis_submitted = True
                        
                        # Initialize chat with AI
                        st.session_state.chat_history = chat_with_ai(
                            results, 
                            issue_description
                        )
                        
                        # Rerun to show results
                        st.rerun()
    
    # Display results if available
    if st.session_state.analysis_submitted and st.session_state.quality_analysis_results:
        # Create tabs for different analysis types
        tabs = st.tabs(["Quality ROI Analysis", "Tariff Calculator", "Marketing ROI", "Monte Carlo Simulation"])
        
        with tabs[0]:  # Quality ROI Analysis tab
            # Add a reset button at the top
            if st.button("New Analysis", key="reset_top_button"):
                reset_analysis()
                
            # Display analysis results
            display_quality_issue_results(st.session_state.quality_analysis_results)
            
            # Display AI chat interface if enabled and in advanced mode
            if st.session_state.view_mode == "advanced":
                with st.expander("Quality Consultant AI Assistant", expanded=True):
                    # Chat container
                    st.markdown('<div style="background-color:#F9FAFB; border-radius:8px; padding:1rem; margin-bottom:1rem; border:1px solid #E5E7EB; max-height:400px; overflow-y:auto;">', unsafe_allow_html=True)
                    
                    if st.session_state.chat_history:
                        for message in st.session_state.chat_history:
                            if message["role"] == "user":
                                st.markdown(f'<div style="padding:0.75rem; background-color:{TERTIARY_COLOR}; border-radius:8px; margin-bottom:0.75rem; margin-left:auto; max-width:85%; border-bottom-right-radius:0;">{message["content"]}</div>', unsafe_allow_html=True)
                            else:
                                st.markdown(f'<div style="padding:0.75rem; background-color:white; border-radius:8px; margin-bottom:0.75rem; margin-right:auto; max-width:85%; border-bottom-left-radius:0; border-left:3px solid {PRIMARY_COLOR};">{message["content"]}</div>', unsafe_allow_html=True)
                    else:
                        st.info("AI consultant not available or initialization failed.")
                        
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                    # Input for new messages
                    user_input = st.text_area(
                        "Ask for advice on the quality issue, potential solutions, or regulatory implications:",
                        placeholder="E.g., What could be causing this issue? What fixes do you recommend? What are the regulatory considerations?",
                        key="chat_input"
                    )
                    
                    # Send button
                    if st.button("Send", key="send_button"):
                        if user_input:
                            # Add user message to chat history
                            st.session_state.chat_history.append({
                                "role": "user",
                                "content": user_input
                            })
                            
                            with st.spinner("Getting AI analysis..."):
                                # Build the message history for API call
                                messages = [
                                    {"role": "system", "content": f"""
                                    You are a Quality Management expert specializing in medical devices. You analyze quality issues, provide insights on cost-benefit analyses, and suggest solutions.
                                    
                                    Product details:
                                    - SKU: {st.session_state.quality_analysis_results["sku"]}
                                    - Type: {st.session_state.quality_analysis_results["product_type"]}
                                    - Issue: {st.session_state.quality_analysis_results["issue_description"]}
                                    
                                    Metrics:
                                    - Return Rate (30 days): {st.session_state.quality_analysis_results["current_metrics"]["return_rate_30d"]:.2f}%
                                    - Monthly Return Cost: ${st.session_state.quality_analysis_results["financial_impact"]["monthly_return_cost"]:.2f}
                                    - Estimated Savings: ${st.session_state.quality_analysis_results["financial_impact"]["estimated_monthly_savings"]:.2f}/month
                                    - Payback Period: {st.session_state.quality_analysis_results["financial_impact"]["payback_months"]:.1f} months
                                    - Medical Risk Level: {st.session_state.quality_analysis_results["medical_device_impact"]["risk_level"]}
                                    - Regulatory Impact: {st.session_state.quality_analysis_results["medical_device_impact"]["regulatory_impact"]}
                                    
                                    Recommendation: {st.session_state.quality_analysis_results["recommendation"]}
                                    
                                    Keep your responses concise, specific, and tailored to the medical device industry and its regulatory requirements.
                                    """}
                                ]
                                
                                # Add conversation history
                                for msg in st.session_state.chat_history:
                                    messages.append({"role": msg["role"], "content": msg["content"]})
                                
                                # Make the API call (excluding the last user message as it's added separately)
                                ai_response = call_openai_api(messages[:-1])
                                
                                # Add AI response to chat history
                                st.session_state.chat_history.append({
                                    "role": "assistant",
                                    "content": ai_response
                                })
                                
                                # Rerun to show updated chat
                                st.rerun()
            
            # Export options
            with st.expander("Export Analysis"):
                col1, col2 = st.columns(2)
                
                with col1:
                    # Export as CSV
                    results_dict = {
                        'Metric': [
                            'SKU',
                            'Product Type',
                            'Return Rate (30d)',
                            'Monthly Return Cost',
                            'Annual Return Cost',
                            'Estimated Monthly Savings',
                            'Annual Savings',
                            'Payback Period (months)',
                            '3-Year ROI',
                            'Current Sales Price',
                            'New Sales Price',
                            'Expected Return Reduction',
                            'Solution Confidence',
                            'Fix Cost Upfront',
                            'Additional Cost Per Unit',
                            'Current Unit Cost',
                            'Recommendation',
                            'Risk Level',
                            'Regulatory Impact'
                        ],
                        'Value': [
                            st.session_state.quality_analysis_results['sku'],
                            st.session_state.quality_analysis_results['product_type'],
                            f"{st.session_state.quality_analysis_results['current_metrics']['return_rate_30d']:.2f}%",
                            f"${st.session_state.quality_analysis_results['financial_impact']['monthly_return_cost']:.2f}",
                            f"${st.session_state.quality_analysis_results['financial_impact']['annual_return_cost']:.2f}",
                            f"${st.session_state.quality_analysis_results['financial_impact']['estimated_monthly_savings']:.2f}",
                            f"${st.session_state.quality_analysis_results['financial_impact']['annual_savings']:.2f}",
                            f"{st.session_state.quality_analysis_results['financial_impact']['payback_months']:.1f}",
                            f"{st.session_state.quality_analysis_results['financial_impact']['roi_3yr']:.1f}%",
                            f"${st.session_state.quality_analysis_results['current_metrics']['sales_price']:.2f}",
                            f"${st.session_state.quality_analysis_results['solution_metrics']['new_sales_price']:.2f}",
                            f"{st.session_state.quality_analysis_results['solution_metrics']['expected_reduction']:.1f}%",
                            f"{st.session_state.quality_analysis_results['solution_metrics']['solution_confidence']:.1f}%",
                            f"${st.session_state.quality_analysis_results['financial_impact']['fix_cost_upfront']:.2f}",
                            f"${st.session_state.quality_analysis_results['financial_impact']['fix_cost_per_unit']:.2f}",
                            f"${st.session_state.quality_analysis_results['financial_impact']['current_unit_cost']:.2f}",
                            st.session_state.quality_analysis_results['recommendation'],
                            st.session_state.quality_analysis_results['medical_device_impact']['risk_level'],
                            st.session_state.quality_analysis_results['medical_device_impact']['regulatory_impact']
                        ]
                    }
                    
                    # Convert to DataFrame
                    results_df = pd.DataFrame(results_dict)
                    
                    # Create download link
                    csv = results_df.to_csv(index=False)
                    b64 = base64.b64encode(csv.encode()).decode()
                    filename = f"{st.session_state.quality_analysis_results['sku']}_analysis.csv"
                    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}" class="download-link">Download Analysis as CSV</a>'
                    st.markdown(href, unsafe_allow_html=True)
                
                with col2:
                    # Export as PDF
                    if st.button("Generate PDF Report"):
                        with st.spinner("Generating PDF report..."):
                            pdf_buffer = export_as_pdf(st.session_state.quality_analysis_results)
                            
                            # Create download link for PDF
                            b64 = base64.b64encode(pdf_buffer.getvalue()).decode()
                            filename = f"{st.session_state.quality_analysis_results['sku']}_analysis.pdf"
                            href = f'<a href="data:application/pdf;base64,{b64}" download="{filename}" class="download-link">Download PDF Report</a>'
                            st.markdown(href, unsafe_allow_html=True)
                    
                    # Add to comparison list
                    if st.button("Add to Comparison List"):
                        if 'compare_list' not in st.session_state:
                            st.session_state.compare_list = []
                        
                        # Check if already in list
                        already_exists = any(item['sku'] == st.session_state.quality_analysis_results['sku'] for item in st.session_state.compare_list)
                        
                        if not already_exists:
                            st.session_state.compare_list.append(st.session_state.quality_analysis_results)
                            st.success(f"Added {st.session_state.quality_analysis_results['sku']} to comparison list. Total items: {len(st.session_state.compare_list)}")
                        else:
                            st.warning(f"{st.session_state.quality_analysis_results['sku']} is already in the comparison list.")
        
        with tabs[1]:  # Tariff Calculator tab
            display_landed_cost_calculator()
        
        with tabs[2]:  # Marketing ROI tab
            calculate_ad_roi_ui()
        
        with tabs[3]:  # Monte Carlo Simulation tab
            run_monte_carlo_simulation_ui()

def display_batch_analysis_page():
    """Display the batch analysis page for analyzing multiple products at once"""
    app_header()
    st.markdown('<div class="main-header">Batch Analysis for Multiple Products</div>', unsafe_allow_html=True)
    
    # Navigation buttons
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Single Analysis", key="single_button", use_container_width=True):
            navigate_to_page("analysis")
    with col2:
        if st.button("Formula Guide", key="help_button", use_container_width=True):
            navigate_to_page("help")
    
    # Instructions for batch upload
    st.markdown("""
    <div class="alert alert-info">
        <p><strong>Batch Analysis Instructions:</strong></p>
        <p>Upload a CSV or Excel file containing multiple products to analyze. Your file must contain the required columns listed below.</p>
        <p>Download a <a href="#" id="template-link">template CSV file</a> to get started.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Required columns
    required_columns = ['sku', 'product_type', 'sales_30d', 'returns_30d', 
                       'issue_description', 'current_unit_cost', 'fix_cost_upfront', 
                       'fix_cost_per_unit', 'sales_price', 'expected_reduction', 
                       'solution_confidence']
    
    # Optional columns
    optional_columns = ['new_sales_price', 'asin', 'ncx_rate', 'sales_365d', 
                        'returns_365d', 'star_rating', 'total_reviews', 'fba_fee',
                        'risk_level', 'regulatory_impact']
    
    # Column descriptions 
    column_descriptions = {
        'sku': 'Product SKU (required)',
        'product_type': 'B2B, B2C, or Both (required)',
        'sales_30d': 'Number of units sold in 30 days (required)',
        'returns_30d': 'Number of units returned in 30 days (required)',
        'issue_description': 'Description of the quality issue (required)',
        'current_unit_cost': 'Cost per unit (required)',
        'fix_cost_upfront': 'One-time cost for implementing fix (required)',
        'fix_cost_per_unit': 'Additional cost per unit after fix (required)',
        'sales_price': 'Current selling price (required)',
        'expected_reduction': 'Expected return rate reduction % (required)',
        'solution_confidence': 'Confidence in solution effectiveness % (required)',
        'new_sales_price': 'New selling price after fix (optional)',
        'asin': 'Alternative product ID (optional)',
        'ncx_rate': 'Negative customer experience rate % (optional)',
        'sales_365d': 'Number of units sold in 365 days (optional)',
        'returns_365d': 'Number of units returned in 365 days (optional)',
        'star_rating': 'Current star rating 1-5 (optional)',
        'total_reviews': 'Total number of reviews (optional)',
        'fba_fee': 'Fulfillment fee per unit (optional)',
        'risk_level': 'Low, Medium, or High (optional, defaults to Medium)',
        'regulatory_impact': 'None, Possible, or Significant (optional, defaults to None)'
    }
    
    # Display column information
    with st.expander("View Column Requirements and Descriptions", expanded=False):
        st.markdown("<h3>Required Columns</h3>", unsafe_allow_html=True)
        
        # Display required columns
        for col in required_columns:
            st.markdown(f"- **{col}**: {column_descriptions[col]}")
        
        st.markdown("<h3>Optional Columns</h3>", unsafe_allow_html=True)
        
        # Display optional columns
        for col in optional_columns:
            st.markdown(f"- **{col}**: {column_descriptions[col]}")
        
        # Example data
        st.markdown("<h3>Example Data</h3>", unsafe_allow_html=True)
        
        # Create example data
        example_data = {
            'sku': ['MED-1001', 'MED-1002'],
            'product_type': ['B2C', 'B2B'],
            'sales_30d': [120, 85],
            'returns_30d': [12, 6],
            'issue_description': ['Packaging damage during shipping', 'Calibration drift issue'],
            'current_unit_cost': [45.0, 120.0],
            'fix_cost_upfront': [5000.0, 8000.0],
            'fix_cost_per_unit': [2.5, 5.0],
            'sales_price': [89.99, 249.99],
            'expected_reduction': [70.0, 80.0],
            'solution_confidence': [85.0, 75.0],
            'risk_level': ['Medium', 'High'],
            'regulatory_impact': ['None', 'Possible']
        }
        
        example_df = pd.DataFrame(example_data)
        st.dataframe(example_df, hide_index=True)
        
        # Generate example CSV for download
        csv = example_df.to_csv(index=False)
        b64 = base64.b64encode(csv.encode()).decode()
        href = f'<a href="data:file/csv;base64,{b64}" download="batch_analysis_template.csv">Download Template CSV</a>'
        st.markdown(href, unsafe_allow_html=True)
    
    # File uploader
    uploaded_file = st.file_uploader("Upload CSV or Excel file", type=["csv", "xlsx", "xls"])
    
    if uploaded_file is not None:
        with st.spinner("Processing file..."):
            df = import_batch_data(uploaded_file)
            
            if df is not None:
                st.success(f"File processed successfully. Found {len(df)} products to analyze.")
                
                # Display preview of data
                with st.expander("Preview Data", expanded=True):
                    st.dataframe(df.head(5), use_container_width=True)
                
                # Run batch analysis
                if st.button("Run Batch Analysis", key="run_batch"):
                    with st.spinner("Running batch analysis..."):
                        results = batch_analyze_quality_issues(df)
                        
                        # Store results in session state
                        st.session_state.batch_analysis_results = results
                        
                        # Display summary
                        st.success(f"Analysis complete: {len(results)} products analyzed.")
                        st.rerun()
    
    # Display batch analysis results if available
    if 'batch_analysis_results' in st.session_state and st.session_state.batch_analysis_results:
        results = st.session_state.batch_analysis_results
        
        st.markdown('<div class="main-header">Batch Analysis Results</div>', unsafe_allow_html=True)
        
        # Summary metrics
        total_products = len(results)
        high_rec_count = sum(1 for r in results if "recommendation_class" in r and r["recommendation_class"] == "recommendation-high")
        medium_rec_count = sum(1 for r in results if "recommendation_class" in r and r["recommendation_class"] == "recommendation-medium")
        low_rec_count = sum(1 for r in results if "recommendation_class" in r and r["recommendation_class"] == "recommendation-low")
        
        # Display summary metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(f"""
            <div class='metric-card'>
                <p class='metric-label'>Total Products</p>
                <p class='metric-value'>{total_products}</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class='metric-card'>
                <p class='metric-label'>Highly Recommended</p>
                <p class='metric-value' style='color:{SUCCESS_COLOR}'>{high_rec_count}</p>
                <p style='font-size:0.8rem;'>({high_rec_count/total_products*100:.1f}%)</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div class='metric-card'>
                <p class='metric-label'>Recommended</p>
                <p class='metric-value' style='color:{WARNING_COLOR}'>{medium_rec_count}</p>
                <p style='font-size:0.8rem;'>({medium_rec_count/total_products*100:.1f}%)</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            st.markdown(f"""
            <div class='metric-card'>
                <p class='metric-label'>Not Recommended</p>
                <p class='metric-value' style='color:{DANGER_COLOR}'>{low_rec_count}</p>
                <p style='font-size:0.8rem;'>({low_rec_count/total_products*100:.1f}%)</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Create summary dataframe
        summary_data = []
        
        for r in results:
            if "error" in r:
                summary_data.append({
                    'SKU': r['sku'],
                    'Recommendation': 'Error',
                    'Return Rate': 'N/A',
                    'Monthly Savings': 'N/A',
                    'Payback Period': 'N/A',
                    '3-Year ROI': 'N/A',
                    'Risk Level': 'N/A',
                    'Error': r['error']
                })
            else:
                summary_data.append({
                    'SKU': r['sku'],
                    'Recommendation': r['recommendation'],
                    'Return Rate': f"{r['current_metrics']['return_rate_30d']:.2f}%",
                    'Monthly Savings': f"${r['financial_impact']['estimated_monthly_savings']:.2f}",
                    'Payback Period': f"{r['financial_impact']['payback_months']:.1f} months",
                    '3-Year ROI': f"{r['financial_impact']['roi_3yr']:.1f}%",
                    'Risk Level': r['medical_device_impact']['risk_level'],
                    'Regulatory': r['medical_device_impact']['regulatory_impact']
                })
        
        summary_df = pd.DataFrame(summary_data)
        
        # Display sortable table
        st.dataframe(summary_df, use_container_width=True)
        
        # Generate CSV download link
        csv = summary_df.to_csv(index=False)
        b64 = base64.b64encode(csv.encode()).decode()
        href = f'<a href="data:file/csv;base64,{b64}" download="batch_analysis_results.csv" class="download-link">Download Results as CSV</a>'
        st.markdown(href, unsafe_allow_html=True)
        
        # Create visualizations
        st.markdown('<div class="section-header">Results Visualization</div>', unsafe_allow_html=True)
        
        # Extract financial metrics for valid results
        valid_results = [r for r in results if "error" not in r]
        
        if valid_results:
            # Create ROI vs Payback scatter chart
            roi_data = {
                'SKU': [r['sku'] for r in valid_results],
                'ROI': [r['financial_impact']['roi_3yr'] for r in valid_results],
                'Payback': [r['financial_impact']['payback_months'] for r in valid_results],
                'Monthly Savings': [r['financial_impact']['estimated_monthly_savings'] for r in valid_results],
                'Initial Cost': [r['financial_impact']['fix_cost_upfront'] for r in valid_results],
                'Risk Level': [r['medical_device_impact']['risk_level'] for r in valid_results],
                'Recommendation': [r['recommendation'] for r in valid_results]
            }
            
            roi_df = pd.DataFrame(roi_data)
            
            # Create scatter plot
            fig = px.scatter(
                roi_df,
                x='Payback',
                y='ROI',
                size='Monthly Savings',
                color='Risk Level',
                hover_name='SKU',
                size_max=30,
                color_discrete_map={
                    'Low': SUCCESS_COLOR,
                    'Medium': WARNING_COLOR,
                    'High': DANGER_COLOR
                },
                labels={
                    'Payback': 'Payback Period (months)',
                    'ROI': 'ROI (%)',
                    'Monthly Savings': 'Monthly Savings ($)'
                },
                title='ROI vs Payback Period by Risk Level'
            )
            
            # Add quadrant lines
            fig.add_hline(y=0, line_dash="dash", line_color=TEXT_SECONDARY)
            fig.add_vline(x=12, line_dash="dash", line_color=TEXT_SECONDARY)
            
            # Add annotations
            fig.add_annotation(
                x=6, y=50,
                text="High Priority<br>(High ROI, Fast Payback)",
                showarrow=False,
                font=dict(color=SUCCESS_COLOR)
            )
            
            fig.add_annotation(
                x=18, y=50,
                text="Medium Priority<br>(High ROI, Slow Payback)",
                showarrow=False,
                font=dict(color=WARNING_COLOR)
            )
            
            fig.add_annotation(
                x=6, y=-20,
                text="Not Recommended<br>(Negative ROI)",
                showarrow=False,
                font=dict(color=DANGER_COLOR)
            )
            
            # Update layout
            fig.update_layout(
                height=500,
                margin=dict(l=20, r=20, t=40, b=20)
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Create bar chart of monthly savings
            savings_df = pd.DataFrame({
                'SKU': [r['sku'] for r in valid_results],
                'Monthly Savings': [r['financial_impact']['estimated_monthly_savings'] for r in valid_results],
                'Risk Level': [r['medical_device_impact']['risk_level'] for r in valid_results]
            })
            
            # Sort by savings
            savings_df = savings_df.sort_values('Monthly Savings', ascending=False).head(10)
            
            # Create bar chart
            bar_fig = px.bar(
                savings_df,
                x='SKU',
                y='Monthly Savings',
                color='Risk Level',
                color_discrete_map={
                    'Low': SUCCESS_COLOR,
                    'Medium': WARNING_COLOR,
                    'High': DANGER_COLOR
                },
                title='Top 10 Products by Monthly Savings'
            )
            
            # Update layout
            bar_fig.update_layout(
                xaxis_title='Product SKU',
                yaxis_title='Estimated Monthly Savings ($)',
                height=400,
                margin=dict(l=20, r=20, t=40, b=20)
            )
            
            st.plotly_chart(bar_fig, use_container_width=True)
            
            # Individual product analysis section
            st.markdown('<div class="section-header">Individual Product Details</div>', unsafe_allow_html=True)
            
            # Let user select a product to view
            selected_sku = st.selectbox(
                "Select Product for Detailed Analysis",
                options=[r['sku'] for r in results],
                format_func=lambda x: f"{x} ({next((r['recommendation'] for r in results if r['sku'] == x), 'Unknown')})"
            )
            
            # Find the selected result
            selected_result = next((r for r in results if r['sku'] == selected_sku), None)
            
            if selected_result:
                # Initialize expanded state if it doesn't exist
                if selected_sku not in st.session_state.analysis_results_expanded:
                    st.session_state.analysis_results_expanded[selected_sku] = True
                
                # Display the selected result
                is_expanded = display_quality_issue_results(
                    selected_result, 
                    expanded=st.session_state.analysis_results_expanded[selected_sku]
                )
                
                # Update expanded state
                st.session_state.analysis_results_expanded[selected_sku] = is_expanded
        else:
            st.warning("No valid results found in the batch analysis.")

def display_help_page():
    """Display the help page with formula explanations"""
    st.markdown('<div class="main-header">QualityROI Formula Guide</div>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="alert alert-info">
        This guide explains all the formulas and calculations used in the QualityROI tool for medical device cost-benefit analysis.
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown('<div class="help-section">', unsafe_allow_html=True)
    st.markdown('<div class="help-title">Basic Metrics</div>', unsafe_allow_html=True)
    
    st.markdown("""
    <p>These fundamental metrics form the basis of our quality analysis:</p>
    
    <p><strong>Return Rate</strong>: The percentage of units returned relative to total sales in a given period.</p>
    <div class="formula-box">Return Rate (%) = (Returns / Sales) × 100</div>
    
    <p><strong>Monthly Return Cost</strong>: The direct financial impact of returns based on unit cost.</p>
    <div class="formula-box">Monthly Return Cost ($) = Returns × Unit Cost</div>
    
    <p><strong>Monthly Revenue</strong>: Total revenue from product sales in the period.</p>
    <div class="formula-box">Monthly Revenue ($) = Sales × Sales Price</div>
    
    <p><strong>Profit Margin</strong>: The percentage of profit relative to revenue.</p>
    <div class="formula-box">Profit Margin (%) = ((Sales × Sales Price) - (Sales × Unit Cost)) / (Sales × Sales Price) × 100</div>
    """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="help-section">', unsafe_allow_html=True)
    st.markdown('<div class="help-title">Solution Impact Calculations</div>', unsafe_allow_html=True)
    
    st.markdown("""
    <p>These calculations estimate the impact of implementing a quality fix:</p>
    
    <p><strong>Adjusted Reduction</strong>: Expected return rate reduction adjusted by confidence level.</p>
    <div class="formula-box">Adjusted Reduction (%) = Expected Reduction (%) × (Solution Confidence (%) / 100)</div>
    
    <p><strong>Reduced Returns</strong>: Estimated number of returns after implementing the fix.</p>
    <div class="formula-box">Reduced Returns = Current Returns × (1 - (Adjusted Reduction / 100))</div>
    
    <p><strong>Returns Saved</strong>: Estimated number of returns prevented by the fix.</p>
    <div class="formula-box">Returns Saved = Current Returns - Reduced Returns</div>
    
    <p><strong>Estimated Monthly Savings</strong>: Financial impact of reduced returns.</p>
    <div class="formula-box">Monthly Savings ($) = Returns Saved × Current Unit Cost</div>
    """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="help-section">', unsafe_allow_html=True)
    st.markdown('<div class="help-title">Price Change Impact</div>', unsafe_allow_html=True)
    
    st.markdown("""
    <p>If the product price changes after the fix, these calculations apply:</p>
    
    <p><strong>New Monthly Revenue</strong>: Revenue after implementing sales price change.</p>
    <div class="formula-box">New Monthly Revenue ($) = Sales × New Sales Price</div>
    
    <p><strong>Revenue Change</strong>: The difference in monthly revenue due to price change.</p>
    <div class="formula-box">Revenue Change ($) = New Monthly Revenue - Current Monthly Revenue</div>
    
    <p><strong>New Unit Cost</strong>: Unit cost after adding the per-unit fix cost.</p>
    <div class="formula-box">New Unit Cost ($) = Current Unit Cost + Fix Cost Per Unit</div>
    
    <p><strong>New Profit Margin</strong>: Profit margin after implementing the fix and price changes.</p>
    <div class="formula-box">New Profit Margin (%) = ((Sales × New Sales Price) - (Sales × New Unit Cost)) / (Sales × New Sales Price) × 100</div>
    """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="help-section">', unsafe_allow_html=True)
    st.markdown('<div class="help-title">ROI Calculations</div>', unsafe_allow_html=True)
    
    st.markdown("""
    <p>These calculations determine the financial return on investment:</p>
    
    <p><strong>Total Monthly Benefit</strong>: Combined impact of return savings and revenue changes.</p>
    <div class="formula-box">Total Monthly Benefit ($) = Monthly Savings + Revenue Change</div>
    
    <p><strong>Payback Period</strong>: Time required to recover the initial investment.</p>
    <div class="formula-box">Payback Period (months) = Fix Cost Upfront / Total Monthly Benefit</div>
    
    <p><strong>Annual Projections</strong>: Yearly projections of financial impact.</p>
    <div class="formula-box">Annual Return Cost ($) = Monthly Return Cost × 12
Annual Savings ($) = Monthly Savings × 12
Annual Revenue Change ($) = Monthly Revenue Change × 12</div>
    
    <p><strong>3-Year ROI</strong>: Return on investment over a 3-year period.</p>
    <div class="formula-box">Projected Sales (36 months) = Monthly Sales × 36
Total Investment ($) = Fix Cost Upfront + (Fix Cost Per Unit × Projected Sales)
Total Savings ($) = (Annual Savings + Annual Revenue Change) × 3
ROI (%) = ((Total Savings - Total Investment) / Total Investment) × 100</div>
    """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="help-section">', unsafe_allow_html=True)
    st.markdown('<div class="help-title">Recommendation Logic</div>', unsafe_allow_html=True)
    
    st.markdown("""
    <p>The system determines recommendations based on these criteria:</p>
    
    <p><strong>Highly Recommended</strong>:</p>
    <ul>
        <li>Significant regulatory impact requiring compliance</li>
        <li>High risk issues with payback period < 12 months</li>
        <li>Any issue with payback period < 3 months</li>
    </ul>
    
    <p><strong>Recommended</strong>:</p>
    <ul>
        <li>Payback period between 3-6 months</li>
        <li>Low star rating products (< 3.5) with brand impact concerns</li>
    </ul>
    
    <p><strong>Consider</strong>:</p>
    <ul>
        <li>Payback period between 6-12 months</li>
        <li>High risk issues despite longer payback</li>
        <li>B2B products with return rate > 10%</li>
    </ul>
    
    <p><strong>Not Recommended</strong>:</p>
    <ul>
        <li>Payback period > 12 months</li>
        <li>Low risk issues with poor financial return</li>
    </ul>
    """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="help-section">', unsafe_allow_html=True)
    st.markdown('<div class="help-title">Using Batch Analysis</div>', unsafe_allow_html=True)
    
    st.markdown("""
    <p>When using the batch analysis feature, your CSV or Excel file must contain these required columns:</p>
    
    <table style="width:100%; border-collapse:collapse; margin:1rem 0;">
        <tr style="background-color:#f8f9fa;">
            <th style="border:1px solid #dee2e6; padding:8px; text-align:left;">Column</th>
            <th style="border:1px solid #dee2e6; padding:8px; text-align:left;">Description</th>
            <th style="border:1px solid #dee2e6; padding:8px; text-align:left;">Required</th>
        </tr>
        <tr>
            <td style="border:1px solid #dee2e6; padding:8px;">sku</td>
            <td style="border:1px solid #dee2e6; padding:8px;">Product SKU identifier</td>
            <td style="border:1px solid #dee2e6; padding:8px;">Yes</td>
        </tr>
        <tr style="background-color:#f8f9fa;">
            <td style="border:1px solid #dee2e6; padding:8px;">product_type</td>
            <td style="border:1px solid #dee2e6; padding:8px;">B2B, B2C, or Both</td>
            <td style="border:1px solid #dee2e6; padding:8px;">Yes</td>
        </tr>
        <tr>
            <td style="border:1px solid #dee2e6; padding:8px;">sales_30d</td>
            <td style="border:1px solid #dee2e6; padding:8px;">Number of units sold in 30 days</td>
            <td style="border:1px solid #dee2e6; padding:8px;">Yes</td>
        </tr>
        <tr style="background-color:#f8f9fa;">
            <td style="border:1px solid #dee2e6; padding:8px;">returns_30d</td>
            <td style="border:1px solid #dee2e6; padding:8px;">Number of returns in 30 days</td>
            <td style="border:1px solid #dee2e6; padding:8px;">Yes</td>
        </tr>
        <tr>
            <td style="border:1px solid #dee2e6; padding:8px;">issue_description</td>
            <td style="border:1px solid #dee2e6; padding:8px;">Description of quality issue</td>
            <td style="border:1px solid #dee2e6; padding:8px;">Yes</td>
        </tr>
        <tr style="background-color:#f8f9fa;">
            <td style="border:1px solid #dee2e6; padding:8px;">current_unit_cost</td>
            <td style="border:1px solid #dee2e6; padding:8px;">Current cost per unit</td>
            <td style="border:1px solid #dee2e6; padding:8px;">Yes</td>
        </tr>
        <tr>
            <td style="border:1px solid #dee2e6; padding:8px;">fix_cost_upfront</td>
            <td style="border:1px solid #dee2e6; padding:8px;">One-time cost for fix implementation</td>
            <td style="border:1px solid #dee2e6; padding:8px;">Yes</td>
        </tr>
        <tr style="background-color:#f8f9fa;">
            <td style="border:1px solid #dee2e6; padding:8px;">fix_cost_per_unit</td>
            <td style="border:1px solid #dee2e6; padding:8px;">Additional cost per unit after fix</td>
            <td style="border:1px solid #dee2e6; padding:8px;">Yes</td>
        </tr>
        <tr>
            <td style="border:1px solid #dee2e6; padding:8px;">sales_price</td>
            <td style="border:1px solid #dee2e6; padding:8px;">Current selling price</td>
            <td style="border:1px solid #dee2e6; padding:8px;">Yes</td>
        </tr>
        <tr style="background-color:#f8f9fa;">
            <td style="border:1px solid #dee2e6; padding:8px;">expected_reduction</td>
            <td style="border:1px solid #dee2e6; padding:8px;">Expected return rate reduction (%)</td>
            <td style="border:1px solid #dee2e6; padding:8px;">Yes</td>
        </tr>
        <tr>
            <td style="border:1px solid #dee2e6; padding:8px;">solution_confidence</td>
            <td style="border:1px solid #dee2e6; padding:8px;">Confidence in solution effectiveness (%)</td>
            <td style="border:1px solid #dee2e6; padding:8px;">Yes</td>
        </tr>
        <tr style="background-color:#f8f9fa;">
            <td style="border:1px solid #dee2e6; padding:8px;">new_sales_price</td>
            <td style="border:1px solid #dee2e6; padding:8px;">New selling price after fix</td>
            <td style="border:1px solid #dee2e6; padding:8px;">No</td>
        </tr>
        <tr>
            <td style="border:1px solid #dee2e6; padding:8px;">risk_level</td>
            <td style="border:1px solid #dee2e6; padding:8px;">Low, Medium, or High</td>
            <td style="border:1px solid #dee2e6; padding:8px;">No</td>
        </tr>
        <tr style="background-color:#f8f9fa;">
            <td style="border:1px solid #dee2e6; padding:8px;">regulatory_impact</td>
            <td style="border:1px solid #dee2e6; padding:8px;">None, Possible, or Significant</td>
            <td style="border:1px solid #dee2e6; padding:8px;">No</td>
        </tr>
    </table>
    
    <p>Additional optional columns include: asin, ncx_rate, sales_365d, returns_365d, star_rating, total_reviews, fba_fee</p>
    """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Navigation back to main page
    if st.button("Return to Analysis Tool"):
        navigate_to_page("analysis")

def safe_divide(a: float, b: float, default: float = 0.0) -> float:
    """Safely divide two numbers, returning a default if divisor is zero."""
    try:
        if b == 0:
            return default
        return a / b
    except:
        return default

def main():
    """Main application function"""
    
    # Set sidebar style
    st.sidebar.markdown('<div class="sidebar-title">Product Profitability Analysis</div>', unsafe_allow_html=True)
    
    # Simple navigation menu in sidebar
    st.sidebar.markdown("### Navigation")
    
    # Navigation buttons
    col1, col2 = st.sidebar.columns(2)
    with col1:
        if st.sidebar.button("Single Analysis", key="nav_single", use_container_width=True):
            navigate_to_page("analysis")
    with col2:
        if st.sidebar.button("Batch Analysis", key="nav_batch", use_container_width=True):
            navigate_to_page("batch_analysis")
    
    if st.sidebar.button("Help & Formulas", key="nav_help", use_container_width=True):
        navigate_to_page("help")
    
    # Version info
    st.sidebar.markdown('---')
    st.sidebar.markdown('### About')
    st.sidebar.info(
        """
        **Product Profitability Analyzer v1.0**  
        Medical Device Quality Management Tool
        
        Combines quality analysis, tariff impact, and marketing ROI 
        to provide comprehensive product profitability insights.
        
        Integrates AI guidance for regulatory and quality management decisions.
        """
    )
    
    # Display main application based on current page
    if st.session_state.current_page == "analysis":
        display_analysis_page()
    elif st.session_state.current_page == "batch_analysis":
        display_batch_analysis_page()
    elif st.session_state.current_page == "help":
        display_help_page()
    else:
        display_analysis_page()  # Default to analysis page

if __name__ == "__main__":
    main()
