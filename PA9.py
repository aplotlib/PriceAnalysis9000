import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import time
import os
from typing import Dict, List, Optional, Tuple, Union, Any
import json
import base64
from io import BytesIO
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import uuid

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="QualityROI - Medical Device CBA Tool",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- THEME COLORS - Lighter, more professional palette ---
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
</style>
""", unsafe_allow_html=True)

# --- SESSION STATE INITIALIZATION ---
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

# --- UTILITY FUNCTIONS ---

def vive_header():
    """Displays a branded header"""
    st.markdown(f"""
    <div style="background-color:{PRIMARY_COLOR}; padding:1rem; border-radius:8px; display:flex; align-items:center; margin-bottom:1.5rem;">
        <div style="font-size:1.5rem; font-weight:600; color:white; margin-right:1rem;">QUALITY ROI</div>
        <div style="color:white; font-weight:500;">Medical Device Cost-Benefit Analysis Tool</div>
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

def toggle_mode():
    """Toggle between basic and advanced view modes"""
    if st.session_state.view_mode == "basic":
        st.session_state.view_mode = "advanced"
    else:
        st.session_state.view_mode = "basic"

def display_mode_toggle():
    """Display the mode toggle UI element"""
    st.markdown("""
    <div class="mode-toggle">
        <div class="basic-mode-btn {}" onclick="handleModeToggle('basic')">Basic Mode</div>
        <div class="advanced-mode-btn {}" onclick="handleModeToggle('advanced')">Advanced Mode</div>
    </div>
    <script>
    function handleModeToggle(mode) {{
        // This will be handled by the button click handler below
        window.parent.postMessage({{type: "streamlit:setComponentValue", value: mode}}, "*");
    }}
    </script>
    """.format(
        "mode-active" if st.session_state.view_mode == "basic" else "mode-inactive",
        "mode-active" if st.session_state.view_mode == "advanced" else "mode-inactive"
    ), unsafe_allow_html=True)
    
    # Since the JavaScript can't directly modify session state, we use buttons
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
        ax.text(0.5, 0.98, "Quality ROI Analysis Report", fontsize=16, fontweight='bold', ha='center')
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

# --- AI ASSISTANT FUNCTIONS ---

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

# --- QUALITY ANALYSIS FUNCTIONS ---

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
            
            if results["solution_metrics"]["revenue_change"] > 0:
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
            risk_color = DANGER_COLOR if risk_level == "High" else (WARNING_COLOR if risk_level == "Medium" else SUCCESS_COLOR)
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
                new_revenue = results["financial_impact"]["monthly_revenue"] + results["financial_impact"]["revenue_change"]
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
                y=[returns_30d := results["financial_impact"]["returns_30d"]],
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
                line=dict(color=TEXT_SECONDARY, width=2, dash="dot"),
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

def get_download_link(df, filename, link_text):
    """Generate a download link for a DataFrame"""
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}" class="download-link">{link_text}</a>'
    return href

def show_batch_analysis_option_button():
    """Display button to navigate to batch analysis"""
    if st.button("Batch Analysis Import", key="batch_button", use_container_width=True, help="Upload a CSV or Excel file to analyze multiple products at once"):
        navigate_to_page("batch_analysis")

def show_help_button():
    """Display button to navigate to help page"""
    if st.button("Formula Guide", key="help_button", use_container_width=True, help="View detailed explanations of formulas used in the analysis"):
        navigate_to_page("help")

# --- MAIN APPLICATION ---

def main():
    """Main application function"""
    
    # Set sidebar style
    st.sidebar.markdown('<div class="sidebar-title">QualityROI CBA Tool</div>', unsafe_allow_html=True)
    
    # Simple navigation menu in sidebar
    st.sidebar.markdown("### Navigation")
    
    # Navigation buttons
    col1, col2 = st.sidebar.columns(2)
    with col1:
        if st.button("Single Analysis", key="nav_single", use_container_width=True):
            navigate_to_page("analysis")
    with col2:
        if st.button("Batch Analysis", key="nav_batch", use_container_width=True):
            navigate_to_page("batch_analysis")
    
    if st.sidebar.button("Help & Formulas", key="nav_help", use_container_width=True):
        navigate_to_page("help")
    
    # Authenticate with simple password (in production you'd use more secure methods)
    password = "MPFvive8955@#@"  # This would normally be stored securely
    
    # Simple password check just for demo
    if 'authenticated' not in st.session_state:
        st.session_state.authenticated = False
    
    if not st.session_state.authenticated:
        st.sidebar.markdown("### Authentication")
        entered_password = st.sidebar.text_input("Enter password:", type="password")
        if st.sidebar.button("Login"):
            if entered_password == password:
                st.session_state.authenticated = True
                st.rerun()
            else:
                st.sidebar.error("Incorrect password")
    else:
        # Show logout button
        if st.sidebar.button("Logout"):
            st.session_state.authenticated = False
            st.rerun()
    
    if not st.session_state.authenticated:
        st.title("QualityROI CBA Tool")
        st.write("Please login to access the tool.")
        return
    
    # Display main application based on current page
    if st.session_state.current_page == "analysis":
        display_analysis_page()
    elif st.session_state.current_page == "batch_analysis":
        display_batch_analysis_page()
    elif st.session_state.current_page == "help":
        display_help_page()
    else:
        display_analysis_page()  # Default to analysis page

def display_analysis_page():
    """Display the single product analysis page"""
    # Display header
    vive_header()
    st.markdown('<div class="main-header">Medical Device Quality ROI Analysis</div>', unsafe_allow_html=True)
    
    # Display mode toggle
    display_mode_toggle()
    
    # Navigation buttons
    col1, col2 = st.columns(2)
    with col1:
        show_batch_analysis_option_button()
    with col2:
        show_help_button()
    
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
                    help="Product SKU (e.g., MPF-1234)"
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
                        "FBA/Fulfillment Fee", 
                        min_value=0.0,
                        help="Amazon FBA fee or other fulfillment fee per unit, if applicable"
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
                            "ASIN", 
                            help="Amazon Standard Identification Number for products sold on Amazon"
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
                            help="Current average star rating on Amazon or other marketplace"
                        )
                        
                        total_reviews = st.number_input(
                            "Total Reviews", 
                            min_value=0,
                            help="Total number of reviews on Amazon or other marketplace"
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
                        f"${st.session_state.quality_analysis
