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
from io import BytesIO

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="Product Profitability Analysis", page_icon="📊", layout="wide", initial_sidebar_state="expanded")
# --- THEME COLORS ---
PRIMARY_COLOR   = "#0096C7"
SECONDARY_COLOR = "#48CAE4"
TERTIARY_COLOR  = "#90E0EF"
BACKGROUND_COLOR= "#F8F9FA"
CARD_BACKGROUND = "#FFFFFF"
TEXT_PRIMARY    = "#212529"
TEXT_SECONDARY  = "#6C757D"
TEXT_MUTED      = "#ADB5BD"
SUCCESS_COLOR   = "#40916C"
WARNING_COLOR   = "#E9C46A"
DANGER_COLOR    = "#E76F51"
BORDER_COLOR    = "#DEE2E6"

# --- SESSION STATE INITIALIZATION ---
if 'quality_analysis_results' not in st.session_state:
    st.session_state.quality_analysis_results = None
if 'analysis_submitted' not in st.session_state:
    st.session_state.analysis_submitted = False
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'current_page' not in st.session_state:
    st.session_state.current_page = "analysis"
if 'view_mode' not in st.session_state:
    st.session_state.view_mode = "basic"
if 'batch_analysis_results' not in st.session_state:
    st.session_state.batch_analysis_results = {}
if 'tariff_calculations' not in st.session_state:
    st.session_state.tariff_calculations = None
if 'monte_carlo_scenario' not in st.session_state:
    st.session_state.monte_carlo_scenario = None
if 'compare_list' not in st.session_state:
    st.session_state.compare_list = []

# --- CSS INJECTION ---
st.markdown(f"""
<style>
    :root {{ --primary: {PRIMARY_COLOR}; --secondary: {SECONDARY_COLOR}; --tertiary: {TERTIARY_COLOR}; 
             --background: {BACKGROUND_COLOR}; --card-bg: {CARD_BACKGROUND}; --text-primary: {TEXT_PRIMARY}; 
             --text-secondary: {TEXT_SECONDARY}; --text-muted: {TEXT_MUTED}; --success: {SUCCESS_COLOR}; 
             --warning: {WARNING_COLOR}; --danger: {DANGER_COLOR}; --border: {BORDER_COLOR}; }}
    .metric-card {{ 
        background-color: var(--card-bg); 
        border-radius: 8px; 
        padding: 1rem; 
        box-shadow: 0 2px 4px rgba(0,0,0,0.1); 
        margin-bottom: 1rem; 
        position: relative;
        border-left: 3px solid var(--primary);
    }}
    .metric-card:hover::after {{
        content: attr(data-tooltip);
        position: absolute;
        bottom: 100%;
        left: 50%;
        transform: translateX(-50%);
        background-color: rgba(0,0,0,0.8);
        color: white;
        padding: 5px 10px;
        border-radius: 5px;
        font-size: 12px;
        white-space: nowrap;
        z-index: 99;
        opacity: 0.9;
    }}
    .metric-label {{ font-size: 0.85rem; color: var(--text-secondary); margin-bottom: 0.25rem; }}
    .metric-value {{ font-size: 1.5rem; font-weight: 600; color: var(--text-primary); }}
    .metric-subvalue {{ font-size: 0.8rem; color: var(--text-muted); }}
    .badge {{ padding: 0.25rem 0.5rem; border-radius: 4px; font-size: 0.75rem; font-weight: 600; }}
    .badge-success {{ background-color: var(--success); color: white; }}
    .badge-warning {{ background-color: var(--warning); color: white; }}
    .badge-danger {{ background-color: var(--danger); color: white; }}
    .assistant-bubble {{ background:white; border-left:3px solid var(--primary); padding:0.75rem; 
                         margin-bottom: 0.5rem; border-radius: 0 4px 4px 0; }}
    .user-bubble {{ background: {TERTIARY_COLOR}; padding:0.75rem; margin-left:auto; 
                    margin-bottom: 0.5rem; border-radius: 4px 0 0 4px; max-width: 80%; }}
    .stTabs [data-baseweb="tab-list"] {{ gap: 1rem; }}
    .stTabs [data-baseweb="tab"] {{ 
        height: 3rem; 
        white-space: pre-wrap; 
        background-color: white;
        border-radius: 4px 4px 0 0; 
        gap: 0.5rem; 
        padding-top: 0.5rem; 
        font-weight: 500;
    }}
    .stTabs [aria-selected="true"] {{ 
        background-color: {PRIMARY_COLOR} !important; 
        color: white !important; 
        font-weight: 600;
    }}
    div.block-container {{ padding-top: 2rem; padding-bottom: 2rem; }}
    .export-button {{ 
        background-color: var(--primary); 
        color: white; 
        padding: 0.5rem 1rem;
        border-radius: 4px; 
        text-decoration: none; 
        display: inline-block; 
        margin-right: 0.5rem;
        transition: background-color 0.3s ease;
    }}
    .export-button:hover {{ 
        background-color: var(--secondary); 
        text-decoration: none;
        color: white;
    }}
    .stButton > button {{
        background-color: var(--primary);
        color: white;
        border: none;
        border-radius: 4px;
        padding: 0.5rem 1rem;
        font-weight: 500;
        transition: background-color 0.3s ease;
    }}
    .stButton > button:hover {{
        background-color: var(--secondary);
    }}
    /* Tooltip styling */
    .tooltip {{
        position: relative;
        display: inline-block;
    }}
    .tooltip .tooltiptext {{
        visibility: hidden;
        width: 200px;
        background-color: #555;
        color: #fff;
        text-align: center;
        border-radius: 6px;
        padding: 5px;
        position: absolute;
        z-index: 1;
        bottom: 125%;
        left: 50%;
        margin-left: -100px;
        opacity: 0;
        transition: opacity 0.3s;
    }}
    .tooltip:hover .tooltiptext {{
        visibility: visible;
        opacity: 1;
    }}
    /* Form styling */
    .stNumberInput input, .stTextInput input, .stSelectbox, .stTextArea textarea {{
        border-radius: 4px;
        border: 1px solid #dee2e6;
    }}
    .stNumberInput input:focus, .stTextInput input:focus, .stSelectbox:focus, .stTextArea textarea:focus {{
        border-color: var(--primary);
        box-shadow: 0 0 0 0.2rem rgba(0,150,199,0.25);
    }}
    /* Header Styling */
    h1, h2, h3, h4, h5, h6 {{
        color: var(--text-primary);
        font-weight: 600;
    }}
    h1 {{
        margin-bottom: 1rem;
        border-bottom: 2px solid var(--primary);
        padding-bottom: 0.5rem;
    }}
    /* Loader styling */
    .stSpinner > div {{
        border-top-color: var(--primary) !important;
    }}
</style>
""", unsafe_allow_html=True)

# --- LOGGING SETUP ---
import logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# --- SENTRY INTEGRATION ---
try:
    import sentry_sdk
    sentry_dsn = st.secrets.get("sentry_dsn", None)
    if sentry_dsn:
        sentry_sdk.init(
            dsn=st.secrets.get("sentry_dsn"),
            traces_sample_rate=0.2,
            release="product-profitability-analysis@1.0.0"
        )
        logger.info("Sentry integration initialized")
    else:
        logger.warning("Sentry DSN not found in secrets")
except ImportError:
    logger.warning("Sentry SDK not installed")
except Exception as e:
    logger.exception("Error initializing Sentry")

# --- UTILITY FUNCTIONS ---
def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """Safely divide two numbers, returning default if denominator is zero."""
    try:
        if denominator == 0:
            return default
        return numerator / denominator
    except Exception:
        return default

def format_currency(value: float) -> str:
    """Format a value as a currency string."""
    if value >= 1000000:
        return f"${value/1000000:.2f}M"
    elif value >= 1000:
        return f"${value/1000:.1f}K"
    else:
        return f"${value:.2f}"

def format_percentage(value: float) -> str:
    """Format a value as a percentage string."""
    return f"{value:.2f}%"

def generate_download_link(df: pd.DataFrame, filename: str, text: str) -> str:
    """Generate a download link for a DataFrame as CSV."""
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}" class="export-button">{text}</a>'
    return href

def generate_color_scale(value: float, good_threshold: float, bad_threshold: float) -> str:
    """Generate a color from green to red based on a value and thresholds."""
    if value >= good_threshold:
        return SUCCESS_COLOR
    elif value <= bad_threshold:
        return DANGER_COLOR
    else:
        # Linear interpolation between warning and success colors
        ratio = (value - bad_threshold) / (good_threshold - bad_threshold)
        return WARNING_COLOR

def get_recommendation_badge(recommendation: str) -> str:
    """Generate an HTML badge for a recommendation."""
    if "Fix Immediately" in recommendation:
        return f'<span class="badge badge-danger">{recommendation}</span>'
    elif "High Priority" in recommendation:
        return f'<span class="badge badge-warning">{recommendation}</span>'
    else:
        return f'<span class="badge badge-success">{recommendation}</span>'

# --- AI ASSISTANT FUNCTIONS ---
def call_openai_api(messages, model="gpt-4o", temperature=0.7, max_tokens=1024):
    """Call the OpenAI API with the given messages."""
    api_key = st.secrets.get("openai_api_key", None)
    if not api_key:
        return "AI assistant not available. API key not configured."
    
    try:
        headers = {"Content-Type":"application/json","Authorization":f"Bearer {api_key}"}
        payload = {"model": model, "messages": messages, "temperature": temperature, "max_tokens": max_tokens}
        response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload, timeout=30)
        
        if response.status_code == 200:
            return response.json()["choices"][0]["message"]["content"]
        else:
            logger.error(f"OpenAI API error: {response.status_code} - {response.text}")
            return f"Error from API: {response.status_code} - {response.text}"
    except Exception as e:
        logger.exception("Error calling OpenAI API")
        return f"Error calling AI assistant: {str(e)}"

def get_system_prompt(results: Dict[str, Any]) -> str:
    """Generate the system prompt for the AI assistant based on analysis results."""
    if not results:
        return """
        You are a Quality Management expert for medical devices. 
        Provide guidance on quality issues, cost analysis, and regulatory compliance.
        """
    
    return f"""
    You are a Quality Management expert for medical devices.
    
    Product details:
    - SKU: {results['sku']}
    - Product Type: {results['product_type']}
    - Issue Description: {results['issue_description']}
    - Return Rate (30d): {results['current_metrics']['return_rate_30d']:.2f}%
    - Current Unit Cost: ${results['current_metrics']['unit_cost']:.2f}
    - Sales Price: ${results['current_metrics']['sales_price']:.2f}
    
    Financial Impact:
    - Annual Loss Due to Returns: ${results['financial_impact']['annual_loss']:.2f}
    - ROI (3yr): {results['financial_impact']['roi_3yr']:.2f}%
    - Payback Period: {results['financial_impact']['payback_period']:.2f} months
    
    Recommendation: {results['recommendation']}
    
    Your task is to provide expert advice on:
    1. Next steps based on the analysis
    2. Additional data that might be needed
    3. Regulatory considerations for medical devices
    4. Best practices for implementing the recommended solution
    
    Be concise but thorough in your responses.
    """

# --- CORE ANALYSIS FUNCTIONS ---
def analyze_quality_issue(
    sku: str,
    product_type: str,
    sales_30d: float,
    returns_30d: float,
    issue_description: str,
    current_unit_cost: float,
    fix_cost_upfront: float,
    fix_cost_per_unit: float,
    sales_price: float,
    expected_reduction: float,
    solution_confidence: float,
    annualized_growth: float = 0.0,
    regulatory_risk: int = 1,
    brand_risk: int = 1,
    medical_risk: int = 1
) -> Dict[str, Any]:
    """
    Analyze a quality issue and calculate financial impact, ROI, and recommendations.
    
    Args:
        sku: Product SKU identifier
        product_type: Type of medical device (B2C, B2B, etc.)
        sales_30d: Number of units sold in the last 30 days
        returns_30d: Number of units returned in the last 30 days
        issue_description: Description of the quality issue
        current_unit_cost: Current manufacturing cost per unit
        fix_cost_upfront: One-time cost to implement the fix
        fix_cost_per_unit: Additional cost per unit after implementing the fix
        sales_price: Sales price per unit
        expected_reduction: Expected percentage reduction in returns after fix (0-100)
        solution_confidence: Confidence level in the solution (0-100)
        annualized_growth: Expected annual growth rate (default: 0%)
        regulatory_risk: Regulatory risk level (1-5, where 5 is highest)
        brand_risk: Brand reputation risk level (1-5, where 5 is highest)
        medical_risk: Medical/patient risk level (1-5, where 5 is highest)
    
    Returns:
        Dictionary containing analysis results
    """
    logger.info(f"Starting analysis for SKU={sku}, sales_30d={sales_30d}")
    
    try:
        # Input validation
        if sales_30d < 0 or returns_30d < 0 or current_unit_cost < 0 or fix_cost_upfront < 0 or fix_cost_per_unit < 0:
            raise ValueError("Input values cannot be negative")
        
        # Calculate current metrics
        return_rate_30d = safe_divide(returns_30d, sales_30d, 0) * 100
        
        # Calculate annualized metrics
        annual_sales = sales_30d * 12
        annual_returns = returns_30d * 12
        
        # Calculate financial impact of current situation
        margin_per_unit = sales_price - current_unit_cost
        margin_percentage = safe_divide(margin_per_unit, sales_price, 0) * 100
        loss_per_return = current_unit_cost + (sales_price * 0.15)  # Cost + 15% of price for handling
        annual_loss = annual_returns * loss_per_return
        
        # Calculate impact of proposed solution
        new_unit_cost = current_unit_cost + fix_cost_per_unit
        new_margin_per_unit = sales_price - new_unit_cost
        new_margin_percentage = safe_divide(new_margin_per_unit, sales_price, 0) * 100
        
        expected_returns_reduction = expected_reduction / 100
        reduced_returns = annual_returns * (1 - expected_returns_reduction)
        returns_prevented = annual_returns - reduced_returns
        savings = returns_prevented * loss_per_return
        
        # Apply confidence factor
        confidence_factor = solution_confidence / 100
        adjusted_savings = savings * confidence_factor
        
        # Calculate ROI and payback period
        implementation_cost = fix_cost_upfront + (annual_sales * fix_cost_per_unit)
        roi_1yr = safe_divide(adjusted_savings - implementation_cost, implementation_cost, 0) * 100
        
        # 3-year ROI with growth
        future_sales = annual_sales
        future_returns = reduced_returns
        cumulative_savings = 0
        cumulative_extra_cost = fix_cost_upfront
        
        for year in range(1, 4):
            future_sales *= (1 + annualized_growth)
            future_returns *= (1 + annualized_growth)
            returns_without_fix = (future_sales / annual_sales) * annual_returns
            returns_prevented_future = returns_without_fix - future_returns
            yearly_savings = returns_prevented_future * loss_per_return * confidence_factor
            yearly_extra_cost = future_sales * fix_cost_per_unit
            
            cumulative_savings += yearly_savings
            cumulative_extra_cost += yearly_extra_cost
        
        total_cost_3yr = fix_cost_upfront + cumulative_extra_cost
        roi_3yr = safe_divide(cumulative_savings - total_cost_3yr, total_cost_3yr, 0) * 100
        
        # Calculate payback period in months
        monthly_net_benefit = safe_divide(adjusted_savings - (annual_sales * fix_cost_per_unit), 12, 0)
        payback_period = safe_divide(fix_cost_upfront, monthly_net_benefit, float('inf'))
        
        # Calculate risk-adjusted metrics
        risk_factor = (regulatory_risk + brand_risk + medical_risk) / 3
        priority_score = (roi_3yr * 0.4) + ((5 - payback_period) * 10 * 0.3) + (risk_factor * 20 * 0.3)
        
        # Generate recommendation
        if medical_risk >= 4 or regulatory_risk >= 4:
            recommendation = "Fix Immediately - Compliance Risk"
        elif roi_3yr >= 200 and payback_period <= 6:
            recommendation = "High Priority - Strong ROI"
        elif roi_3yr >= 100 or risk_factor >= 3:
            recommendation = "Medium Priority - Good ROI"
        elif roi_3yr > 0:
            recommendation = "Consider Fix - Positive ROI"
        else:
            recommendation = "Monitor - Negative ROI"
        
        # Compile results
        results = {
            "sku": sku,
            "product_type": product_type,
            "issue_description": issue_description,
            "analysis_date": datetime.now().strftime("%Y-%m-%d"),
            "current_metrics": {
                "sales_30d": sales_30d,
                "returns_30d": returns_30d,
                "return_rate_30d": return_rate_30d,
                "annual_sales": annual_sales,
                "annual_returns": annual_returns,
                "unit_cost": current_unit_cost,
                "sales_price": sales_price,
                "margin_per_unit": margin_per_unit,
                "margin_percentage": margin_percentage,
                "loss_per_return": loss_per_return  # Added this field
            },
            "solution_metrics": {
                "fix_cost_upfront": fix_cost_upfront,
                "fix_cost_per_unit": fix_cost_per_unit,
                "new_unit_cost": new_unit_cost,
                "new_margin_per_unit": new_margin_per_unit,
                "new_margin_percentage": new_margin_percentage,
                "expected_reduction": expected_reduction,
                "solution_confidence": solution_confidence
            },
            "financial_impact": {
                "annual_loss": annual_loss,
                "returns_prevented": returns_prevented,
                "savings": savings,
                "adjusted_savings": adjusted_savings,
                "implementation_cost": implementation_cost,
                "roi_1yr": roi_1yr,
                "roi_3yr": roi_3yr,
                "payback_period": payback_period
            },
            "risk_assessment": {
                "regulatory_risk": regulatory_risk,
                "brand_risk": brand_risk,
                "medical_risk": medical_risk,
                "risk_factor": risk_factor,
                "priority_score": priority_score
            },
            "recommendation": recommendation
        }
        
        logger.debug(f"Computed ROI: {roi_3yr:.2f}%")
        return results
    
    except Exception as e:
        logger.exception("Error in analyze_quality_issue")
        raise

def calculate_landed_cost(
    sales_price: float,
    cogs: float,
    tariff_rate: float,
    shipping_cost: float = 0.0,
    storage_cost: float = 0.0,
    customs_fee: float = 0.0,
    broker_fee: float = 0.0,
    other_costs: float = 0.0,
    units_per_shipment: int = 1
) -> Dict[str, Any]:
    """
    Calculate landed cost and impact on margins with tariffs.
    
    Args:
        sales_price: Sales price per unit
        cogs: Cost of goods sold per unit (manufacturing cost)
        tariff_rate: Tariff percentage (0-100)
        shipping_cost: Cost of shipping entire shipment
        storage_cost: Cost of storage per unit
        customs_fee: Customs processing fee per shipment
        broker_fee: Customs broker fee per shipment
        other_costs: Other costs per unit
        units_per_shipment: Number of units per shipment
    
    Returns:
        Dictionary containing landed cost analysis
    """
    try:
        # Calculate per-unit costs
        per_unit_shipping = safe_divide(shipping_cost, units_per_shipment, 0)
        per_unit_customs = safe_divide(customs_fee, units_per_shipment, 0)
        per_unit_broker = safe_divide(broker_fee, units_per_shipment, 0)
        
        # Calculate tariff amount
        tariff_amount = (cogs * tariff_rate) / 100
        
        # Calculate total landed cost
        landed_cost = (
            cogs + 
            tariff_amount + 
            per_unit_shipping + 
            storage_cost + 
            per_unit_customs + 
            per_unit_broker + 
            other_costs
        )
        
        # Calculate margins
        original_margin = sales_price - cogs
        original_margin_percentage = safe_divide(original_margin, sales_price, 0) * 100
        
        new_margin = sales_price - landed_cost
        new_margin_percentage = safe_divide(new_margin, sales_price, 0) * 100
        
        margin_impact = original_margin_percentage - new_margin_percentage
        margin_impact_dollars = original_margin - new_margin
        
        # Breakeven analysis
        breakeven_price = landed_cost / (1 - (original_margin_percentage / 100))
        price_increase_needed = breakeven_price - sales_price
        price_increase_percentage = safe_divide(price_increase_needed, sales_price, 0) * 100
        
        return {
            "original_cogs": cogs,
            "tariff_rate": tariff_rate,
            "tariff_amount": tariff_amount,
            "shipping_cost": shipping_cost,
            "per_unit_shipping": per_unit_shipping,
            "storage_cost": storage_cost,
            "customs_fee": customs_fee,
            "per_unit_customs": per_unit_customs,
            "broker_fee": broker_fee,
            "per_unit_broker": per_unit_broker,
            "other_costs": other_costs,
            "landed_cost": landed_cost,
            "sales_price": sales_price,
            "original_margin": original_margin,
            "original_margin_percentage": original_margin_percentage,
            "new_margin": new_margin,
            "new_margin_percentage": new_margin_percentage,
            "margin_impact": margin_impact,
            "margin_impact_dollars": margin_impact_dollars,
            "breakeven_price": breakeven_price,
            "price_increase_needed": price_increase_needed,
            "price_increase_percentage": price_increase_percentage
        }
    except Exception as e:
        logger.exception("Error in calculate_landed_cost")
        raise

def generate_tariff_scenarios(
    sales_price: float,
    cogs: float,
    base_tariff: float,
    shipping_cost: float = 0.0,
    storage_cost: float = 0.0,
    customs_fee: float = 0.0,
    broker_fee: float = 0.0,
    other_costs: float = 0.0,
    units_per_shipment: int = 1
) -> Dict[str, Dict[str, Any]]:
    """
    Generate multiple tariff scenarios for analysis.
    
    Args:
        sales_price: Sales price per unit
        cogs: Cost of goods sold per unit
        base_tariff: Current tariff percentage
        Other parameters: Same as calculate_landed_cost
    
    Returns:
        Dictionary of scenarios with landed cost calculations
    """
    tariff_rates = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
    scenarios = {}
    
    for rate in tariff_rates:
        scenario_name = f"{rate}% Tariff"
        scenarios[scenario_name] = calculate_landed_cost(
            sales_price=sales_price,
            cogs=cogs,
            tariff_rate=rate,
            shipping_cost=shipping_cost,
            storage_cost=storage_cost,
            customs_fee=customs_fee,
            broker_fee=broker_fee,
            other_costs=other_costs,
            units_per_shipment=units_per_shipment
        )
    
    return scenarios

def calculate_ad_roi(
    ad_spend: float,
    impressions: float,
    clicks: float,
    conversions: float,
    avg_order_value: float,
    contribution_margin_percent: float
) -> Dict[str, Any]:
    """
    Calculate ROI and key metrics for advertising campaigns.
    
    Args:
        ad_spend: Total advertising spend
        impressions: Number of ad impressions
        clicks: Number of clicks on ads
        conversions: Number of conversions (sales)
        avg_order_value: Average order value
        contribution_margin_percent: Contribution margin percentage (0-100)
    
    Returns:
        Dictionary containing ad performance metrics
    """
    try:
        # Calculate basic metrics
        ctr = safe_divide(clicks, impressions, 0) * 100  # Click-through rate
        conversion_rate = safe_divide(conversions, clicks, 0) * 100  # Conversion rate
        
        # Calculate cost metrics
        cpm = safe_divide(ad_spend * 1000, impressions, 0)  # Cost per thousand impressions
        cpc = safe_divide(ad_spend, clicks, 0)  # Cost per click
        cpa = safe_divide(ad_spend, conversions, 0)  # Cost per acquisition
        
        # Calculate revenue and profit
        revenue = conversions * avg_order_value
        contribution_margin = contribution_margin_percent / 100
        profit = revenue * contribution_margin
        
        # Calculate ROI
        roi = safe_divide(profit - ad_spend, ad_spend, -100) * 100
        
        # Calculate ROAS (Return on Ad Spend)
        roas = safe_divide(revenue, ad_spend, 0)
        
        return {
            "ad_spend": ad_spend,
            "impressions": impressions,
            "clicks": clicks,
            "conversions": conversions,
            "avg_order_value": avg_order_value,
            "contribution_margin_percent": contribution_margin_percent,
            "ctr": ctr,
            "conversion_rate": conversion_rate,
            "cpm": cpm,
            "cpc": cpc,
            "cpa": cpa,
            "revenue": revenue,
            "profit": profit,
            "roi": roi,
            "roas": roas
        }
    except Exception as e:
        logger.exception("Error in calculate_ad_roi")
        raise

def run_monte_carlo_simulation(
    base_unit_cost: float,
    base_sales_price: float,
    base_monthly_sales: float,
    base_return_rate: float,
    fix_cost_upfront: float,
    fix_cost_per_unit: float,
    expected_reduction: float,
    iterations: int = 1000,
    cost_std_dev: float = 0.05,
    price_std_dev: float = 0.03,
    sales_std_dev: float = 0.10,
    return_std_dev: float = 0.15,
    reduction_std_dev: float = 0.20
) -> Dict[str, Any]:
    """
    Run a Monte Carlo simulation to analyze risk and probability distributions.
    
    Args:
        base_unit_cost: Base manufacturing cost per unit
        base_sales_price: Base sales price per unit
        base_monthly_sales: Base monthly sales units
        base_return_rate: Base return rate (0-100)
        fix_cost_upfront: One-time cost to implement the fix
        fix_cost_per_unit: Additional cost per unit after fix
        expected_reduction: Expected percentage reduction in returns (0-100)
        iterations: Number of simulation iterations
        cost_std_dev: Standard deviation for unit cost variations
        price_std_dev: Standard deviation for price variations
        sales_std_dev: Standard deviation for sales volume variations
        return_std_dev: Standard deviation for return rate variations
        reduction_std_dev: Standard deviation for expected reduction variations
    
    Returns:
        Dictionary containing simulation results
    """
    np.random.seed(42)  # For reproducibility
    
    try:
        # Initialize arrays to store simulation results
        roi_results = np.zeros(iterations)
        payback_results = np.zeros(iterations)
        savings_results = np.zeros(iterations)
        
        for i in range(iterations):
            # Generate random variations of input parameters
            unit_cost = np.random.normal(base_unit_cost, base_unit_cost * cost_std_dev)
            unit_cost = max(unit_cost, 0.1)  # Ensure positive
            
            sales_price = np.random.normal(base_sales_price, base_sales_price * price_std_dev)
            sales_price = max(sales_price, unit_cost * 1.1)  # Ensure price > cost
            
            monthly_sales = np.random.normal(base_monthly_sales, base_monthly_sales * sales_std_dev)
            monthly_sales = max(monthly_sales, 1)  # Ensure positive
            
            return_rate = np.random.normal(base_return_rate, base_return_rate * return_std_dev)
            return_rate = np.clip(return_rate, 0.1, 100)  # Ensure between 0.1% and 100%
            
            reduction = np.random.normal(expected_reduction, expected_reduction * reduction_std_dev)
            reduction = np.clip(reduction, 0, 100)  # Ensure between 0% and 100%
            
            # Calculate financial metrics for this iteration
            annual_sales = monthly_sales * 12
            annual_returns = (annual_sales * return_rate) / 100
            
            loss_per_return = unit_cost + (sales_price * 0.15)
            annual_loss = annual_returns * loss_per_return
            
            new_unit_cost = unit_cost + fix_cost_per_unit
            
            reduced_returns = annual_returns * (1 - (reduction / 100))
            returns_prevented = annual_returns - reduced_returns
            savings = returns_prevented * loss_per_return
            
            implementation_cost = fix_cost_upfront + (annual_sales * fix_cost_per_unit)
            roi = safe_divide(savings - implementation_cost, implementation_cost, -100) * 100
            
            monthly_net_benefit = safe_divide(savings - (annual_sales * fix_cost_per_unit), 12, 0.01)
            payback = safe_divide(fix_cost_upfront, monthly_net_benefit, 120)  # Cap at 10 years
            
            # Store results
            roi_results[i] = roi
            payback_results[i] = payback
            savings_results[i] = savings
        
        # Analyze results
        roi_mean = np.mean(roi_results)
        roi_median = np.median(roi_results)
        roi_std = np.std(roi_results)
        roi_min = np.min(roi_results)
        roi_max = np.max(roi_results)
        
        payback_mean = np.mean(payback_results)
        payback_median = np.median(payback_results)
        payback_std = np.std(payback_results)
        payback_min = np.min(payback_results)
        payback_max = np.max(payback_results)
        
        savings_mean = np.mean(savings_results)
        savings_median = np.median(savings_results)
        savings_std = np.std(savings_results)
        savings_min = np.min(savings_results)
        savings_max = np.max(savings_results)
        
        # Calculate probability of positive ROI
        prob_positive_roi = np.sum(roi_results > 0) / iterations * 100
        
        # Calculate probability of payback within 12 months
        prob_payback_1yr = np.sum(payback_results <= 12) / iterations * 100
        
        # Calculate percentiles for ROI
        roi_percentiles = {
            "p10": np.percentile(roi_results, 10),
            "p25": np.percentile(roi_results, 25),
            "p50": np.percentile(roi_results, 50),
            "p75": np.percentile(roi_results, 75),
            "p90": np.percentile(roi_results, 90)
        }
        
        # Calculate percentiles for payback
        payback_percentiles = {
            "p10": np.percentile(payback_results, 10),
            "p25": np.percentile(payback_results, 25),
            "p50": np.percentile(payback_results, 50),
            "p75": np.percentile(payback_results, 75),
            "p90": np.percentile(payback_results, 90)
        }
        
        # Prepare histogram data for ROI and payback
        roi_hist, roi_bins = np.histogram(roi_results, bins=20)
        payback_hist, payback_bins = np.histogram(payback_results, bins=20)
        
        return {
            "iterations": iterations,
            "input_parameters": {
                "base_unit_cost": base_unit_cost,
                "base_sales_price": base_sales_price,
                "base_monthly_sales": base_monthly_sales,
                "base_return_rate": base_return_rate,
                "fix_cost_upfront": fix_cost_upfront,
                "fix_cost_per_unit": fix_cost_per_unit,
                "expected_reduction": expected_reduction
            },
            "roi_stats": {
                "mean": roi_mean,
                "median": roi_median,
                "std_dev": roi_std,
                "min": roi_min,
                "max": roi_max,
                "percentiles": roi_percentiles,
                "histogram": {
                    "counts": roi_hist.tolist(),
                    "bins": roi_bins.tolist()
                }
            },
            "payback_stats": {
                "mean": payback_mean,
                "median": payback_median,
                "std_dev": payback_std,
                "min": payback_min,
                "max": payback_max,
                "percentiles": payback_percentiles,
                "histogram": {
                    "counts": payback_hist.tolist(),
                    "bins": payback_bins.tolist()
                }
            },
            "savings_stats": {
                "mean": savings_mean,
                "median": savings_median,
                "std_dev": savings_std,
                "min": savings_min,
                "max": savings_max
            },
            "probability_metrics": {
                "prob_positive_roi": prob_positive_roi,
                "prob_payback_1yr": prob_payback_1yr
            }
        }
    except Exception as e:
        logger.exception("Error in run_monte_carlo_simulation")
        raise

# --- EXPORT FUNCTIONS ---
def export_as_csv(results: Dict[str, Any]) -> pd.DataFrame:
    """Convert analysis results to a DataFrame for CSV export."""
    if not results:
        return pd.DataFrame()
    
    data = {
        "Metric": [
            "SKU",
            "Product Type",
            "Analysis Date",
            "Return Rate (30d)",
            "Annual Return Rate",
            "Annual Loss Due to Returns",
            "Fix Cost (Upfront)",
            "Fix Cost (Per Unit)",
            "Expected Reduction in Returns",
            "ROI (1 Year)",
            "ROI (3 Year)",
            "Payback Period (Months)",
            "Recommendation"
        ],
        "Value": [
            results["sku"],
            results["product_type"],
            results["analysis_date"],
            f"{results['current_metrics']['return_rate_30d']:.2f}%",
            f"{results['current_metrics']['annual_return_rate']:.2f}%",
            f"${results['financial_impact']['annual_loss']:.2f}",
            f"${results['solution_metrics']['fix_cost_upfront']:.2f}",
            f"${results['solution_metrics']['fix_cost_per_unit']:.2f}",
            f"{results['solution_metrics']['expected_reduction']:.2f}%",
            f"{results['financial_impact']['roi_1yr']:.2f}%",
            f"{results['financial_impact']['roi_3yr']:.2f}%",
            f"{results['financial_impact']['payback_period']:.2f}",
            results["recommendation"]
        ]
    }
    
    return pd.DataFrame(data)

def export_as_pdf(results: Dict[str, Any]) -> BytesIO:
    """Generate a PDF report of the analysis results."""
    if not results:
        return BytesIO()
    
    buffer = BytesIO()
    
    try:
        with PdfPages(buffer) as pdf:
            # Set up the figure for the first page
            plt.figure(figsize=(8.5, 11))
            plt.suptitle(f"Quality Issue Analysis: {results['sku']}", fontsize=16)
            plt.text(0.1, 0.9, f"Product: {results['product_type']}", fontsize=12)
            plt.text(0.1, 0.85, f"Analysis Date: {results['analysis_date']}", fontsize=12)
            plt.text(0.1, 0.8, f"Issue: {results['issue_description']}", fontsize=12)
            
            # Key metrics
            plt.text(0.1, 0.7, "Key Metrics:", fontsize=14, weight='bold')
            plt.text(0.1, 0.65, f"Return Rate (30d): {results['current_metrics']['return_rate_30d']:.2f}%", fontsize=12)
            plt.text(0.1, 0.6, f"Annual Loss: ${results['financial_impact']['annual_loss']:.2f}", fontsize=12)
            plt.text(0.1, 0.55, f"ROI (3 Year): {results['financial_impact']['roi_3yr']:.2f}%", fontsize=12)
            plt.text(0.1, 0.5, f"Payback Period: {results['financial_impact']['payback_period']:.2f} months", fontsize=12)
            plt.text(0.1, 0.45, f"Recommendation: {results['recommendation']}", fontsize=12, weight='bold')
            
            # Add charts
            ax1 = plt.axes([0.1, 0.1, 0.35, 0.25])
            ax1.bar(['Current', 'After Fix'], 
                   [results['current_metrics']['return_rate_30d'], 
                    results['current_metrics']['return_rate_30d'] * (1 - results['solution_metrics']['expected_reduction']/100)])
            ax1.set_title('Return Rate')
            ax1.set_ylabel('Percentage')
            
            ax2 = plt.axes([0.55, 0.1, 0.35, 0.25])
            ax2.bar(['Current', 'After Fix'], 
                   [results['current_metrics']['margin_percentage'], 
                    results['solution_metrics']['new_margin_percentage']])
            ax2.set_title('Margin Percentage')
            ax2.set_ylabel('Percentage')
            
            plt.tight_layout(rect=[0, 0, 1, 0.95])
            pdf.savefig()
            plt.close()
            
            # Second page with financial details
            plt.figure(figsize=(8.5, 11))
            plt.suptitle("Financial Impact Details", fontsize=16)
            
            # Current state
            plt.text(0.1, 0.9, "Current State:", fontsize=14, weight='bold')
            plt.text(0.1, 0.85, f"Sales (30d): {results['current_metrics']['sales_30d']}", fontsize=12)
            plt.text(0.1, 0.8, f"Returns (30d): {results['current_metrics']['returns_30d']}", fontsize=12)
            plt.text(0.1, 0.75, f"Unit Cost: ${results['current_metrics']['unit_cost']:.2f}", fontsize=12)
            plt.text(0.1, 0.7, f"Sales Price: ${results['current_metrics']['sales_price']:.2f}", fontsize=12)
            plt.text(0.1, 0.65, f"Margin: ${results['current_metrics']['margin_per_unit']:.2f} ({results['current_metrics']['margin_percentage']:.2f}%)", fontsize=12)
            
            # Proposed solution
            plt.text(0.1, 0.55, "Proposed Solution:", fontsize=14, weight='bold')
            plt.text(0.1, 0.5, f"Upfront Cost: ${results['solution_metrics']['fix_cost_upfront']:.2f}", fontsize=12)
            plt.text(0.1, 0.45, f"Cost Per Unit: ${results['solution_metrics']['fix_cost_per_unit']:.2f}", fontsize=12)
            plt.text(0.1, 0.4, f"New Unit Cost: ${results['solution_metrics']['new_unit_cost']:.2f}", fontsize=12)
            plt.text(0.1, 0.35, f"New Margin: ${results['solution_metrics']['new_margin_per_unit']:.2f} ({results['solution_metrics']['new_margin_percentage']:.2f}%)", fontsize=12)
            plt.text(0.1, 0.3, f"Expected Reduction: {results['solution_metrics']['expected_reduction']:.2f}%", fontsize=12)
            plt.text(0.1, 0.25, f"Confidence: {results['solution_metrics']['solution_confidence']:.2f}%", fontsize=12)
            
            # Financial impact
            plt.text(0.5, 0.55, "Financial Impact:", fontsize=14, weight='bold')
            plt.text(0.5, 0.5, f"Annual Loss: ${results['financial_impact']['annual_loss']:.2f}", fontsize=12)
            plt.text(0.5, 0.45, f"Returns Prevented: {results['financial_impact']['returns_prevented']:.2f}", fontsize=12)
            plt.text(0.5, 0.4, f"Savings: ${results['financial_impact']['savings']:.2f}", fontsize=12)
            plt.text(0.5, 0.35, f"Adjusted Savings: ${results['financial_impact']['adjusted_savings']:.2f}", fontsize=12)
            plt.text(0.5, 0.3, f"Implementation Cost: ${results['financial_impact']['implementation_cost']:.2f}", fontsize=12)
            plt.text(0.5, 0.25, f"ROI (1yr): {results['financial_impact']['roi_1yr']:.2f}%", fontsize=12)
            plt.text(0.5, 0.2, f"ROI (3yr): {results['financial_impact']['roi_3yr']:.2f}%", fontsize=12)
            plt.text(0.5, 0.15, f"Payback Period: {results['financial_impact']['payback_period']:.2f} months", fontsize=12)
            
            plt.tight_layout(rect=[0, 0, 1, 0.95])
            pdf.savefig()
            plt.close()
            
            # Third page with charts
            plt.figure(figsize=(8.5, 11))
            plt.suptitle("Visual Analysis", fontsize=16)
            
            # ROI chart
            ax1 = plt.axes([0.1, 0.7, 0.8, 0.2])
            roi_values = [results['financial_impact']['roi_1yr'], results['financial_impact']['roi_3yr']]
            ax1.bar(['1 Year ROI', '3 Year ROI'], roi_values, color=['#48CAE4', '#0096C7'])
            for i, v in enumerate(roi_values):
                ax1.text(i, v + 5, f"{v:.1f}%", ha='center')
            ax1.set_title('Return on Investment')
            ax1.set_ylabel('Percentage')
            
            # Return rate reduction chart
            ax2 = plt.axes([0.1, 0.4, 0.8, 0.2])
            labels = ['Before', 'After']
            values = [
                results['current_metrics']['return_rate_30d'],
                results['current_metrics']['return_rate_30d'] * (1 - results['solution_metrics']['expected_reduction'] / 100)
            ]
            ax2.bar(labels, values, color=['#E76F51', '#40916C'])
            for i, v in enumerate(values):
                ax2.text(i, v + 0.5, f"{v:.2f}%", ha='center')
            ax2.set_title('Return Rate Reduction')
            ax2.set_ylabel('Return Rate (%)')
            
            # Margin impact chart
            ax3 = plt.axes([0.1, 0.1, 0.8, 0.2])
            margin_labels = ['Original Margin', 'New Margin']
            margin_values = [
                results['current_metrics']['margin_percentage'],
                results['solution_metrics']['new_margin_percentage']
            ]
            ax3.bar(margin_labels, margin_values, color=['#E9C46A', '#48CAE4'])
            for i, v in enumerate(margin_values):
                ax3.text(i, v + 0.5, f"{v:.2f}%", ha='center')
            ax3.set_title('Margin Impact')
            ax3.set_ylabel('Margin (%)')
            
            plt.tight_layout(rect=[0, 0, 1, 0.95])
            pdf.savefig()
            plt.close()
            
        buffer.seek(0)
        return buffer
    except Exception as e:
        logger.exception("Error generating PDF report")
        plt.close('all')  # Close any open figures
        raise

# --- UI COMPONENTS ---
def display_header():
    """Display the application header."""
    col1, col2 = st.columns([6, 1])
    with col1:
        st.title("🔍 Product Profitability Analysis")
        st.markdown("""
        <div style="margin-bottom: 1rem;">
            Analyze product quality issues, calculate ROI for improvement projects, and simulate financial scenarios.
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        with st.popover("📋 Help", use_container_width=True):
            st.markdown("""
            ### How to use this tool
            
            1. **Quality ROI Analysis**: Enter product details and quality issue information to calculate ROI of potential fixes
            2. **Tariff Calculator**: Determine the impact of tariffs and import costs on product margins
            3. **Marketing ROI**: Analyze advertising campaign performance metrics
            4. **Monte Carlo Simulation**: Understand risks and probabilities with statistical modeling
            
            Each tab provides specialized analysis tools with interactive visualizations and export options.
            
            #### Tips
            - Hover over charts and metrics for additional information
            - Use the AI assistant to get expert recommendations
            - Export results as CSV or PDF for reporting
            
            For additional help, contact the Quality Management team.
            """)
    
    # Add app navigation and user info in a status bar
    st.markdown("""
    <div style="background-color: #f8f9fa; padding: 0.5rem; border-radius: 4px; margin-bottom: 1rem; display: flex; justify-content: space-between; align-items: center;">
        <div>
            <span style="color: #6c757d; margin-right: 1rem;">User: Quality Manager</span>
            <span style="color: #6c757d;">Department: Product Development</span>
        </div>
        <div>
            <span style="color: #6c757d;">Last updated: {}</span>
        </div>
    </div>
    """.format(datetime.now().strftime("%Y-%m-%d")), unsafe_allow_html=True)

def display_quality_issue_results(results: Dict[str, Any], expanded: bool = True):
    """Display the results of a quality issue analysis."""
    if not results:
        return
    
    st.markdown("### 📊 Analysis Results")
    
    # Summary metrics in columns
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        # Use tooltip for extra information
        with st.container():
            st.markdown(f"""
            <div class="metric-card" data-tooltip="30-day return rate based on recent sales data">
                <div class="metric-label">Return Rate (30d)</div>
                <div class="metric-value">{results['current_metrics']['return_rate_30d']:.2f}%</div>
                <div class="metric-subvalue">Monthly average</div>
            </div>
            """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card" data-tooltip="Estimated annual financial impact from returns">
            <div class="metric-label">Annual Loss</div>
            <div class="metric-value">{format_currency(results['financial_impact']['annual_loss'])}</div>
            <div class="metric-subvalue">Due to returns</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        roi_color = generate_color_scale(
            results['financial_impact']['roi_3yr'], 200, 0
        )
        st.markdown(f"""
        <div class="metric-card" data-tooltip="Return on investment over a 3-year period">
            <div class="metric-label">3-Year ROI</div>
            <div class="metric-value" style="color: {roi_color};">{results['financial_impact']['roi_3yr']:.2f}%</div>
            <div class="metric-subvalue">1-Year: {results['financial_impact']['roi_1yr']:.2f}%</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        payback_color = generate_color_scale(
            24 - min(results['financial_impact']['payback_period'], 24), 18, 6
        )
        st.markdown(f"""
        <div class="metric-card" data-tooltip="Time required to recover the investment">
            <div class="metric-label">Payback Period</div>
            <div class="metric-value" style="color: {payback_color};">{results['financial_impact']['payback_period']:.1f} months</div>
            <div class="metric-subvalue">To break even</div>
        </div>
        """, unsafe_allow_html=True)
    
    # Recommendation banner
    st.markdown(f"""
    <div style="background-color: {PRIMARY_COLOR}; padding: 1rem; border-radius: 4px; margin: 1rem 0; 
                color: white; display: flex; justify-content: space-between; align-items: center;">
        <div>
            <span style="font-size: 1.1rem; font-weight: 600;">Recommendation:</span> 
            <span style="font-size: 1.1rem;">{results['recommendation']}</span>
        </div>
        <div>
            {get_recommendation_badge(results['recommendation'])}
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Detailed results in expandable section
    with st.expander("View Detailed Analysis", expanded=expanded):
        tabs = st.tabs(["Financial Impact", "Visualizations", "Solution Details", "Risk Assessment"])
        
        with tabs[0]:
            # Financial Impact tab
            financial_impact = results['financial_impact']
            st.subheader("Financial Impact")
            
            # Calculate values needed for display
            annual_returns = results['current_metrics']['annual_returns']
            annual_sales = results['current_metrics']['annual_sales']
            loss_per_return = financial_impact['annual_loss'] / annual_returns if annual_returns > 0 else 0
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### Current Situation")
                st.markdown(f"""
                - **Annual Returns:** {annual_returns:.0f} units
                - **Return Rate:** {results['current_metrics']['return_rate_30d']:.2f}%
                - **Loss Per Return:** {format_currency(loss_per_return)}
                - **Annual Loss:** {format_currency(financial_impact['annual_loss'])}
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown("#### After Improvement")
                st.markdown(f"""
                - **Returns Prevented:** {financial_impact['returns_prevented']:.0f} units
                - **Gross Savings:** {format_currency(financial_impact['savings'])}
                - **Adjusted Savings:** {format_currency(financial_impact['adjusted_savings'])} (with {results['solution_metrics']['solution_confidence']}% confidence)
                - **Implementation Cost:** {format_currency(financial_impact['implementation_cost'])} (includes {format_currency(results['solution_metrics']['fix_cost_upfront'])} upfront)
                """, unsafe_allow_html=True)
            
            # Waterfall chart with hover tooltips
            fig = go.Figure(go.Waterfall(
                name="Financial Impact",
                orientation="v",
                measure=["absolute", "relative", "relative", "relative", "total"],
                x=["Current Loss", "Prevented Returns", "Implementation Cost", "Ongoing Costs", "Net Impact"],
                textposition="outside",
                text=[
                    f"${financial_impact['annual_loss']:,.0f}",
                    f"+${financial_impact['adjusted_savings']:,.0f}",
                    f"-${results['solution_metrics']['fix_cost_upfront']:,.0f}",
                    f"-${(annual_sales * results['solution_metrics']['fix_cost_per_unit']):,.0f}",
                    f"${(financial_impact['adjusted_savings'] - financial_impact['implementation_cost']):,.0f}"
                ],
                y=[
                    -financial_impact['annual_loss'],
                    financial_impact['adjusted_savings'],
                    -results['solution_metrics']['fix_cost_upfront'],
                    -(annual_sales * results['solution_metrics']['fix_cost_per_unit']),
                    0
                ],
                connector={"line": {"color": "rgb(63, 63, 63)"}},
                decreasing={"marker": {"color": DANGER_COLOR}},
                increasing={"marker": {"color": SUCCESS_COLOR}},
                totals={"marker": {"color": SECONDARY_COLOR}},
                hoverinfo="text",
                hovertext=[
                    f"Current annual loss due to returns: ${financial_impact['annual_loss']:,.0f}",
                    f"Savings from prevented returns: +${financial_impact['adjusted_savings']:,.0f}",
                    f"One-time implementation cost: -${results['solution_metrics']['fix_cost_upfront']:,.0f}",
                    f"Annual ongoing costs: -${(annual_sales * results['solution_metrics']['fix_cost_per_unit']):,.0f}",
                    f"Net annual impact: ${(financial_impact['adjusted_savings'] - financial_impact['implementation_cost']):,.0f}"
                ]
            ))
            
            fig.update_layout(
                title="Financial Impact Waterfall",
                showlegend=False,
                height=400,
                hoverlabel=dict(
                    bgcolor="white",
                    font_size=12,
                    font_family="Arial"
                )
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        with tabs[1]:
            # Visualizations tab
            st.subheader("Visualizations")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Return rate reduction
                current_return_rate = results['current_metrics']['return_rate_30d']
                improved_return_rate = current_return_rate * (1 - results['solution_metrics']['expected_reduction'] / 100)
                
                fig = go.Figure()
                fig.add_trace(go.Bar(
                    x=["Current", "After Fix"],
                    y=[current_return_rate, improved_return_rate],
                    text=[f"{current_return_rate:.2f}%", f"{improved_return_rate:.2f}%"],
                    textposition='auto',
                    marker_color=[DANGER_COLOR, SUCCESS_COLOR],
                    hoverinfo="text",
                    hovertext=[
                        f"Current return rate: {current_return_rate:.2f}%<br>Based on {results['current_metrics']['returns_30d']} returns in 30 days",
                        f"Projected return rate: {improved_return_rate:.2f}%<br>Expected reduction: {results['solution_metrics']['expected_reduction']}%<br>Confidence: {results['solution_metrics']['solution_confidence']}%"
                    ]
                ))
                fig.update_layout(
                    title="Return Rate Reduction",
                    yaxis_title="Return Rate (%)",
                    height=350,
                    hoverlabel=dict(
                        bgcolor="white",
                        font_size=12,
                        font_family="Arial"
                    )
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # ROI comparison
                fig = go.Figure()
                fig.add_trace(go.Bar(
                    x=["1-Year ROI", "3-Year ROI"],
                    y=[results['financial_impact']['roi_1yr'], results['financial_impact']['roi_3yr']],
                    text=[f"{results['financial_impact']['roi_1yr']:.2f}%", f"{results['financial_impact']['roi_3yr']:.2f}%"],
                    textposition='auto',
                    marker_color=[SECONDARY_COLOR, PRIMARY_COLOR],
                    hoverinfo="text",
                    hovertext=[
                        f"1-Year ROI: {results['financial_impact']['roi_1yr']:.2f}%<br>Implementation cost: ${results['financial_impact']['implementation_cost']:,.2f}<br>First year savings: ${results['financial_impact']['adjusted_savings']:,.2f}",
                        f"3-Year ROI: {results['financial_impact']['roi_3yr']:.2f}%<br>Includes annual growth of {annualized_growth:.1f}%<br>Cumulative 3-year benefit: ${cumulative_savings:,.2f}"
                    ]
                ))
                fig.update_layout(
                    title="Return on Investment",
                    yaxis_title="ROI (%)",
                    height=350,
                    hoverlabel=dict(
                        bgcolor="white",
                        font_size=12,
                        font_family="Arial"
                    )
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # Payback period gauge
            fig = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=results['financial_impact']['payback_period'],
                title={"text": "Payback Period (Months)"},
                delta={'reference': 12, 'decreasing': {'color': SUCCESS_COLOR}, 'increasing': {'color': DANGER_COLOR}},
                gauge={
                    "axis": {"range": [None, 36], "tickwidth": 1, "tickcolor": "darkblue"},
                    "bar": {"color": PRIMARY_COLOR},
                    "bgcolor": "white",
                    "borderwidth": 2,
                    "bordercolor": "gray",
                    "steps": [
                        {"range": [0, 6], "color": SUCCESS_COLOR},
                        {"range": [6, 12], "color": "#88D498"},
                        {"range": [12, 24], "color": WARNING_COLOR},
                        {"range": [24, 36], "color": DANGER_COLOR}
                    ],
                    "threshold": {
                        "line": {"color": "red", "width": 4},
                        "thickness": 0.75,
                        "value": 12
                    }
                }
            ))
            fig.update_layout(
                height=300,
                margin=dict(l=30, r=30, t=50, b=30),
                annotations=[
                    dict(
                        x=0.5,
                        y=-0.15,
                        text=f"Break-even in {results['financial_impact']['payback_period']:.1f} months<br>Monthly net benefit: ${monthly_net_benefit:,.2f}",
                        showarrow=False,
                        align="center"
                    )
                ]
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with tabs[2]:
            # Solution Details tab
            st.subheader("Solution Details")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### Cost Structure")
                st.markdown(f"""
                - **Current Unit Cost:** {format_currency(results['current_metrics']['unit_cost'])}
                - **Fix Cost (Upfront):** {format_currency(results['solution_metrics']['fix_cost_upfront'])}
                - **Fix Cost (Per Unit):** {format_currency(results['solution_metrics']['fix_cost_per_unit'])}
                - **New Unit Cost:** {format_currency(results['solution_metrics']['new_unit_cost'])}
                """)
            
            with col2:
                st.markdown("#### Margin Impact")
                st.markdown(f"""
                - **Current Margin:** {format_currency(results['current_metrics']['margin_per_unit'])} ({results['current_metrics']['margin_percentage']:.2f}%)
                - **New Margin:** {format_currency(results['solution_metrics']['new_margin_per_unit'])} ({results['solution_metrics']['new_margin_percentage']:.2f}%)
                - **Margin Change:** {format_currency(results['solution_metrics']['new_margin_per_unit'] - results['current_metrics']['margin_per_unit'])}
                - **Expected Reduction:** {results['solution_metrics']['expected_reduction']:.2f}% (with {results['solution_metrics']['solution_confidence']:.0f}% confidence)
                """)
            
            # Margin comparison chart
            fig = go.Figure()
            fig.add_trace(go.Bar(
                name="Unit Cost",
                x=["Current", "After Fix"],
                y=[results['current_metrics']['unit_cost'], results['solution_metrics']['new_unit_cost']],
                marker_color=DANGER_COLOR
            ))
            fig.add_trace(go.Bar(
                name="Margin",
                x=["Current", "After Fix"],
                y=[results['current_metrics']['margin_per_unit'], results['solution_metrics']['new_margin_per_unit']],
                marker_color=SUCCESS_COLOR
            ))
            
            fig.update_layout(
                title="Cost and Margin Comparison",
                yaxis_title="Amount ($)",
                barmode='stack',
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        with tabs[3]:
            # Risk Assessment tab
            st.subheader("Risk Assessment")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=results['risk_assessment']['regulatory_risk'],
                    title={"text": "Regulatory Risk"},
                    gauge={
                        "axis": {"range": [None, 5], "tickwidth": 1},
                        "bar": {"color": "darkblue"},
                        "steps": [
                            {"range": [0, 2], "color": SUCCESS_COLOR},
                            {"range": [2, 4], "color": WARNING_COLOR},
                            {"range": [4, 5], "color": DANGER_COLOR}
                        ],
                        "threshold": {
                            "line": {"color": "red", "width": 4},
                            "thickness": 0.75,
                            "value": 4
                        }
                    }
                ))
                fig.update_layout(height=250)
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=results['risk_assessment']['brand_risk'],
                    title={"text": "Brand Risk"},
                    gauge={
                        "axis": {"range": [None, 5], "tickwidth": 1},
                        "bar": {"color": "darkblue"},
                        "steps": [
                            {"range": [0, 2], "color": SUCCESS_COLOR},
                            {"range": [2, 4], "color": WARNING_COLOR},
                            {"range": [4, 5], "color": DANGER_COLOR}
                        ],
                        "threshold": {
                            "line": {"color": "red", "width": 4},
                            "thickness": 0.75,
                            "value": 4
                        }
                    }
                ))
                fig.update_layout(height=250)
                st.plotly_chart(fig, use_container_width=True)
            
            with col3:
                fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=results['risk_assessment']['medical_risk'],
                    title={"text": "Medical Risk"},
                    gauge={
                        "axis": {"range": [None, 5], "tickwidth": 1},
                        "bar": {"color": "darkblue"},
                        "steps": [
                            {"range": [0, 2], "color": SUCCESS_COLOR},
                            {"range": [2, 4], "color": WARNING_COLOR},
                            {"range": [4, 5], "color": DANGER_COLOR}
                        ],
                        "threshold": {
                            "line": {"color": "red", "width": 4},
                            "thickness": 0.75,
                            "value": 4
                        }
                    }
                ))
                fig.update_layout(height=250)
                st.plotly_chart(fig, use_container_width=True)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### Risk Factor Analysis")
                st.markdown(f"""
                - **Overall Risk Factor:** {results['risk_assessment']['risk_factor']:.2f}/5.0
                - **Priority Score:** {results['risk_assessment']['priority_score']:.2f}/100
                
                *The priority score combines financial metrics and risk factors to determine the overall priority of this improvement project.*
                """)
            
            with col2:
                st.markdown("#### Risk Mitigation")
                if results['risk_assessment']['medical_risk'] >= 4:
                    st.error("⚠️ High medical risk detected. Immediate action recommended.")
                elif results['risk_assessment']['regulatory_risk'] >= 4:
                    st.warning("⚠️ High regulatory risk detected. Prioritize compliance actions.")
                elif results['risk_assessment']['brand_risk'] >= 4:
                    st.info("⚠️ High brand risk detected. Consider customer communication plan.")
                else:
                    st.success("✅ Risk levels are manageable with standard protocols.")
    
    # Export options
    col1, col2 = st.columns(2)
    
    with col1:
        if results:
            df = export_as_csv(results)
            csv_download = generate_download_link(
                df, 
                f"quality_analysis_{results['sku']}_{datetime.now().strftime('%Y%m%d')}.csv", 
                "📥 Export as CSV"
            )
            st.markdown(csv_download, unsafe_allow_html=True)
    
    with col2:
        if results:
            try:
                pdf_buffer = export_as_pdf(results)
                pdf_data = base64.b64encode(pdf_buffer.read()).decode('utf-8')
                pdf_download = f'<a href="data:application/pdf;base64,{pdf_data}" download="quality_analysis_{results["sku"]}_{datetime.now().strftime("%Y%m%d")}.pdf" class="export-button">📄 Export as PDF</a>'
                st.markdown(pdf_download, unsafe_allow_html=True)
            except Exception as e:
                st.error(f"Error generating PDF: {e}")

def display_quality_analysis_form():
    """Display the form for quality issue analysis."""
    with st.form(key="quality_analysis_form"):
        st.markdown("""
        <h3 style="color: #0096C7; border-bottom: 2px solid #0096C7; padding-bottom: 0.5rem;">
            <i class="fas fa-clipboard-list"></i> Product Information
        </h3>
        """, unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            sku = st.text_input("SKU", placeholder="Enter product SKU", help="Unique identifier for the product")
        
        with col2:
            product_type_options = ["B2C - Consumer", "B2B - Professional", "B2B - Healthcare", "OEM"]
            product_type = st.selectbox(
                "Product Type",
                options=product_type_options,
                index=0,
                help="Select the market segment for this product"
            )
        
        with col3:
            issue_description = st.text_input(
                "Issue Description", 
                placeholder="Brief description of the quality issue",
                help="Describe the quality problem being addressed"
            )
        
        st.markdown("""
        <h3 style="color: #0096C7; border-bottom: 2px solid #0096C7; padding-bottom: 0.5rem; margin-top: 1.5rem;">
            <i class="fas fa-chart-line"></i> Sales & Returns Data
        </h3>
        """, unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            sales_30d = st.number_input(
                "Sales (Last 30 Days)",
                min_value=0,
                value=1000,
                step=100,
                help="Number of units sold in the last 30 days"
            )
        
        with col2:
            returns_30d = st.number_input(
                "Returns (Last 30 Days)",
                min_value=0,
                value=50,
                step=10,
                help="Number of units returned in the last 30 days"
            )
        
        with col3:
            annualized_growth = st.slider(
                "Expected Annual Growth (%)",
                min_value=0.0,
                max_value=50.0,
                value=5.0,
                step=1.0,
                format="%.1f%%",
                help="Projected annual growth rate for sales volume"
            )
            
            # Calculate and display current return rate for user reference
            if sales_30d > 0:
                return_rate = (returns_30d / sales_30d) * 100
                st.caption(f"Current return rate: {return_rate:.2f}%")
        
        st.markdown("""
        <h3 style="color: #0096C7; border-bottom: 2px solid #0096C7; padding-bottom: 0.5rem; margin-top: 1.5rem;">
            <i class="fas fa-dollar-sign"></i> Cost & Pricing Information
        </h3>
        """, unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            current_unit_cost = st.number_input(
                "Current Unit Cost ($)",
                min_value=0.0,
                value=10.0,
                step=1.0,
                help="Manufacturing cost per unit before improvements"
            )
        
        with col2:
            sales_price = st.number_input(
                "Sales Price ($)",
                min_value=0.0,
                value=25.0,
                step=1.0,
                help="Retail or wholesale price per unit"
            )
        
        with col3:
            margin = sales_price - current_unit_cost
            margin_percentage = safe_divide(margin, sales_price, 0) * 100
            
            st.markdown(
                f"""
                <div style="background-color: {CARD_BACKGROUND}; padding: 0.75rem; border-radius: 4px; 
                        border-left: 3px solid {PRIMARY_COLOR}; margin-top: 1.55rem;">
                    <div style="font-size: 0.85rem; color: {TEXT_SECONDARY}; margin-bottom: 0.25rem;">Current Margin</div>
                    <div style="font-size: 1.2rem; font-weight: 600; color: {TEXT_PRIMARY};">${margin:.2f} ({margin_percentage:.2f}%)</div>
                </div>
                """, 
                unsafe_allow_html=True
            )
        
        st.markdown("""
        <h3 style="color: #0096C7; border-bottom: 2px solid #0096C7; padding-bottom: 0.5rem; margin-top: 1.5rem;">
            <i class="fas fa-tools"></i> Proposed Solution
        </h3>
        """, unsafe_allow_html=True)
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            fix_cost_upfront = st.number_input(
                "Fix Cost - Upfront ($)",
                min_value=0.0,
                value=5000.0,
                step=500.0,
                help="One-time cost to implement the solution"
            )
        
        with col2:
            fix_cost_per_unit = st.number_input(
                "Fix Cost - Per Unit ($)",
                min_value=0.0,
                value=0.5,
                step=0.1,
                help="Additional cost per unit after implementing the solution"
            )
        
        with col3:
            expected_reduction = st.slider(
                "Expected Return Reduction (%)",
                min_value=0.0,
                max_value=100.0,
                value=50.0,
                step=5.0,
                format="%.1f%%",
                help="Projected percentage reduction in return rate"
            )
        
        with col4:
            solution_confidence = st.slider(
                "Solution Confidence (%)",
                min_value=0.0,
                max_value=100.0,
                value=80.0,
                step=5.0,
                format="%.1f%%",
                help="Confidence level in the effectiveness of the solution"
            )
        
        st.markdown("""
        <h3 style="color: #0096C7; border-bottom: 2px solid #0096C7; padding-bottom: 0.5rem; margin-top: 1.5rem;">
            <i class="fas fa-exclamation-triangle"></i> Risk Assessment
        </h3>
        """, unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            regulatory_risk = st.slider(
                "Regulatory Risk",
                min_value=1,
                max_value=5,
                value=2,
                step=1,
                help="1 = Low, 5 = High. Considers potential regulatory impact."
            )
        
        with col2:
            brand_risk = st.slider(
                "Brand Risk",
                min_value=1,
                max_value=5,
                value=3,
                step=1,
                help="1 = Low, 5 = High. Considers impact on brand reputation."
            )
        
        with col3:
            medical_risk = st.slider(
                "Medical Risk",
                min_value=1,
                max_value=5,
                value=2,
                step=1,
                help="1 = Low, 5 = High. Considers potential patient impact."
            )
        
        col1, col2 = st.columns([3, 1])
        with col2:
            submitted = st.form_submit_button("Run Analysis", use_container_width=True)
        
        if submitted:
            if not sku:
                st.error("⚠️ SKU is required")
                return None
            
            if sales_30d <= 0:
                st.error("⚠️ Sales must be greater than zero")
                return None
            
            if current_unit_cost <= 0 or sales_price <= 0:
                st.error("⚠️ Cost and price must be greater than zero")
                return None
            
            with st.spinner("Analyzing quality issue... Please wait"):
                try:
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
                        annualized_growth=annualized_growth,
                        regulatory_risk=regulatory_risk,
                        brand_risk=brand_risk,
                        medical_risk=medical_risk
                    )
                    
                    st.session_state.quality_analysis_results = results
                    st.session_state.analysis_submitted = True
                    
                    # Store variables needed for charts in session state
                    st.session_state.annualized_growth = annualized_growth
                    st.session_state.monthly_net_benefit = safe_divide(
                        results['financial_impact']['adjusted_savings'] - 
                        (results['current_metrics']['annual_sales'] * results['solution_metrics']['fix_cost_per_unit']), 
                        12, 0
                    )
                    st.session_state.cumulative_savings = results['financial_impact']['adjusted_savings'] * 3  # Simplified calculation
                    
                    # Prepare initial message for AI assistant
                    if 'chat_history' not in st.session_state:
                        st.session_state.chat_history = []
                    
                    return results
                except Exception as e:
                    st.error(f"Error analyzing quality issue: {str(e)}")
                    logger.exception("Error in quality analysis")
                    return None
        
        return None

def display_ai_assistant(results: Dict[str, Any]):
    """Display the AI assistant chat interface."""
    if not results:
        return
    
    system_prompt = get_system_prompt(results)
    
    st.markdown("""
    <h3 style="color: #0096C7; border-bottom: 2px solid #0096C7; padding-bottom: 0.5rem; margin-top: 1rem;">
        <i class="fas fa-robot"></i> Quality Consultant AI Assistant
    </h3>
    """, unsafe_allow_html=True)
    
    # Add a brief explanation of the assistant's capabilities
    st.markdown("""
    <div style="background-color: #f8f9fa; padding: 0.75rem; border-radius: 4px; margin-bottom: 1rem; border-left: 3px solid #0096C7;">
        Your AI quality management consultant can provide expert guidance on:
        <ul style="margin-top: 0.5rem; margin-bottom: 0.5rem;">
            <li>Regulatory considerations for medical devices</li>
            <li>Implementation strategies for quality improvements</li>
            <li>Cost-benefit analysis interpretation</li>
            <li>Risk mitigation recommendations</li>
        </ul>
        Ask specific questions about your analysis results to get targeted advice.
    </div>
    """, unsafe_allow_html=True)
    
    # Display chat history
    chat_container = st.container()
    with chat_container:
        for msg in st.session_state.chat_history:
            if msg["role"] == "user":
                st.markdown(f'<div class="user-bubble">{msg["content"]}</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="assistant-bubble">{msg["content"]}</div>', unsafe_allow_html=True)
    
    # Input area with suggested prompts
    col1, col2 = st.columns([4, 1])
    
    with col1:
        user_input = st.text_area(
            "Ask the AI consultant...",
            key="chat_input", 
            placeholder="Type your question here or select a suggestion.",
            height=80
        )
    
    with col2:
        st.markdown("<div style='margin-top: 1.7rem;'></div>", unsafe_allow_html=True)
        send_button = st.button("📤 Send", key="send_msg_btn", use_container_width=True)
    
    # Suggested prompt buttons
    st.markdown("<p style='margin-bottom: 0.5rem; font-size: 0.85rem; color: #6c757d;'>Suggested questions:</p>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("What are the next steps I should take?", key="prompt1", use_container_width=True):
            user_input = "What are the next steps I should take based on this analysis?"
            send_button = True
    
    with col2:
        if st.button("Regulatory considerations?", key="prompt2", use_container_width=True):
            user_input = "What regulatory considerations should I keep in mind for this quality issue?"
            send_button = True
    
    with col3:
        if st.button("How to improve confidence?", key="prompt3", use_container_width=True):
            user_input = "How can I improve my confidence in the proposed solution?"
            send_button = True
    
    # Process message
    if send_button and user_input:
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        
        messages = [{"role": "system", "content": system_prompt}] + [
            {"role": m["role"], "content": m["content"]} for m in st.session_state.chat_history
        ]
        
        with st.spinner("AI consultant is thinking..."):
            ai_resp = call_openai_api(messages)
        
        st.session_state.chat_history.append({"role": "assistant", "content": ai_resp})
        st.rerun()
    
    # Add initial message if chat is empty
    if not st.session_state.chat_history:
        st.info("💡 Ask the AI assistant questions about your analysis, regulatory considerations, or best practices for implementation.")
    
    # Add a clear conversation button
    if st.session_state.chat_history and st.button("Clear Conversation", key="clear_chat_btn"):
        st.session_state.chat_history = []
        st.rerun()

def display_landed_cost_calculator():
    """Display the landed cost calculator UI."""
    st.markdown("### 🚢 Landed Cost & Tariff Calculator")
    st.markdown("""
    Calculate the impact of tariffs, shipping, and other import costs on your product margins.
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        with st.form(key="landed_cost_form"):
            st.markdown("#### Product & Pricing")
            
            sales_price = st.number_input(
                "Sales Price ($)",
                min_value=0.01,
                value=25.0,
                step=1.0
            )
            
            cogs = st.number_input(
                "Manufacturing Cost (COGS) ($)",
                min_value=0.01,
                value=10.0,
                step=1.0
            )
            
            tariff_rate = st.number_input(
                "Tariff Rate (%)",
                min_value=0.0,
                max_value=100.0,
                value=25.0,
                step=1.0
            )
            
            st.markdown("#### Logistics Costs")
            
            col_a, col_b = st.columns(2)
            
            with col_a:
                shipping_cost = st.number_input(
                    "Shipping Cost per Shipment ($)",
                    min_value=0.0,
                    value=1000.0,
                    step=100.0
                )
                
                customs_fee = st.number_input(
                    "Customs Processing Fee ($)",
                    min_value=0.0,
                    value=150.0,
                    step=10.0
                )
            
            with col_b:
                units_per_shipment = st.number_input(
                    "Units per Shipment",
                    min_value=1,
                    value=1000,
                    step=100
                )
                
                broker_fee = st.number_input(
                    "Customs Broker Fee ($)",
                    min_value=0.0,
                    value=200.0,
                    step=10.0
                )
            
            storage_cost = st.number_input(
                "Storage Cost per Unit ($)",
                min_value=0.0,
                value=0.2,
                step=0.1
            )
            
            other_costs = st.number_input(
                "Other Costs per Unit ($)",
                min_value=0.0,
                value=0.0,
                step=0.1
            )
            
            submitted = st.form_submit_button("Calculate Landed Cost")
            
            if submitted:
                with st.spinner("Calculating landed cost..."):
                    try:
                        results = calculate_landed_cost(
                            sales_price=sales_price,
                            cogs=cogs,
                            tariff_rate=tariff_rate,
                            shipping_cost=shipping_cost,
                            storage_cost=storage_cost,
                            customs_fee=customs_fee,
                            broker_fee=broker_fee,
                            other_costs=other_costs,
                            units_per_shipment=units_per_shipment
                        )
                        
                        # Generate tariff scenarios
                        scenarios = generate_tariff_scenarios(
                            sales_price=sales_price,
                            cogs=cogs,
                            base_tariff=tariff_rate,
                            shipping_cost=shipping_cost,
                            storage_cost=storage_cost,
                            customs_fee=customs_fee,
                            broker_fee=broker_fee,
                            other_costs=other_costs,
                            units_per_shipment=units_per_shipment
                        )
                        
                        st.session_state.tariff_calculations = {
                            "results": results,
                            "scenarios": scenarios
                        }
                    except Exception as e:
                        st.error(f"Error calculating landed cost: {str(e)}")
                        logger.exception("Error in landed cost calculation")
    
    with col2:
        if st.session_state.tariff_calculations:
            results = st.session_state.tariff_calculations["results"]
            scenarios = st.session_state.tariff_calculations["scenarios"]
            
            st.markdown("#### Results")
            
            # Summary cards
            col_a, col_b = st.columns(2)
            
            with col_a:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Landed Cost</div>
                    <div class="metric-value">${results['landed_cost']:.2f}</div>
                    <div class="metric-subvalue">Per unit</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col_b:
                margin_color = generate_color_scale(
                    results['new_margin_percentage'], 40, 15
                )
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">New Margin</div>
                    <div class="metric-value" style="color: {margin_color};">{results['new_margin_percentage']:.2f}%</div>
                    <div class="metric-subvalue">Was: {results['original_margin_percentage']:.2f}%</div>
                </div>
                """, unsafe_allow_html=True)
            
            # Cost breakdown
            st.markdown("#### Cost Breakdown")
            
            fig = go.Figure()
            
            # Sort components by value for better visualization
            cost_components = [
                ("COGS", results['original_cogs']),
                ("Tariff", results['tariff_amount']),
                ("Shipping", results['per_unit_shipping']),
                ("Storage", results['storage_cost']),
                ("Customs", results['per_unit_customs']),
                ("Broker", results['per_unit_broker']),
                ("Other", results['other_costs'])
            ]
            
            # Filter out zero values and sort by value
            cost_components = [(name, value) for name, value in cost_components if value > 0]
            cost_components.sort(key=lambda x: x[1], reverse=True)
            
            labels = [name for name, _ in cost_components]
            values = [value for _, value in cost_components]
            
            fig.add_trace(go.Pie(
                labels=labels,
                values=values,
                hole=0.4,
                textinfo='label+percent',
                insidetextorientation='radial'
            ))
            
            fig.update_layout(
                title="Landed Cost Components",
                height=300
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Tariff impact
            st.markdown("#### Tariff Impact")
            
            # Create a line chart showing margin % at different tariff rates
            tariff_rates = [float(s.split('%')[0]) for s in scenarios.keys()]
            margin_percentages = [s['new_margin_percentage'] for s in scenarios.values()]
            
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=tariff_rates,
                y=margin_percentages,
                mode='lines+markers',
                name='Margin %',
                line=dict(color=PRIMARY_COLOR, width=3),
                marker=dict(size=8)
            ))
            
            # Add a horizontal line for the target margin (e.g., 30%)
            target_margin = 30
            fig.add_trace(go.Scatter(
                x=[min(tariff_rates), max(tariff_rates)],
                y=[target_margin, target_margin],
                mode='lines',
                name=f'Target Margin ({target_margin}%)',
                line=dict(color='red', width=2, dash='dash')
            ))
            
            fig.update_layout(
                title="Margin at Different Tariff Rates",
                xaxis_title="Tariff Rate (%)",
                yaxis_title="Margin (%)",
                height=350
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Price adjustment analysis
            st.markdown("#### Price Adjustment Analysis")
            
            st.markdown(f"""
            To maintain your original margin of **{results['original_margin_percentage']:.2f}%**, 
            you would need to increase your price by **{results['price_increase_percentage']:.2f}%** 
            (${results['price_increase_needed']:.2f} per unit).
            
            New breakeven price: **${results['breakeven_price']:.2f}**
            """)
            
            # Create a waterfall chart for price adjustment
            fig = go.Figure(go.Waterfall(
                name="Price Adjustment",
                orientation="v",
                measure=["absolute", "relative", "total"],
                x=["Original Price", "Required Increase", "New Price"],
                textposition="outside",
                text=[
                    f"${results['sales_price']:.2f}",
                    f"+${results['price_increase_needed']:.2f}",
                    f"${results['breakeven_price']:.2f}"
                ],
                y=[
                    results['sales_price'],
                    results['price_increase_needed'],
                    0
                ],
                connector={"line": {"color": "rgb(63, 63, 63)"}},
                increasing={"marker": {"color": WARNING_COLOR}},
                totals={"marker": {"color": PRIMARY_COLOR}}
            ))
            
            fig.update_layout(
                title="Price Adjustment to Maintain Margin",
                showlegend=False,
                height=300
            )
            
            st.plotly_chart(fig, use_container_width=True)

def calculate_ad_roi_ui():
    """Display the advertising ROI calculator UI."""
    st.markdown("### 📈 Advertising ROI Calculator")
    st.markdown("""
    Calculate and visualize the return on investment for your advertising campaigns.
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        with st.form(key="ad_roi_form"):
            st.markdown("#### Campaign Metrics")
            
            ad_spend = st.number_input(
                "Ad Spend ($)",
                min_value=0.01,
                value=5000.0,
                step=500.0
            )
            
            impressions = st.number_input(
                "Impressions",
                min_value=1,
                value=500000,
                step=10000
            )
            
            clicks = st.number_input(
                "Clicks",
                min_value=0,
                value=15000,
                step=1000
            )
            
            conversions = st.number_input(
                "Conversions (Sales)",
                min_value=0,
                value=300,
                step=10
            )
            
            avg_order_value = st.number_input(
                "Average Order Value ($)",
                min_value=0.01,
                value=75.0,
                step=5.0
            )
            
            contribution_margin_percent = st.slider(
                "Contribution Margin (%)",
                min_value=0.0,
                max_value=100.0,
                value=40.0,
                step=1.0
            )
            
            submitted = st.form_submit_button("Calculate ROI")
            
            if submitted:
                with st.spinner("Calculating ad ROI..."):
                    try:
                        results = calculate_ad_roi(
                            ad_spend=ad_spend,
                            impressions=impressions,
                            clicks=clicks,
                            conversions=conversions,
                            avg_order_value=avg_order_value,
                            contribution_margin_percent=contribution_margin_percent
                        )
                        
                        st.session_state.ad_roi_results = results
                    except Exception as e:
                        st.error(f"Error calculating ad ROI: {str(e)}")
                        logger.exception("Error in ad ROI calculation")
    
    with col2:
        if hasattr(st.session_state, 'ad_roi_results') and st.session_state.ad_roi_results:
            results = st.session_state.ad_roi_results
            
            st.markdown("#### Results")
            
            # Summary cards
            col_a, col_b, col_c = st.columns(3)
            
            with col_a:
                roi_color = generate_color_scale(
                    results['roi'], 200, 0
                )
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">ROI</div>
                    <div class="metric-value" style="color: {roi_color};">{results['roi']:.2f}%</div>
                    <div class="metric-subvalue">Return on Investment</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col_b:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">ROAS</div>
                    <div class="metric-value">{results['roas']:.2f}x</div>
                    <div class="metric-subvalue">Return on Ad Spend</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col_c:
                cpa_color = generate_color_scale(
                    50 - min(results['cpa'], 50), 40, 10
                )
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">CPA</div>
                    <div class="metric-value" style="color: {cpa_color};">${results['cpa']:.2f}</div>
                    <div class="metric-subvalue">Cost Per Acquisition</div>
                </div>
                """, unsafe_allow_html=True)
            
            # Funnel visualization
            st.markdown("#### Campaign Funnel")
            
            stages = ['Impressions', 'Clicks', 'Conversions']
            values = [results['impressions'], results['clicks'], results['conversions']]
            
            # Create logarithmic scale for better visualization
            log_values = [max(1, v) for v in values]
            log_values = [np.log10(v) for v in log_values]
            
            fig = go.Figure()
            
            fig.add_trace(go.Funnel(
                name='Funnel',
                y=stages,
                x=values,
                textinfo="value+percent initial",
                marker={"color": [TERTIARY_COLOR, SECONDARY_COLOR, PRIMARY_COLOR]}
            ))
            
            fig.update_layout(
                title="Campaign Conversion Funnel",
                height=300
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Performance metrics
            st.markdown("#### Performance Metrics")
            
            col_a, col_b = st.columns(2)
            
            with col_a:
                st.markdown(f"""
                - **CTR (Click-Through Rate):** {results['ctr']:.2f}%
                - **Conversion Rate:** {results['conversion_rate']:.2f}%
                - **CPM (Cost per 1000 Impressions):** ${results['cpm']:.2f}
                """)
            
            with col_b:
                st.markdown(f"""
                - **CPC (Cost per Click):** ${results['cpc']:.2f}
                - **Revenue:** ${results['revenue']:.2f}
                - **Profit:** ${results['profit']:.2f}
                """)
            
            # Financial visualization
            st.markdown("#### Financial Breakdown")
            
            fig = go.Figure()
            
            fig.add_trace(go.Bar(
                name='Ad Spend',
                x=['Cost'],
                y=[results['ad_spend']],
                marker_color=DANGER_COLOR
            ))
            
            fig.add_trace(go.Bar(
                name='Revenue',
                x=['Revenue'],
                y=[results['revenue']],
                marker_color=SECONDARY_COLOR
            ))
            
            fig.add_trace(go.Bar(
                name='Profit',
                x=['Profit'],
                y=[results['profit']],
                marker_color=SUCCESS_COLOR
            ))
            
            fig.update_layout(
                title="Financial Performance",
                height=350,
                yaxis_title="Amount ($)"
            )
            
            st.plotly_chart(fig, use_container_width=True)

def run_monte_carlo_simulation_ui():
    """Display the Monte Carlo simulation UI."""
    st.markdown("### 🎲 Monte Carlo Simulation")
    st.markdown("""
    Run a Monte Carlo simulation to understand risks and probabilities in your quality improvement project.
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        with st.form(key="monte_carlo_form"):
            st.markdown("#### Base Parameters")
            
            base_unit_cost = st.number_input(
                "Base Unit Cost ($)",
                min_value=0.01,
                value=10.0,
                step=1.0
            )
            
            base_sales_price = st.number_input(
                "Base Sales Price ($)",
                min_value=0.01,
                value=25.0,
                step=1.0
            )
            
            base_monthly_sales = st.number_input(
                "Base Monthly Sales (Units)",
                min_value=1,
                value=1000,
                step=100
            )
            
            base_return_rate = st.number_input(
                "Base Return Rate (%)",
                min_value=0.1,
                max_value=100.0,
                value=5.0,
                step=0.5
            )
            
            st.markdown("#### Solution Parameters")
            
            fix_cost_upfront = st.number_input(
                "Fix Cost - Upfront ($)",
                min_value=0.0,
                value=5000.0,
                step=500.0
            )
            
            fix_cost_per_unit = st.number_input(
                "Fix Cost - Per Unit ($)",
                min_value=0.0,
                value=0.5,
                step=0.1
            )
            
            expected_reduction = st.number_input(
                "Expected Return Reduction (%)",
                min_value=0.0,
                max_value=100.0,
                value=50.0,
                step=5.0
            )
            
            st.markdown("#### Simulation Settings")
            
            iterations = st.slider(
                "Number of Iterations",
                min_value=100,
                max_value=10000,
                value=1000,
                step=100
            )
            
            with st.expander("Advanced Variability Settings"):
                cost_std_dev = st.slider(
                    "Cost Variability",
                    min_value=0.01,
                    max_value=0.50,
                    value=0.05,
                    step=0.01,
                    format="%.2f",
                    help="Standard deviation as a fraction of base cost"
                )
                
                price_std_dev = st.slider(
                    "Price Variability",
                    min_value=0.01,
                    max_value=0.50,
                    value=0.03,
                    step=0.01,
                    format="%.2f",
                    help="Standard deviation as a fraction of base price"
                )
                
                sales_std_dev = st.slider(
                    "Sales Variability",
                    min_value=0.01,
                    max_value=0.50,
                    value=0.10,
                    step=0.01,
                    format="%.2f",
                    help="Standard deviation as a fraction of base sales"
                )
                
                return_std_dev = st.slider(
                    "Return Rate Variability",
                    min_value=0.01,
                    max_value=0.50,
                    value=0.15,
                    step=0.01,
                    format="%.2f",
                    help="Standard deviation as a fraction of base return rate"
                )
                
                reduction_std_dev = st.slider(
                    "Reduction Variability",
                    min_value=0.01,
                    max_value=0.50,
                    value=0.20,
                    step=0.01,
                    format="%.2f",
                    help="Standard deviation as a fraction of expected reduction"
                )
            
            submitted = st.form_submit_button("Run Simulation")
            
            if submitted:
                with st.spinner(f"Running Monte Carlo simulation with {iterations} iterations..."):
                    try:
                        results = run_monte_carlo_simulation(
                            base_unit_cost=base_unit_cost,
                            base_sales_price=base_sales_price,
                            base_monthly_sales=base_monthly_sales,
                            base_return_rate=base_return_rate,
                            fix_cost_upfront=fix_cost_upfront,
                            fix_cost_per_unit=fix_cost_per_unit,
                            expected_reduction=expected_reduction,
                            iterations=iterations,
                            cost_std_dev=cost_std_dev,
                            price_std_dev=price_std_dev,
                            sales_std_dev=sales_std_dev,
                            return_std_dev=return_std_dev,
                            reduction_std_dev=reduction_std_dev
                        )
                        
                        st.session_state.monte_carlo_scenario = results
                    except Exception as e:
                        st.error(f"Error running Monte Carlo simulation: {str(e)}")
                        logger.exception("Error in Monte Carlo simulation")
    
    with col2:
        if hasattr(st.session_state, 'monte_carlo_scenario') and st.session_state.monte_carlo_scenario:
            results = st.session_state.monte_carlo_scenario
            
            st.markdown("#### Simulation Results")
            
            # Probability metrics
            col_a, col_b = st.columns(2)
            
            with col_a:
                prob_positive_roi_color = generate_color_scale(
                    results['probability_metrics']['prob_positive_roi'], 90, 50
                )
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Probability of Positive ROI</div>
                    <div class="metric-value" style="color: {prob_positive_roi_color};">{results['probability_metrics']['prob_positive_roi']:.1f}%</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col_b:
                prob_payback_1yr_color = generate_color_scale(
                    results['probability_metrics']['prob_payback_1yr'], 90, 50
                )
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Probability of Payback < 1 Year</div>
                    <div class="metric-value" style="color: {prob_payback_1yr_color};">{results['probability_metrics']['prob_payback_1yr']:.1f}%</div>
                </div>
                """, unsafe_allow_html=True)
            
            # ROI distribution
            st.markdown("#### ROI Distribution")
            
            roi_hist = results['roi_stats']['histogram']['counts']
            roi_bins = results['roi_stats']['histogram']['bins']
            roi_bin_centers = [(roi_bins[i] + roi_bins[i+1])/2 for i in range(len(roi_bins)-1)]
            
            fig = go.Figure()
            
            fig.add_trace(go.Bar(
                x=roi_bin_centers,
                y=roi_hist,
                marker_color=PRIMARY_COLOR,
                name="ROI Distribution"
            ))
            
            # Add a vertical line at ROI = 0
            fig.add_shape(
                type="line",
                x0=0, y0=0,
                x1=0, y1=max(roi_hist),
                line=dict(color="red", width=2, dash="dash")
            )
            
            fig.update_layout(
                title="ROI Distribution",
                xaxis_title="ROI (%)",
                yaxis_title="Frequency",
                height=350
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Payback distribution
            st.markdown("#### Payback Period Distribution")
            
            payback_hist = results['payback_stats']['histogram']['counts']
            payback_bins = results['payback_stats']['histogram']['bins']
            payback_bin_centers = [(payback_bins[i] + payback_bins[i+1])/2 for i in range(len(payback_bins)-1)]
            
            fig = go.Figure()
            
            fig.add_trace(go.Bar(
                x=payback_bin_centers,
                y=payback_hist,
                marker_color=SECONDARY_COLOR,
                name="Payback Distribution"
            ))
            
            # Add a vertical line at Payback = 12 months
            fig.add_shape(
                type="line",
                x0=12, y0=0,
                x1=12, y1=max(payback_hist),
                line=dict(color="red", width=2, dash="dash")
            )
            
            fig.update_layout(
                title="Payback Period Distribution",
                xaxis_title="Payback Period (Months)",
                yaxis_title="Frequency",
                height=350
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Statistics table
            st.markdown("#### Statistical Summary")
            
            stats_df = pd.DataFrame({
                "Statistic": ["Mean", "Median", "Std Dev", "Min", "Max", "10th Percentile", "25th Percentile", "75th Percentile", "90th Percentile"],
                "ROI (%)": [
                    f"{results['roi_stats']['mean']:.2f}",
                    f"{results['roi_stats']['median']:.2f}",
                    f"{results['roi_stats']['std_dev']:.2f}",
                    f"{results['roi_stats']['min']:.2f}",
                    f"{results['roi_stats']['max']:.2f}",
                    f"{results['roi_stats']['percentiles']['p10']:.2f}",
                    f"{results['roi_stats']['percentiles']['p25']:.2f}",
                    f"{results['roi_stats']['percentiles']['p75']:.2f}",
                    f"{results['roi_stats']['percentiles']['p90']:.2f}"
                ],
                "Payback (Months)": [
                    f"{results['payback_stats']['mean']:.2f}",
                    f"{results['payback_stats']['median']:.2f}",
                    f"{results['payback_stats']['std_dev']:.2f}",
                    f"{results['payback_stats']['min']:.2f}",
                    f"{results['payback_stats']['max']:.2f}",
                    f"{results['payback_stats']['percentiles']['p10']:.2f}",
                    f"{results['payback_stats']['percentiles']['p25']:.2f}",
                    f"{results['payback_stats']['percentiles']['p75']:.2f}",
                    f"{results['payback_stats']['percentiles']['p90']:.2f}"
                ]
            })
            
            st.dataframe(stats_df, use_container_width=True)
            
            # Insights section
            st.markdown("#### Simulation Insights")
            
            if results['probability_metrics']['prob_positive_roi'] >= 90:
                st.success("✅ This project has a **high probability of success** with over 90% chance of positive ROI.")
            elif results['probability_metrics']['prob_positive_roi'] >= 70:
                st.info("ℹ️ This project has a **good probability of success** with over 70% chance of positive ROI.")
            elif results['probability_metrics']['prob_positive_roi'] >= 50:
                st.warning("⚠️ This project has a **moderate risk** with just over 50% chance of positive ROI.")
            else:
                st.error("❌ This project has a **high risk** with less than 50% chance of positive ROI.")
            
            st.markdown(f"""
            **Key Insights:**
            
            - There is a **{results['probability_metrics']['prob_positive_roi']:.1f}%** probability of achieving a positive ROI
            - There is a **{results['probability_metrics']['prob_payback_1yr']:.1f}%** probability of achieving payback within 1 year
            - The median ROI is **{results['roi_stats']['median']:.2f}%**
            - The median payback period is **{results['payback_stats']['median']:.2f}** months
            
            The simulation accounts for variability in costs, pricing, sales volumes, return rates, and expected reduction effectiveness.
            """)

def display_analysis_page():
    """Display the main analysis page with tabs."""
    display_header()
    
    tabs = st.tabs(["Quality ROI Analysis", "Tariff Calculator", "Marketing ROI", "Monte Carlo Simulation"])
    
    with tabs[0]:
        if not st.session_state.analysis_submitted:
            results = display_quality_analysis_form()
            if results:
                display_quality_issue_results(results)
                display_ai_assistant(results)
        else:
            col1, col2 = st.columns([1, 6])
            with col1:
                if st.button("Start New Analysis", key="new_analysis_btn"):
                    st.session_state.analysis_submitted = False
                    st.session_state.quality_analysis_results = None
                    st.session_state.chat_history = []
                    st.rerun()
            
            # Display a badge with the SKU being analyzed
            with col2:
                st.markdown(f"""
                <div style="background-color: {TERTIARY_COLOR}; padding: 0.5rem 1rem; border-radius: 20px; 
                      display: inline-block; margin-bottom: 1rem;">
                    <span style="font-weight: 600;">Analyzing SKU:</span> {st.session_state.quality_analysis_results['sku']} | 
                    <span style="font-weight: 600;">Type:</span> {st.session_state.quality_analysis_results['product_type']}
                </div>
                """, unsafe_allow_html=True)
            
            # Add tabs for results sections
            result_tabs = st.tabs(["Analysis Results", "AI Assistant", "Export Options"])
            
            with result_tabs[0]:
                display_quality_issue_results(st.session_state.quality_analysis_results)
            
            with result_tabs[1]:
                display_ai_assistant(st.session_state.quality_analysis_results)
            
            with result_tabs[2]:
                st.markdown("### Export Analysis Results")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("#### CSV Export")
                    if st.session_state.quality_analysis_results:
                        results = st.session_state.quality_analysis_results
                        df = export_as_csv(results)
                        csv_download = generate_download_link(
                            df, 
                            f"quality_analysis_{results['sku']}_{datetime.now().strftime('%Y%m%d')}.csv", 
                            "📥 Export as CSV"
                        )
                        st.markdown(csv_download, unsafe_allow_html=True)
                        
                        st.markdown("Preview:")
                        st.dataframe(df, use_container_width=True, height=300)
                
                with col2:
                    st.markdown("#### PDF Report")
                    if st.session_state.quality_analysis_results:
                        results = st.session_state.quality_analysis_results
                        try:
                            pdf_buffer = export_as_pdf(results)
                            pdf_data = base64.b64encode(pdf_buffer.read()).decode('utf-8')
                            pdf_download = f'<a href="data:application/pdf;base64,{pdf_data}" download="quality_analysis_{results["sku"]}_{datetime.now().strftime("%Y%m%d")}.pdf" class="export-button">📄 Export as PDF</a>'
                            st.markdown(pdf_download, unsafe_allow_html=True)
                            
                            st.markdown("The PDF report includes:")
                            st.markdown("""
                            - Executive summary with key metrics
                            - Detailed financial analysis
                            - Visualizations of return rates and ROI
                            - Risk assessment breakdown
                            - Recommendations for action
                            """)
                        except Exception as e:
                            st.error(f"Error generating PDF: {e}")
    
    with tabs[1]:
        display_landed_cost_calculator()
    
    with tabs[2]:
        calculate_ad_roi_ui()
    
    with tabs[3]:
        run_monte_carlo_simulation_ui()

# --- MAIN APPLICATION ---
def main():
    """Main application function."""
    try:
        # Add FontAwesome to the page for icons
        st.markdown("""
        <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css">
        """, unsafe_allow_html=True)
        
        # Handle page navigation
        if st.session_state.current_page == "analysis":
            display_analysis_page()
        
        # Add footer with version and additional info
        st.markdown("""
        <div style="background-color: #f8f9fa; padding: 1rem; border-radius: 4px; margin-top: 2rem; 
                    text-align: center; border-top: 1px solid #dee2e6;">
            <div style="color: #6c757d; font-size: 0.8rem;">
                Product Profitability Analysis Tool v1.0.1 | © 2025 Medical Device Quality Management
            </div>
            <div style="color: #6c757d; font-size: 0.8rem; margin-top: 0.5rem;">
                For support contact: <a href="mailto:quality@meddevice.com">quality@meddevice.com</a>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
    except Exception as e:
        st.error(f"An unexpected error occurred: {str(e)}")
        logger.exception("Unexpected application error")
        
        # Show detailed error message and recovery options
        with st.expander("Error Details", expanded=True):
            st.code(traceback.format_exc())
            
            st.markdown("""
            ### Troubleshooting Options
            
            1. **Refresh the page** to restart the application
            2. **Clear browser cache** and try again
            3. If the problem persists, please contact support with the error details above
            """)
            
            if st.button("Reset Application State"):
                for key in list(st.session_state.keys()):
                    del st.session_state[key]
                st.experimental_rerun()

if __name__ == "__main__":
    main()
