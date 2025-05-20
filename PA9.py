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
import logging

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
    
    .metric-card {{ 
        background-color: var(--card-bg); 
        border-radius: 8px; 
        padding: 1rem; 
        box-shadow: 0 2px 4px rgba(0,0,0,0.1); 
        margin-bottom: 1rem; 
    }}
    
    .metric-label {{
        font-size: 0.85rem;
        color: var(--text-secondary);
        margin-bottom: 0.25rem;
    }}
    
    .metric-value {{
        font-size: 1.5rem;
        font-weight: 600;
        color: var(--text-primary);
    }}
    
    .metric-subvalue {{
        font-size: 0.8rem;
        color: var(--text-muted);
    }}
    
    .badge {{
        padding: 0.25rem 0.5rem;
        border-radius: 4px;
        font-size: 0.75rem;
        font-weight: 600;
    }}
    
    .badge-success {{
        background-color: var(--success);
        color: white;
    }}
    
    .badge-warning {{
        background-color: var(--warning);
        color: white;
    }}
    
    .badge-danger {{
        background-color: var(--danger);
        color: white;
    }}
    
    .assistant-bubble {{ 
        background: white; 
        border-left: 3px solid var(--primary); 
        padding: 0.75rem; 
        margin-bottom: 0.5rem;
        border-radius: 0 4px 4px 0;
    }}
    
    .user-bubble {{ 
        background: {TERTIARY_COLOR}; 
        padding: 0.75rem; 
        margin-left: auto; 
        margin-bottom: 0.5rem;
        border-radius: 4px 0 0 4px;
        max-width: 80%;
    }}
    
    .stTabs [data-baseweb="tab-list"] {{
        gap: 1rem;
    }}
    
    .stTabs [data-baseweb="tab"] {{
        height: 3rem;
        white-space: pre-wrap;
        background-color: white;
        border-radius: 4px 4px 0 0;
        gap: 0.5rem;
        padding-top: 0.5rem;
    }}
    
    .stTabs [aria-selected="true"] {{
        background-color: {PRIMARY_COLOR} !important;
        color: white !important;
    }}
    
    div.block-container {{
        padding-top: 2rem;
        padding-bottom: 2rem;
    }}
    
    .export-button {{
        background-color: var(--primary);
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 4px;
        text-decoration: none;
        display: inline-block;
        margin-right: 0.5rem;
    }}
    
    .export-button:hover {{
        background-color: var(--secondary);
    }}
</style>
""", unsafe_allow_html=True)

# --- LOGGING SETUP ---
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
            dsn=sentry_dsn,
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
        headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}
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
    
    Be concise but thorough in your responses. Prioritize patient safety and regulatory compliance.
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
    expected_reduction: float = 50.0,
    solution_confidence: float = 90.0,
    monthly_growth_rate: float = 5.0,
    additional_params: Dict[str, Any] = None
) -> Dict[str, Any]:
    """
    Analyze a quality issue and calculate financial impact, ROI, and recommendations.
    
    Parameters:
    -----------
    sku : str
        Product SKU
    product_type : str
        Product type (B2B, B2C, etc.)
    sales_30d : float
        Sales volume in the last 30 days
    returns_30d : float
        Returns volume in the last 30 days
    issue_description : str
        Description of the quality issue
    current_unit_cost : float
        Current unit cost of the product
    fix_cost_upfront : float
        Upfront cost to implement the fix
    fix_cost_per_unit : float
        Additional cost per unit after implementing the fix
    sales_price : float
        Sales price of the product
    expected_reduction : float, optional
        Expected percentage reduction in returns after the fix (default: 50.0)
    solution_confidence : float, optional
        Confidence level in the solution (default: 90.0)
    monthly_growth_rate : float, optional
        Expected monthly sales growth rate percentage (default: 5.0)
    additional_params : Dict[str, Any], optional
        Additional parameters for the analysis
    
    Returns:
    --------
    Dict[str, Any]
        Dictionary with analysis results
    """
    logger.info(f"Starting analysis for SKU={sku}, sales_30d={sales_30d}")
    
    try:
        # Initialize additional parameters if not provided
        if additional_params is None:
            additional_params = {}
        
        # Initialize results dictionary
        results = {
            "sku": sku,
            "product_type": product_type,
            "issue_description": issue_description,
            "analysis_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "current_metrics": {},
            "projected_metrics": {},
            "financial_impact": {},
            "risk_assessment": {},
            "recommendation": ""
        }
        
        # Calculate current metrics
        return_rate_30d = safe_divide(returns_30d * 100, sales_30d)
        unit_margin = sales_price - current_unit_cost
        margin_percentage = safe_divide(unit_margin * 100, sales_price)
        monthly_return_cost = returns_30d * current_unit_cost
        
        results["current_metrics"] = {
            "sales_30d": sales_30d,
            "returns_30d": returns_30d,
            "return_rate_30d": return_rate_30d,
            "unit_cost": current_unit_cost,
            "sales_price": sales_price,
            "unit_margin": unit_margin,
            "margin_percentage": margin_percentage,
            "monthly_return_cost": monthly_return_cost
        }
        
        # Project future metrics
        annual_sales = sales_30d * 12 * (1 + (monthly_growth_rate / 100)) ** 6  # Average growth over year
        annual_returns_current = (return_rate_30d / 100) * annual_sales
        annual_returns_after_fix = annual_returns_current * (1 - (expected_reduction / 100))
        annual_returns_reduction = annual_returns_current - annual_returns_after_fix
        
        new_unit_cost = current_unit_cost + fix_cost_per_unit
        new_unit_margin = sales_price - new_unit_cost
        new_margin_percentage = safe_divide(new_unit_margin * 100, sales_price)
        
        results["projected_metrics"] = {
            "annual_sales": annual_sales,
            "annual_returns_current": annual_returns_current,
            "annual_returns_after_fix": annual_returns_after_fix,
            "annual_returns_reduction": annual_returns_reduction,
            "new_unit_cost": new_unit_cost,
            "new_unit_margin": new_unit_margin,
            "new_margin_percentage": new_margin_percentage,
            "expected_reduction": expected_reduction,
            "solution_confidence": solution_confidence
        }
        
        # Calculate financial impact
        annual_loss_current = annual_returns_current * current_unit_cost
        annual_loss_after_fix = annual_returns_after_fix * new_unit_cost
        annual_savings = annual_loss_current - annual_loss_after_fix
        
        # Apply solution confidence factor
        risk_adjusted_savings = annual_savings * (solution_confidence / 100)
        total_investment = fix_cost_upfront + (fix_cost_per_unit * annual_sales)
        
        # Calculate ROI and payback period
        roi_1yr = safe_divide(risk_adjusted_savings * 100, total_investment)
        roi_3yr = safe_divide(risk_adjusted_savings * 3 * 100, total_investment)
        payback_period = safe_divide(total_investment * 12, risk_adjusted_savings)  # in months
        
        results["financial_impact"] = {
            "annual_loss": annual_loss_current,
            "annual_loss_after_fix": annual_loss_after_fix,
            "annual_savings": annual_savings,
            "risk_adjusted_savings": risk_adjusted_savings,
            "total_investment": total_investment,
            "roi_1yr": roi_1yr,
            "roi_3yr": roi_3yr,
            "payback_period": payback_period
        }
        
        # Risk assessment
        brand_impact = additional_params.get("brand_impact", "Medium")
        medical_impact = additional_params.get("medical_impact", "Low")
        regulatory_risk = additional_params.get("regulatory_risk", "Low")
        
        results["risk_assessment"] = {
            "brand_impact": brand_impact,
            "medical_impact": medical_impact,
            "regulatory_risk": regulatory_risk
        }
        
        # Generate recommendation
        if roi_3yr > 200 or return_rate_30d > 10 or medical_impact == "High":
            recommendation = "Fix Immediately"
        elif roi_3yr > 100 or return_rate_30d > 5 or medical_impact == "Medium":
            recommendation = "High Priority Fix"
        elif roi_3yr > 50:
            recommendation = "Recommended Fix"
        else:
            recommendation = "Monitor Issue"
        
        results["recommendation"] = recommendation
        
        # Log results
        logger.debug(f"Computed ROI: {roi_3yr:.2f}%")
        logger.info(f"Analysis completed for SKU={sku} with recommendation: {recommendation}")
        
        return results
    
    except Exception as e:
        logger.exception(f"Error in analyze_quality_issue for SKU={sku}")
        raise

def calculate_landed_cost(
    sales_price: float,
    unit_cost: float,
    tariff_percentage: float,
    shipping_cost: float = 0.0,
    storage_cost: float = 0.0,
    customs_fee: float = 0.0,
    broker_fee: float = 0.0,
    other_costs: float = 0.0,
    units_per_shipment: int = 1,
    currency_exchange_rate: float = 1.0
) -> Dict[str, float]:
    """
    Calculate the landed cost and relevant metrics for a product.
    
    Parameters:
    -----------
    sales_price : float
        Selling price of the product
    unit_cost : float
        Base cost of the product
    tariff_percentage : float
        Tariff percentage to apply
    shipping_cost : float, optional
        Shipping cost per shipment (default: 0.0)
    storage_cost : float, optional
        Storage cost per unit (default: 0.0)
    customs_fee : float, optional
        Customs fees per shipment (default: 0.0)
    broker_fee : float, optional
        Broker fees per shipment (default: 0.0)
    other_costs : float, optional
        Other miscellaneous costs per shipment (default: 0.0)
    units_per_shipment : int, optional
        Number of units per shipment (default: 1)
    currency_exchange_rate : float, optional
        Exchange rate to apply to costs (default: 1.0)
    
    Returns:
    --------
    Dict[str, float]
        Dictionary with landed cost calculations
    """
    logger.info(f"Calculating landed cost for unit_cost={unit_cost}, tariff_percentage={tariff_percentage}")
    
    try:
        # Apply exchange rate to costs
        adjusted_unit_cost = unit_cost * currency_exchange_rate
        
        # Calculate tariff amount
        tariff_amount = adjusted_unit_cost * (tariff_percentage / 100)
        
        # Calculate per-unit shipping and fees
        per_unit_shipping = safe_divide(shipping_cost, units_per_shipment)
        per_unit_customs = safe_divide(customs_fee, units_per_shipment)
        per_unit_broker = safe_divide(broker_fee, units_per_shipment)
        per_unit_other = safe_divide(other_costs, units_per_shipment)
        
        # Calculate total landed cost
        landed_cost = (
            adjusted_unit_cost + 
            tariff_amount + 
            per_unit_shipping + 
            storage_cost + 
            per_unit_customs + 
            per_unit_broker + 
            per_unit_other
        )
        
        # Calculate margins
        gross_margin = sales_price - landed_cost
        margin_percentage = safe_divide(gross_margin * 100, sales_price)
        
        # Calculate percentage breakdown
        total_extra_costs = landed_cost - adjusted_unit_cost
        base_cost_pct = safe_divide(adjusted_unit_cost * 100, landed_cost)
        tariff_pct = safe_divide(tariff_amount * 100, landed_cost)
        shipping_pct = safe_divide(per_unit_shipping * 100, landed_cost)
        storage_pct = safe_divide(storage_cost * 100, landed_cost)
        customs_pct = safe_divide(per_unit_customs * 100, landed_cost)
        broker_pct = safe_divide(per_unit_broker * 100, landed_cost)
        other_pct = safe_divide(per_unit_other * 100, landed_cost)
        
        return {
            "sales_price": sales_price,
            "base_unit_cost": adjusted_unit_cost,
            "tariff_amount": tariff_amount,
            "per_unit_shipping": per_unit_shipping,
            "storage_cost": storage_cost,
            "per_unit_customs": per_unit_customs,
            "per_unit_broker": per_unit_broker,
            "per_unit_other": per_unit_other,
            "landed_cost": landed_cost,
            "gross_margin": gross_margin,
            "margin_percentage": margin_percentage,
            "base_cost_pct": base_cost_pct,
            "tariff_pct": tariff_pct,
            "shipping_pct": shipping_pct,
            "storage_pct": storage_pct,
            "customs_pct": customs_pct,
            "broker_pct": broker_pct,
            "other_pct": other_pct
        }
    
    except Exception as e:
        logger.exception("Error in calculate_landed_cost")
        raise

def generate_tariff_scenarios(
    base_result: Dict[str, float],
    tariff_ranges: List[float]
) -> Dict[str, Dict[str, float]]:
    """
    Generate multiple tariff scenarios based on different tariff percentages.
    
    Parameters:
    -----------
    base_result : Dict[str, float]
        Base landed cost calculation result
    tariff_ranges : List[float]
        List of tariff percentages to simulate
    
    Returns:
    --------
    Dict[str, Dict[str, float]]
        Dictionary with scenarios for each tariff percentage
    """
    logger.info(f"Generating tariff scenarios for {len(tariff_ranges)} different rates")
    
    try:
        scenarios = {}
        
        sales_price = base_result["sales_price"]
        unit_cost = base_result["base_unit_cost"]
        shipping_cost = base_result["per_unit_shipping"] * (units_per_shipment if "units_per_shipment" in base_result else 1)
        storage_cost = base_result["storage_cost"]
        customs_fee = base_result["per_unit_customs"] * (units_per_shipment if "units_per_shipment" in base_result else 1)
        broker_fee = base_result["per_unit_broker"] * (units_per_shipment if "units_per_shipment" in base_result else 1)
        other_costs = base_result["per_unit_other"] * (units_per_shipment if "units_per_shipment" in base_result else 1)
        units_per_shipment = 1  # Default if not in base_result
        
        for tariff in tariff_ranges:
            scenario = calculate_landed_cost(
                sales_price=sales_price,
                unit_cost=unit_cost,
                tariff_percentage=tariff,
                shipping_cost=shipping_cost,
                storage_cost=storage_cost,
                customs_fee=customs_fee,
                broker_fee=broker_fee,
                other_costs=other_costs,
                units_per_shipment=units_per_shipment
            )
            scenarios[f"{tariff:.1f}%"] = scenario
        
        return scenarios
    
    except Exception as e:
        logger.exception("Error in generate_tariff_scenarios")
        raise

def calculate_ad_roi(
    campaign_cost: float,
    average_order_value: float,
    conversion_rate: float,
    impressions: float,
    profit_margin_percentage: float,
    existing_conversion_rate: float = 0.0,
    existing_impressions: float = 0.0
) -> Dict[str, float]:
    """
    Calculate ROI and metrics for advertising campaigns.
    
    Parameters:
    -----------
    campaign_cost : float
        Total cost of the campaign
    average_order_value : float
        Average value of an order
    conversion_rate : float
        Conversion rate percentage from impression to purchase
    impressions : float
        Number of campaign impressions
    profit_margin_percentage : float
        Profit margin percentage on sales
    existing_conversion_rate : float, optional
        Existing baseline conversion rate percentage (default: 0.0)
    existing_impressions : float, optional
        Existing baseline impressions (default: 0.0)
    
    Returns:
    --------
    Dict[str, float]
        Dictionary with calculated ROI metrics
    """
    logger.info(f"Calculating ad ROI for campaign_cost={campaign_cost}, impressions={impressions}")
    
    try:
        # Convert percentages to decimals
        conv_rate_decimal = conversion_rate / 100
        profit_margin_decimal = profit_margin_percentage / 100
        existing_conv_rate_decimal = existing_conversion_rate / 100
        
        # Calculate conversions (purchases)
        campaign_conversions = impressions * conv_rate_decimal
        existing_conversions = existing_impressions * existing_conv_rate_decimal
        incremental_conversions = campaign_conversions - existing_conversions
        
        # Calculate revenue and profit
        campaign_revenue = campaign_conversions * average_order_value
        campaign_profit = campaign_revenue * profit_margin_decimal
        
        # Calculate ROI and other metrics
        roi_percentage = safe_divide((campaign_profit - campaign_cost) * 100, campaign_cost)
        cpa = safe_divide(campaign_cost, campaign_conversions)  # Cost per acquisition
        cpc = safe_divide(campaign_cost, impressions * 0.1)  # Estimated cost per click (10% CTR)
        cpm = safe_divide(campaign_cost * 1000, impressions)  # Cost per thousand impressions
        
        return {
            "campaign_cost": campaign_cost,
            "impressions": impressions,
            "conversions": campaign_conversions,
            "conversion_rate": conversion_rate,
            "average_order_value": average_order_value,
            "campaign_revenue": campaign_revenue,
            "campaign_profit": campaign_profit,
            "roi_percentage": roi_percentage,
            "cpa": cpa,
            "cpc": cpc,
            "cpm": cpm,
            "profit_margin_percentage": profit_margin_percentage
        }
    
    except Exception as e:
        logger.exception("Error in calculate_ad_roi")
        raise

def run_monte_carlo_simulation(
    base_unit_cost: float,
    sales_price: float,
    monthly_sales: float,
    monthly_returns: float,
    fix_cost_upfront: float,
    fix_cost_per_unit: float,
    expected_reduction: float,
    solution_confidence: float,
    num_simulations: int = 1000,
    time_horizon_months: int = 36
) -> Dict[str, Any]:
    """
    Run a Monte Carlo simulation for quality issue fixes.
    
    Parameters:
    -----------
    base_unit_cost : float
        Current unit cost of the product
    sales_price : float
        Sales price of the product
    monthly_sales : float
        Current monthly sales volume
    monthly_returns : float
        Current monthly returns volume
    fix_cost_upfront : float
        Upfront cost to implement the fix
    fix_cost_per_unit : float
        Additional cost per unit after implementing the fix
    expected_reduction : float
        Expected percentage reduction in returns after the fix
    solution_confidence : float
        Confidence level in the solution
    num_simulations : int, optional
        Number of Monte Carlo simulations to run (default: 1000)
    time_horizon_months : int, optional
        Time horizon in months for the simulation (default: 36)
    
    Returns:
    --------
    Dict[str, Any]
        Dictionary with simulation results
    """
    logger.info(f"Running Monte Carlo simulation with {num_simulations} iterations")
    
    try:
        # Initialize results and random generator
        np.random.seed(int(time.time()))
        
        # Initialize arrays to store simulation results
        roi_results = np.zeros(num_simulations)
        payback_results = np.zeros(num_simulations)
        npv_results = np.zeros(num_simulations)
        
        # Define variation ranges
        sales_variation = 0.15  # ±15% monthly sales fluctuation
        return_rate_variation = 0.2  # ±20% return rate fluctuation
        fix_effectiveness_variation = 0.3  # ±30% fix effectiveness variation
        discount_rate = 0.1 / 12  # Monthly discount rate (10% annual)
        
        # Calculate base return rate
        base_return_rate = safe_divide(monthly_returns, monthly_sales)
        
        # Run simulations
        for i in range(num_simulations):
            # Randomize parameters for this simulation
            sim_monthly_sales = monthly_sales * np.random.uniform(1 - sales_variation, 1 + sales_variation)
            sim_return_rate = base_return_rate * np.random.uniform(1 - return_rate_variation, 1 + return_rate_variation)
            sim_fix_effectiveness = expected_reduction / 100 * np.random.uniform(
                1 - fix_effectiveness_variation, 
                1 + fix_effectiveness_variation
            )
            
            # Cap effectiveness at 95%
            sim_fix_effectiveness = min(sim_fix_effectiveness, 0.95)
            
            # Calculate monthly cash flows
            monthly_cashflows = np.zeros(time_horizon_months)
            
            # Initial investment (negative cash flow)
            monthly_cashflows[0] = -fix_cost_upfront
            
            for month in range(time_horizon_months):
                # Sales grow slightly each month (random between 0-2%)
                month_growth = 1 + np.random.uniform(0, 0.02)
                if month > 0:
                    sim_monthly_sales *= month_growth
                
                # Calculate returns before and after fix
                monthly_returns_before = sim_monthly_sales * sim_return_rate
                monthly_returns_after = monthly_returns_before * (1 - sim_fix_effectiveness)
                returns_reduction = monthly_returns_before - monthly_returns_after
                
                # Calculate savings from reduced returns
                savings = returns_reduction * base_unit_cost
                
                # Calculate additional costs from fix
                additional_costs = sim_monthly_sales * fix_cost_per_unit
                
                # Net monthly benefit
                net_benefit = savings - additional_costs
                
                # Add to cash flow (offset by 1 for initial investment)
                if month > 0:
                    monthly_cashflows[month] = net_benefit
            
            # Calculate NPV
            npv = -fix_cost_upfront  # Start with initial investment
            for month in range(1, time_horizon_months):
                npv += monthly_cashflows[month] / ((1 + discount_rate) ** month)
            
            # Calculate ROI (using NPV)
            total_investment = fix_cost_upfront + (fix_cost_per_unit * sim_monthly_sales * time_horizon_months)
            roi = safe_divide(npv * 100, total_investment)
            
            # Calculate payback period
            cumulative_cashflow = np.cumsum(monthly_cashflows)
            payback_month = np.argmax(cumulative_cashflow >= 0)
            if payback_month == 0 and cumulative_cashflow[-1] < 0:
                payback_month = time_horizon_months  # Never pays back
            
            # Store results
            roi_results[i] = roi
            payback_results[i] = payback_month
            npv_results[i] = npv
        
        # Calculate statistics
        roi_mean = np.mean(roi_results)
        roi_median = np.median(roi_results)
        roi_std = np.std(roi_results)
        roi_percentiles = np.percentile(roi_results, [5, 25, 75, 95])
        
        payback_mean = np.mean(payback_results)
        payback_median = np.median(payback_results)
        payback_std = np.std(payback_results)
        payback_percentiles = np.percentile(payback_results, [5, 25, 75, 95])
        
        npv_mean = np.mean(npv_results)
        npv_median = np.median(npv_results)
        npv_std = np.std(npv_results)
        npv_percentiles = np.percentile(npv_results, [5, 25, 75, 95])
        
        # Calculate success probabilities
        prob_positive_roi = np.mean(roi_results > 0) * 100
        prob_roi_above_50 = np.mean(roi_results > 50) * 100
        prob_roi_above_100 = np.mean(roi_results > 100) * 100
        prob_payback_under_12 = np.mean(payback_results < 12) * 100
        
        # Prepare histogram data
        roi_histogram = np.histogram(roi_results, bins=20)
        payback_histogram = np.histogram(payback_results, bins=20)
        npv_histogram = np.histogram(npv_results, bins=20)
        
        results = {
            "simulation_parameters": {
                "base_unit_cost": base_unit_cost,
                "sales_price": sales_price,
                "monthly_sales": monthly_sales,
                "monthly_returns": monthly_returns,
                "fix_cost_upfront": fix_cost_upfront,
                "fix_cost_per_unit": fix_cost_per_unit,
                "expected_reduction": expected_reduction,
                "solution_confidence": solution_confidence,
                "num_simulations": num_simulations,
                "time_horizon_months": time_horizon_months
            },
            "roi_statistics": {
                "mean": roi_mean,
                "median": roi_median,
                "std": roi_std,
                "percentile_5": roi_percentiles[0],
                "percentile_25": roi_percentiles[1],
                "percentile_75": roi_percentiles[2],
                "percentile_95": roi_percentiles[3]
            },
            "payback_statistics": {
                "mean": payback_mean,
                "median": payback_median,
                "std": payback_std,
                "percentile_5": payback_percentiles[0],
                "percentile_25": payback_percentiles[1],
                "percentile_75": payback_percentiles[2],
                "percentile_95": payback_percentiles[3]
            },
            "npv_statistics": {
                "mean": npv_mean,
                "median": npv_median,
                "std": npv_std,
                "percentile_5": npv_percentiles[0],
                "percentile_25": npv_percentiles[1],
                "percentile_75": npv_percentiles[2],
                "percentile_95": npv_percentiles[3]
            },
            "probabilities": {
                "positive_roi": prob_positive_roi,
                "roi_above_50": prob_roi_above_50,
                "roi_above_100": prob_roi_above_100,
                "payback_under_12": prob_payback_under_12
            },
            "histogram_data": {
                "roi_histogram": {
                    "counts": roi_histogram[0].tolist(),
                    "bins": roi_histogram[1].tolist()
                },
                "payback_histogram": {
                    "counts": payback_histogram[0].tolist(),
                    "bins": payback_histogram[1].tolist()
                },
                "npv_histogram": {
                    "counts": npv_histogram[0].tolist(),
                    "bins": npv_histogram[1].tolist()
                }
            },
            "raw_data": {
                "roi_results": roi_results.tolist(),
                "payback_results": payback_results.tolist(),
                "npv_results": npv_results.tolist()
            }
        }
        
        return results
    
    except Exception as e:
        logger.exception("Error in run_monte_carlo_simulation")
        raise

# --- EXPORT FUNCTIONS ---
def export_as_csv(results: Dict[str, Any]) -> str:
    """Export analysis results as a CSV download link."""
    try:
        if not results:
            return ""
        
        # Create a DataFrame with key metrics
        data = {
            "Metric": [
                "SKU",
                "Product Type",
                "Return Rate (30d)",
                "Monthly Sales",
                "Monthly Returns",
                "Unit Cost",
                "Sales Price",
                "Margin (%)",
                "Monthly Return Cost",
                "Annual Return Cost",
                "Fix Cost (Upfront)",
                "Fix Cost (Per Unit)",
                "Expected Reduction (%)",
                "Solution Confidence (%)",
                "Annual Savings",
                "ROI (1 Year)",
                "ROI (3 Year)",
                "Payback Period (Months)",
                "Recommendation"
            ],
            "Value": [
                results["sku"],
                results["product_type"],
                f"{results['current_metrics']['return_rate_30d']:.2f}%",
                results["current_metrics"]["sales_30d"],
                results["current_metrics"]["returns_30d"],
                f"${results['current_metrics']['unit_cost']:.2f}",
                f"${results['current_metrics']['sales_price']:.2f}",
                f"{results['current_metrics']['margin_percentage']:.2f}%",
                f"${results['current_metrics']['monthly_return_cost']:.2f}",
                f"${results['financial_impact']['annual_loss']:.2f}",
                f"${results.get('fix_cost_upfront', 0):.2f}",
                f"${results.get('fix_cost_per_unit', 0):.2f}",
                f"{results['projected_metrics']['expected_reduction']:.2f}%",
                f"{results['projected_metrics']['solution_confidence']:.2f}%",
                f"${results['financial_impact']['annual_savings']:.2f}",
                f"{results['financial_impact']['roi_1yr']:.2f}%",
                f"{results['financial_impact']['roi_3yr']:.2f}%",
                f"{results['financial_impact']['payback_period']:.2f}",
                results["recommendation"]
            ]
        }
        
        df = pd.DataFrame(data)
        
        # Generate timestamp for filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"quality_analysis_{results['sku']}_{timestamp}.csv"
        
        # Generate download link
        download_link = generate_download_link(df, filename, "📊 Download CSV Report")
        
        return download_link
    
    except Exception as e:
        logger.exception("Error in export_as_csv")
        return ""

def export_as_pdf(results: Dict[str, Any]) -> str:
    """Export analysis results as a PDF download link."""
    try:
        if not results:
            return ""
        
        # Create a BytesIO object to store the PDF
        pdf_buffer = BytesIO()
        
        # Create PDF with matplotlib
        with PdfPages(pdf_buffer) as pdf:
            # Create figures for the report
            
            # Figure 1: Summary page
            fig1, ax1 = plt.subplots(figsize=(8.5, 11))
            ax1.axis('off')
            
            # Add title and metadata
            plt.suptitle("Quality Issue Analysis Report", fontsize=16, y=0.98)
            plt.figtext(0.1, 0.95, f"SKU: {results['sku']} | Product Type: {results['product_type']}", fontsize=12)
            plt.figtext(0.1, 0.92, f"Analysis Date: {results['analysis_date']}", fontsize=10)
            plt.figtext(0.1, 0.89, f"Issue: {results['issue_description']}", fontsize=10)
            
            # Add recommendation
            recommendation = results["recommendation"]
            rec_color = "green" if "Monitor" in recommendation else "red" if "Immediately" in recommendation else "orange"
            plt.figtext(0.1, 0.85, "Recommendation:", fontsize=12, fontweight='bold')
            plt.figtext(0.3, 0.85, recommendation, fontsize=12, color=rec_color, fontweight='bold')
            
            # Current metrics section
            plt.figtext(0.1, 0.80, "Current Metrics", fontsize=12, fontweight='bold')
            plt.figtext(0.1, 0.77, f"Monthly Sales: {results['current_metrics']['sales_30d']}")
            plt.figtext(0.1, 0.75, f"Monthly Returns: {results['current_metrics']['returns_30d']}")
            plt.figtext(0.1, 0.73, f"Return Rate: {results['current_metrics']['return_rate_30d']:.2f}%")
            plt.figtext(0.1, 0.71, f"Unit Cost: ${results['current_metrics']['unit_cost']:.2f}")
            plt.figtext(0.1, 0.69, f"Sales Price: ${results['current_metrics']['sales_price']:.2f}")
            plt.figtext(0.1, 0.67, f"Margin: ${results['current_metrics']['unit_margin']:.2f} ({results['current_metrics']['margin_percentage']:.2f}%)")
            
            # Financial impact section
            plt.figtext(0.5, 0.77, f"Annual Loss: ${results['financial_impact']['annual_loss']:.2f}")
            plt.figtext(0.5, 0.75, f"Annual Savings: ${results['financial_impact']['annual_savings']:.2f}")
            plt.figtext(0.5, 0.73, f"Total Investment: ${results['financial_impact']['total_investment']:.2f}")
            plt.figtext(0.5, 0.71, f"ROI (1yr): {results['financial_impact']['roi_1yr']:.2f}%")
            plt.figtext(0.5, 0.69, f"ROI (3yr): {results['financial_impact']['roi_3yr']:.2f}%")
            plt.figtext(0.5, 0.67, f"Payback: {results['financial_impact']['payback_period']:.2f} months")
            
            # Projected metrics section
            plt.figtext(0.1, 0.62, "Projected Improvements", fontsize=12, fontweight='bold')
            plt.figtext(0.1, 0.59, f"Expected Reduction: {results['projected_metrics']['expected_reduction']:.2f}%")
            plt.figtext(0.1, 0.57, f"Solution Confidence: {results['projected_metrics']['solution_confidence']:.2f}%")
            plt.figtext(0.1, 0.55, f"New Unit Cost: ${results['projected_metrics']['new_unit_cost']:.2f}")
            plt.figtext(0.1, 0.53, f"New Margin: ${results['projected_metrics']['new_unit_margin']:.2f} ({results['projected_metrics']['new_margin_percentage']:.2f}%)")
            
            # Risk assessment section
            plt.figtext(0.5, 0.59, f"Brand Impact: {results['risk_assessment']['brand_impact']}")
            plt.figtext(0.5, 0.57, f"Medical Impact: {results['risk_assessment']['medical_impact']}")
            plt.figtext(0.5, 0.55, f"Regulatory Risk: {results['risk_assessment']['regulatory_risk']}")
            
            # Add page to PDF
            pdf.savefig(fig1)
            plt.close(fig1)
            
            # Figure 2: ROI Chart
            fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.5, 11))
            
            # ROI bar chart
            roi_data = [results['financial_impact']['roi_1yr'], results['financial_impact']['roi_3yr']]
            ax1.bar(['1 Year', '3 Year'], roi_data, color=[SECONDARY_COLOR, PRIMARY_COLOR])
            ax1.set_title('Return on Investment')
            ax1.set_ylabel('ROI (%)')
            ax1.grid(axis='y', linestyle='--', alpha=0.7)
            
            # Highlight levels
            ax1.axhline(y=50, color='green', linestyle='--', alpha=0.5)
            ax1.axhline(y=100, color='blue', linestyle='--', alpha=0.5)
            
            # Payback chart
            months = np.arange(1, 25)
            upfront = results.get('fix_cost_upfront', 0)
            monthly_savings = results['financial_impact']['annual_savings'] / 12
            cumulative_cashflow = [-upfront + (i * monthly_savings) for i in months]
            
            ax2.plot(months, cumulative_cashflow, marker='o', color=PRIMARY_COLOR)
            ax2.set_title('Payback Period')
            ax2.set_xlabel('Months')
            ax2.set_ylabel('Cumulative Cashflow ($)')
            ax2.grid(True, linestyle='--', alpha=0.7)
            ax2.axhline(y=0, color='red', linestyle='-', alpha=0.3)
            
            plt.tight_layout()
            pdf.savefig(fig2)
            plt.close(fig2)
        
        # Reset buffer position
        pdf_buffer.seek(0)
        
        # Generate timestamp for filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"quality_analysis_{results['sku']}_{timestamp}.pdf"
        
        # Create download link
        b64 = base64.b64encode(pdf_buffer.read()).decode()
        href = f'<a href="data:application/pdf;base64,{b64}" download="{filename}" class="export-button">📄 Download PDF Report</a>'
        
        return href
    
    except Exception as e:
        logger.exception("Error in export_as_pdf")
        return ""

# --- UI COMPONENTS ---
def display_header():
    """Display the application header."""
    st.markdown("""
    <div style="display: flex; justify-content: space-between; align-items: center;">
        <h1>Product Profitability Analysis</h1>
        <div>
            <span style="background-color: #F8F9FA; padding: 0.5rem; border-radius: 4px; margin-right: 1rem;">
                <span style="color: #6C757D; font-weight: 500;">Quality Manager</span>
            </span>
        </div>
    </div>
    <hr style="margin-top: 0;">
    """, unsafe_allow_html=True)

def display_sidebar():
    """Display the sidebar navigation."""
    with st.sidebar:
        st.image("https://via.placeholder.com/150x80?text=Logo", use_column_width=True)
        st.markdown("### Navigation")
        
        if st.button("📊 Analysis Dashboard", use_container_width=True):
            st.session_state.current_page = "analysis"
            st.rerun()
        
        if st.button("📈 Historical Reports", use_container_width=True):
            st.session_state.current_page = "reports"
            st.rerun()
        
        if st.button("⚙️ Settings", use_container_width=True):
            st.session_state.current_page = "settings"
            st.rerun()
        
        st.markdown("---")
        
        st.markdown("### View Mode")
        view_mode = st.radio(
            "Select detail level:",
            ["Basic", "Advanced"],
            index=0 if st.session_state.view_mode == "basic" else 1,
            label_visibility="collapsed"
        )
        
        if view_mode == "Basic" and st.session_state.view_mode != "basic":
            st.session_state.view_mode = "basic"
            st.rerun()
        elif view_mode == "Advanced" and st.session_state.view_mode != "advanced":
            st.session_state.view_mode = "advanced"
            st.rerun()
        
        st.markdown("---")
        
        # Add help section
        with st.expander("Help"):
            st.markdown("""
            ### Quick Guide
            
            - **Analysis Dashboard**: Analyze quality issues and their financial impact
            - **Historical Reports**: View previous analyses and compare results
            - **Settings**: Configure application settings and preferences
            
            For more help, contact support@vivehealth.com
            """)

def display_metric_card(label, value, subvalue=None, delta=None, delta_suffix=None, good_delta_up=True):
    """Display a metric card with label, value, and optional delta."""
    delta_html = ""
    if delta is not None:
        delta_value = float(delta.strip('%').strip('$').replace(',', ''))
        delta_color = SUCCESS_COLOR if (delta_value > 0 and good_delta_up) or (delta_value < 0 and not good_delta_up) else DANGER_COLOR
        delta_prefix = "+" if delta_value > 0 else ""
        delta_html = f"""
        <div style="font-size: 0.8rem; color: {delta_color}; margin-top: 0.25rem;">
            {delta_prefix}{delta}{" " + delta_suffix if delta_suffix else ""}
        </div>
        """
    
    subvalue_html = ""
    if subvalue:
        subvalue_html = f'<div class="metric-subvalue">{subvalue}</div>'
    
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">{label}</div>
        <div class="metric-value">{value}</div>
        {subvalue_html}
        {delta_html}
    </div>
    """, unsafe_allow_html=True)

def display_badge(text, badge_type="success"):
    """Display a badge with the specified text and type."""
    st.markdown(f'<span class="badge badge-{badge_type}">{text}</span>', unsafe_allow_html=True)

def display_quality_issue_results(results, expanded=True):
    """Display the results of a quality issue analysis."""
    if not results:
        return
    
    # Create columns for metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        display_metric_card(
            "Return Rate (30d)",
            f"{results['current_metrics']['return_rate_30d']:.2f}%",
            subvalue=f"{results['current_metrics']['returns_30d']} of {results['current_metrics']['sales_30d']} units"
        )
    
    with col2:
        display_metric_card(
            "Annual Loss From Returns",
            format_currency(results['financial_impact']['annual_loss']),
            subvalue=f"Monthly: {format_currency(results['current_metrics']['monthly_return_cost'])}"
        )
    
    with col3:
        display_metric_card(
            "Recommendation",
            results['recommendation'],
            subvalue=f"Confidence: {results['projected_metrics']['solution_confidence']}%"
        )
    
    # Financial impact section
    st.markdown("### Financial Impact")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        display_metric_card(
            "ROI (3 Year)",
            f"{results['financial_impact']['roi_3yr']:.2f}%",
            subvalue=f"1 Year: {results['financial_impact']['roi_1yr']:.2f}%"
        )
    
    with col2:
        display_metric_card(
            "Payback Period",
            f"{results['financial_impact']['payback_period']:.1f} months",
            subvalue=f"For ${format_currency(results['financial_impact']['total_investment'])}"
        )
    
    with col3:
        display_metric_card(
            "Annual Savings",
            format_currency(results['financial_impact']['annual_savings']),
            subvalue=f"Risk-adjusted: {format_currency(results['financial_impact']['risk_adjusted_savings'])}"
        )
    
    with col4:
        display_metric_card(
            "Expected Return Reduction",
            f"{results['projected_metrics']['expected_reduction']:.1f}%",
            subvalue=f"From {results['current_metrics']['return_rate_30d']:.2f}% to " + 
                    f"{results['current_metrics']['return_rate_30d'] * (1 - results['projected_metrics']['expected_reduction']/100):.2f}%"
        )
    
    # Visualizations
    st.markdown("### Visualizations")
    
    tab1, tab2, tab3 = st.tabs(["ROI Analysis", "Return Rate Impact", "Cost Breakdown"])
    
    with tab1:
        # ROI Visualization
        fig = go.Figure()
        
        # Add ROI bars
        roi_data = [
            results['financial_impact']['roi_1yr'],
            results['financial_impact']['roi_3yr']
        ]
        
        colors = [SECONDARY_COLOR, PRIMARY_COLOR]
        
        fig.add_trace(go.Bar(
            x=['1 Year ROI', '3 Year ROI'],
            y=roi_data,
            text=[f"{roi:.1f}%" for roi in roi_data],
            textposition='auto',
            marker_color=colors
        ))
        
        # Add threshold lines
        fig.add_shape(
            type="line",
            x0=-0.5,
            x1=1.5,
            y0=50,
            y1=50,
            line=dict(color=SUCCESS_COLOR, width=2, dash="dash"),
            name="Good ROI"
        )
        
        fig.add_shape(
            type="line",
            x0=-0.5,
            x1=1.5,
            y0=100,
            y1=100,
            line=dict(color=SUCCESS_COLOR, width=3),
            name="Excellent ROI"
        )
        
        # Update layout
        fig.update_layout(
            title="Return on Investment Analysis",
            xaxis_title="Time Period",
            yaxis_title="ROI (%)",
            yaxis=dict(
                ticksuffix="%",
                zeroline=True,
                zerolinewidth=2,
                zerolinecolor=BORDER_COLOR
            ),
            height=400,
            margin=dict(l=40, r=40, t=50, b=40),
            plot_bgcolor="white"
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        # Return Rate Impact Visualization
        current_return_rate = results['current_metrics']['return_rate_30d']
        projected_return_rate = current_return_rate * (1 - results['projected_metrics']['expected_reduction'] / 100)
        
        fig = go.Figure()
        
        fig.add_trace(go.Indicator(
            mode="gauge+number+delta",
            value=projected_return_rate,
            title={"text": "Return Rate After Fix"},
            delta={"reference": current_return_rate, "valueformat": ".2f", "relative": False, "decreasing": {"color": SUCCESS_COLOR}, "increasing": {"color": DANGER_COLOR}},
            gauge={
                "axis": {"range": [0, max(current_return_rate * 1.2, 10)]},
                "bar": {"color": SUCCESS_COLOR},
                "bgcolor": "white",
                "borderwidth": 2,
                "bordercolor": "gray",
                "steps": [
                    {"range": [0, 1], "color": SUCCESS_COLOR},
                    {"range": [1, 3], "color": WARNING_COLOR},
                    {"range": [3, 10], "color": DANGER_COLOR}
                ],
                "threshold": {
                    "line": {"color": "red", "width": 4},
                    "thickness": 0.75,
                    "value": current_return_rate
                }
            }
        ))
        
        fig.update_layout(
            height=300,
            margin=dict(l=40, r=40, t=50, b=40)
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Monthly projection
        months = list(range(1, 13))
        current_returns = [current_return_rate] * 12
        
        # Calculate gradual improvement over 12 months
        improvement_rate = results['projected_metrics']['expected_reduction'] / 100
        projected_returns = [current_return_rate * (1 - (improvement_rate * min(i / 3, 1))) for i in months]
        
        fig2 = go.Figure()
        
        fig2.add_trace(go.Scatter(
            x=months,
            y=current_returns,
            mode='lines',
            name='Without Fix',
            line=dict(color=DANGER_COLOR, dash='dash')
        ))
        
        fig2.add_trace(go.Scatter(
            x=months,
            y=projected_returns,
            mode='lines+markers',
            name='With Fix',
            line=dict(color=SUCCESS_COLOR)
        ))
        
        fig2.update_layout(
            title="Projected Return Rate Over Time",
            xaxis_title="Month",
            yaxis_title="Return Rate (%)",
            yaxis=dict(
                ticksuffix="%",
                zeroline=True,
                zerolinewidth=2,
                zerolinecolor=BORDER_COLOR
            ),
            height=300,
            margin=dict(l=40, r=40, t=50, b=40),
            plot_bgcolor="white",
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        st.plotly_chart(fig2, use_container_width=True)
    
    with tab3:
        # Cost Breakdown Visualization
        fig = go.Figure()
        
        # Create waterfall chart for cost breakdown
        measures = ["absolute", "relative", "relative", "relative", "total"]
        
        original_cost = results['current_metrics']['monthly_return_cost'] * 12  # Annual return cost
        reduced_returns_savings = original_cost * (results['projected_metrics']['expected_reduction'] / 100)
        increased_unit_cost = results['projected_metrics'].get('annual_sales', results['current_metrics']['sales_30d'] * 12) * results.get('fix_cost_per_unit', 0)
        upfront_cost_annual = results.get('fix_cost_upfront', 0) / 3  # Amortized over 3 years
        
        values = [original_cost, -reduced_returns_savings, increased_unit_cost, upfront_cost_annual, 
                 original_cost - reduced_returns_savings + increased_unit_cost + upfront_cost_annual]
        
        labels = ["Current Annual Loss", "Reduced Returns", "Increased Unit Cost", "Upfront Cost (Annual)", "New Annual Cost"]
        
        colors = [DANGER_COLOR, SUCCESS_COLOR, WARNING_COLOR, WARNING_COLOR, 
                 DANGER_COLOR if values[-1] > original_cost else SUCCESS_COLOR]
        
        fig.add_trace(go.Waterfall(
            name="Cost Breakdown",
            orientation="v",
            measure=measures,
            x=labels,
            textposition="outside",
            text=[format_currency(abs(val)) for val in values],
            y=values,
            connector={"line": {"color": "rgb(63, 63, 63)"}},
            marker={"color": colors}
        ))
        
        fig.update_layout(
            title="Cost Breakdown Analysis (Annual)",
            yaxis_title="Cost ($)",
            height=500,
            margin=dict(l=40, r=40, t=50, b=120),
            plot_bgcolor="white",
            showlegend=False
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Detailed metrics (expandable section)
    with st.expander("Detailed Metrics", expanded=expanded):
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Current Metrics")
            metrics_df = pd.DataFrame({
                "Metric": [
                    "Monthly Sales", 
                    "Monthly Returns", 
                    "Return Rate", 
                    "Unit Cost", 
                    "Sales Price", 
                    "Unit Margin", 
                    "Margin Percentage"
                ],
                "Value": [
                    f"{results['current_metrics']['sales_30d']:.0f} units",
                    f"{results['current_metrics']['returns_30d']:.0f} units",
                    f"{results['current_metrics']['return_rate_30d']:.2f}%",
                    f"${results['current_metrics']['unit_cost']:.2f}",
                    f"${results['current_metrics']['sales_price']:.2f}",
                    f"${results['current_metrics']['unit_margin']:.2f}",
                    f"{results['current_metrics']['margin_percentage']:.2f}%"
                ]
            })
            st.table(metrics_df)
        
        with col2:
            st.markdown("#### Projected Metrics")
            projected_df = pd.DataFrame({
                "Metric": [
                    "Annual Sales",
                    "Annual Returns (Current)",
                    "Annual Returns (After Fix)",
                    "Returns Reduction",
                    "New Unit Cost",
                    "New Unit Margin",
                    "New Margin Percentage"
                ],
                "Value": [
                    f"{results['projected_metrics']['annual_sales']:.0f} units",
                    f"{results['projected_metrics']['annual_returns_current']:.0f} units",
                    f"{results['projected_metrics']['annual_returns_after_fix']:.0f} units",
                    f"{results['projected_metrics']['annual_returns_reduction']:.0f} units",
                    f"${results['projected_metrics']['new_unit_cost']:.2f}",
                    f"${results['projected_metrics']['new_unit_margin']:.2f}",
                    f"{results['projected_metrics']['new_margin_percentage']:.2f}%"
                ]
            })
            st.table(projected_df)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Financial Impact")
            financial_df = pd.DataFrame({
                "Metric": [
                    "Annual Loss (Current)",
                    "Annual Loss (After Fix)",
                    "Annual Savings",
                    "Risk-Adjusted Savings",
                    "Total Investment",
                    "ROI (1 Year)",
                    "ROI (3 Year)",
                    "Payback Period"
                ],
                "Value": [
                    f"${results['financial_impact']['annual_loss']:.2f}",
                    f"${results['financial_impact']['annual_loss_after_fix']:.2f}",
                    f"${results['financial_impact']['annual_savings']:.2f}",
                    f"${results['financial_impact']['risk_adjusted_savings']:.2f}",
                    f"${results['financial_impact']['total_investment']:.2f}",
                    f"{results['financial_impact']['roi_1yr']:.2f}%",
                    f"{results['financial_impact']['roi_3yr']:.2f}%",
                    f"{results['financial_impact']['payback_period']:.2f} months"
                ]
            })
            st.table(financial_df)
        
        with col2:
            st.markdown("#### Risk Assessment")
            risk_df = pd.DataFrame({
                "Factor": [
                    "Brand Impact",
                    "Medical Impact",
                    "Regulatory Risk",
                    "Expected Reduction",
                    "Solution Confidence"
                ],
                "Assessment": [
                    results['risk_assessment']['brand_impact'],
                    results['risk_assessment']['medical_impact'],
                    results['risk_assessment']['regulatory_risk'],
                    f"{results['projected_metrics']['expected_reduction']:.2f}%",
                    f"{results['projected_metrics']['solution_confidence']:.2f}%"
                ]
            })
            st.table(risk_df)
    
    # Export options
    st.markdown("### Export Results")
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(export_as_csv(results), unsafe_allow_html=True)
    
    with col2:
        st.markdown(export_as_pdf(results), unsafe_allow_html=True)

def display_chat_assistant(results=None):
    """Display the AI chat assistant with conversation history."""
    with st.expander("Quality Consultant AI Assistant", expanded=True):
        # Display conversation history
        for msg in st.session_state.chat_history:
            if msg["role"] == "user":
                st.markdown(f'<div class="user-bubble">{msg["content"]}</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="assistant-bubble">{msg["content"]}</div>', unsafe_allow_html=True)
        
        # Input for new message
        user_input = st.text_area("Ask the AI assistant about quality issues, regulations, or recommendations:", key="chat_input", placeholder="e.g., What regulatory considerations should I keep in mind?")
        
        # Send button
        if st.button("Send", key="send_chat"):
            if user_input.strip():
                # Add user message to history
                st.session_state.chat_history.append({"role": "user", "content": user_input})
                
                # Create system prompt if results are available
                system_prompt = get_system_prompt(results)
                
                # Build messages for API call
                messages = [{"role": "system", "content": system_prompt}] + [
                    {"role": m["role"], "content": m["content"]} for m in st.session_state.chat_history
                ]
                
                # Get response from AI
                with st.spinner("Getting expert advice..."):
                    ai_resp = call_openai_api(messages)
                
                # Add assistant response to history
                st.session_state.chat_history.append({"role": "assistant", "content": ai_resp})
                
                # Clear input and refresh
                st.rerun()

def display_quality_analysis_form():
    """Display the quality issue analysis form."""
    with st.form("quality_analysis_form"):
        st.markdown("### Quality Issue Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            sku = st.text_input("SKU Number", placeholder="e.g., VH-MED-1001")
            product_type = st.selectbox(
                "Product Type",
                options=["B2C", "B2B", "Hospital", "Home Health", "Specialty"],
                index=0
            )
            sales_30d = st.number_input("Sales (Last 30 Days)", min_value=0, value=1000)
            returns_30d = st.number_input("Returns (Last 30 Days)", min_value=0, value=50)
            current_unit_cost = st.number_input("Current Unit Cost ($)", min_value=0.0, value=25.0, format="%.2f")
            sales_price = st.number_input("Sales Price ($)", min_value=0.0, value=59.99, format="%.2f")
        
        with col2:
            fix_cost_upfront = st.number_input("Fix Cost - Upfront ($)", min_value=0.0, value=5000.0, format="%.2f")
            fix_cost_per_unit = st.number_input("Fix Cost - Per Unit ($)", min_value=0.0, value=1.50, format="%.2f")
            expected_reduction = st.slider("Expected Returns Reduction (%)", min_value=0, max_value=100, value=50)
            solution_confidence = st.slider("Solution Confidence (%)", min_value=0, max_value=100, value=90)
            monthly_growth_rate = st.slider("Monthly Growth Rate (%)", min_value=0.0, max_value=20.0, value=5.0)
        
        # Additional parameters in advanced mode
        if st.session_state.view_mode == "advanced":
            st.markdown("### Advanced Parameters")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                brand_impact = st.selectbox(
                    "Brand Impact",
                    options=["Low", "Medium", "High"],
                    index=1
                )
            
            with col2:
                medical_impact = st.selectbox(
                    "Medical Impact",
                    options=["Low", "Medium", "High"],
                    index=0
                )
            
            with col3:
                regulatory_risk = st.selectbox(
                    "Regulatory Risk",
                    options=["Low", "Medium", "High"],
                    index=0
                )
            
            additional_params = {
                "brand_impact": brand_impact,
                "medical_impact": medical_impact,
                "regulatory_risk": regulatory_risk
            }
        else:
            additional_params = None
        
        issue_description = st.text_area(
            "Issue Description",
            placeholder="Describe the quality issue in detail...",
            height=100
        )
        
        submitted = st.form_submit_button("Analyze Quality Issue")
        
        if submitted:
            # Validation
            if not sku:
                st.error("SKU Number is required")
                return
            
            if sales_30d <= 0:
                st.error("Sales must be greater than zero")
                return
            
            if current_unit_cost <= 0 or sales_price <= 0:
                st.error("Unit cost and sales price must be greater than zero")
                return
            
            # Run analysis
            with st.spinner("Analyzing quality issue..."):
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
                        monthly_growth_rate=monthly_growth_rate,
                        additional_params=additional_params
                    )
                    
                    st.session_state.quality_analysis_results = results
                    st.session_state.analysis_submitted = True
                    st.rerun()
                except Exception as e:
                    logger.exception("Error running quality analysis")
                    st.error(f"Error analyzing quality issue: {str(e)}")

def display_landed_cost_calculator():
    """Display the landed cost calculator form and results."""
    st.markdown("### Landed Cost & Tariff Calculator")
    
    col1, col2 = st.columns(2)
    
    with col1:
        with st.form("landed_cost_form"):
            st.markdown("#### Product Details")
            
            sales_price = st.number_input("Sales Price ($)", min_value=0.0, value=59.99, format="%.2f")
            unit_cost = st.number_input("Unit Cost ($)", min_value=0.0, value=25.0, format="%.2f")
            tariff_percentage = st.slider("Tariff Percentage (%)", min_value=0.0, max_value=50.0, value=20.0, step=0.5)
            
            st.markdown("#### Shipping & Logistics")
            shipping_cost = st.number_input("Shipping Cost per Shipment ($)", min_value=0.0, value=250.0, format="%.2f")
            units_per_shipment = st.number_input("Units per Shipment", min_value=1, value=100)
            storage_cost = st.number_input("Storage Cost per Unit ($)", min_value=0.0, value=0.5, format="%.2f")
            
            st.markdown("#### Fees & Other Costs")
            customs_fee = st.number_input("Customs Fees per Shipment ($)", min_value=0.0, value=75.0, format="%.2f")
            broker_fee = st.number_input("Broker Fees per Shipment ($)", min_value=0.0, value=100.0, format="%.2f")
            other_costs = st.number_input("Other Costs per Shipment ($)", min_value=0.0, value=50.0, format="%.2f")
            
            if st.session_state.view_mode == "advanced":
                st.markdown("#### Advanced Options")
                currency_exchange_rate = st.number_input("Currency Exchange Rate", min_value=0.1, value=1.0, format="%.4f")
            else:
                currency_exchange_rate = 1.0
            
            submitted = st.form_submit_button("Calculate Landed Cost")
            
            if submitted:
                try:
                    with st.spinner("Calculating landed cost..."):
                        results = calculate_landed_cost(
                            sales_price=sales_price,
                            unit_cost=unit_cost,
                            tariff_percentage=tariff_percentage,
                            shipping_cost=shipping_cost,
                            storage_cost=storage_cost,
                            customs_fee=customs_fee,
                            broker_fee=broker_fee,
                            other_costs=other_costs,
                            units_per_shipment=units_per_shipment,
                            currency_exchange_rate=currency_exchange_rate
                        )
                        
                        # Run alternative tariff scenarios
                        tariff_ranges = [0.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0]
                        scenarios = generate_tariff_scenarios(results, tariff_ranges)
                        
                        st.session_state.tariff_calculations = {
                            "main_result": results,
                            "scenarios": scenarios
                        }
                        st.rerun()
                except Exception as e:
                    logger.exception("Error calculating landed cost")
                    st.error(f"Error calculating landed cost: {str(e)}")
    
    with col2:
        if st.session_state.tariff_calculations:
            results = st.session_state.tariff_calculations["main_result"]
            scenarios = st.session_state.tariff_calculations["scenarios"]
            
            st.markdown("#### Landed Cost Results")
            
            # Key metrics
            col1, col2, col3 = st.columns(3)
            
            with col1:
                display_metric_card(
                    "Landed Cost",
                    f"${results['landed_cost']:.2f}",
                    subvalue=f"Base: ${results['base_unit_cost']:.2f} + Extras: ${results['landed_cost'] - results['base_unit_cost']:.2f}"
                )
            
            with col2:
                display_metric_card(
                    "Gross Margin",
                    f"${results['gross_margin']:.2f}",
                    subvalue=f"{results['margin_percentage']:.2f}%"
                )
            
            with col3:
                display_metric_card(
                    "Tariff Amount",
                    f"${results['tariff_amount']:.2f}",
                    subvalue=f"{results['tariff_pct']:.2f}% of landed cost"
                )
            
            # Cost breakdown chart
            st.markdown("#### Cost Breakdown")
            
            labels = ['Base Cost', 'Tariff', 'Shipping', 'Storage', 'Customs', 'Broker', 'Other']
            values = [
                results['base_unit_cost'],
                results['tariff_amount'],
                results['per_unit_shipping'],
                results['storage_cost'],
                results['per_unit_customs'],
                results['per_unit_broker'],
                results['per_unit_other']
            ]
            
            # Create pie chart
            fig = go.Figure(data=[go.Pie(
                labels=labels,
                values=values,
                hole=.4,
                marker_colors=[PRIMARY_COLOR, DANGER_COLOR, WARNING_COLOR, TERTIARY_COLOR, SECONDARY_COLOR, SUCCESS_COLOR, TEXT_MUTED]
            )])
            
            fig.update_layout(
                title="Landed Cost Breakdown",
                height=300,
                margin=dict(l=40, r=40, t=50, b=40)
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Tariff scenario comparison
            st.markdown("#### Tariff Scenario Comparison")
            
            tariff_rates = list(scenarios.keys())
            margin_percentages = [scenario['margin_percentage'] for scenario in scenarios.values()]
            landed_costs = [scenario['landed_cost'] for scenario in scenarios.values()]
            
            # Create dual-axis chart
            fig = make_subplots(specs=[[{"secondary_y": True}]])
            
            fig.add_trace(
                go.Bar(
                    x=tariff_rates,
                    y=landed_costs,
                    name="Landed Cost",
                    marker_color=SECONDARY_COLOR
                ),
                secondary_y=False,
            )
            
            fig.add_trace(
                go.Scatter(
                    x=tariff_rates,
                    y=margin_percentages,
                    name="Margin %",
                    mode='lines+markers',
                    marker=dict(color=PRIMARY_COLOR),
                    line=dict(width=3)
                ),
                secondary_y=True,
            )
            
            fig.update_layout(
                title="Impact of Different Tariff Rates",
                xaxis_title="Tariff Rate",
                height=400,
                margin=dict(l=40, r=40, t=50, b=40),
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                )
            )
            
            fig.update_yaxes(title_text="Landed Cost ($)", secondary_y=False)
            fig.update_yaxes(title_text="Gross Margin (%)", secondary_y=True)
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Detailed breakdown in expandable section
            with st.expander("Detailed Cost Breakdown"):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("#### Cost Components")
                    cost_df = pd.DataFrame({
                        "Component": [
                            "Base Unit Cost",
                            "Tariff Amount",
                            "Shipping (Per Unit)",
                            "Storage Cost",
                            "Customs Fee (Per Unit)",
                            "Broker Fee (Per Unit)",
                            "Other Costs (Per Unit)",
                            "Total Landed Cost"
                        ],
                        "Amount": [
                            f"${results['base_unit_cost']:.2f}",
                            f"${results['tariff_amount']:.2f}",
                            f"${results['per_unit_shipping']:.2f}",
                            f"${results['storage_cost']:.2f}",
                            f"${results['per_unit_customs']:.2f}",
                            f"${results['per_unit_broker']:.2f}",
                            f"${results['per_unit_other']:.2f}",
                            f"${results['landed_cost']:.2f}"
                        ],
                        "Percentage": [
                            f"{results['base_cost_pct']:.2f}%",
                            f"{results['tariff_pct']:.2f}%",
                            f"{results['shipping_pct']:.2f}%",
                            f"{results['storage_pct']:.2f}%",
                            f"{results['customs_pct']:.2f}%",
                            f"{results['broker_pct']:.2f}%",
                            f"{results['other_pct']:.2f}%",
                            "100.00%"
                        ]
                    })
                    st.table(cost_df)
                
                with col2:
                    st.markdown("#### Margin Analysis")
                    margin_df = pd.DataFrame({
                        "Metric": [
                            "Sales Price",
                            "Landed Cost",
                            "Gross Margin",
                            "Margin Percentage"
                        ],
                        "Value": [
                            f"${results['sales_price']:.2f}",
                            f"${results['landed_cost']:.2f}",
                            f"${results['gross_margin']:.2f}",
                            f"{results['margin_percentage']:.2f}%"
                        ]
                    })
                    st.table(margin_df)
                
                st.markdown("#### Tariff Scenarios")
                scenarios_data = {
                    "Tariff Rate": [],
                    "Landed Cost": [],
                    "Gross Margin": [],
                    "Margin %": []
                }
                
                for rate, scenario in scenarios.items():
                    scenarios_data["Tariff Rate"].append(rate)
                    scenarios_data["Landed Cost"].append(f"${scenario['landed_cost']:.2f}")
                    scenarios_data["Gross Margin"].append(f"${scenario['gross_margin']:.2f}")
                    scenarios_data["Margin %"].append(f"{scenario['margin_percentage']:.2f}%")
                
                st.table(pd.DataFrame(scenarios_data))

def calculate_ad_roi_ui():
    """Display the advertising ROI calculator form and results."""
    st.markdown("### Marketing ROI Calculator")
    
    col1, col2 = st.columns(2)
    
    with col1:
        with st.form("ad_roi_form"):
            st.markdown("#### Campaign Details")
            
            campaign_name = st.text_input("Campaign Name", placeholder="e.g., Summer Promotion")
            campaign_cost = st.number_input("Campaign Cost ($)", min_value=0.0, value=5000.0, format="%.2f")
            impressions = st.number_input("Estimated Impressions", min_value=0, value=100000)
            
            st.markdown("#### Conversion Metrics")
            conversion_rate = st.slider("Conversion Rate (%)", min_value=0.0, max_value=10.0, value=2.0, step=0.1)
            average_order_value = st.number_input("Average Order Value ($)", min_value=0.0, value=60.0, format="%.2f")
            profit_margin_percentage = st.slider("Profit Margin (%)", min_value=0.0, max_value=100.0, value=40.0, step=1.0)
            
            if st.session_state.view_mode == "advanced":
                st.markdown("#### Comparative Analysis")
                existing_conversion_rate = st.slider("Existing Conversion Rate (%)", min_value=0.0, max_value=10.0, value=1.0, step=0.1)
                existing_impressions = st.number_input("Existing Impressions", min_value=0, value=50000)
            else:
                existing_conversion_rate = 0.0
                existing_impressions = 0.0
            
            submitted = st.form_submit_button("Calculate ROI")
            
            if submitted:
                try:
                    with st.spinner("Calculating advertising ROI..."):
                        results = calculate_ad_roi(
                            campaign_cost=campaign_cost,
                            average_order_value=average_order_value,
                            conversion_rate=conversion_rate,
                            impressions=impressions,
                            profit_margin_percentage=profit_margin_percentage,
                            existing_conversion_rate=existing_conversion_rate,
                            existing_impressions=existing_impressions
                        )
                        
                        results["campaign_name"] = campaign_name
                        
                        st.session_state.ad_roi_results = results
                        st.rerun()
                except Exception as e:
                    logger.exception("Error calculating ad ROI")
                    st.error(f"Error calculating ROI: {str(e)}")
    
    with col2:
        if hasattr(st.session_state, 'ad_roi_results') and st.session_state.ad_roi_results:
            results = st.session_state.ad_roi_results
            
            st.markdown(f"#### Results: {results['campaign_name']}")
            
            # Key metrics
            col1, col2, col3 = st.columns(3)
            
            with col1:
                roi_value = results['roi_percentage']
                roi_color = SUCCESS_COLOR if roi_value > 0 else DANGER_COLOR
                
                display_metric_card(
                    "Campaign ROI",
                    f"{roi_value:.2f}%",
                    subvalue=f"Profit: ${results['campaign_profit']:.2f}"
                )
            
            with col2:
                display_metric_card(
                    "Conversions",
                    f"{results['conversions']:.0f}",
                    subvalue=f"Rate: {results['conversion_rate']:.2f}%"
                )
            
            with col3:
                display_metric_card(
                    "Revenue",
                    f"${results['campaign_revenue']:.2f}",
                    subvalue=f"Cost: ${results['campaign_cost']:.2f}"
                )
            
            # ROI visualization
            st.markdown("#### ROI Breakdown")
            
            # Create waterfall chart for ROI breakdown
            fig = go.Figure()
            
            fig.add_trace(go.Waterfall(
                name="ROI Breakdown",
                orientation="v",
                measure=["absolute", "relative", "relative", "total"],
                x=["Campaign Cost", "Revenue", "Profit Margin", "Net Profit"],
                textposition="outside",
                text=[
                    f"-${results['campaign_cost']:.2f}", 
                    f"+${results['campaign_revenue']:.2f}", 
                    f"-${results['campaign_revenue'] - results['campaign_profit']:.2f}", 
                    f"${results['campaign_profit'] - results['campaign_cost']:.2f}"
                ],
                y=[
                    -results['campaign_cost'], 
                    results['campaign_revenue'], 
                    -(results['campaign_revenue'] - results['campaign_profit']), 
                    results['campaign_profit'] - results['campaign_cost']
                ],
                connector={"line": {"color": "rgb(63, 63, 63)"}},
                increasing={"marker": {"color": SUCCESS_COLOR}},
                decreasing={"marker": {"color": DANGER_COLOR}},
                totals={"marker": {"color": PRIMARY_COLOR}}
            ))
            
            fig.update_layout(
                title="Campaign ROI Breakdown",
                height=400,
                margin=dict(l=40, r=40, t=50, b=40),
                plot_bgcolor="white",
                showlegend=False
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Cost metrics
            st.markdown("#### Cost Efficiency Metrics")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                display_metric_card(
                    "Cost Per Acquisition (CPA)",
                    f"${results['cpa']:.2f}",
                    subvalue=f"AOV: ${results['average_order_value']:.2f}"
                )
            
            with col2:
                display_metric_card(
                    "Estimated Cost Per Click (CPC)",
                    f"${results['cpc']:.2f}",
                    subvalue="Based on 10% CTR"
                )
            
            with col3:
                display_metric_card(
                    "Cost Per Mille (CPM)",
                    f"${results['cpm']:.2f}",
                    subvalue=f"Impressions: {results['impressions']:,}"
                )
            
            # Detailed metrics in expandable section
            with st.expander("Detailed Metrics"):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("#### Campaign Metrics")
                    metrics_df = pd.DataFrame({
                        "Metric": [
                            "Campaign Cost",
                            "Impressions",
                            "Conversion Rate",
                            "Conversions",
                            "Average Order Value",
                            "Revenue",
                            "Profit Margin",
                            "Profit",
                            "ROI"
                        ],
                        "Value": [
                            f"${results['campaign_cost']:.2f}",
                            f"{results['impressions']:,}",
                            f"{results['conversion_rate']:.2f}%",
                            f"{results['conversions']:.0f}",
                            f"${results['average_order_value']:.2f}",
                            f"${results['campaign_revenue']:.2f}",
                            f"{results['profit_margin_percentage']:.2f}%",
                            f"${results['campaign_profit']:.2f}",
                            f"{results['roi_percentage']:.2f}%"
                        ]
                    })
                    st.table(metrics_df)
                
                with col2:
                    st.markdown("#### Cost Efficiency")
                    cost_df = pd.DataFrame({
                        "Metric": [
                            "Cost Per Acquisition (CPA)",
                            "Cost Per Click (CPC est.)",
                            "Cost Per Mille (CPM)",
                            "Revenue Per Impression",
                            "Profit Per Impression",
                            "Conversion Per Impression"
                        ],
                        "Value": [
                            f"${results['cpa']:.2f}",
                            f"${results['cpc']:.2f}",
                            f"${results['cpm']:.2f}",
                            f"${results['campaign_revenue'] / results['impressions']:.4f}",
                            f"${results['campaign_profit'] / results['impressions']:.4f}",
                            f"{results['conversion_rate'] / 100:.4f}"
                        ]
                    })
                    st.table(cost_df)
                
                # Breakeven analysis
                st.markdown("#### Breakeven Analysis")
                
                breakeven_conv_rate = (results['campaign_cost'] / (results['impressions'] * results['average_order_value'] * (results['profit_margin_percentage'] / 100))) * 100
                
                breakeven_df = pd.DataFrame({
                    "Metric": [
                        "Breakeven Conversion Rate",
                        "Current Conversion Rate",
                        "Difference",
                        "Safety Margin"
                    ],
                    "Value": [
                        f"{breakeven_conv_rate:.2f}%",
                        f"{results['conversion_rate']:.2f}%",
                        f"{results['conversion_rate'] - breakeven_conv_rate:.2f}%",
                        f"{((results['conversion_rate'] / breakeven_conv_rate) - 1) * 100:.2f}%"
                    ]
                })
                st.table(breakeven_df)

def run_monte_carlo_simulation_ui():
    """Display the Monte Carlo simulation form and results."""
    st.markdown("### Monte Carlo Simulation")
    
    col1, col2 = st.columns(2)
    
    with col1:
        with st.form("monte_carlo_form"):
            st.markdown("#### Simulation Parameters")
            
            # Inherit values from quality analysis if available
            default_results = st.session_state.get('quality_analysis_results', None)
            
            if default_results:
                default_unit_cost = default_results['current_metrics']['unit_cost']
                default_sales_price = default_results['current_metrics']['sales_price']
                default_monthly_sales = default_results['current_metrics']['sales_30d']
                default_monthly_returns = default_results['current_metrics']['returns_30d']
                default_expected_reduction = default_results['projected_metrics']['expected_reduction']
                default_solution_confidence = default_results['projected_metrics']['solution_confidence']
            else:
                default_unit_cost = 25.0
                default_sales_price = 59.99
                default_monthly_sales = 1000
                default_monthly_returns = 50
                default_expected_reduction = 50
                default_solution_confidence = 90
            
            base_unit_cost = st.number_input("Unit Cost ($)", min_value=0.01, value=default_unit_cost, format="%.2f")
            sales_price = st.number_input("Sales Price ($)", min_value=0.01, value=default_sales_price, format="%.2f")
            monthly_sales = st.number_input("Monthly Sales (Units)", min_value=1, value=default_monthly_sales)
            monthly_returns = st.number_input("Monthly Returns (Units)", min_value=0, value=default_monthly_returns)
            
            st.markdown("#### Fix Parameters")
            fix_cost_upfront = st.number_input("Fix Cost - Upfront ($)", min_value=0.0, value=5000.0, format="%.2f")
            fix_cost_per_unit = st.number_input("Fix Cost - Per Unit ($)", min_value=0.0, value=1.50, format="%.2f")
            expected_reduction = st.slider("Expected Returns Reduction (%)", min_value=0, max_value=100, value=default_expected_reduction)
            solution_confidence = st.slider("Solution Confidence (%)", min_value=0, max_value=100, value=default_solution_confidence)
            
            st.markdown("#### Simulation Options")
            num_simulations = st.slider("Number of Simulations", min_value=100, max_value=10000, value=1000, step=100)
            time_horizon_months = st.slider("Time Horizon (Months)", min_value=12, max_value=60, value=36, step=12)
            
            submitted = st.form_submit_button("Run Simulation")
            
            if submitted:
                try:
                    with st.spinner(f"Running {num_simulations} simulations..."):
                        results = run_monte_carlo_simulation(
                            base_unit_cost=base_unit_cost,
                            sales_price=sales_price,
                            monthly_sales=monthly_sales,
                            monthly_returns=monthly_returns,
                            fix_cost_upfront=fix_cost_upfront,
                            fix_cost_per_unit=fix_cost_per_unit,
                            expected_reduction=expected_reduction,
                            solution_confidence=solution_confidence,
                            num_simulations=num_simulations,
                            time_horizon_months=time_horizon_months
                        )
                        
                        st.session_state.monte_carlo_scenario = results
                        st.rerun()
                except Exception as e:
                    logger.exception("Error in Monte Carlo simulation")
                    st.error(f"Error running simulation: {str(e)}")
    
    with col2:
        if hasattr(st.session_state, 'monte_carlo_scenario') and st.session_state.monte_carlo_scenario:
            results = st.session_state.monte_carlo_scenario
            
            st.markdown("#### Simulation Results")
            
            # Key metrics
            col1, col2, col3 = st.columns(3)
            
            with col1:
                display_metric_card(
                    "Average ROI",
                    f"{results['roi_statistics']['mean']:.2f}%",
                    subvalue=f"Median: {results['roi_statistics']['median']:.2f}%"
                )
            
            with col2:
                display_metric_card(
                    "Avg. Payback Period",
                    f"{results['payback_statistics']['mean']:.1f} months",
                    subvalue=f"Median: {results['payback_statistics']['median']:.1f} months"
                )
            
            with col3:
                display_metric_card(
                    "Probability of Positive ROI",
                    f"{results['probabilities']['positive_roi']:.1f}%",
                    subvalue=f"ROI > 100%: {results['probabilities']['roi_above_100']:.1f}%"
                )
            
            # ROI distribution histogram
            st.markdown("#### ROI Distribution")
            
            roi_bins = results['histogram_data']['roi_histogram']['bins']
            roi_counts = results['histogram_data']['roi_histogram']['counts']
            
            fig = go.Figure()
            
            fig.add_trace(go.Bar(
                x=[(roi_bins[i] + roi_bins[i+1])/2 for i in range(len(roi_bins)-1)],
                y=roi_counts,
                marker_color=PRIMARY_COLOR,
                name="ROI Distribution"
            ))
            
            # Add mean and median lines
            fig.add_trace(go.Scatter(
                x=[results['roi_statistics']['mean'], results['roi_statistics']['mean']],
                y=[0, max(roi_counts) * 1.1],
                mode="lines",
                line=dict(color=SUCCESS_COLOR, width=2, dash="dash"),
                name="Mean ROI"
            ))
            
            fig.add_trace(go.Scatter(
                x=[results['roi_statistics']['median'], results['roi_statistics']['median']],
                y=[0, max(roi_counts) * 1.1],
                mode="lines",
                line=dict(color=SECONDARY_COLOR, width=2, dash="dash"),
                name="Median ROI"
            ))
            
            # Add zero line
            fig.add_trace(go.Scatter(
                x=[0, 0],
                y=[0, max(roi_counts) * 1.1],
                mode="lines",
                line=dict(color=DANGER_COLOR, width=2),
                name="Breakeven"
            ))
            
            fig.update_layout(
                title="ROI Distribution",
                xaxis_title="ROI (%)",
                yaxis_title="Frequency",
                height=300,
                margin=dict(l=40, r=40, t=50, b=40),
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                )
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Payback period distribution
            st.markdown("#### Payback Period Distribution")
            
            payback_bins = results['histogram_data']['payback_histogram']['bins']
            payback_counts = results['histogram_data']['payback_histogram']['counts']
            
            fig = go.Figure()
            
            fig.add_trace(go.Bar(
                x=[(payback_bins[i] + payback_bins[i+1])/2 for i in range(len(payback_bins)-1)],
                y=payback_counts,
                marker_color=SECONDARY_COLOR,
                name="Payback Distribution"
            ))
            
            # Add mean and median lines
            fig.add_trace(go.Scatter(
                x=[results['payback_statistics']['mean'], results['payback_statistics']['mean']],
                y=[0, max(payback_counts) * 1.1],
                mode="lines",
                line=dict(color=SUCCESS_COLOR, width=2, dash="dash"),
                name="Mean Payback"
            ))
            
            fig.add_trace(go.Scatter(
                x=[results['payback_statistics']['median'], results['payback_statistics']['median']],
                y=[0, max(payback_counts) * 1.1],
                mode="lines",
                line=dict(color=PRIMARY_COLOR, width=2, dash="dash"),
                name="Median Payback"
            ))
            
            # Add 12-month line
            fig.add_trace(go.Scatter(
                x=[12, 12],
                y=[0, max(payback_counts) * 1.1],
                mode="lines",
                line=dict(color=WARNING_COLOR, width=2),
                name="12 Months"
            ))
            
            fig.update_layout(
                title="Payback Period Distribution",
                xaxis_title="Payback Period (Months)",
                yaxis_title="Frequency",
                height=300,
                margin=dict(l=40, r=40, t=50, b=40),
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                )
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Detailed metrics in expandable section
            with st.expander("Detailed Statistics"):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("#### ROI Statistics")
                    roi_df = pd.DataFrame({
                        "Statistic": [
                            "Mean",
                            "Median",
                            "Standard Deviation",
                            "5th Percentile",
                            "25th Percentile",
                            "75th Percentile",
                            "95th Percentile"
                        ],
                        "Value": [
                            f"{results['roi_statistics']['mean']:.2f}%",
                            f"{results['roi_statistics']['median']:.2f}%",
                            f"{results['roi_statistics']['std']:.2f}%",
                            f"{results['roi_statistics']['percentile_5']:.2f}%",
                            f"{results['roi_statistics']['percentile_25']:.2f}%",
                            f"{results['roi_statistics']['percentile_75']:.2f}%",
                            f"{results['roi_statistics']['percentile_95']:.2f}%"
                        ]
                    })
                    st.table(roi_df)
                
                with col2:
                    st.markdown("#### Payback Statistics")
                    payback_df = pd.DataFrame({
                        "Statistic": [
                            "Mean",
                            "Median",
                            "Standard Deviation",
                            "5th Percentile",
                            "25th Percentile",
                            "75th Percentile",
                            "95th Percentile"
                        ],
                        "Value": [
                            f"{results['payback_statistics']['mean']:.2f} months",
                            f"{results['payback_statistics']['median']:.2f} months",
                            f"{results['payback_statistics']['std']:.2f} months",
                            f"{results['payback_statistics']['percentile_5']:.2f} months",
                            f"{results['payback_statistics']['percentile_25']:.2f} months",
                            f"{results['payback_statistics']['percentile_75']:.2f} months",
                            f"{results['payback_statistics']['percentile_95']:.2f} months"
                        ]
                    })
                    st.table(payback_df)
                
                st.markdown("#### Probability Analysis")
                prob_df = pd.DataFrame({
                    "Probability": [
                        "Positive ROI",
                        "ROI > 50%",
                        "ROI > 100%",
                        "Payback < 12 months"
                    ],
                    "Value": [
                        f"{results['probabilities']['positive_roi']:.2f}%",
                        f"{results['probabilities']['roi_above_50']:.2f}%",
                        f"{results['probabilities']['roi_above_100']:.2f}%",
                        f"{results['probabilities']['payback_under_12']:.2f}%"
                    ]
                })
                st.table(prob_df)
                
                st.markdown("#### Simulation Parameters")
                params_df = pd.DataFrame({
                    "Parameter": [
                        "Base Unit Cost",
                        "Sales Price",
                        "Monthly Sales",
                        "Monthly Returns",
                        "Fix Cost (Upfront)",
                        "Fix Cost (Per Unit)",
                        "Expected Reduction",
                        "Solution Confidence",
                        "Number of Simulations",
                        "Time Horizon"
                    ],
                    "Value": [
                        f"${results['simulation_parameters']['base_unit_cost']:.2f}",
                        f"${results['simulation_parameters']['sales_price']:.2f}",
                        f"{results['simulation_parameters']['monthly_sales']:.0f} units",
                        f"{results['simulation_parameters']['monthly_returns']:.0f} units",
                        f"${results['simulation_parameters']['fix_cost_upfront']:.2f}",
                        f"${results['simulation_parameters']['fix_cost_per_unit']:.2f}",
                        f"{results['simulation_parameters']['expected_reduction']:.2f}%",
                        f"{results['simulation_parameters']['solution_confidence']:.2f}%",
                        f"{results['simulation_parameters']['num_simulations']:.0f}",
                        f"{results['simulation_parameters']['time_horizon_months']:.0f} months"
                    ]
                })
                st.table(params_df)

def display_analysis_page():
    """Display the main analysis page with tabs."""
    st.markdown("## Product Profitability Analysis Dashboard")
    
    # Create tabs
    tabs = st.tabs(["Quality ROI Analysis", "Tariff Calculator", "Marketing ROI", "Monte Carlo Simulation"])
    
    with tabs[0]:
        # Quality ROI Analysis tab
        display_quality_analysis_form()
        
        if st.session_state.analysis_submitted and st.session_state.quality_analysis_results:
            st.markdown("---")
            st.markdown("## Analysis Results")
            display_quality_issue_results(st.session_state.quality_analysis_results, expanded=False)
            
            # Add AI assistant
            st.markdown("---")
            display_chat_assistant(st.session_state.quality_analysis_results)
    
    with tabs[1]:
        # Tariff Calculator tab
        display_landed_cost_calculator()
    
    with tabs[2]:
        # Marketing ROI tab
        calculate_ad_roi_ui()
    
    with tabs[3]:
        # Monte Carlo Simulation tab
        run_monte_carlo_simulation_ui()

def display_reports_page():
    """Display the historical reports page."""
    st.markdown("## Historical Reports")
    
    st.info("This feature will allow you to view and compare previous analyses. Currently in development.")
    
    # Placeholder for historical reports functionality
    st.markdown("### Recent Analyses")
    
    if hasattr(st.session_state, 'batch_analysis_results') and st.session_state.batch_analysis_results:
        # Display saved analyses
        for sku, result in st.session_state.batch_analysis_results.items():
            with st.expander(f"Analysis for {sku} ({result['analysis_date']})"):
                display_quality_issue_results(result, expanded=False)
    else:
        st.warning("No historical analyses found. Complete an analysis to save it here.")

def display_settings_page():
    """Display the settings page."""
    st.markdown("## Application Settings")
    
    st.info("This page allows you to configure application settings. Currently in development.")
    
    with st.form("settings_form"):
        st.markdown("### Display Settings")
        theme = st.selectbox(
            "Application Theme",
            options=["Default", "Dark Mode", "High Contrast"],
            index=0
        )
        
        decimal_places = st.slider("Decimal Places for Calculations", min_value=0, max_value=4, value=2)
        
        st.markdown("### Default Values")
        default_margin = st.slider("Default Profit Margin (%)", min_value=0, max_value=100, value=40)
        default_growth = st.slider("Default Monthly Growth Rate (%)", min_value=0, max_value=20, value=5)
        
        st.markdown("### API Configuration")
        api_key = st.text_input("OpenAI API Key", type="password", value="•" * 10)
        
        submitted = st.form_submit_button("Save Settings")
        
        if submitted:
            st.success("Settings saved successfully!")

# --- MAIN APPLICATION ---
def main():
    """Main application entry point."""
    # Display header and sidebar
    display_header()
    display_sidebar()
    
    # Display the appropriate page based on session state
    if st.session_state.current_page == "analysis":
        display_analysis_page()
    elif st.session_state.current_page == "reports":
        display_reports_page()
    elif st.session_state.current_page == "settings":
        display_settings_page()

if __name__ == "__main__":
    main()
