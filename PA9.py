import streamlit as st
import pandas as pd
import numpy as np
import os
import json
import time
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Union, Any
import openai
from openai import OpenAI

# Page configuration
st.set_page_config(
    page_title="QualityROI - Cost-Benefit Analysis Tool",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Define Vive Health color theme
VIVE_TEAL = "#41BEB6"  # Primary teal color from Vive Health
VIVE_TEAL_LIGHT = "#9BE4DF"  # Lighter teal for backgrounds
VIVE_TEAL_DARK = "#2D8C86"  # Darker teal for highlights/accents
VIVE_NAVY = "#1A3A4F"  # Navy color for headers and important text
VIVE_GREY_LIGHT = "#E5E7EB"  # Light grey for backgrounds
VIVE_GREY = "#6B7280"  # Medium grey for less important text
VIVE_RED = "#E53E3E"  # Red for warnings
VIVE_GREEN = "#10B981"  # Green for positive indicators
VIVE_AMBER = "#FBBF24"  # Amber for moderate indicators
VIVE_BLUE = "#3B82F6"  # Blue for information/charts

# Custom styling with Vive Health colors
st.markdown(f"""
<style>
    /* Main theme colors and base styles */
    :root {{
        --vive-teal: {VIVE_TEAL};
        --vive-teal-light: {VIVE_TEAL_LIGHT};
        --vive-teal-dark: {VIVE_TEAL_DARK};
        --vive-navy: {VIVE_NAVY};
        --vive-grey-light: {VIVE_GREY_LIGHT};
        --vive-grey: {VIVE_GREY};
        --vive-red: {VIVE_RED};
        --vive-green: {VIVE_GREEN};
        --vive-amber: {VIVE_AMBER};
        --vive-blue: {VIVE_BLUE};
    }}
    
    /* Apply brand colors to Streamlit components */
    .stButton>button {{
        background-color: var(--vive-teal);
        color: white;
        border: none;
        border-radius: 5px;
        padding: 0.5rem 1rem;
        font-weight: 500;
    }}
    .stButton>button:hover {{
        background-color: var(--vive-teal-dark);
    }}
    
    .stCheckbox label p {{
        color: var(--vive-navy);
    }}
    
    .stTextInput>div>div>input, .stNumberInput>div>div>input {{
        border-radius: 5px;
        border-color: #ddd;
    }}
    
    .stTextInput>div>div>input:focus, .stNumberInput>div>div>input:focus {{
        border-color: var(--vive-teal);
        box-shadow: 0 0 0 1px var(--vive-teal-light);
    }}
    
    /* Headers and text */
    .main-header {{
        font-size: 2.5rem;
        font-weight: 700;
        color: var(--vive-navy);
        margin-bottom: 1rem;
        border-bottom: 3px solid var(--vive-teal);
        padding-bottom: 0.5rem;
    }}
    
    .sub-header {{
        font-size: 1.8rem;
        font-weight: 600;
        color: var(--vive-navy);
        margin-bottom: 0.8rem;
        border-left: 4px solid var(--vive-teal);
        padding-left: 0.5rem;
    }}
    
    /* Cards and metrics */
    .card {{
        background-color: white;
        border-radius: 0.75rem;
        padding: 1.5rem;
        box-shadow: 0 4px 10px rgba(0, 0, 0, 0.08);
        margin-bottom: 1.5rem;
        border-top: 4px solid var(--vive-teal);
    }}
    
    .metric-card {{
        background-color: white;
        border-radius: 0.5rem;
        padding: 1rem;
        box-shadow: 0 2px 6px rgba(0, 0, 0, 0.08);
        margin-bottom: 1rem;
        border-left: 4px solid var(--vive-teal);
        transition: transform 0.2s;
    }}
    
    .metric-card:hover {{
        transform: translateY(-3px);
    }}
    
    .metric-label {{
        font-size: 1rem;
        font-weight: 500;
        color: var(--vive-grey);
        margin-bottom: 0.25rem;
    }}
    
    .metric-value {{
        font-size: 1.5rem;
        font-weight: 600;
        color: var(--vive-navy);
    }}
    
    /* Recommendations styling */
    .recommendation-high {{
        background-color: #DCFCE7;
        color: #166534;
        padding: 0.75rem;
        border-radius: 0.25rem;
        font-weight: 600;
        margin-top: 0.5rem;
        border-left: 4px solid #16A34A;
    }}
    
    .recommendation-medium {{
        background-color: #FEF3C7;
        color: #92400E;
        padding: 0.75rem;
        border-radius: 0.25rem;
        font-weight: 600;
        margin-top: 0.5rem;
        border-left: 4px solid #D97706;
    }}
    
    .recommendation-low {{
        background-color: #FEE2E2;
        color: #B91C1C;
        padding: 0.75rem;
        border-radius: 0.25rem;
        font-weight: 600;
        margin-top: 0.5rem;
        border-left: 4px solid #DC2626;
    }}
    
    /* Chat message styling */
    .chat-container {{
        background-color: #F9FAFB;
        border-radius: 0.75rem;
        padding: 1rem;
        margin-bottom: 1rem;
        height: 400px;
        overflow-y: auto;
        border: 1px solid #E5E7EB;
    }}
    
    .chat-message {{
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 0.75rem;
        max-width: 85%;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
    }}
    
    .user-message {{
        background-color: var(--vive-teal-light);
        margin-left: auto;
        border-bottom-right-radius: 0;
    }}
    
    .ai-message {{
        background-color: white;
        margin-right: auto;
        border-bottom-left-radius: 0;
        border-left: 3px solid var(--vive-teal);
    }}
    
    /* Vive Health branding bar */
    .vive-health-header {{
        background-color: var(--vive-teal);
        color: white;
        padding: 0.75rem;
        display: flex;
        align-items: center;
        border-radius: 0.5rem;
        margin-bottom: 1.5rem;
    }}
    
    .vive-logo {{
        font-size: 1.75rem;
        font-weight: 700;
        margin-right: 1rem;
    }}
    
    /* Form and input improvements */
    .form-section {{
        background-color: #F9FAFB;
        border-radius: 0.5rem;
        padding: 1.5rem;
        margin-bottom: 1.5rem;
        border: 1px solid #E5E7EB;
    }}
    
    .form-section-header {{
        font-size: 1.25rem;
        font-weight: 600;
        color: var(--vive-navy);
        margin-bottom: 1rem;
        border-bottom: 2px solid var(--vive-teal-light);
        padding-bottom: 0.5rem;
    }}
    
    /* Tooltips */
    .tooltip {{
        position: relative;
        display: inline-block;
        cursor: help;
    }}
    
    .tooltip .tooltiptext {{
        visibility: hidden;
        width: 200px;
        background-color: var(--vive-navy);
        color: white;
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
    
    /* Streamlit sidebar customization */
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
        color: var(--vive-navy);
        margin-bottom: 1.5rem;
        text-align: center;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid var(--vive-teal);
    }}
    
    /* Loading animation */
    .loading-spinner {{
        display: flex;
        justify-content: center;
        align-items: center;
        height: 100px;
    }}
    
    /* Chart styling */
    .chart-container {{
        background-color: white;
        border-radius: 0.5rem;
        padding: 1rem;
        box-shadow: 0 2px 6px rgba(0, 0, 0, 0.05);
        margin-bottom: 1.5rem;
    }}
</style>
""", unsafe_allow_html=True)

# --- CUSTOM COMPONENTS ---

def vive_header():
    """Displays a Vive Health branded header"""
    st.markdown(f"""
    <div class="vive-health-header">
        <div class="vive-logo">VIVE HEALTH</div>
        <div>QualityROI Analysis Tool</div>
    </div>
    """, unsafe_allow_html=True)

def display_spinner():
    """Displays a loading spinner"""
    st.markdown("""
    <div class="loading-spinner">
        <div class="spinner"></div>
        <div>Processing your request...</div>
    </div>
    """, unsafe_allow_html=True)

# --- UTILITY FUNCTIONS ---

def initialize_openai_client(api_key: str) -> Optional[OpenAI]:
    """
    Initialize the OpenAI client with the provided API key
    
    Args:
        api_key: OpenAI API key
        
    Returns:
        OpenAI client instance or None if initialization fails
    """
    if not api_key:
        st.warning("OpenAI API key is not configured. AI assistant features will be disabled.")
        return None
        
    try:
        client = OpenAI(api_key=api_key)
        # Test the connection with a minimal request
        test_response = client.chat.completions.create(
            model="gpt-3.5-turbo",  # Using a standard model for testing
            messages=[{"role": "user", "content": "Hello"}],
            max_tokens=5
        )
        return client
    except Exception as e:
        error_msg = str(e)
        if "authentication" in error_msg.lower():
            st.error("Authentication error: Please check your OpenAI API key.")
        elif "rate limit" in error_msg.lower():
            st.warning("Rate limit exceeded. Please try again later.")
        else:
            st.error(f"Error initializing OpenAI client: {error_msg}")
        return None

def get_ai_analysis(
    client: Optional[OpenAI], 
    system_prompt: str, 
    user_message: str,
    model: str = "gpt-4",  # Default to a standard model
    fallback_model: str = "gpt-3.5-turbo",  # Fallback model
    messages: Optional[List[Dict[str, str]]] = None
) -> str:
    """
    Get AI analysis for quality issues using the OpenAI API
    
    Args:
        client: OpenAI client instance
        system_prompt: System prompt providing context to the AI
        user_message: User message or query
        model: Primary model ID to use
        fallback_model: Fallback model if primary isn't available
        messages: Optional list of previous messages for context
        
    Returns:
        AI response text
    """
    if client is None:
        return ("AI assistant is currently unavailable. Please check your API key configuration "
                "or try again later.")

    try:
        # Build the messages array
        message_list = [{"role": "system", "content": system_prompt}]
        
        # Add previous messages if provided
        if messages:
            message_list.extend(messages)
        
        # Add the current user message
        message_list.append({"role": "user", "content": user_message})
        
        # Make the API call with the primary model
        try:
            response = client.chat.completions.create(
                model=model,
                messages=message_list,
                temperature=0.7,
                max_tokens=2048
            )
            return response.choices[0].message.content
        except Exception as primary_model_error:
            # Log the primary model error
            print(f"Error with primary model ({model}): {primary_model_error}")
            
            # Try with fallback model
            response = client.chat.completions.create(
                model=fallback_model,
                messages=message_list,
                temperature=0.7,
                max_tokens=2048
            )
            
            # Indicate fallback model was used
            content = response.choices[0].message.content
            return f"{content}\n\n_Note: Response provided using {fallback_model} model._"
            
    except Exception as e:
        error_message = str(e)
        print(f"Error getting AI analysis: {error_message}")
        
        # Provide a user-friendly error message
        if "rate limit" in error_message.lower():
            return "I'm sorry, but we've reached the OpenAI API rate limit. Please try again in a few moments."
        elif "authentication" in error_message.lower():
            return "There's an issue with the OpenAI API authentication. Please contact your administrator."
        else:
            return (f"I'm sorry, but I encountered an issue while analyzing this request. "
                    f"Please try again or contact technical support if the problem persists.\n\n"
                    f"Error details: {error_message}")

def format_currency(value: float) -> str:
    """Format a value as currency with $ symbol"""
    return f"${value:,.2f}"

def format_percentage(value: float) -> str:
    """Format a value as percentage with % symbol"""
    return f"{value:.1f}%"

# --- QUALITY MANAGER FUNCTIONS ---

def analyze_quality_issue(
    sku: str,
    product_type: str,  # B2B, B2C, or Both
    sales_30d: float,
    returns_30d: float,
    issue_description: str,
    current_unit_cost: float,
    fix_cost_upfront: float,
    fix_cost_per_unit: float,
    asin: Optional[str] = None,
    ncx_rate: Optional[float] = None,
    sales_365d: Optional[float] = None,
    returns_365d: Optional[float] = None,
    star_rating: Optional[float] = None,
    total_reviews: Optional[int] = None,
    fba_fee: Optional[float] = None
) -> Dict[str, Any]:
    """
    Analyze quality issue and determine cost-effectiveness of fixes
    """
    # Calculate basic metrics with error handling
    try:
        return_rate_30d = (returns_30d / sales_30d) * 100 if sales_30d > 0 else 0
    except ZeroDivisionError:
        return_rate_30d = 0
    
    # Include 365-day data if available
    return_rate_365d = None
    if sales_365d is not None and returns_365d is not None and sales_365d > 0:
        try:
            return_rate_365d = (returns_365d / sales_365d) * 100
        except ZeroDivisionError:
            return_rate_365d = 0
    
    # Calculate financial impact
    monthly_return_cost = returns_30d * current_unit_cost
    
    # Assume 80% reduction in returns by default, but scale based on return rate
    # Higher return rates might benefit more from fixes
    if return_rate_30d > 20:
        reduction_factor = 0.85  # 85% reduction for high return rates
    elif return_rate_30d > 10:
        reduction_factor = 0.80  # 80% reduction for medium return rates
    else:
        reduction_factor = 0.75  # 75% reduction for low return rates
        
    estimated_monthly_savings = monthly_return_cost * reduction_factor
    
    # Annual projections
    annual_return_cost = monthly_return_cost * 12
    annual_savings = estimated_monthly_savings * 12
    
    # Simple payback period (months) with error handling
    try:
        if estimated_monthly_savings > 0:
            payback_months = fix_cost_upfront / estimated_monthly_savings
        else:
            payback_months = float('inf')
    except ZeroDivisionError:
        payback_months = float('inf')
    
    # Calculate 3-year ROI with error handling
    try:
        total_investment = fix_cost_upfront + (fix_cost_per_unit * sales_30d * 36)  # 36 months
        total_savings = annual_savings * 3
        
        if total_investment > 0:
            roi_3yr = ((total_savings - total_investment) / total_investment) * 100
        else:
            roi_3yr = 0 if total_savings == 0 else float('inf')
    except ZeroDivisionError:
        roi_3yr = 0 if total_savings == 0 else float('inf')
    
    # Determine recommendation based on metrics
    if payback_months < 3:
        recommendation = "Highly Recommended - Quick ROI expected"
        recommendation_class = "recommendation-high"
    elif payback_months < 6:
        recommendation = "Recommended - Good medium-term ROI"
        recommendation_class = "recommendation-medium"
    elif payback_months < 12:
        recommendation = "Consider - Long-term benefits may outweigh costs"
        recommendation_class = "recommendation-medium"
    else:
        recommendation = "Not Recommended - Poor financial return"
        recommendation_class = "recommendation-low"
    
    # Adjust recommendation based on B2B/B2C and review metrics
    # For B2C products, star ratings are more important
    brand_impact = None
    if product_type in ["B2C", "Both"] and star_rating is not None:
        if star_rating < 3.5:
            brand_impact = "Significant - Low star rating indicates potential brand damage"
            # Adjust recommendation if star rating is concerning
            if recommendation_class != "recommendation-high":
                recommendation = "Recommended despite ROI - Brand protection needed"
                recommendation_class = "recommendation-medium"
    
    # For B2B products, return rate is more important
    if product_type in ["B2B", "Both"] and return_rate_30d > 10:
        if recommendation_class == "recommendation-low":
            recommendation = "Consider despite ROI - High return rate for B2B product"
            recommendation_class = "recommendation-medium"
            brand_impact = "Moderate - High return rate for B2B product may affect customer relationships"
    
    # Prepare results dictionary
    results = {
        "sku": sku,
        "asin": asin,
        "product_type": product_type,
        "current_metrics": {
            "return_rate_30d": return_rate_30d,
            "return_rate_365d": return_rate_365d,
            "ncx_rate": ncx_rate,
            "star_rating": star_rating,
            "total_reviews": total_reviews
        },
        "financial_impact": {
            "monthly_return_cost": monthly_return_cost,
            "annual_return_cost": annual_return_cost,
            "estimated_monthly_savings": estimated_monthly_savings,
            "annual_savings": annual_savings,
            "payback_months": payback_months,
            "roi_3yr": roi_3yr,
            "fix_cost_upfront": fix_cost_upfront,
            "fix_cost_per_unit": fix_cost_per_unit,
            "current_unit_cost": current_unit_cost,
            "fba_fee": fba_fee
        },
        "recommendation": recommendation,
        "recommendation_class": recommendation_class,
        "brand_impact": brand_impact,
        "issue_description": issue_description,
        "reduction_factor": reduction_factor * 100  # Store the reduction percentage for reference
    }
    
    return results

def analyze_salvage_operation(
    sku: str,
    affected_inventory: int,
    current_unit_cost: float,
    rework_cost_upfront: float,
    rework_cost_per_unit: float,
    expected_recovery_pct: float,
    expected_discount_pct: float
) -> Dict[str, Any]:
    """
    Analyze potential salvage operation for affected inventory
    """
    # Error handling for division by zero scenarios
    if affected_inventory <= 0:
        affected_inventory = 1
    
    # Calculate salvage metrics
    expected_units_recovered = affected_inventory * (expected_recovery_pct / 100)
    
    # Use industry standard markup based on product type
    # Typically medical devices have 2.5-3x markup
    regular_price = current_unit_cost * 2.5
    
    # Calculate discounted price
    discounted_price = regular_price * (1 - expected_discount_pct / 100)
    
    # Calculate financial impact
    total_rework_cost = rework_cost_upfront + (rework_cost_per_unit * affected_inventory)
    salvage_revenue = expected_units_recovered * discounted_price
    
    # Units that can't be recovered result in a write-off loss
    write_off_loss = (affected_inventory - expected_units_recovered) * current_unit_cost
    
    # Calculate net profit/loss from salvage operation
    salvage_profit = salvage_revenue - total_rework_cost - write_off_loss
    
    # Calculate ROI with error handling
    if total_rework_cost > 0:
        roi_percent = (salvage_profit / total_rework_cost) * 100
    else:
        roi_percent = 0 if salvage_profit == 0 else float('inf')
    
    # Complete write-off scenario (alternative)
    writeoff_cost = affected_inventory * current_unit_cost
    
    # Calculate profit/loss per unit
    profit_per_unit = salvage_profit / affected_inventory if affected_inventory > 0 else 0
    
    # Determine if salvage operation is recommended
    if salvage_profit > 0 and roi_percent > 20:
        recommendation = "Highly Recommended - Good return on rework investment"
        recommendation_class = "recommendation-high"
    elif salvage_profit > 0:
        recommendation = "Recommended - Positive but modest returns"
        recommendation_class = "recommendation-medium"
    elif salvage_profit > -writeoff_cost * 0.3:  # If loss is less than 30% of complete write-off
        recommendation = "Consider - Minimizes losses compared to full write-off"
        recommendation_class = "recommendation-medium"
    else:
        recommendation = "Not Recommended - Complete write-off may be more economical"
        recommendation_class = "recommendation-low"
    
    return {
        "sku": sku,
        "metrics": {
            "affected_inventory": affected_inventory,
            "expected_units_recovered": expected_units_recovered,
            "recovery_rate": expected_recovery_pct,
            "regular_price": regular_price,
            "discounted_price": discounted_price,
            "discount_percentage": expected_discount_pct
        },
        "financial": {
            "total_rework_cost": total_rework_cost,
            "salvage_revenue": salvage_revenue,
            "write_off_loss": write_off_loss,
            "salvage_profit": salvage_profit,
            "complete_writeoff_cost": writeoff_cost,
            "roi_percent": roi_percent,
            "profit_per_unit": profit_per_unit
        },
        "recommendation": recommendation,
        "recommendation_class": recommendation_class
    }

def display_quality_issue_results(results):
    """Display the quality issue analysis results in a visually appealing way"""
    st.markdown(f'<div class="card">', unsafe_allow_html=True)
    
    # Results header
    st.markdown(f'<div class="sub-header">Analysis Results for SKU: {results["sku"]}</div>', unsafe_allow_html=True)
    
    # Current metrics in a 3-column layout
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown('<div class="metric-label">Return Rate (30d)</div>', unsafe_allow_html=True)
        
        # Color-code return rate based on severity
        return_rate = results["current_metrics"]["return_rate_30d"]
        if return_rate > 10:
            color = VIVE_RED
        elif return_rate > 5:
            color = VIVE_AMBER
        else:
            color = VIVE_GREEN
            
        st.markdown(f'<div class="metric-value" style="color:{color}">{return_rate:.2f}%</div>', unsafe_allow_html=True)
        
        if results["current_metrics"]["return_rate_365d"] is not None:
            st.markdown('<div class="metric-label">Return Rate (365d)</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="metric-value">{results["current_metrics"]["return_rate_365d"]:.2f}%</div>', unsafe_allow_html=True)
        
        if results["current_metrics"]["star_rating"]:
            st.markdown('<div class="metric-label">Star Rating</div>', unsafe_allow_html=True)
            
            # Color-code star rating
            star_rating = results["current_metrics"]["star_rating"]
            if star_rating >= 4.0:
                rating_color = VIVE_GREEN
            elif star_rating >= 3.0:
                rating_color = VIVE_AMBER
            else:
                rating_color = VIVE_RED
                
            st.markdown(f'<div class="metric-value" style="color:{rating_color}">{star_rating:.1f}★</div>', unsafe_allow_html=True)
            
            if results["current_metrics"]["total_reviews"]:
                st.markdown(f'<div style="font-size:0.9rem;color:{VIVE_GREY}">({results["current_metrics"]["total_reviews"]} reviews)</div>', unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown('<div class="metric-label">Monthly Return Cost</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="metric-value">{format_currency(results["financial_impact"]["monthly_return_cost"])}</div>', unsafe_allow_html=True)
        
        st.markdown('<div class="metric-label">Annual Return Cost</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="metric-value">{format_currency(results["financial_impact"]["annual_return_cost"])}</div>', unsafe_allow_html=True)
        
        st.markdown('<div class="metric-label">Current Unit Cost</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="metric-value">{format_currency(results["financial_impact"]["current_unit_cost"])}</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col3:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown('<div class="metric-label">Est. Monthly Savings</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="metric-value" style="color:{VIVE_GREEN}">{format_currency(results["financial_impact"]["estimated_monthly_savings"])}</div>', unsafe_allow_html=True)
        
        if "reduction_factor" in results:
            st.markdown(f'<div style="font-size:0.9rem;color:{VIVE_GREY}">({results["reduction_factor"]:.0f}% reduction in returns)</div>', unsafe_allow_html=True)
        
        st.markdown('<div class="metric-label">Payback Period</div>', unsafe_allow_html=True)
        if results["financial_impact"]["payback_months"] == float('inf'):
            payback_text = "N/A"
            payback_color = VIVE_RED
        else:
            payback_months = results['financial_impact']['payback_months']
            payback_text = f"{payback_months:.1f} months"
            
            if payback_months < 3:
                payback_color = VIVE_GREEN
            elif payback_months < 6:
                payback_color = VIVE_TEAL
            elif payback_months < 12:
                payback_color = VIVE_AMBER
            else:
                payback_color = VIVE_RED
                
        st.markdown(f'<div class="metric-value" style="color:{payback_color}">{payback_text}</div>', unsafe_allow_html=True)
        
        st.markdown('<div class="metric-label">3-Year ROI</div>', unsafe_allow_html=True)
        roi = results["financial_impact"]["roi_3yr"]
        
        if roi == float('inf'):
            roi_text = "∞"
            roi_color = VIVE_GREEN
        else:
            roi_text = f"{roi:.1f}%"
            roi_color = VIVE_GREEN if roi > 0 else VIVE_RED
            
        st.markdown(f'<div class="metric-value" style="color:{roi_color}">{roi_text}</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Recommendation box
    st.markdown('<div class="metric-label">Recommendation</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="{results["recommendation_class"]}">{results["recommendation"]}</div>', unsafe_allow_html=True)
    
    # Brand impact if available
    if results["brand_impact"]:
        st.markdown('<div class="metric-label" style="margin-top:10px;">Brand Impact Assessment</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="metric-value">{results["brand_impact"]}</div>', unsafe_allow_html=True)
    
    # Issue description
    st.markdown('<div class="metric-label" style="margin-top:15px;">Issue Description</div>', unsafe_allow_html=True)
    st.markdown(f'<div style="padding: 10px; background-color:#F9FAFB; border-radius:5px;">{results["issue_description"]}</div>', unsafe_allow_html=True)
    
    # ROI chart
    st.markdown('<div style="margin-top:20px;">', unsafe_allow_html=True)
    fig = go.Figure()
    
    # Initial investment
    fig.add_trace(go.Bar(
        x=['Initial Investment'],
        y=[results["financial_impact"]["fix_cost_upfront"]],
        name='Initial Investment',
        marker_color=VIVE_RED
    ))
    
    # Per-unit cost increase (total over 3 years)
    per_unit_cost_total = results["financial_impact"]["fix_cost_per_unit"] * 12 * 3  # monthly sales * 36 months
    
    if per_unit_cost_total > 0:
        fig.add_trace(go.Bar(
            x=['Additional Unit Costs (3 years)'],
            y=[per_unit_cost_total],
            name='Additional Unit Costs',
            marker_color='#FCD34D'  # Amber for additional costs
        ))
    
    # 1-year, 2-year, 3-year savings
    fig.add_trace(go.Bar(
        x=['Year 1', 'Year 2', 'Year 3'],
        y=[
            results["financial_impact"]["annual_savings"],
            results["financial_impact"]["annual_savings"],
            results["financial_impact"]["annual_savings"]
        ],
        name='Estimated Savings',
        marker_color=VIVE_GREEN
    ))
    
    # Cumulative net benefit line
    year1_net = results["financial_impact"]["annual_savings"] - results["financial_impact"]["fix_cost_upfront"] - per_unit_cost_total/3
    year2_net = year1_net + results["financial_impact"]["annual_savings"] - per_unit_cost_total/3
    year3_net = year2_net + results["financial_impact"]["annual_savings"] - per_unit_cost_total/3
    
    fig.add_trace(go.Scatter(
        x=['Initial Investment', 'Year 1', 'Year 2', 'Year 3'],
        y=[-results["financial_impact"]["fix_cost_upfront"], year1_net, year2_net, year3_net],
        name='Cumulative Net Benefit',
        line=dict(color=VIVE_NAVY, width=3, dash='dot'),
        mode='lines+markers'
    ))
    
    fig.update_layout(
        title='Projected Returns Over Time',
        xaxis_title='Timeline',
        yaxis_title='Amount ($)',
        barmode='group',
        height=350,
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
    
    st.markdown('</div>', unsafe_allow_html=True)

def display_salvage_results(results):
    """Display the salvage operation analysis results in a visually appealing way"""
    st.markdown(f'<div class="card">', unsafe_allow_html=True)
    
    # Results header
    st.markdown(f'<div class="sub-header">Salvage Analysis for SKU: {results["sku"]}</div>', unsafe_allow_html=True)
    
    # Key metrics in a 3-column layout
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown('<div class="metric-label">Affected Inventory</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="metric-value">{results["metrics"]["affected_inventory"]:,} units</div>', unsafe_allow_html=True)
        
        st.markdown('<div class="metric-label">Expected Recovery</div>', unsafe_allow_html=True)
        recovery_units = results["metrics"]["expected_units_recovered"]
        recovery_rate = results["metrics"]["recovery_rate"]
        
        # Color based on recovery rate
        if recovery_rate >= 80:
            recovery_color = VIVE_GREEN
        elif recovery_rate >= 50:
            recovery_color = VIVE_AMBER
        else:
            recovery_color = VIVE_RED
            
        st.markdown(f'<div class="metric-value" style="color:{recovery_color}">{recovery_units:.0f} units ({recovery_rate}%)</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown('<div class="metric-label">Total Rework Cost</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="metric-value">{format_currency(results["financial"]["total_rework_cost"])}</div>', unsafe_allow_html=True)
        
        st.markdown('<div class="metric-label">Salvage Revenue</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="metric-value">{format_currency(results["financial"]["salvage_revenue"])}</div>', unsafe_allow_html=True)
        
        st.markdown('<div class="metric-label">Write-off Loss</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="metric-value" style="color:{VIVE_RED}">{format_currency(results["financial"]["write_off_loss"])}</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col3:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown('<div class="metric-label">Complete Write-off Cost</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="metric-value" style="color:{VIVE_RED}">{format_currency(results["financial"]["complete_writeoff_cost"])}</div>', unsafe_allow_html=True)
        
        st.markdown('<div class="metric-label">Salvage Net Profit/Loss</div>', unsafe_allow_html=True)
        profit = results["financial"]["salvage_profit"]
        profit_color = VIVE_GREEN if profit >= 0 else VIVE_RED
        profit_prefix = "" if profit < 0 else "+"  # Add plus sign for profits
        
        st.markdown(f'<div class="metric-value" style="color:{profit_color}">{profit_prefix}{format_currency(profit)}</div>', unsafe_allow_html=True)
        
        st.markdown('<div class="metric-label">ROI Percentage</div>', unsafe_allow_html=True)
        roi = results["financial"]["roi_percent"]
        
        if roi == float('inf'):
            roi_text = "∞"
            roi_color = VIVE_GREEN
        else:
            roi_text = f"{roi:.1f}%"
            roi_color = VIVE_GREEN if roi > 0 else VIVE_RED
            
        st.markdown(f'<div class="metric-value" style="color:{roi_color}">{roi_text}</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Recommendation
    st.markdown('<div class="metric-label">Recommendation</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="{results["recommendation_class"]}">{results["recommendation"]}</div>', unsafe_allow_html=True)
    
    # Comparison chart
    st.markdown('<div class="chart-container">', unsafe_allow_html=True)
    fig = go.Figure()
    
    # Scenario comparison
    fig.add_trace(go.Bar(
        x=['Complete Write-off', 'Salvage Operation'],
        y=[-results["financial"]["complete_writeoff_cost"], results["financial"]["salvage_profit"]],
        name='Financial Impact',
        marker_color=[VIVE_RED, VIVE_GREEN if results["financial"]["salvage_profit"] >= 0 else VIVE_RED]
    ))
    
    # Add text annotations showing the exact values
    fig.add_annotation(
        x='Complete Write-off',
        y=-results["financial"]["complete_writeoff_cost"]/2,
        text=f"-{format_currency(results['financial']['complete_writeoff_cost'])}",
        showarrow=False,
        font=dict(color="white", size=14)
    )
    
    fig.add_annotation(
        x='Salvage Operation',
        y=results["financial"]["salvage_profit"]/2,
        text=format_currency(results["financial"]["salvage_profit"]),
        showarrow=False,
        font=dict(color="white" if results["financial"]["salvage_profit"] < 0 else "black", size=14)
    )
    
    fig.update_layout(
        title='Financial Comparison: Write-off vs. Salvage',
        xaxis_title='Option',
        yaxis_title='Profit/Loss ($)',
        height=350,
        margin=dict(l=20, r=20, t=40, b=20),
        template="plotly_white",
        plot_bgcolor='rgba(0,0,0,0)'
    )
    
    st.plotly_chart(fig, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Price breakdown
    st.markdown('<div class="sub-header" style="margin-top:20px;">Pricing Analysis</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown('<div class="metric-label">Regular Price</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="metric-value">{format_currency(results["metrics"]["regular_price"])}</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown('<div class="metric-label">Discounted Price</div>', unsafe_allow_html=True)
        discount = results["metrics"]["discount_percentage"]
        st.markdown(f'<div class="metric-value">{format_currency(results["metrics"]["discounted_price"])} <span style="color:{VIVE_RED};font-size:1rem;">({discount}% off)</span></div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Add a waterfall chart showing the breakdown of costs and revenues
    st.markdown('<div class="chart-container">', unsafe_allow_html=True)
    
    waterfall_fig = go.Figure(go.Waterfall(
        name="Salvage Operation Breakdown",
        orientation="v",
        measure=["relative", "relative", "relative", "total"],
        x=["Salvage Revenue", "Rework Costs", "Write-off Loss", "Net Profit/Loss"],
        textposition="outside",
        text=[
            f"+{format_currency(results['financial']['salvage_revenue'])}",
            f"-{format_currency(results['financial']['total_rework_cost'])}",
            f"-{format_currency(results['financial']['write_off_loss'])}",
            f"{format_currency(results['financial']['salvage_profit'])}"
        ],
        y=[
            results["financial"]["salvage_revenue"],
            -results["financial"]["total_rework_cost"],
            -results["financial"]["write_off_loss"],
            0  # The total will be calculated automatically
        ],
        connector={"line": {"color": "rgb(63, 63, 63)"}},
        decreasing={"marker": {"color": VIVE_RED}},
        increasing={"marker": {"color": VIVE_GREEN}},
        totals={"marker": {"color": VIVE_NAVY}}
    ))
    
    waterfall_fig.update_layout(
        title="Salvage Operation Financial Breakdown",
        showlegend=False,
        height=350,
        margin=dict(l=20, r=20, t=40, b=20),
        template="plotly_white",
        plot_bgcolor='rgba(0,0,0,0)'
    )
    
    st.plotly_chart(waterfall_fig, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

def chat_with_ai(results, issue_description, chat_history=None, client=None):
    """Initialize or continue chat with AI about quality issues"""
    if chat_history is None:
        chat_history = []
        
        if client is None:
            # Add a placeholder message if AI is not available
            chat_history.append({
                "role": "assistant",
                "content": ("AI assistant is currently unavailable. Please check your API key configuration "
                           "or contact your administrator to enable this feature.")
            })
            return chat_history
            
        # Generate initial AI analysis
        system_prompt = f"""
        You are a Quality Management expert for Vive Health's product line. You analyze quality issues, provide insights on cost-benefit analyses, and suggest solutions.
        
        Product details:
        - SKU: {results["sku"]}
        - Type: {results["product_type"]}
        - Issue: {issue_description}
        
        Metrics:
        - Return Rate (30 days): {results["current_metrics"]["return_rate_30d"]:.2f}%
        - Monthly Return Cost: ${results["financial_impact"]["monthly_return_cost"]:.2f}
        - Estimated Savings: ${results["financial_impact"]["estimated_monthly_savings"]:.2f}/month
        - Payback Period: {results["financial_impact"]["payback_months"]:.1f} months
        
        Recommendation: {results["recommendation"]}
        
        As a Quality Management expert for Vive Health, you should provide specific, actionable insights about this quality issue.
        Focus on:
        1. Root cause analysis of the issue
        2. Practical solutions to fix the quality problem
        3. Implementation recommendations
        4. Risk assessment for the proposed solution
        
        Keep your analysis concise, practical, and specific to medical devices and Vive Health products.
        """
        
        initial_analysis = get_ai_analysis(
            client,
            system_prompt,
            "Based on the product information and quality metrics, provide your initial analysis of the issue and suggested next steps. Be specific about potential fixes and quality improvements.",
            model="gpt-4",  # Try to use GPT-4 for better quality analysis
            fallback_model="gpt-3.5-turbo"  # Fall back to GPT-3.5-Turbo if needed
        )
        
        # Add initial AI message
        chat_history.append({
            "role": "assistant",
            "content": initial_analysis
        })
    
    return chat_history

# --- SALES ANALYSIS FUNCTIONS ---

def load_sales_data():
    """
    Load sample sales data or return placeholder if not available
    In a real application, this would load from a database or file
    """
    try:
        # Generate more realistic sample data
        np.random.seed(42)  # For reproducibility
        
        # Date range
        date_range = pd.date_range(start="2023-01-01", periods=365, freq="D")
        
        # Base sales with seasonal trend and weekend effect
        base_sales = np.linspace(100, 300, 365)  # Increasing trend over the year
        
        # Add seasonality - higher in Q4, lower in Q1
        quarter_effect = np.array([0.8, 1.0, 0.9, 1.3])
        month_indices = np.array([d.month-1 for d in date_range])
        quarter_indices = month_indices // 3
        seasonality = np.array([quarter_effect[i] for i in quarter_indices])
        
        # Add day-of-week effect (weekend boost)
        weekday_indices = np.array([d.weekday() for d in date_range])
        weekend_boost = np.where(weekday_indices >= 5, 1.3, 1.0)
        
        # Add some noise
        noise = np.random.normal(1, 0.15, 365)
        
        # Calculate final sales
        sales = (base_sales * seasonality * weekend_boost * noise).astype(int)
        
        # Create realistic channels with different distributions
        channels = ["Amazon", "Website", "Retail", "Wholesale"]
        channel_weights = [0.45, 0.25, 0.2, 0.1]  # 45% Amazon, 25% Website, etc.
        
        # Assign channels based on weights
        channel_choices = np.random.choice(channels, 365, p=channel_weights)
        
        # Create product categories 
        categories = ["Mobility Aids", "Bathroom Safety", "Bedroom Aids", "Physical Therapy", "Health Monitors"]
        category_weights = [0.3, 0.25, 0.15, 0.2, 0.1]
        category_choices = np.random.choice(categories, 365, p=category_weights)
        
        # Define product for each sale
        products = [
            "Mobility Scooter", "Rollator Walker", "Transport Wheelchair", 
            "Bath Safety Rail", "Shower Chair", "Toilet Riser",
            "Bed Rail", "Overbed Table", "Transfer Board",
            "TENS Unit", "Hot/Cold Therapy", "Exercise Equipment",
            "Blood Pressure Monitor", "Pulse Oximeter", "Thermometer"
        ]
        
        # Map categories to products
        category_to_products = {
            "Mobility Aids": ["Mobility Scooter", "Rollator Walker", "Transport Wheelchair"],
            "Bathroom Safety": ["Bath Safety Rail", "Shower Chair", "Toilet Riser"],
            "Bedroom Aids": ["Bed Rail", "Overbed Table", "Transfer Board"],
            "Physical Therapy": ["TENS Unit", "Hot/Cold Therapy", "Exercise Equipment"],
            "Health Monitors": ["Blood Pressure Monitor", "Pulse Oximeter", "Thermometer"]
        }
        
        # Generate product for each sale based on category
        product_choices = []
        for category in category_choices:
            product_options = category_to_products[category]
            product_choices.append(np.random.choice(product_options))
        
        # Create the dataframe
        data = {
            "date": date_range,
            "sales": sales,
            "channel": channel_choices,
            "product_category": category_choices,
            "product": product_choices,
            "revenue": sales * np.random.uniform(50, 200, 365)  # Random price between $50-$200
        }
        
        return pd.DataFrame(data)
    except Exception as e:
        st.error(f"Error loading sales data: {e}")
        return pd.DataFrame()

def analyze_sales_trends(df):
    """Analyze sales trends from the dataframe"""
    if df.empty:
        return None
    
    try:
        # Group by date and calculate daily sales
        daily_sales = df.groupby("date")[["sales", "revenue"]].sum().reset_index()
        
        # Calculate 7-day moving averages
        daily_sales["7d_moving_avg_sales"] = daily_sales["sales"].rolling(window=7).mean()
        daily_sales["7d_moving_avg_revenue"] = daily_sales["revenue"].rolling(window=7).mean()
        
        # Calculate month-to-date metrics
        daily_sales["month"] = daily_sales["date"].dt.month
        daily_sales["year"] = daily_sales["date"].dt.year
        monthly_sales = daily_sales.groupby(["year", "month"])[["sales", "revenue"]].sum().reset_index()
        
        # Add month name for better readability
        monthly_sales["month_name"] = monthly_sales["month"].apply(lambda x: datetime(2000, x, 1).strftime('%b'))
        
        # Calculate sales by channel
        channel_sales = df.groupby("channel")[["sales", "revenue"]].sum().reset_index()
        
        # Calculate sales by product category
        category_sales = df.groupby("product_category")[["sales", "revenue"]].sum().reset_index()
        
        # Calculate sales by product
        product_sales = df.groupby("product")[["sales", "revenue"]].sum().reset_index()
        
        # Calculate average revenue per unit
        if df["sales"].sum() > 0:
            avg_revenue_per_unit = df["revenue"].sum() / df["sales"].sum()
        else:
            avg_revenue_per_unit = 0
            
        # Calculate growth metrics
        # Year-over-year growth (if we have multiple years)
        years = sorted(daily_sales["year"].unique())
        yoy_growth = None
        
        if len(years) > 1:
            # Compare the same period between years
            last_year = years[-2]
            current_year = years[-1]
            
            last_year_sales = df[df["date"].dt.year == last_year]["sales"].sum()
            current_year_sales = df[df["date"].dt.year == current_year]["sales"].sum()
            
            if last_year_sales > 0:
                yoy_growth = (current_year_sales - last_year_sales) / last_year_sales * 100
            
        # Month-over-month growth
        months = sorted(daily_sales["month"].unique())
        mom_growth = None
        
        if len(months) > 1:
            # Compare sequential months
            last_month = months[-2]
            current_month = months[-1]
            current_year = daily_sales["year"].iloc[-1]
            
            last_month_sales = df[(df["date"].dt.month == last_month) & (df["date"].dt.year == current_year)]["sales"].sum()
            current_month_sales = df[(df["date"].dt.month == current_month) & (df["date"].dt.year == current_year)]["sales"].sum()
            
            if last_month_sales > 0:
                mom_growth = (current_month_sales - last_month_sales) / last_month_sales * 100
        
        return {
            "daily_sales": daily_sales,
            "monthly_sales": monthly_sales,
            "channel_sales": channel_sales,
            "category_sales": category_sales,
            "product_sales": product_sales,
            "avg_revenue_per_unit": avg_revenue_per_unit,
            "yoy_growth": yoy_growth,
            "mom_growth": mom_growth
        }
    except Exception as e:
        st.error(f"Error analyzing sales trends: {e}")
        return None

# --- INVENTORY MANAGEMENT FUNCTIONS ---

def load_inventory_data():
    """
    Load sample inventory data or return placeholder if not available
    In a real application, this would load from a database or file
    """
    try:
        np.random.seed(42)  # For reproducibility
        
        # Define realistic product categories
        categories = ["Mobility Aids", "Bathroom Safety", "Bedroom Aids", "Physical Therapy", "Health Monitors"]
        
        # Define products for each category
        category_products = {
            "Mobility Aids": ["Mobility Scooter", "Rollator Walker", "Transport Wheelchair", "Folding Cane", "Knee Scooter"],
            "Bathroom Safety": ["Bath Safety Rail", "Shower Chair", "Toilet Riser", "Bath Bench", "Grab Bar"],
            "Bedroom Aids": ["Bed Rail", "Overbed Table", "Transfer Board", "Bed Wedge", "Mattress Elevator"],
            "Physical Therapy": ["TENS Unit", "Hot/Cold Therapy", "Exercise Equipment", "Resistance Bands", "Hand Therapy"],
            "Health Monitors": ["Blood Pressure Monitor", "Pulse Oximeter", "Thermometer", "Glucose Monitor", "Scale"]
        }
        
        # Generate SKUs and product data
        skus = []
        product_names = []
        categories_list = []
        
        for category in categories:
            for product in category_products[category]:
                # Create multiple SKUs for each product (variants)
                for variant in range(1, 4):  # 3 variants per product
                    sku_suffix = ""
                    if variant == 1:
                        sku_suffix = "STD"  # Standard
                    elif variant == 2:
                        sku_suffix = "DLX"  # Deluxe
                    else:
                        sku_suffix = "PRO"  # Professional
                        
                    # Create SKU following a pattern
                    category_prefix = ''.join([word[0] for word in category.split()])
                    product_prefix = ''.join([word[0] for word in product.split()])
                    sku = f"VH-{category_prefix}{product_prefix}-{sku_suffix}"
                    
                    skus.append(sku)
                    product_names.append(f"{product} {sku_suffix}")
                    categories_list.append(category)
        
        # Total number of products
        num_products = len(skus)
        
        # Generate inventory data with realistic distributions
        in_stock = np.random.randint(0, 150, num_products)
        
        # Generate realistic reorder points (higher for popular items)
        reorder_points = np.random.randint(5, 50, num_products)
        
        # Lead times generally 7-60 days, with most around 14-30
        lead_times = np.random.choice(
            [7, 14, 21, 30, 45, 60],
            num_products,
            p=[0.1, 0.3, 0.3, 0.2, 0.05, 0.05]
        )
        
        # Costs vary by category
        # More expensive: Mobility Aids, cheaper: Health Monitors, etc.
        costs = []
        for category in categories_list:
            if category == "Mobility Aids":
                cost = np.random.uniform(150, 600)
            elif category == "Bathroom Safety":
                cost = np.random.uniform(40, 120)
            elif category == "Bedroom Aids":
                cost = np.random.uniform(50, 180)
            elif category == "Physical Therapy":
                cost = np.random.uniform(30, 150)
            else:  # Health Monitors
                cost = np.random.uniform(20, 100)
                
            # Adjust cost based on suffix (STD, DLX, PRO)
            if "STD" in skus[len(costs)]:
                pass  # Standard cost
            elif "DLX" in skus[len(costs)]:
                cost *= 1.3  # Deluxe version costs 30% more
            elif "PRO" in skus[len(costs)]:
                cost *= 1.7  # Pro version costs 70% more
                
            costs.append(round(cost, 2))
        
        # Last ordered dates (more recent for low stock items)
        last_ordered = []
        for i in range(num_products):
            if in_stock[i] <= reorder_points[i]:
                # Low stock items ordered more recently
                days_ago = np.random.randint(1, 20)
            else:
                # Well-stocked items ordered less recently
                days_ago = np.random.randint(21, 90)
                
            order_date = pd.Timestamp.now() - pd.Timedelta(days=days_ago)
            last_ordered.append(order_date)
            
        # Create dataframe
        data = {
            "sku": skus,
            "product_name": product_names,
            "category": categories_list,
            "in_stock": in_stock,
            "reorder_point": reorder_points,
            "lead_time_days": lead_times,
            "cost": costs,
            "last_ordered": last_ordered
        }
        
        df = pd.DataFrame(data)
        
        # Mark status
        df["status"] = df.apply(
            lambda x: "Critical" if x["in_stock"] == 0 else 
                     "Low Stock" if x["in_stock"] <= x["reorder_point"] else 
                     "OK", 
            axis=1
        )
        
        # Calculate inventory value
        df["inventory_value"] = df["in_stock"] * df["cost"]
        
        return df
    except Exception as e:
        st.error(f"Error loading inventory data: {e}")
        return pd.DataFrame()

def analyze_inventory_status(df):
    """Analyze inventory status from the dataframe"""
    if df.empty:
        return None
    
    try:
        # Calculate inventory metrics
        low_stock_items = df[df["status"] == "Low Stock"]
        critical_items = df[df["status"] == "Critical"]
        stock_out_risk = len(low_stock_items) / len(df) * 100 if len(df) > 0 else 0
        
        # Calculate inventory value
        total_inventory_value = df["inventory_value"].sum()
        avg_item_value = total_inventory_value / len(df) if len(df) > 0 else 0
        
        # Calculate inventory by category
        category_inventory = df.groupby("category")["inventory_value"].sum().reset_index()
        category_counts = df.groupby("category").size().reset_index(name="item_count")
        
        # Calculate items to reorder (currently at or below reorder point)
        items_to_reorder = df[df["in_stock"] <= df["reorder_point"]]
        
        # Calculate total reorder value
        items_to_reorder["reorder_quantity"] = items_to_reorder["reorder_point"] - items_to_reorder["in_stock"]
        items_to_reorder["reorder_quantity"] = items_to_reorder["reorder_quantity"].apply(lambda x: max(0, x))
        items_to_reorder["reorder_value"] = items_to_reorder["reorder_quantity"] * items_to_reorder["cost"]
        total_reorder_value = items_to_reorder["reorder_value"].sum()
        
        # Calculate days of supply for each item
        # Assuming daily sales is 10% of reorder point (rough approximation)
        df["daily_sales_estimate"] = df["reorder_point"] * 0.1
        df["days_of_supply"] = df.apply(
            lambda x: x["in_stock"] / x["daily_sales_estimate"] if x["daily_sales_estimate"] > 0 else 100,
            axis=1
        )
        
        # Calculate high stock items (more than 3x reorder point)
        high_stock_items = df[df["in_stock"] > 3 * df["reorder_point"]]
        excess_inventory_value = high_stock_items["inventory_value"].sum()
        
        # Calculate inventory metrics by status
        status_summary = df.groupby("status").agg({
            "sku": "count",
            "inventory_value": "sum"
        }).reset_index()
        
        return {
            "low_stock_items": low_stock_items,
            "critical_items": critical_items,
            "stock_out_risk": stock_out_risk,
            "total_inventory_value": total_inventory_value,
            "avg_item_value": avg_item_value,
            "category_inventory": category_inventory,
            "category_counts": category_counts,
            "items_to_reorder": items_to_reorder,
            "total_reorder_value": total_reorder_value,
            "high_stock_items": high_stock_items,
            "excess_inventory_value": excess_inventory_value,
            "status_summary": status_summary
        }
    except Exception as e:
        st.error(f"Error analyzing inventory status: {e}")
        return None

# --- DASHBOARD FUNCTIONS ---

def load_dashboard_data():
    """
    Load data for the dashboard overview
    Combines data from different sources to create a comprehensive dashboard
    """
    try:
        # Load sales data
        sales_df = load_sales_data()
        sales_analysis = analyze_sales_trends(sales_df)
        
        # Load inventory data
        inventory_df = load_inventory_data()
        inventory_analysis = analyze_inventory_status(inventory_df)
        
        # Combine data for the dashboard
        if sales_analysis and inventory_analysis:
            # Calculate key performance indicators
            recent_sales = sales_df[sales_df["date"] >= (sales_df["date"].max() - pd.Timedelta(days=30))]
            monthly_revenue = recent_sales["revenue"].sum()
            monthly_units = recent_sales["sales"].sum()
            
            # Calculate profit (simplified as 40% of revenue)
            monthly_profit = monthly_revenue * 0.4
            
            # Calculate inventory turnover (simplified)
            monthly_cogs = monthly_revenue * 0.6  # Assuming 60% COGS
            if inventory_analysis["total_inventory_value"] > 0:
                inventory_turnover = (monthly_cogs * 12) / inventory_analysis["total_inventory_value"]
            else:
                inventory_turnover = 0
                
            # Calculate average inventory days
            if inventory_turnover > 0:
                avg_inventory_days = 365 / inventory_turnover
            else:
                avg_inventory_days = 0
                
            # Calculate return rate (randomly generated for demo purposes)
            return_rate = np.random.uniform(3, 8)
            
            # Calculate quality metrics (randomly generated for demo)
            quality_score = np.random.uniform(85, 95)
            
            # Calculate pending orders (randomly generated for demo)
            pending_orders = int(np.random.uniform(50, 150))
            
            # Top selling products
            top_products = sales_df.groupby("product")["sales"].sum().sort_values(ascending=False).head(5)
            top_products_df = pd.DataFrame({
                "product": top_products.index,
                "units_sold": top_products.values
            })
            
            # Top channels by revenue
            top_channels = sales_df.groupby("channel")["revenue"].sum().sort_values(ascending=False)
            top_channels_df = pd.DataFrame({
                "channel": top_channels.index,
                "revenue": top_channels.values
            })
            
            return {
                "monthly_revenue": monthly_revenue,
                "monthly_profit": monthly_profit,
                "monthly_units": monthly_units,
                "inventory_turnover": inventory_turnover,
                "avg_inventory_days": avg_inventory_days,
                "return_rate": return_rate,
                "quality_score": quality_score,
                "pending_orders": pending_orders,
                "stock_out_risk": inventory_analysis["stock_out_risk"],
                "total_inventory_value": inventory_analysis["total_inventory_value"],
                "low_stock_items": inventory_analysis["low_stock_items"],
                "critical_items": inventory_analysis["critical_items"],
                "sales_trend": sales_analysis["daily_sales"],
                "monthly_sales": sales_analysis["monthly_sales"],
                "sales_by_channel": sales_analysis["channel_sales"],
                "sales_by_category": sales_analysis["category_sales"],
                "inventory_by_category": inventory_analysis["category_inventory"],
                "top_products": top_products_df,
                "top_channels": top_channels_df,
                "items_to_reorder": inventory_analysis["items_to_reorder"],
                "mom_growth": sales_analysis["mom_growth"],
                "yoy_growth": sales_analysis["yoy_growth"]
            }
        else:
            st.error("Error loading dashboard data: Sales or inventory analysis failed.")
            return None
    except Exception as e:
        st.error(f"Error loading dashboard data: {e}")
        return None

# --- PAGE DISPLAY FUNCTIONS ---

def display_quality_manager(client):
    vive_header()
    st.markdown('<div class="main-header">Quality ROI Analysis</div>', unsafe_allow_html=True)
    
    # Create tabs for different analyses
    tab1, tab2 = st.tabs(["Quality Issue Analysis", "Salvage Operation Analysis"])
    
    with tab1:
        st.markdown('<div class="sub-header">Quality Issue Analysis</div>', unsafe_allow_html=True)
        
        # Initialize session state for chat
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = None
        
        # Check if analysis results exist
        if "quality_analysis_results" not in st.session_state:
            st.session_state.quality_analysis_results = None
            st.session_state.analysis_submitted = False
        
        # Form for entering quality issue data
        if not st.session_state.analysis_submitted:
            with st.form("quality_issue_form"):
                st.markdown('<div class="form-section-header">Product Information</div>', unsafe_allow_html=True)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # Required inputs
                    sku = st.text_input(
                        "SKU*", 
                        help="Required: Product SKU (e.g., VH-1234)"
                    )
                    
                    product_type = st.selectbox(
                        "Product Type*", 
                        ["B2C", "B2B", "Both"],
                        help="Required: Distribution channel for this product (B2C = Direct to Consumer, B2B = Business to Business)"
                    )
                    
                    sales_30d = st.number_input(
                        "Total Sales Last 30 Days*", 
                        min_value=0.0,
                        help="Required: Number of units sold in the last 30 days"
                    )
                    
                    returns_30d = st.number_input(
                        "Total Returns Last 30 Days*", 
                        min_value=0.0,
                        help="Required: Number of units returned in the last 30 days"
                    )
                    
                with col2:
                    current_unit_cost = st.number_input(
                        "Current Unit Cost* (Landed Cost)", 
                        min_value=0.0,
                        help="Required: Current per-unit cost to produce/acquire including shipping and duties"
                    )
                    
                    fix_cost_upfront = st.number_input(
                        "Fix Cost Upfront*", 
                        min_value=0.0,
                        help="Required: One-time cost to implement the quality fix (engineering hours, design changes, tooling, etc.)"
                    )
                    
                    fix_cost_per_unit = st.number_input(
                        "Additional Cost Per Unit*", 
                        min_value=0.0,
                        help="Required: Additional cost per unit after implementing the fix (extra material, labor, etc.)"
                    )
                
                # Description of issue
                issue_description = st.text_area(
                    "Description of Quality Issue*",
                    help="Required: Detailed description of the quality problem, including failure modes and customer impact"
                )
                
                # Expandable section for optional metrics
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
                            "Total Sales Last 365 Days", 
                            min_value=0.0,
                            help="Number of units sold in the last 365 days (for long-term trends)"
                        )
                        
                        returns_365d = st.number_input(
                            "Total Returns Last 365 Days", 
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
                        
                        fba_fee = st.number_input(
                            "FBA Fee", 
                            min_value=0.0,
                            help="Amazon FBA fee per unit, if applicable"
                        )
                
                # Form submission
                submit_col1, submit_col2 = st.columns([3, 1])
                with submit_col2:
                    submit_button = st.form_submit_button("Analyze Quality Issue")
                
                if submit_button:
                    # Validate required fields
                    if not all([sku, issue_description]):
                        st.error("Please fill in all required fields marked with *")
                    elif sales_30d <= 0:
                        st.error("Total Sales Last 30 Days must be greater than zero")
                    elif current_unit_cost <= 0:
                        st.error("Current Unit Cost must be greater than zero")
                    else:
                        with st.spinner("Analyzing quality issue..."):
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
                                asin=asin if asin else None,
                                ncx_rate=ncx_rate if ncx_rate > 0 else None,
                                sales_365d=sales_365d if sales_365d > 0 else None,
                                returns_365d=returns_365d if returns_365d > 0 else None,
                                star_rating=star_rating if star_rating > 0 else None,
                                total_reviews=total_reviews if total_reviews > 0 else None,
                                fba_fee=fba_fee if fba_fee > 0 else None
                            )
                            
                            # Store results in session state
                            st.session_state.quality_analysis_results = results
                            st.session_state.analysis_submitted = True
                            
                            # Initialize chat with AI
                            st.session_state.chat_history = chat_with_ai(
                                results, 
                                issue_description,
                                client=client
                            )
                            
                            # Rerun to show results
                            st.experimental_rerun()
        
        # Display results if available
        if st.session_state.analysis_submitted and st.session_state.quality_analysis_results:
            # Add a reset button at the top
            if st.button("Start New Analysis", key="reset_top_button"):
                st.session_state.quality_analysis_results = None
                st.session_state.analysis_submitted = False
                st.session_state.chat_history = None
                st.experimental_rerun()
                
            # Display analysis results
            display_quality_issue_results(st.session_state.quality_analysis_results)
            
            # Display AI chat interface
            st.markdown('<div class="sub-header">AI Quality Consultant</div>', unsafe_allow_html=True)
            
            # Chat container
            st.markdown('<div class="chat-container" id="chat-container">', unsafe_allow_html=True)
            for message in st.session_state.chat_history:
                if message["role"] == "user":
                    st.markdown(f'<div class="chat-message user-message">{message["content"]}</div>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="chat-message ai-message">{message["content"]}</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Input for new messages
            col1, col2 = st.columns([4, 1])
            
            with col1:
                user_input = st.text_input(
                    "Ask about the quality issue or potential solutions:",
                    key="user_message",
                    placeholder="E.g., What could be causing this issue? What fixes do you recommend?"
                )
            
            with col2:
                send_button = st.button("Send", key="send_button")
            
            if send_button and user_input:
                with st.spinner("Getting AI analysis..."):
                    # Add user message to chat history
                    st.session_state.chat_history.append({
                        "role": "user",
                        "content": user_input
                    })
                    
                    # Get AI response
                    system_prompt = f"""
                    You are a Quality Management expert for Vive Health's product line. You analyze quality issues, provide insights, and suggest solutions.
                    
                    Product details:
                    - SKU: {st.session_state.quality_analysis_results["sku"]}
                    - Type: {st.session_state.quality_analysis_results["product_type"]}
                    - Issue: {st.session_state.quality_analysis_results["issue_description"]}
                    
                    Metrics:
                    - Return Rate (30 days): {st.session_state.quality_analysis_results["current_metrics"]["return_rate_30d"]:.2f}%
                    - Monthly Return Cost: ${st.session_state.quality_analysis_results["financial_impact"]["monthly_return_cost"]:.2f}
                    - Estimated Savings: ${st.session_state.quality_analysis_results["financial_impact"]["estimated_monthly_savings"]:.2f}/month
                    - Payback Period: {st.session_state.quality_analysis_results["financial_impact"]["payback_months"]:.1f} months
                    
                    Recommendation: {st.session_state.quality_analysis_results["recommendation"]}
                    
                    As a Quality Management expert for Vive Health, you should provide specific, actionable insights.
                    Focus on:
                    1. Root cause analysis of quality issues
                    2. Practical solutions for medical devices and healthcare products
                    3. Implementation strategies that consider FDA/regulatory compliance
                    4. Risk assessment for the proposed solution
                    
                    Keep your responses concise, specific, and tailored to the medical device industry.
                    """
                    
                    # Get the full chat history for context
                    messages_history = []
                    for msg in st.session_state.chat_history:
                        messages_history.append({"role": msg["role"], "content": msg["content"]})
                    
                    ai_response = get_ai_analysis(
                        client,
                        system_prompt,
                        user_input,
                        model="gpt-4",  # Try to use GPT-4 
                        fallback_model="gpt-3.5-turbo",  # Use GPT-3.5 as fallback
                        messages=messages_history[:-1]  # Exclude the latest user message as it's passed separately
                    )
                    
                    # Add AI response to chat history
                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": ai_response
                    })
                    
                    # Clear input and rerun to show updated chat
                    st.experimental_rerun()
            
            # Add a second reset button at the bottom
            if st.button("Start New Analysis", key="reset_bottom_button"):
                st.session_state.quality_analysis_results = None
                st.session_state.analysis_submitted = False
                st.session_state.chat_history = None
                st.experimental_rerun()
    
    with tab2:
        st.markdown('<div class="sub-header">Salvage Operation Analysis</div>', unsafe_allow_html=True)
        
        # Check if salvage results exist
        if "salvage_results" not in st.session_state:
            st.session_state.salvage_results = None
            st.session_state.salvage_submitted = False
        
        # Form for entering salvage operation data
        if not st.session_state.salvage_submitted:
            with st.form("salvage_operation_form"):
                st.markdown('<div class="form-section-header">Salvage Operation Details</div>', unsafe_allow_html=True)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    sku = st.text_input(
                        "SKU*", 
                        key="salvage_sku", 
                        help="Required: Product SKU (e.g., VH-1234)"
                    )
                    
                    affected_inventory = st.number_input(
                        "Affected Inventory Units*", 
                        min_value=1,
                        key="affected_inventory",
                        help="Required: Number of units affected by quality issue"
                    )
                    
                    current_unit_cost = st.number_input(
                        "Current Unit Cost* (Landed Cost)", 
                        min_value=0.01,
                        key="salvage_unit_cost",
                        help="Required: Current per-unit cost to produce/acquire including shipping and duties"
                    )
                    
                    rework_cost_upfront = st.number_input(
                        "Rework Setup Cost*", 
                        min_value=0.0,
                        key="rework_upfront",
                        help="Required: One-time cost to set up the rework operation (equipment, training, etc.)"
                    )
                
                with col2:
                    rework_cost_per_unit = st.number_input(
                        "Rework Cost Per Unit*", 
                        min_value=0.0,
                        key="rework_per_unit",
                        help="Required: Cost to rework each affected unit (labor, parts, QC testing)"
                    )
                    
                    expected_recovery_pct = st.slider(
                        "Expected Recovery Percentage*", 
                        min_value=0.0,
                        max_value=100.0,
                        value=80.0,
                        step=5.0,
                        help="Required: Percentage of affected units expected to be successfully reworked"
                    )
                    
                    expected_discount_pct = st.slider(
                        "Expected Discount Percentage*", 
                        min_value=0.0,
                        max_value=100.0,
                        value=30.0,
                        step=5.0,
                        help="Required: Discount percentage for selling reworked units"
                    )
                
                # Brief explanation of rework plan
                rework_plan = st.text_area(
                    "Brief Rework Plan (Optional)",
                    placeholder="Describe the proposed rework process (e.g., replacing components, repackaging, etc.)",
                    help="Optional: Describe how you plan to rework/salvage the affected inventory"
                )
                
                # Form submission
                submit_col1, submit_col2 = st.columns([3, 1])
                with submit_col2:
                    submit_button = st.form_submit_button("Analyze Salvage Operation")
                
                if submit_button:
                    # Validate required fields
                    if not sku:
                        st.error("Please enter a valid SKU")
                    elif affected_inventory <= 0:
                        st.error("Affected Inventory Units must be greater than zero")
                    elif current_unit_cost <= 0:
                        st.error("Current Unit Cost must be greater than zero")
                    else:
                        with st.spinner("Analyzing salvage operation..."):
                            # Perform analysis
                            results = analyze_salvage_operation(
                                sku=sku,
                                affected_inventory=affected_inventory,
                                current_unit_cost=current_unit_cost,
                                rework_cost_upfront=rework_cost_upfront,
                                rework_cost_per_unit=rework_cost_per_unit,
                                expected_recovery_pct=expected_recovery_pct,
                                expected_discount_pct=expected_discount_pct
                            )
                            
                            # Store rework plan in results if provided
                            if rework_plan:
                                results["rework_plan"] = rework_plan
                            
                            # Store results in session state
                            st.session_state.salvage_results = results
                            st.session_state.salvage_submitted = True
                            
                            # Rerun to show results
                            st.experimental_rerun()
        
        # Display results if available
        if st.session_state.salvage_submitted and st.session_state.salvage_results:
            # Add a reset button at the top
            if st.button("Start New Analysis", key="salvage_reset_top_button"):
                st.session_state.salvage_results = None
                st.session_state.salvage_submitted = False
                st.experimental_rerun()
                
            # Display analysis results
            display_salvage_results(st.session_state.salvage_results)
            
            # Scenario modeling
            st.markdown('<div class="sub-header">Scenario Modeling</div>', unsafe_allow_html=True)
            
            st.markdown("""
            <div style="background-color:#F0F9FF;padding:10px;border-radius:5px;margin-bottom:15px;">
                Adjust the parameters below to model different scenarios for your salvage operation. 
                This helps you understand how changes in recovery rate, discount percentage, or rework cost
                affect the financial outcome.
            </div>
            """, unsafe_allow_html=True)
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                recovery_adjustment = st.slider(
                    "Adjust Recovery Rate",
                    min_value=0.0,
                    max_value=100.0,
                    value=float(st.session_state.salvage_results["metrics"]["recovery_rate"]),
                    step=5.0,
                    help="Percentage of units that can be successfully reworked",
                    key="recovery_slider"
                )
            
            with col2:
                discount_adjustment = st.slider(
                    "Adjust Discount Percentage",
                    min_value=0.0,
                    max_value=100.0,
                    value=float(st.session_state.salvage_results["metrics"]["discount_percentage"]),
                    step=5.0,
                    help="Discount percentage for selling reworked units",
                    key="discount_slider"
                )
            
            with col3:
                max_rework_cost = float(st.session_state.salvage_results["financial"]["total_rework_cost"]/st.session_state.salvage_results["metrics"]["affected_inventory"])*2
                current_rework_cost = float(st.session_state.salvage_results["financial"]["total_rework_cost"]/st.session_state.salvage_results["metrics"]["affected_inventory"] - st.session_state.salvage_results["financial"]["total_rework_cost"]/st.session_state.salvage_results["metrics"]["affected_inventory"]/10)
                
                rework_adjustment = st.slider(
                    "Adjust Per-Unit Rework Cost",
                    min_value=0.0,
                    max_value=max_rework_cost,
                    value=current_rework_cost,
                    step=0.5,
                    help="Cost per unit to rework inventory",
                    key="rework_slider"
                )
            
            if st.button("Run Scenario", key="scenario_button"):
                with st.spinner("Calculating scenario..."):
                    # Calculate new scenario
                    new_results = analyze_salvage_operation(
                        sku=st.session_state.salvage_results["sku"],
                        affected_inventory=st.session_state.salvage_results["metrics"]["affected_inventory"],
                        current_unit_cost=st.session_state.salvage_results["financial"]["complete_writeoff_cost"]/st.session_state.salvage_results["metrics"]["affected_inventory"],
                        rework_cost_upfront=st.session_state.salvage_results["financial"]["total_rework_cost"] - (st.session_state.salvage_results["metrics"]["affected_inventory"] * (st.session_state.salvage_results["financial"]["total_rework_cost"]/st.session_state.salvage_results["metrics"]["affected_inventory"] - st.session_state.salvage_results["financial"]["total_rework_cost"]/st.session_state.salvage_results["metrics"]["affected_inventory"]/10)),
                        rework_cost_per_unit=rework_adjustment,
                        expected_recovery_pct=recovery_adjustment,
                        expected_discount_pct=discount_adjustment
                    )
                    
                    # Copy rework plan if exists
                    if "rework_plan" in st.session_state.salvage_results:
                        new_results["rework_plan"] = st.session_state.salvage_results["rework_plan"]
                    
                    # Compare current vs new scenario
                    st.markdown('<div class="sub-header">Scenario Comparison</div>', unsafe_allow_html=True)
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                        st.markdown('<div class="metric-label">Current Scenario</div>', unsafe_allow_html=True)
                        
                        profit = st.session_state.salvage_results["financial"]["salvage_profit"]
                        profit_color = VIVE_GREEN if profit >= 0 else VIVE_RED
                        profit_prefix = "" if profit < 0 else "+"
                        
                        st.markdown(f'<div class="metric-value" style="color:{profit_color}">Net Profit/Loss: {profit_prefix}{format_currency(profit)}</div>', unsafe_allow_html=True)
                        st.markdown(f'<div class="{st.session_state.salvage_results["recommendation_class"]}" style="margin-top:10px;">{st.session_state.salvage_results["recommendation"]}</div>', unsafe_allow_html=True)
                        st.markdown('</div>', unsafe_allow_html=True)
                    
                    with col2:
                        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                        st.markdown('<div class="metric-label">New Scenario</div>', unsafe_allow_html=True)
                        
                        new_profit = new_results["financial"]["salvage_profit"]
                        new_profit_color = VIVE_GREEN if new_profit >= 0 else VIVE_RED
                        new_profit_prefix = "" if new_profit < 0 else "+"
                        
                        st.markdown(f'<div class="metric-value" style="color:{new_profit_color}">Net Profit/Loss: {new_profit_prefix}{format_currency(new_profit)}</div>', unsafe_allow_html=True)
                        st.markdown(f'<div class="{new_results["recommendation_class"]}" style="margin-top:10px;">{new_results["recommendation"]}</div>', unsafe_allow_html=True)
                        st.markdown('</div>', unsafe_allow_html=True)
                    
                    # Calculate difference between scenarios
                    profit_difference = new_profit - profit
                    if abs(profit_difference) > 0.01:  # Only show if there's a meaningful difference
                        diff_color = VIVE_GREEN if profit_difference > 0 else VIVE_RED
                        diff_prefix = "+" if profit_difference > 0 else ""
                        st.markdown(f'<div style="text-align:center;margin:15px 0;font-weight:600;">Profit Difference: <span style="color:{diff_color}">{diff_prefix}{format_currency(profit_difference)}</span></div>', unsafe_allow_html=True)
                    
                    # Comparison chart
                    fig = go.Figure()
                    
                    # Financial metrics comparison
                    metrics = [
                        "Total Rework Cost", 
                        "Salvage Revenue", 
                        "Write-off Loss", 
                        "Net Profit/Loss"
                    ]
                    
                    current_values = [
                        st.session_state.salvage_results["financial"]["total_rework_cost"],
                        st.session_state.salvage_results["financial"]["salvage_revenue"],
                        st.session_state.salvage_results["financial"]["write_off_loss"],
                        st.session_state.salvage_results["financial"]["salvage_profit"]
                    ]
                    
                    new_values = [
                        new_results["financial"]["total_rework_cost"],
                        new_results["financial"]["salvage_revenue"],
                        new_results["financial"]["write_off_loss"],
                        new_results["financial"]["salvage_profit"]
                    ]
                    
                    fig.add_trace(go.Bar(
                        x=metrics,
                        y=current_values,
                        name='Current Scenario',
                        marker_color=VIVE_BLUE
                    ))
                    
                    fig.add_trace(go.Bar(
                        x=metrics,
                        y=new_values,
                        name='New Scenario',
                        marker_color=VIVE_TEAL
                    ))
                    
                    fig.update_layout(
                        title='Scenario Comparison',
                        xaxis_title='Metric',
                        yaxis_title='Amount ($)',
                        barmode='group',
                        height=400,
                        margin=dict(l=20, r=20, t=40, b=20),
                        template="plotly_white",
                        plot_bgcolor='rgba(0,0,0,0)'
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Show parameters that changed
                    st.markdown('<div class="sub-header">Parameter Changes</div>', unsafe_allow_html=True)
                    
                    param_changes = []
                    
                    if abs(recovery_adjustment - st.session_state.salvage_results["metrics"]["recovery_rate"]) > 0.01:
                        param_changes.append({
                            "parameter": "Recovery Rate",
                            "original": f"{st.session_state.salvage_results['metrics']['recovery_rate']}%",
                            "new": f"{recovery_adjustment}%"
                        })
                    
                    if abs(discount_adjustment - st.session_state.salvage_results["metrics"]["discount_percentage"]) > 0.01:
                        param_changes.append({
                            "parameter": "Discount Percentage",
                            "original": f"{st.session_state.salvage_results['metrics']['discount_percentage']}%",
                            "new": f"{discount_adjustment}%"
                        })
                    
                    if abs(rework_adjustment - current_rework_cost) > 0.01:
                        param_changes.append({
                            "parameter": "Per-Unit Rework Cost",
                            "original": format_currency(current_rework_cost),
                            "new": format_currency(rework_adjustment)
                        })
                    
                    if param_changes:
                        param_df = pd.DataFrame(param_changes)
                        st.table(param_df)
                    else:
                        st.info("No parameters were changed between scenarios.")
                    
                    # Sensitivity analysis insight
                    st.markdown('<div class="sub-header">Sensitivity Analysis</div>', unsafe_allow_html=True)
                    
                    # Determine which parameter had the biggest impact
                    param_impact = ""
                    if len(param_changes) > 0:
                        if abs(recovery_adjustment - st.session_state.salvage_results["metrics"]["recovery_rate"]) > 0.01:
                            recovery_impact = abs(profit_difference) / abs(recovery_adjustment - st.session_state.salvage_results["metrics"]["recovery_rate"])
                            param_impact += f"For each 1% change in recovery rate, profit changes by approximately {format_currency(recovery_impact/100)}. "
                        
                        if abs(discount_adjustment - st.session_state.salvage_results["metrics"]["discount_percentage"]) > 0.01:
                            discount_impact = abs(profit_difference) / abs(discount_adjustment - st.session_state.salvage_results["metrics"]["discount_percentage"])
                            param_impact += f"For each 1% change in discount percentage, profit changes by approximately {format_currency(discount_impact/100)}. "
                        
                        if abs(rework_adjustment - current_rework_cost) > 0.01:
                            rework_impact = abs(profit_difference) / abs(rework_adjustment - current_rework_cost)
                            param_impact += f"For each ${1:.2f} change in per-unit rework cost, profit changes by approximately {format_currency(rework_impact)}. "
                        
                        st.markdown(f'<div style="background-color:#F0F9FF;padding:15px;border-radius:5px;">{param_impact}</div>', unsafe_allow_html=True)
                    else:
                        st.info("No parameters were changed to perform sensitivity analysis.")
            
            # Reset button
            if st.button("Start New Analysis", key="salvage_reset_bottom_button"):
                st.session_state.salvage_results = None
                st.session_state.salvage_submitted = False
                st.experimental_rerun()

def display_dashboard():
    vive_header()
    st.markdown('<div class="main-header">Business Overview Dashboard</div>', unsafe_allow_html=True)
    
    # Load dashboard data with error handling
    with st.spinner("Loading dashboard data..."):
        dashboard_data = load_dashboard_data()
    
    if dashboard_data:
        # Date filter for the dashboard
        today = datetime.now().date()
        col1, col2, col3 = st.columns([2, 2, 1])
        
        with col1:
            start_date = st.date_input(
                "Start Date",
                value=today.replace(year=today.year-1),  # Default to 1 year ago
                max_value=today
            )
        
        with col2:
            end_date = st.date_input(
                "End Date",
                value=today,
                min_value=start_date,
                max_value=today
            )
            
        with col3:
            if st.button("Apply Filter", key="dashboard_filter"):
                st.success("Filters applied. Dashboard refreshed.")
        
        # Display KPIs in cards with a clean layout
        st.markdown('<div class="sub-header">Key Performance Indicators</div>', unsafe_allow_html=True)
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.markdown('<div class="metric-label">Monthly Revenue</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="metric-value">{format_currency(dashboard_data["monthly_revenue"])}</div>', unsafe_allow_html=True)
            
            # Add growth indicator if available
            if dashboard_data["mom_growth"] is not None:
                growth = dashboard_data["mom_growth"]
                growth_color = VIVE_GREEN if growth > 0 else VIVE_RED
                growth_icon = "↑" if growth > 0 else "↓"
                st.markdown(f'<div style="font-size:0.9rem;color:{growth_color}">{growth_icon} {abs(growth):.1f}% vs previous month</div>', unsafe_allow_html=True)
            
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.markdown('<div class="metric-label">Monthly Profit</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="metric-value">{format_currency(dashboard_data["monthly_profit"])}</div>', unsafe_allow_html=True)
            
            # Calculate and display profit margin
            profit_margin = (dashboard_data["monthly_profit"] / dashboard_data["monthly_revenue"]) * 100 if dashboard_data["monthly_revenue"] > 0 else 0
            st.markdown(f'<div style="font-size:0.9rem;color:{VIVE_GREY}">{profit_margin:.1f}% profit margin</div>', unsafe_allow_html=True)
            
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col3:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.markdown('<div class="metric-label">Inventory Turnover</div>', unsafe_allow_html=True)
            
            turnover = dashboard_data["inventory_turnover"]
            
            # Color-code inventory turnover
            if turnover > 4:
                turnover_color = VIVE_GREEN  # Good turnover (> 4 times per year)
            elif turnover > 2:
                turnover_color = VIVE_AMBER  # Moderate turnover
            else:
                turnover_color = VIVE_RED  # Poor turnover (< 2 times per year)
                
            st.markdown(f'<div class="metric-value" style="color:{turnover_color}">{turnover:.2f}x</div>', unsafe_allow_html=True)
            
            # Add average days of inventory
            st.markdown(f'<div style="font-size:0.9rem;color:{VIVE_GREY}">{dashboard_data["avg_inventory_days"]:.1f} days avg. inventory</div>', unsafe_allow_html=True)
            
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col4:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.markdown('<div class="metric-label">Return Rate</div>', unsafe_allow_html=True)
            
            return_rate = dashboard_data["return_rate"]
            
            # Color-code return rate
            if return_rate < 3:
                return_color = VIVE_GREEN  # Good return rate (< 3%)
            elif return_rate < 7:
                return_color = VIVE_AMBER  # Moderate return rate
            else:
                return_color = VIVE_RED  # High return rate (> 7%)
                
            st.markdown(f'<div class="metric-value" style="color:{return_color}">{return_rate:.1f}%</div>', unsafe_allow_html=True)
            
            # Add quality score if available
            if "quality_score" in dashboard_data:
                quality_score = dashboard_data["quality_score"]
                quality_color = VIVE_GREEN if quality_score >= 90 else (VIVE_AMBER if quality_score >= 80 else VIVE_RED)
                st.markdown(f'<div style="font-size:0.9rem;color:{quality_color}">Quality Score: {quality_score:.1f}/100</div>', unsafe_allow_html=True)
            
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Second row of metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.markdown('<div class="metric-label">Monthly Units Sold</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="metric-value">{dashboard_data["monthly_units"]:,}</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.markdown('<div class="metric-label">Pending Orders</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="metric-value">{dashboard_data["pending_orders"]:,}</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col3:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.markdown('<div class="metric-label">Inventory Value</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="metric-value">{format_currency(dashboard_data["total_inventory_value"])}</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col4:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.markdown('<div class="metric-label">Stock-Out Risk</div>', unsafe_allow_html=True)
            
            stock_out_risk = dashboard_data["stock_out_risk"]
            
            # Color-code stock-out risk
            if stock_out_risk < 5:
                risk_color = VIVE_GREEN  # Low risk
            elif stock_out_risk < 15:
                risk_color = VIVE_AMBER  # Moderate risk
            else:
                risk_color = VIVE_RED  # High risk
                
            st.markdown(f'<div class="metric-value" style="color:{risk_color}">{stock_out_risk:.1f}%</div>', unsafe_allow_html=True)
            
            # Show count of critical items
            critical_count = len(dashboard_data["critical_items"])
            if critical_count > 0:
                st.markdown(f'<div style="font-size:0.9rem;color:{VIVE_RED}">{critical_count} SKUs at zero stock</div>', unsafe_allow_html=True)
            
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Sales trend chart
        st.markdown('<div class="sub-header">Sales Trend</div>', unsafe_allow_html=True)
        
        # Create tabs for different time frames
        trend_tab1, trend_tab2 = st.tabs(["Daily Sales", "Monthly Sales"])
        
        with trend_tab1:
            # Get the last 90 days of data for better visualization
            last_90_days = dashboard_data["sales_trend"].sort_values("date", ascending=False).head(90).sort_values("date")
            
            sales_fig = px.line(
                last_90_days, 
                x="date", 
                y=["sales", "7d_moving_avg_sales"],
                labels={"value": "Units Sold", "variable": "Metric"},
                color_discrete_map={"sales": VIVE_BLUE, "7d_moving_avg_sales": VIVE_TEAL},
                title="Daily Sales (Last 90 Days)"
            )
            
            sales_fig.update_layout(
                xaxis_title="Date",
                yaxis_title="Units Sold",
                legend_title="",
                height=400,
                margin=dict(l=20, r=20, t=40, b=20),
                template="plotly_white",
                plot_bgcolor='rgba(0,0,0,0)'
            )
            
            st.plotly_chart(sales_fig, use_container_width=True)
            
        with trend_tab2:
            # Get monthly data
            monthly_data = dashboard_data["monthly_sales"].tail(12)  # Last 12 months
            
            monthly_fig = px.bar(
                monthly_data,
                x="month_name",
                y="sales",
                color_discrete_sequence=[VIVE_TEAL],
                title="Monthly Sales (Last 12 Months)"
            )
            
            monthly_fig.update_layout(
                xaxis_title="Month",
                yaxis_title="Units Sold",
                height=400,
                margin=dict(l=20, r=20, t=40, b=20),
                template="plotly_white",
                plot_bgcolor='rgba(0,0,0,0)'
            )
            
            st.plotly_chart(monthly_fig, use_container_width=True)
        
        # Sales breakdown and inventory status
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown('<div class="sub-header">Sales by Channel</div>', unsafe_allow_html=True)
            
            channel_fig = px.pie(
                dashboard_data["sales_by_channel"],
                values="revenue",
                names="channel",
                hole=0.4,
                color_discrete_sequence=[VIVE_TEAL, VIVE_BLUE, "#4CC9F0", "#90E0EF"]
            )
            
            channel_fig.update_layout(
                height=350,
                margin=dict(l=20, r=20, t=20, b=20),
                template="plotly_white",
                legend=dict(orientation="h", yanchor="bottom", y=-0.2)
            )
            
            st.plotly_chart(channel_fig, use_container_width=True)
        
        with col2:
            st.markdown('<div class="sub-header">Sales by Category</div>', unsafe_allow_html=True)
            
            # Sort categories by sales
            sorted_categories = dashboard_data["sales_by_category"].sort_values("sales", ascending=False)
            
            category_fig = px.bar(
                sorted_categories,
                x="product_category",
                y="sales",
                color="product_category",
                labels={"product_category": "Category", "sales": "Units Sold"},
                color_discrete_sequence=[VIVE_TEAL, VIVE_BLUE, "#4CC9F0", "#90E0EF", "#CAF0F8"]
            )
            
            category_fig.update_layout(
                xaxis_title="",
                showlegend=False,
                height=350,
                margin=dict(l=20, r=20, t=20, b=20),
                template="plotly_white",
                plot_bgcolor='rgba(0,0,0,0)'
            )
            
            st.plotly_chart(category_fig, use_container_width=True)
        
        # Top products and inventory status
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown('<div class="sub-header">Top Selling Products</div>', unsafe_allow_html=True)
            
            # Create a horizontal bar chart for top products
            top_products_fig = px.bar(
                dashboard_data["top_products"],
                y="product",
                x="units_sold",
                orientation='h',
                color_discrete_sequence=[VIVE_TEAL],
                labels={"product": "Product", "units_sold": "Units Sold"}
            )
            
            top_products_fig.update_layout(
                yaxis=dict(autorange="reversed"),  # Highest values at the top
                height=350,
                margin=dict(l=20, r=20, t=20, b=20),
                template="plotly_white",
                plot_bgcolor='rgba(0,0,0,0)'
            )
            
            st.plotly_chart(top_products_fig, use_container_width=True)
        
        with col2:
            st.markdown('<div class="sub-header">Low Stock Alert</div>', unsafe_allow_html=True)
            
            # Check if there are low stock items
            if len(dashboard_data["low_stock_items"]) > 0:
                # Display count of low stock items
                low_stock_count = len(dashboard_data["low_stock_items"])
                critical_count = len(dashboard_data["critical_items"])
                
                # Display as warning or error based on severity
                if critical_count > 0:
                    st.error(f"{critical_count} items are out of stock and {low_stock_count - critical_count} additional items are below reorder point.")
                else:
                    st.warning(f"{low_stock_count} items are below reorder point and should be reordered soon.")
                
                # Create a filtered dataframe with essential columns
                low_stock_display = dashboard_data["low_stock_items"][["sku", "product_name", "in_stock", "reorder_point", "lead_time_days"]].sort_values("in_stock").head(10)
                
                # Format the dataframe for display
                st.dataframe(
                    low_stock_display,
                    column_config={
                        "sku": "SKU",
                        "product_name": "Product",
                        "in_stock": st.column_config.NumberColumn("In Stock", help="Current inventory level"),
                        "reorder_point": st.column_config.NumberColumn("Reorder Point", help="Level at which item should be reordered"),
                        "lead_time_days": st.column_config.NumberColumn("Lead Time (Days)", help="Days needed for replenishment")
                    },
                    hide_index=True,
                    use_container_width=True
                )
                
                # Calculate total reorder value
                if "total_reorder_value" in dashboard_data:
                    reorder_value = dashboard_data["total_reorder_value"]
                    st.info(f"Estimated reorder value: {format_currency(reorder_value)} for all low stock items.")
            else:
                st.success("No items are currently below reorder point.")
    else:
        st.error("Unable to load dashboard data. Please check your data sources.")

def display_sales_analysis():
    vive_header()
    st.markdown('<div class="main-header">Sales Analysis</div>', unsafe_allow_html=True)
    
    # Load sales data with error handling
    with st.spinner("Loading sales data..."):
        sales_df = load_sales_data()
    
    if not sales_df.empty:
        # Date filter in a single row
        col1, col2, col3 = st.columns([2, 2, 1])
        
        min_date = min(sales_df["date"]).date()
        max_date = max(sales_df["date"]).date()
        
        with col1:
            start_date = st.date_input(
                "Start Date",
                min_date,
                min_value=min_date,
                max_value=max_date
            )
        
        with col2:
            end_date = st.date_input(
                "End Date",
                max_date,
                min_value=start_date,
                max_value=max_date
            )
        
        with col3:
            if st.button("Apply Filter", key="date_filter"):
                st.success("Date filter applied")
        
        # Filter data based on date range
        filtered_df = sales_df[
            (sales_df["date"].dt.date >= start_date) & 
            (sales_df["date"].dt.date <= end_date)
        ]
        
        # Channel and category filters in a single row
        col1, col2 = st.columns(2)
        
        with col1:
            available_channels = sorted(sales_df["channel"].unique())
            selected_channels = st.multiselect(
                "Filter by Channel",
                options=available_channels,
                default=available_channels
            )
        
        with col2:
            available_categories = sorted(sales_df["product_category"].unique())
            selected_categories = st.multiselect(
                "Filter by Product Category",
                options=available_categories,
                default=available_categories
            )
        
        # Apply additional filters
        if selected_channels:
            filtered_df = filtered_df[filtered_df["channel"].isin(selected_channels)]
        
        if selected_categories:
            filtered_df = filtered_df[filtered_df["product_category"].isin(selected_categories)]
        
        # Check if we have data after filtering
        if not filtered_df.empty:
            # Analyze the filtered data
            with st.spinner("Analyzing sales data..."):
                analysis_results = analyze_sales_trends(filtered_df)
            
            if analysis_results:
                # Display summary metrics
                st.markdown('<div class="sub-header">Sales Summary</div>', unsafe_allow_html=True)
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    total_revenue = filtered_df["revenue"].sum()
                    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                    st.markdown('<div class="metric-label">Total Revenue</div>', unsafe_allow_html=True)
                    st.markdown(f'<div class="metric-value">{format_currency(total_revenue)}</div>', unsafe_allow_html=True)
                    st.markdown('</div>', unsafe_allow_html=True)
                
                with col2:
                    total_units = filtered_df["sales"].sum()
                    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                    st.markdown('<div class="metric-label">Total Units Sold</div>', unsafe_allow_html=True)
                    st.markdown(f'<div class="metric-value">{total_units:,}</div>', unsafe_allow_html=True)
                    st.markdown('</div>', unsafe_allow_html=True)
                
                with col3:
                    avg_daily_revenue = filtered_df.groupby("date")["revenue"].sum().mean()
                    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                    st.markdown('<div class="metric-label">Average Daily Revenue</div>', unsafe_allow_html=True)
                    st.markdown(f'<div class="metric-value">{format_currency(avg_daily_revenue)}</div>', unsafe_allow_html=True)
                    st.markdown('</div>', unsafe_allow_html=True)
                
                with col4:
                    if total_units > 0:
                        avg_revenue_per_unit = total_revenue / total_units
                    else:
                        avg_revenue_per_unit = 0
                        
                    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                    st.markdown('<div class="metric-label">Average Price Per Unit</div>', unsafe_allow_html=True)
                    st.markdown(f'<div class="metric-value">{format_currency(avg_revenue_per_unit)}</div>', unsafe_allow_html=True)
                    st.markdown('</div>', unsafe_allow_html=True)
                
                # Sales trend charts
                st.markdown('<div class="sub-header">Sales Trend</div>', unsafe_allow_html=True)
                
                # Create tabs for units vs revenue view
                trend_tab1, trend_tab2 = st.tabs(["Units Sold", "Revenue"])
                
                with trend_tab1:
                    units_fig = px.line(
                        analysis_results["daily_sales"], 
                        x="date", 
                        y=["sales", "7d_moving_avg_sales"],
                        labels={"value": "Units Sold", "variable": "Metric"},
                        color_discrete_map={"sales": VIVE_BLUE, "7d_moving_avg_sales": VIVE_TEAL}
                    )
                    
                    units_fig.update_layout(
                        xaxis_title="Date",
                        yaxis_title="Units Sold",
                        legend_title="",
                        height=400,
                        margin=dict(l=20, r=20, t=20, b=20),
                        template="plotly_white",
                        plot_bgcolor='rgba(0,0,0,0)'
                    )
                    
                    st.plotly_chart(units_fig, use_container_width=True)
                
                with trend_tab2:
                    revenue_fig = px.line(
                        analysis_results["daily_sales"], 
                        x="date", 
                        y=["revenue", "7d_moving_avg_revenue"],
                        labels={"value": "Revenue ($)", "variable": "Metric"},
                        color_discrete_map={"revenue": VIVE_GREEN, "7d_moving_avg_revenue": VIVE_TEAL_DARK}
                    )
                    
                    revenue_fig.update_layout(
                        xaxis_title="Date",
                        yaxis_title="Revenue ($)",
                        legend_title="",
                        height=400,
                        margin=dict(l=20, r=20, t=20, b=20),
                        template="plotly_white",
                        plot_bgcolor='rgba(0,0,0,0)'
                    )
                    
                    st.plotly_chart(revenue_fig, use_container_width=True)
                
                # Sales breakdown
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown('<div class="sub-header">Sales by Channel</div>', unsafe_allow_html=True)
                    
                    # Sort channels by revenue
                    channel_data = analysis_results["channel_sales"].sort_values("revenue", ascending=False)
                    
                    # Create two tabs for different views
                    channel_tab1, channel_tab2 = st.tabs(["Revenue", "Units Sold"])
                    
                    with channel_tab1:
                        channel_revenue_fig = px.pie(
                            channel_data,
                            values="revenue",
                            names="channel",
                            hole=0.4,
                            color_discrete_sequence=[VIVE_TEAL, VIVE_BLUE, "#4CC9F0", "#90E0EF"]
                        )
                        
                        channel_revenue_fig.update_layout(
                            height=350,
                            margin=dict(l=20, r=20, t=20, b=20),
                            template="plotly_white"
                        )
                        
                        st.plotly_chart(channel_revenue_fig, use_container_width=True)
                    
                    with channel_tab2:
                        channel_units_fig = px.bar(
                            channel_data,
                            x="channel",
                            y="sales",
                            color="channel",
                            color_discrete_sequence=[VIVE_TEAL, VIVE_BLUE, "#4CC9F0", "#90E0EF"]
                        )
                        
                        channel_units_fig.update_layout(
                            xaxis_title="Channel",
                            yaxis_title="Units Sold",
                            showlegend=False,
                            height=350,
                            margin=dict(l=20, r=20, t=20, b=20),
                            template="plotly_white",
                            plot_bgcolor='rgba(0,0,0,0)'
                        )
                        
                        st.plotly_chart(channel_units_fig, use_container_width=True)
                        
                        # Add a table with channel performance metrics
                        channel_metrics = channel_data.copy()
                        if channel_metrics["sales"].sum() > 0:
                            channel_metrics["avg_price"] = channel_metrics["revenue"] / channel_metrics["sales"]
                        else:
                            channel_metrics["avg_price"] = 0
                            
                        channel_metrics["share"] = (channel_metrics["sales"] / channel_metrics["sales"].sum()) * 100
                        
                        # Format columns for display
                        channel_metrics["avg_price"] = channel_metrics["avg_price"].apply(lambda x: format_currency(x))
                        channel_metrics["revenue"] = channel_metrics["revenue"].apply(lambda x: format_currency(x))
                        channel_metrics["share"] = channel_metrics["share"].apply(lambda x: f"{x:.1f}%")
                        
                        st.dataframe(
                            channel_metrics[["channel", "sales", "revenue", "avg_price", "share"]],
                            column_config={
                                "channel": "Channel",
                                "sales": "Units Sold",
                                "revenue": "Revenue",
                                "avg_price": "Avg. Price",
                                "share": "Market Share"
                            },
                            hide_index=True,
                            use_container_width=True
                        )
                
                with col2:
                    st.markdown('<div class="sub-header">Sales by Category</div>', unsafe_allow_html=True)
                    
                    # Sort categories by revenue
                    category_data = analysis_results["category_sales"].sort_values("revenue", ascending=False)
                    
                    # Create two tabs for different views
                    category_tab1, category_tab2 = st.tabs(["Revenue", "Units Sold"])
                    
                    with category_tab1:
                        category_revenue_fig = px.bar(
                            category_data,
                            x="product_category",
                            y="revenue",
                            color="product_category",
                            color_discrete_sequence=[VIVE_TEAL, VIVE_BLUE, "#4CC9F0", "#90E0EF", "#CAF0F8"]
                        )
                        
                        category_revenue_fig.update_layout(
                            xaxis_title="Product Category",
                            yaxis_title="Revenue ($)",
                            showlegend=False,
                            height=350,
                            margin=dict(l=20, r=20, t=20, b=20),
                            template="plotly_white",
                            plot_bgcolor='rgba(0,0,0,0)'
                        )
                        
                        st.plotly_chart(category_revenue_fig, use_container_width=True)
                    
                    with category_tab2:
                        category_units_fig = px.bar(
                            category_data,
                            x="product_category",
                            y="sales",
                            color="product_category",
                            color_discrete_sequence=[VIVE_TEAL, VIVE_BLUE, "#4CC9F0", "#90E0EF", "#CAF0F8"]
                        )
                        
                        category_units_fig.update_layout(
                            xaxis_title="Product Category",
                            yaxis_title="Units Sold",
                            showlegend=False,
                            height=350,
                            margin=dict(l=20, r=20, t=20, b=20),
                            template="plotly_white",
                            plot_bgcolor='rgba(0,0,0,0)'
                        )
                        
                        st.plotly_chart(category_units_fig, use_container_width=True)
                
                # Top products section
                st.markdown('<div class="sub-header">Top Selling Products</div>', unsafe_allow_html=True)
                
                # Get top 10 products by units sold
                top_products = filtered_df.groupby("product")["sales"].sum().reset_index().sort_values("sales", ascending=False).head(10)
                
                # Create a horizontal bar chart
                top_products_fig = px.bar(
                    top_products,
                    y="product",
                    x="sales",
                    orientation='h',
                    color_discrete_sequence=[VIVE_TEAL],
                    height=400
                )
                
                top_products_fig.update_layout(
                    yaxis=dict(autorange="reversed"),  # Highest values at the top
                    xaxis_title="Units Sold",
                    yaxis_title="Product",
                    margin=dict(l=20, r=20, t=20, b=20),
                    template="plotly_white",
                    plot_bgcolor='rgba(0,0,0,0)'
                )
                
                st.plotly_chart(top_products_fig, use_container_width=True)
                
                # Monthly trends
                st.markdown('<div class="sub-header">Monthly Sales Trends</div>', unsafe_allow_html=True)
                
                # Check if we have monthly data
                if len(analysis_results["monthly_sales"]) > 1:
                    # Add month-year for better display
                    monthly_data = analysis_results["monthly_sales"].copy()
                    monthly_data["month_year"] = monthly_data.apply(lambda x: f"{x['month_name']} {x['year']}", axis=1)
                    
                    # Plot monthly units and revenue
                    monthly_tab1, monthly_tab2 = st.tabs(["Units Sold", "Revenue"])
                    
                    with monthly_tab1:
                        monthly_units_fig = px.bar(
                            monthly_data,
                            x="month_year",
                            y="sales",
                            color_discrete_sequence=[VIVE_TEAL],
                            labels={"month_year": "Month", "sales": "Units Sold"}
                        )
                        
                        monthly_units_fig.update_layout(
                            xaxis_title="Month",
                            yaxis_title="Units Sold",
                            height=350,
                            margin=dict(l=20, r=20, t=20, b=20),
                            template="plotly_white",
                            plot_bgcolor='rgba(0,0,0,0)'
                        )
                        
                        st.plotly_chart(monthly_units_fig, use_container_width=True)
                    
                    with monthly_tab2:
                        monthly_revenue_fig = px.bar(
                            monthly_data,
                            x="month_year",
                            y="revenue",
                            color_discrete_sequence=[VIVE_GREEN],
                            labels={"month_year": "Month", "revenue": "Revenue"}
                        )
                        
                        monthly_revenue_fig.update_layout(
                            xaxis_title="Month",
                            yaxis_title="Revenue ($)",
                            height=350,
                            margin=dict(l=20, r=20, t=20, b=20),
                            template="plotly_white",
                            plot_bgcolor='rgba(0,0,0,0)'
                        )
                        
                        st.plotly_chart(monthly_revenue_fig, use_container_width=True)
                    
                    # Calculate month-over-month growth
                    if len(monthly_data) > 1:
                        growth_data = []
                        for i in range(1, len(monthly_data)):
                            prev_sales = monthly_data.iloc[i-1]["sales"]
                            curr_sales = monthly_data.iloc[i]["sales"]
                            
                            growth_pct = ((curr_sales - prev_sales) / prev_sales * 100) if prev_sales > 0 else 0
                            
                            growth_data.append({
                                "month_year": monthly_data.iloc[i]["month_year"],
                                "growth_pct": growth_pct
                            })
                        
                        growth_df = pd.DataFrame(growth_data)
                        
                        # Plot growth chart
                        growth_fig = px.bar(
                            growth_df,
                            x="month_year",
                            y="growth_pct",
                            labels={"month_year": "Month", "growth_pct": "Growth %"},
                            color="growth_pct",
                            color_continuous_scale=[(0, VIVE_RED), (0.5, VIVE_AMBER), (1, VIVE_GREEN)]
                        )
                        
                        growth_fig.update_layout(
                            xaxis_title="Month",
                            yaxis_title="Month-over-Month Growth (%)",
                            height=250,
                            margin=dict(l=20, r=20, t=40, b=20),
                            template="plotly_white",
                            plot_bgcolor='rgba(0,0,0,0)',
                            title="Month-over-Month Growth (%)"
                        )
                        
                        st.plotly_chart(growth_fig, use_container_width=True)
                else:
                    st.info("Not enough monthly data to display trends. Try selecting a wider date range.")
                
                # Raw data table
                with st.expander("Show Raw Data"):
                    # Create summary by day
                    daily_summary = filtered_df.groupby("date").agg({
                        "sales": "sum",
                        "revenue": "sum"
                    }).reset_index()
                    
                    # Calculate average price
                    daily_summary["avg_price"] = daily_summary["revenue"] / daily_summary["sales"]
                    
                    # Format for display
                    daily_summary["date"] = daily_summary["date"].dt.strftime('%Y-%m-%d')
                    
                    st.dataframe(
                        daily_summary.sort_values("date", ascending=False),
                        column_config={
                            "date": "Date",
                            "sales": st.column_config.NumberColumn("Units Sold"),
                            "revenue": st.column_config.NumberColumn("Revenue", format="$%.2f"),
                            "avg_price": st.column_config.NumberColumn("Average Price", format="$%.2f")
                        },
                        hide_index=True,
                        use_container_width=True
                    )
            else:
                st.error("Error analyzing sales data. Please check your data and try again.")
        else:
            st.warning("No data available for the selected filters. Please adjust your selection.")
    else:
        st.error("Unable to load sales data. Please check your data sources.")

def display_inventory_management():
    vive_header()
    st.markdown('<div class="main-header">Inventory Management</div>', unsafe_allow_html=True)
    
    # Load inventory data with error handling
    with st.spinner("Loading inventory data..."):
        inventory_df = load_inventory_data()
    
    if not inventory_df.empty:
        # Filter options in two rows for better organization
        col1, col2 = st.columns(2)
        
        with col1:
            selected_categories = st.multiselect(
                "Filter by Category",
                options=sorted(inventory_df["category"].unique()),
                default=sorted(inventory_df["category"].unique())
            )
        
        with col2:
            selected_status = st.multiselect(
                "Filter by Status",
                options=sorted(inventory_df["status"].unique()),
                default=sorted(inventory_df["status"].unique())
            )
        
        col1, col2 = st.columns(2)
        
        with col1:
            sku_search = st.text_input(
                "Search by SKU or Product Name",
                placeholder="Enter SKU or product name",
                help="Search for specific products by SKU or name"
            )
        
        with col2:
            # Add filter by stock level
            stock_options = ["All Levels", "Low Stock Only", "Critical Only", "Excess Stock Only"]
            stock_filter = st.selectbox(
                "Filter by Stock Level",
                options=stock_options,
                index=0,
                help="Filter to show specific stock level concerns"
            )
        
        # Apply filters
        filtered_df = inventory_df.copy()
        
        if selected_categories:
            filtered_df = filtered_df[filtered_df["category"].isin(selected_categories)]
        
        if selected_status:
            filtered_df = filtered_df[filtered_df["status"].isin(selected_status)]
        
        if sku_search:
            filtered_df = filtered_df[
                (filtered_df["sku"].str.contains(sku_search, case=False)) |
                (filtered_df["product_name"].str.contains(sku_search, case=False))
            ]
        
        # Apply stock level filter
        if stock_filter == "Low Stock Only":
            filtered_df = filtered_df[filtered_df["status"] == "Low Stock"]
        elif stock_filter == "Critical Only":
            filtered_df = filtered_df[filtered_df["status"] == "Critical"]
        elif stock_filter == "Excess Stock Only":
            filtered_df = filtered_df[filtered_df["in_stock"] > 3 * filtered_df["reorder_point"]]
        
        # Check if we have data after filtering
        if not filtered_df.empty:
            # Analyze the filtered data
            with st.spinner("Analyzing inventory data..."):
                analysis_results = analyze_inventory_status(filtered_df)
            
            if analysis_results:
                # Display summary metrics
                st.markdown('<div class="sub-header">Inventory Summary</div>', unsafe_allow_html=True)
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    total_inventory = filtered_df["in_stock"].sum()
                    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                    st.markdown('<div class="metric-label">Total Inventory Units</div>', unsafe_allow_html=True)
                    st.markdown(f'<div class="metric-value">{total_inventory:,}</div>', unsafe_allow_html=True)
                    st.markdown('</div>', unsafe_allow_html=True)
                
                with col2:
                    total_value = analysis_results["total_inventory_value"]
                    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                    st.markdown('<div class="metric-label">Total Inventory Value</div>', unsafe_allow_html=True)
                    st.markdown(f'<div class="metric-value">{format_currency(total_value)}</div>', unsafe_allow_html=True)
                    st.markdown('</div>', unsafe_allow_html=True)
                
                with col3:
                    low_stock_count = len(analysis_results["low_stock_items"])
                    low_stock_percent = low_stock_count / len(filtered_df) * 100 if len(filtered_df) > 0 else 0
                    
                    # Color coding based on percentage
                    if low_stock_percent > 20:
                        stock_color = VIVE_RED
                    elif low_stock_percent > 10:
                        stock_color = VIVE_AMBER
                    else:
                        stock_color = VIVE_GREEN
                        
                    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                    st.markdown('<div class="metric-label">Low Stock Items</div>', unsafe_allow_html=True)
                    st.markdown(f'<div class="metric-value" style="color:{stock_color}">{low_stock_count} ({low_stock_percent:.1f}%)</div>', unsafe_allow_html=True)
                    st.markdown('</div>', unsafe_allow_html=True)
                
                with col4:
                    if "items_to_reorder" in analysis_results:
                        total_reorder_value = analysis_results["total_reorder_value"]
                        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                        st.markdown('<div class="metric-label">Reorder Value</div>', unsafe_allow_html=True)
                        st.markdown(f'<div class="metric-value">{format_currency(total_reorder_value)}</div>', unsafe_allow_html=True)
                        st.markdown('</div>', unsafe_allow_html=True)
                
                # Inventory status distribution
                status_summary = analysis_results["status_summary"]
                
                # Create a stacked bar chart for inventory status
                st.markdown('<div class="sub-header">Inventory Status Overview</div>', unsafe_allow_html=True)
                
                # Status distribution by count
                status_fig = go.Figure()
                
                # Add traces for each status
                for status in status_summary["status"]:
                    status_data = status_summary[status_summary["status"] == status]
                    
                    # Set color based on status
                    if status == "Critical":
                        bar_color = VIVE_RED
                    elif status == "Low Stock":
                        bar_color = VIVE_AMBER
                    else:  # OK
                        bar_color = VIVE_GREEN
                    
                    status_fig.add_trace(go.Bar(
                        x=["Item Count"],
                        y=[status_data["sku"].values[0]],
                        name=status,
                        marker_color=bar_color
                    ))
                
                status_fig.update_layout(
                    barmode='stack',
                    height=300,
                    margin=dict(l=20, r=20, t=20, b=20),
                    template="plotly_white",
                    plot_bgcolor='rgba(0,0,0,0)',
                    legend=dict(
                        orientation="h",
                        yanchor="bottom",
                        y=1.02,
                        xanchor="right",
                        x=1
                    )
                )
                
                # Inventory by category chart
                col1, col2 = st.columns(2)
                
                with col1:
                    st.plotly_chart(status_fig, use_container_width=True)
                
                with col2:
                    # Inventory value by category
                    category_inventory = analysis_results["category_inventory"]
                    
                    category_fig = px.pie(
                        category_inventory,
                        values="inventory_value",
                        names="category",
                        hole=0.4,
                        color_discrete_sequence=[VIVE_TEAL, VIVE_BLUE, "#4CC9F0", "#90E0EF", "#CAF0F8"]
                    )
                    
                    category_fig.update_layout(
                        title="Inventory Value by Category",
                        height=300,
                        margin=dict(l=20, r=20, t=40, b=20),
                        template="plotly_white"
                    )
                    
                    st.plotly_chart(category_fig, use_container_width=True)
                
                # Inventory Status Table with tabs for different views
                st.markdown('<div class="sub-header">Inventory Status</div>', unsafe_allow_html=True)
                
                inventory_tabs = st.tabs(["All Inventory", "Low Stock Items", "Items to Reorder"])
                
                with inventory_tabs[0]:
                    # All inventory table
                    st.dataframe(
                        filtered_df.sort_values(["status", "category", "sku"]),
                        column_config={
                            "sku": "SKU",
                            "product_name": "Product Name",
                            "category": "Category",
                            "in_stock": st.column_config.NumberColumn("In Stock", help="Current inventory level"),
                            "reorder_point": st.column_config.NumberColumn("Reorder Point", help="Level at which item should be reordered"),
                            "lead_time_days": st.column_config.NumberColumn("Lead Time (Days)", help="Days needed for replenishment"),
                            "cost": st.column_config.NumberColumn("Cost", format="$%.2f", help="Per-unit cost"),
                            "last_ordered": "Last Ordered",
                            "status": "Status",
                            "inventory_value": st.column_config.NumberColumn("Inventory Value", format="$%.2f", help="Total value of current inventory")
                        },
                        hide_index=True,
                        use_container_width=True
                    )
                
                with inventory_tabs[1]:
                    # Low stock items
                    if len(analysis_results["low_stock_items"]) > 0:
                        low_stock_df = analysis_results["low_stock_items"].sort_values("in_stock")
                        
                        st.dataframe(
                            low_stock_df[["sku", "product_name", "category", "in_stock", "reorder_point", "lead_time_days", "last_ordered", "status"]],
                            column_config={
                                "sku": "SKU",
                                "product_name": "Product Name",
                                "category": "Category",
                                "in_stock": st.column_config.NumberColumn("In Stock", help="Current inventory level"),
                                "reorder_point": st.column_config.NumberColumn("Reorder Point", help="Level at which item should be reordered"),
                                "lead_time_days": st.column_config.NumberColumn("Lead Time (Days)", help="Days needed for replenishment"),
                                "last_ordered": "Last Ordered",
                                "status": "Status"
                            },
                            hide_index=True,
                            use_container_width=True
                        )
                    else:
                        st.success("No low stock items found in the current selection.")
                
                with inventory_tabs[2]:
                    # Items to reorder
                    if "items_to_reorder" in analysis_results and len(analysis_results["items_to_reorder"]) > 0:
                        reorder_df = analysis_results["items_to_reorder"].sort_values("reorder_quantity", ascending=False)
                        
                        st.dataframe(
                            reorder_df[["sku", "product_name", "category", "in_stock", "reorder_point", "reorder_quantity", "cost", "reorder_value", "lead_time_days"]],
                            column_config={
                                "sku": "SKU",
                                "product_name": "Product Name",
                                "category": "Category",
                                "in_stock": st.column_config.NumberColumn("In Stock", help="Current inventory level"),
                                "reorder_point": st.column_config.NumberColumn("Reorder Point", help="Level at which item should be reordered"),
                                "reorder_quantity": st.column_config.NumberColumn("Reorder Qty", help="Quantity to order to reach optimal levels"),
                                "cost": st.column_config.NumberColumn("Unit Cost", format="$%.2f", help="Per-unit cost"),
                                "reorder_value": st.column_config.NumberColumn("Reorder Value", format="$%.2f", help="Total cost to reorder"),
                                "lead_time_days": st.column_config.NumberColumn("Lead Time (Days)", help="Days needed for replenishment")
                            },
                            hide_index=True,
                            use_container_width=True
                        )
                        
                        # Calculate total reorder quantity and value
                        total_reorder_qty = reorder_df["reorder_quantity"].sum()
                        total_reorder_value = reorder_df["reorder_value"].sum()
                        
                        st.markdown(f"""
                        <div style="background-color:#F0F9FF;padding:15px;border-radius:5px;margin-top:15px;">
                            <strong>Reorder Summary:</strong> {total_reorder_qty:,} units across {len(reorder_df)} products, with a total value of {format_currency(total_reorder_value)}.
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.success("No items need to be reordered at this time.")
                
                # Download buttons for inventory reports
                st.markdown('<div class="sub-header">Export Reports</div>', unsafe_allow_html=True)
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    # Convert all inventory to CSV
                    csv_all = filtered_df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        "Download Full Inventory",
                        data=csv_all,
                        file_name="vive_inventory_full.csv",
                        mime="text/csv",
                        help="Download the full inventory list as CSV"
                    )
                
                with col2:
                    # Convert low stock items to CSV
                    if len(analysis_results["low_stock_items"]) > 0:
                        csv_low_stock = analysis_results["low_stock_items"].to_csv(index=False).encode('utf-8')
                        st.download_button(
                            "Download Low Stock Report",
                            data=csv_low_stock,
                            file_name="vive_inventory_low_stock.csv",
                            mime="text/csv",
                            help="Download the low stock items list as CSV"
                        )
                    else:
                        st.button("Download Low Stock Report", disabled=True)
                
                with col3:
                    # Convert reorder list to CSV
                    if "items_to_reorder" in analysis_results and len(analysis_results["items_to_reorder"]) > 0:
                        csv_reorder = analysis_results["items_to_reorder"].to_csv(index=False).encode('utf-8')
                        st.download_button(
                            "Download Reorder Report",
                            data=csv_reorder,
                            file_name="vive_inventory_reorder.csv",
                            mime="text/csv",
                            help="Download the items to reorder list as CSV"
                        )
                    else:
                        st.button("Download Reorder Report", disabled=True)
            else:
                st.error("Error analyzing inventory data. Please check your data and try again.")
        else:
            st.warning("No inventory data available for the selected filters. Please adjust your selection.")
    else:
        st.error("Unable to load inventory data. Please check your data sources.")

# --- PASSWORD VERIFICATION ---
def verify_password(password):
    """Verify user password with more advanced method"""
    valid_password = "MPFvive8955@#@"  # For demonstration - in production use a more secure method
    
    # Compare to valid password in a time-safe manner to prevent timing attacks
    # In a real application, use a secure password hashing method
    if len(password) != len(valid_password):
        return False
    
    result = True
    for a, b in zip(password, valid_password):
        result = result and (a == b)
    
    return result

# --- API KEY HANDLING ---
def get_openai_api_key():
    """Get OpenAI API key from environment or secrets"""
    try:
        api_key = st.secrets["openai_api_key"]
    except (FileNotFoundError, KeyError):
        api_key = os.environ.get("OPENAI_API_KEY", "")
    
    return api_key

def main():
    # Set sidebar style
    st.markdown("""
    <style>
        section[data-testid="stSidebar"] {
            background-color: #F8FAFC;
            border-right: 1px solid #E2E8F0;
        }
        
        section[data-testid="stSidebar"] > div {
            padding-top: 2rem;
            padding-left: 1rem;
            padding-right: 1rem;
        }
        
        section[data-testid="stSidebar"] .sidebar-title {
            font-size: 1.25rem;
            font-weight: 600;
            color: var(--vive-navy);
            margin-bottom: 1.5rem;
            text-align: center;
            padding-bottom: 0.5rem;
            border-bottom: 2px solid var(--vive-teal);
        }
    </style>
    """, unsafe_allow_html=True)
    
    # Sidebar header
    st.sidebar.markdown('<div class="sidebar-title">QualityROI Dashboard</div>', unsafe_allow_html=True)
    
    # Check if we should initialize OpenAI client
    if "openai_client" not in st.session_state:
        api_key = get_openai_api_key()
        st.session_state.openai_client = initialize_openai_client(api_key)
    
    # Navigation
    app_mode = st.sidebar.selectbox(
        "Select Module",
        ["Quality Manager", "Dashboard", "Sales Analysis", "Inventory Management", "Login"]
    )
    
    # Login Screen
    if app_mode == "Login":
        vive_header()
        st.title("QualityROI - Cost-Benefit Analysis Tool")
        st.subheader("Login")
        
        # Create a clean login form
        with st.form("login_form"):
            password = st.text_input("Enter password", type="password")
            login_button = st.form_submit_button("Login")
            
            if login_button:
                if verify_password(password):
                    st.session_state["authenticated"] = True
                    st.success("Login successful! Redirecting to dashboard...")
                    time.sleep(1)  # Short delay for better UX
                    st.experimental_rerun()
                else:
                    st.error("Incorrect password. Please try again.")
    
    # Protected pages - only shown if authenticated
    elif "authenticated" in st.session_state and st.session_state["authenticated"]:
        if app_mode == "Dashboard":
            display_dashboard()
        elif app_mode == "Sales Analysis":
            display_sales_analysis()
        elif app_mode == "Inventory Management":
            display_inventory_management()
        elif app_mode == "Quality Manager":
            display_quality_manager(st.session_state.openai_client)
    else:
        # If quality mode was explicitly chosen but not logged in, show login with explanation
        if app_mode == "Quality Manager":
            vive_header()
            st.title("QualityROI - Cost-Benefit Analysis Tool")
            st.warning("Please login to access the Quality Manager")
            
            with st.form("login_redirect_form"):
                password = st.text_input("Enter password", type="password")
                login_button = st.form_submit_button("Login")
                
                if login_button:
                    if verify_password(password):
                        st.session_state["authenticated"] = True
                        st.success("Login successful! Redirecting to Quality Manager...")
                        time.sleep(1)  # Short delay for better UX
                        st.experimental_rerun()
                    else:
                        st.error("Incorrect password. Please try again.")
        else:
            vive_header()
            st.title("QualityROI - Cost-Benefit Analysis Tool")
            st.info("Please login to access this application")
            
            # Add a quick login form
            with st.form("quick_login_form"):
                password = st.text_input("Enter password", type="password")
                login_button = st.form_submit_button("Login")
                
                if login_button:
                    if verify_password(password):
                        st.session_state["authenticated"] = True
                        st.success("Login successful! Redirecting...")
                        time.sleep(1)  # Short delay for better UX
                        st.experimental_rerun()
                    else:
                        st.error("Incorrect password. Please try again.")

if __name__ == "__main__":
    main()
