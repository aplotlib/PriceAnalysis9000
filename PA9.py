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
from openai import OpenAI
import base64
import io

# Page configuration
st.set_page_config(
    page_title="QualityROI - Cost-Benefit Analysis Tool",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.8rem;
        font-weight: 600;
        color: #2563EB;
        margin-bottom: 0.8rem;
    }
    .card {
        background-color: #FFFFFF;
        border-radius: 0.5rem;
        padding: 1.5rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-bottom: 1rem;
    }
    .metric-label {
        font-size: 1rem;
        font-weight: 500;
        color: #6B7280;
    }
    .metric-value {
        font-size: 1.5rem;
        font-weight: 600;
        color: #111827;
    }
    .recommendation-high {
        background-color: #DCFCE7;
        color: #166534;
        padding: 0.5rem;
        border-radius: 0.25rem;
        font-weight: 600;
    }
    .recommendation-medium {
        background-color: #FEF3C7;
        color: #92400E;
        padding: 0.5rem;
        border-radius: 0.25rem;
        font-weight: 600;
    }
    .recommendation-low {
        background-color: #FEE2E2;
        color: #B91C1C;
        padding: 0.5rem;
        border-radius: 0.25rem;
        font-weight: 600;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 0.5rem;
        max-width: 80%;
    }
    .user-message {
        background-color: #E2E8F0;
        margin-left: auto;
    }
    .ai-message {
        background-color: #DBEAFE;
        margin-right: auto;
    }
    .stTextInput>div>div>input {
        background-color: #F9FAFB;
        border: 1px solid #D1D5DB;
    }
    .stNumberInput>div>div>input {
        background-color: #F9FAFB;
        border: 1px solid #D1D5DB;
    }
    .stButton>button {
        background-color: #2563EB;
        color: white;
        font-weight: 600;
        border-radius: 0.375rem;
        border: none;
        padding: 0.5rem 1rem;
        box-shadow: 0 1px 2px rgba(0, 0, 0, 0.05);
    }
    .stButton>button:hover {
        background-color: #1D4ED8;
    }
    .sidebar .sidebar-content {
        background-image: linear-gradient(180deg, #E0F2FE 0%, #F0F9FF 75%);
    }
    .required-field::after {
        content: " *";
        color: red;
    }
    .data-warning {
        background-color: #FEF3C7;
        border-left: 4px solid #D97706;
        color: #92400E;
        padding: 0.75rem;
        border-radius: 0.25rem;
        margin-bottom: 1rem;
    }
    .sidebar-info {
        background-color: #EFF6FF;
        border-radius: 0.375rem;
        padding: 1rem;
        margin-bottom: 1rem;
    }
    .export-button {
        color: #2563EB;
        background-color: #EFF6FF;
        border: 1px solid #BFDBFE;
        padding: 0.5rem 0.75rem;
        border-radius: 0.375rem;
        font-size: 0.875rem;
        display: inline-flex;
        align-items: center;
        justify-content: center;
        cursor: pointer;
    }
    .export-button:hover {
        background-color: #DBEAFE;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 1rem;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: #F3F4F6;
        border-radius: 0.375rem;
        padding: 0.25rem 1rem;
        font-weight: 500;
    }
    .stTabs [aria-selected="true"] {
        background-color: #2563EB;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# --- UTILITY FUNCTIONS ---

def initialize_openai_client(api_key: str) -> OpenAI:
    """
    Initialize the OpenAI client with the provided API key
    
    Args:
        api_key: OpenAI API key
        
    Returns:
        OpenAI client instance
    """
    try:
        client = OpenAI(api_key=api_key)
        return client
    except Exception as e:
        # Handle initialization errors gracefully
        print(f"Error initializing OpenAI client: {e}")
        return None

def get_ai_analysis(
    client: OpenAI, 
    system_prompt: str, 
    user_message: str,
    model: str = "gpt-4o",
    messages: Optional[List[Dict[str, str]]] = None
) -> str:
    """
    Get AI analysis for quality issues using the OpenAI API
    
    Args:
        client: OpenAI client instance
        system_prompt: System prompt providing context to the AI
        user_message: User message or query
        model: Model ID to use
        messages: Optional list of previous messages for context
        
    Returns:
        AI response text
    """
    try:
        # Build the messages array
        message_list = [{"role": "system", "content": system_prompt}]
        
        # Add previous messages if provided
        if messages:
            message_list.extend(messages)
        
        # Add the current user message
        message_list.append({"role": "user", "content": user_message})
        
        # Make the API call
        response = client.chat.completions.create(
            model=model,
            messages=message_list,
            temperature=0.7,
            max_tokens=2048
        )
        
        # Extract and return the response text
        return response.choices[0].message.content
    except Exception as e:
        # Handle API errors gracefully
        error_message = f"Error getting AI analysis: {str(e)}"
        print(error_message)
        return f"I apologize, but I encountered an error while analyzing this issue. Please try again or contact technical support if the problem persists.\n\nError details: {error_message}"

def format_currency(value: float) -> str:
    """Format a value as currency with $ symbol"""
    return f"${value:.2f}"

def format_percentage(value: float) -> str:
    """Format a value as percentage with % symbol"""
    return f"{value:.1f}%"

def get_csv_download_link(df: pd.DataFrame, filename: str, button_text: str) -> str:
    """
    Generate a download link for a CSV file
    
    Args:
        df: DataFrame to export
        filename: Name of the file
        button_text: Text to display on the button
        
    Returns:
        HTML string with download link
    """
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}" class="export-button">📥 {button_text}</a>'
    return href

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
    msrp: float,  # Added MSRP as required parameter
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
    
    Args:
        sku: Product SKU
        product_type: B2B, B2C, or Both
        sales_30d: Units sold in last 30 days
        returns_30d: Units returned in last 30 days
        issue_description: Description of the quality issue
        current_unit_cost: Current landed cost per unit
        fix_cost_upfront: One-time cost to implement fix
        fix_cost_per_unit: Additional cost per unit after fix
        msrp: Manufacturer's Suggested Retail Price
        asin: Amazon Standard Identification Number
        ncx_rate: Negative Customer Experience rate
        sales_365d: Units sold in last 365 days
        returns_365d: Units returned in last 365 days
        star_rating: Current star rating on Amazon
        total_reviews: Total number of reviews
        fba_fee: Amazon FBA fee per unit
        
    Returns:
        Dict with analysis results
    """
    # Calculate basic metrics
    return_rate_30d = (returns_30d / sales_30d) * 100 if sales_30d > 0 else 0
    
    # Include 365-day data if available
    return_rate_365d = None
    if sales_365d is not None and returns_365d is not None and sales_365d > 0:
        return_rate_365d = (returns_365d / sales_365d) * 100
    
    # Calculate current financial metrics
    current_profit_per_unit = msrp - current_unit_cost
    if fba_fee is not None and fba_fee > 0:
        current_profit_per_unit -= fba_fee
    
    current_margin_percentage = (current_profit_per_unit / msrp) * 100 if msrp > 0 else 0
    
    # Calculate return costs and impact
    monthly_return_cost = returns_30d * current_unit_cost
    
    # Estimate future return rate after fix (assuming 80% reduction in returns)
    estimated_future_return_rate = return_rate_30d * 0.2  # 80% reduction
    estimated_future_returns = sales_30d * (estimated_future_return_rate / 100)
    
    # Calculate new unit cost after fix
    new_unit_cost = current_unit_cost + fix_cost_per_unit
    
    # Calculate new profit metrics
    new_profit_per_unit = msrp - new_unit_cost
    if fba_fee is not None and fba_fee > 0:
        new_profit_per_unit -= fba_fee
    
    new_margin_percentage = (new_profit_per_unit / msrp) * 100 if msrp > 0 else 0
    
    # Calculate monthly savings from reduced returns
    current_monthly_return_units = returns_30d
    estimated_monthly_return_units = estimated_future_returns
    monthly_return_units_saved = current_monthly_return_units - estimated_monthly_return_units
    
    # Financial impact
    monthly_return_cost = current_monthly_return_units * current_unit_cost
    estimated_future_return_cost = estimated_monthly_return_units * new_unit_cost
    estimated_monthly_savings = monthly_return_cost - estimated_future_return_cost
    
    # Calculate impact on monthly profit
    current_monthly_profit = (current_profit_per_unit * sales_30d) - (current_profit_per_unit * returns_30d)
    new_monthly_profit = (new_profit_per_unit * sales_30d) - (new_profit_per_unit * estimated_future_returns)
    
    # Annual projections
    annual_return_cost = monthly_return_cost * 12
    annual_savings = estimated_monthly_savings * 12
    current_annual_profit = current_monthly_profit * 12
    new_annual_profit = new_monthly_profit * 12
    profit_improvement = new_annual_profit - current_annual_profit
    
    # Simple payback period (months)
    if estimated_monthly_savings > 0:
        payback_months = fix_cost_upfront / estimated_monthly_savings
    else:
        payback_months = float('inf')
    
    # Calculate 3-year ROI
    total_investment = fix_cost_upfront + (fix_cost_per_unit * sales_30d * 36)  # 36 months
    total_savings = annual_savings * 3
    roi_3yr = ((total_savings - total_investment) / total_investment) * 100 if total_investment > 0 else 0
    
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
    
    # Check if fix makes product unprofitable
    profitability_impact = None
    if new_margin_percentage < 0:
        profitability_impact = "Warning - Fix will make product unprofitable at current MSRP"
        recommendation = "Not Recommended - Fix makes product unprofitable"
        recommendation_class = "recommendation-low"
    elif new_margin_percentage < current_margin_percentage * 0.5:
        profitability_impact = "Caution - Fix will significantly reduce profit margin"
        if recommendation_class == "recommendation-high":
            recommendation = "Consider - Good ROI but significant impact on margins"
            recommendation_class = "recommendation-medium"
    
    # Prepare results dictionary
    results = {
        "sku": sku,
        "asin": asin,
        "product_type": product_type,
        "msrp": msrp,
        "current_metrics": {
            "return_rate_30d": return_rate_30d,
            "return_rate_365d": return_rate_365d,
            "ncx_rate": ncx_rate,
            "star_rating": star_rating,
            "total_reviews": total_reviews,
            "profit_per_unit": current_profit_per_unit,
            "margin_percentage": current_margin_percentage
        },
        "future_metrics": {
            "estimated_return_rate": estimated_future_return_rate,
            "profit_per_unit": new_profit_per_unit,
            "margin_percentage": new_margin_percentage
        },
        "financial_impact": {
            "monthly_return_cost": monthly_return_cost,
            "annual_return_cost": annual_return_cost,
            "estimated_monthly_savings": estimated_monthly_savings,
            "annual_savings": annual_savings,
            "current_monthly_profit": current_monthly_profit,
            "new_monthly_profit": new_monthly_profit,
            "current_annual_profit": current_annual_profit,
            "new_annual_profit": new_annual_profit,
            "profit_improvement": profit_improvement,
            "payback_months": payback_months,
            "roi_3yr": roi_3yr,
            "fix_cost_upfront": fix_cost_upfront,
            "fix_cost_per_unit": fix_cost_per_unit,
            "current_unit_cost": current_unit_cost,
            "new_unit_cost": new_unit_cost,
            "fba_fee": fba_fee
        },
        "recommendation": recommendation,
        "recommendation_class": recommendation_class,
        "brand_impact": brand_impact,
        "profitability_impact": profitability_impact,
        "issue_description": issue_description
    }
    
    return results

def analyze_salvage_operation(
    sku: str,
    affected_inventory: int,
    current_unit_cost: float,
    rework_cost_upfront: float,
    rework_cost_per_unit: float,
    expected_recovery_pct: float,
    expected_discount_pct: float,
    msrp: float  # Added MSRP as required parameter
) -> Dict[str, Any]:
    """
    Analyze potential salvage operation for affected inventory
    
    Args:
        sku: Product SKU
        affected_inventory: Number of affected units
        current_unit_cost: Current landed cost per unit
        rework_cost_upfront: One-time cost to set up rework operation
        rework_cost_per_unit: Cost to rework each unit
        expected_recovery_pct: Percentage of units expected to be recovered
        expected_discount_pct: Discount percentage for selling reworked units
        msrp: Manufacturer's Suggested Retail Price
        
    Returns:
        Dict with analysis results
    """
    # Calculate salvage metrics
    expected_units_recovered = affected_inventory * (expected_recovery_pct / 100)
    regular_price = msrp  # Use actual MSRP instead of assuming markup
    discounted_price = regular_price * (1 - expected_discount_pct / 100)
    
    total_rework_cost = rework_cost_upfront + (rework_cost_per_unit * affected_inventory)
    salvage_revenue = expected_units_recovered * discounted_price
    write_off_loss = (affected_inventory - expected_units_recovered) * current_unit_cost
    
    # Calculate profit metrics
    original_profit_per_unit = msrp - current_unit_cost
    total_original_value = affected_inventory * current_unit_cost
    salvage_cost_per_unit = (rework_cost_upfront / affected_inventory) + rework_cost_per_unit if affected_inventory > 0 else 0
    salvage_profit_per_unit = discounted_price - current_unit_cost - salvage_cost_per_unit
    
    salvage_profit = salvage_revenue - total_rework_cost - write_off_loss
    roi_percent = (salvage_profit / total_rework_cost) * 100 if total_rework_cost > 0 else 0
    
    # Complete write-off scenario
    writeoff_cost = affected_inventory * current_unit_cost
    
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
            "discount_percentage": expected_discount_pct,
            "original_profit_per_unit": original_profit_per_unit,
            "salvage_profit_per_unit": salvage_profit_per_unit
        },
        "financial": {
            "total_rework_cost": total_rework_cost,
            "salvage_revenue": salvage_revenue,
            "write_off_loss": write_off_loss,
            "salvage_profit": salvage_profit,
            "complete_writeoff_cost": writeoff_cost,
            "roi_percent": roi_percent,
            "profit_per_unit": salvage_profit / affected_inventory if affected_inventory > 0 else 0,
            "total_original_value": total_original_value
        },
        "recommendation": recommendation,
        "recommendation_class": recommendation_class
    }

def display_quality_issue_results(results):
    """Display the quality issue analysis results in a visually appealing way"""
    st.markdown(f'<div class="card">', unsafe_allow_html=True)
    
    # Results header
    st.markdown(f'<div class="sub-header">Analysis Results for SKU: {results["sku"]}</div>', unsafe_allow_html=True)
    
    # Current metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown('<div class="metric-label">Return Rate (30d)</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="metric-value">{results["current_metrics"]["return_rate_30d"]:.2f}%</div>', unsafe_allow_html=True)
        
        if results["current_metrics"]["star_rating"]:
            st.markdown('<div class="metric-label">Star Rating</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="metric-value">{results["current_metrics"]["star_rating"]:.1f}★</div>', unsafe_allow_html=True)
        
        st.markdown('<div class="metric-label">Current Profit Margin</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="metric-value">{results["current_metrics"]["margin_percentage"]:.2f}%</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="metric-label">Monthly Return Cost</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="metric-value">${results["financial_impact"]["monthly_return_cost"]:.2f}</div>', unsafe_allow_html=True)
        
        st.markdown('<div class="metric-label">Annual Return Cost</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="metric-value">${results["financial_impact"]["annual_return_cost"]:.2f}</div>', unsafe_allow_html=True)
        
        st.markdown('<div class="metric-label">Future Profit Margin</div>', unsafe_allow_html=True)
        margin_color = "#10B981" if results["future_metrics"]["margin_percentage"] >= results["current_metrics"]["margin_percentage"] else "#EF4444"
        st.markdown(f'<div class="metric-value" style="color:{margin_color}">{results["future_metrics"]["margin_percentage"]:.2f}%</div>', unsafe_allow_html=True)
    
    with col3:
        st.markdown('<div class="metric-label">Est. Monthly Savings</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="metric-value">${results["financial_impact"]["estimated_monthly_savings"]:.2f}</div>', unsafe_allow_html=True)
        
        st.markdown('<div class="metric-label">Payback Period</div>', unsafe_allow_html=True)
        if results["financial_impact"]["payback_months"] == float('inf'):
            payback_text = "N/A"
        else:
            payback_text = f"{results['financial_impact']['payback_months']:.1f} months"
        st.markdown(f'<div class="metric-value">{payback_text}</div>', unsafe_allow_html=True)
        
        st.markdown('<div class="metric-label">Annual Profit Improvement</div>', unsafe_allow_html=True)
        profit_color = "#10B981" if results["financial_impact"]["profit_improvement"] > 0 else "#EF4444"
        st.markdown(f'<div class="metric-value" style="color:{profit_color}">${results["financial_impact"]["profit_improvement"]:.2f}</div>', unsafe_allow_html=True)
    
    # Recommendation
    st.markdown('<div class="metric-label">Recommendation</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="{results["recommendation_class"]}">{results["recommendation"]}</div>', unsafe_allow_html=True)
    
    # Impacts
    if results["brand_impact"]:
        st.markdown('<div class="metric-label">Brand Impact Assessment</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="metric-value">{results["brand_impact"]}</div>', unsafe_allow_html=True)
    
    if results["profitability_impact"]:
        st.markdown('<div class="metric-label">Profitability Impact</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="metric-value" style="color:#EF4444">{results["profitability_impact"]}</div>', unsafe_allow_html=True)
    
    # Comparison table of current vs. future metrics
    st.markdown('<div class="sub-header">Current vs. Future Metrics</div>', unsafe_allow_html=True)
    
    comparison_data = {
        "Metric": ["Unit Cost", "Profit per Unit", "Profit Margin", "Monthly Return Rate", "Monthly Returns", "Monthly Profit"],
        "Current": [
            f"${results['financial_impact']['current_unit_cost']:.2f}",
            f"${results['current_metrics']['profit_per_unit']:.2f}",
            f"{results['current_metrics']['margin_percentage']:.2f}%",
            f"{results['current_metrics']['return_rate_30d']:.2f}%",
            "N/A",  # We don't have the current returns in the result
            f"${results['financial_impact']['current_monthly_profit']:.2f}"
        ],
        "After Fix": [
            f"${results['financial_impact']['new_unit_cost']:.2f}",
            f"${results['future_metrics']['profit_per_unit']:.2f}",
            f"{results['future_metrics']['margin_percentage']:.2f}%",
            f"{results['future_metrics']['estimated_return_rate']:.2f}%",
            "N/A",  # We don't have the future returns in the result
            f"${results['financial_impact']['new_monthly_profit']:.2f}"
        ],
        "Change": [
            f"${results['financial_impact']['fix_cost_per_unit']:.2f}",
            f"${results['future_metrics']['profit_per_unit'] - results['current_metrics']['profit_per_unit']:.2f}",
            f"{results['future_metrics']['margin_percentage'] - results['current_metrics']['margin_percentage']:.2f}%",
            f"-{results['current_metrics']['return_rate_30d'] - results['future_metrics']['estimated_return_rate']:.2f}%",
            "N/A",
            f"${results['financial_impact']['new_monthly_profit'] - results['financial_impact']['current_monthly_profit']:.2f}"
        ]
    }
    
    comparison_df = pd.DataFrame(comparison_data)
    st.dataframe(comparison_df, hide_index=True)
    
    # ROI chart
    fig = go.Figure()
    
    # Initial investment
    fig.add_trace(go.Bar(
        x=['Initial Investment'],
        y=[results["financial_impact"]["fix_cost_upfront"]],
        name='Initial Investment',
        marker_color='#EF4444'
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
        marker_color='#10B981'
    ))
    
    # Profit improvement
    fig.add_trace(go.Bar(
        x=['Year 1 Profit Improvement', 'Year 2 Profit Improvement', 'Year 3 Profit Improvement'],
        y=[
            results["financial_impact"]["profit_improvement"],
            results["financial_impact"]["profit_improvement"],
            results["financial_impact"]["profit_improvement"]
        ],
        name='Profit Improvement',
        marker_color='#3B82F6'
    ))
    
    fig.update_layout(
        title='Projected Returns Over Time',
        xaxis_title='Timeline',
        yaxis_title='Amount ($)',
        barmode='group',
        height=300,
        margin=dict(l=20, r=20, t=40, b=20),
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Create data for export
    if 'quality_analysis_data' not in st.session_state:
        st.session_state.quality_analysis_data = []
    
    analysis_data = {
        "SKU": results["sku"],
        "ASIN": results["asin"] if results["asin"] else "N/A",
        "MSRP": results["msrp"],
        "Product Type": results["product_type"],
        "Current Unit Cost": results["financial_impact"]["current_unit_cost"],
        "New Unit Cost": results["financial_impact"]["new_unit_cost"],
        "Current Return Rate": results["current_metrics"]["return_rate_30d"],
        "Estimated Future Return Rate": results["future_metrics"]["estimated_return_rate"],
        "Current Profit Margin": results["current_metrics"]["margin_percentage"],
        "Future Profit Margin": results["future_metrics"]["margin_percentage"],
        "Monthly Savings": results["financial_impact"]["estimated_monthly_savings"],
        "Annual Savings": results["financial_impact"]["annual_savings"],
        "Payback Period (Months)": results["financial_impact"]["payback_months"],
        "3-Year ROI": results["financial_impact"]["roi_3yr"],
        "Annual Profit Improvement": results["financial_impact"]["profit_improvement"],
        "Recommendation": results["recommendation"],
        "Analysis Date": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    
    st.session_state.quality_analysis_data.append(analysis_data)
    
    # Export button
    if len(st.session_state.quality_analysis_data) > 0:
        export_df = pd.DataFrame(st.session_state.quality_analysis_data)
        st.markdown(get_csv_download_link(export_df, "quality_analysis_results.csv", "Export Analysis Results to CSV"), unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

def display_salvage_results(results):
    """Display the salvage operation analysis results in a visually appealing way"""
    st.markdown(f'<div class="card">', unsafe_allow_html=True)
    
    # Results header
    st.markdown(f'<div class="sub-header">Salvage Analysis for SKU: {results["sku"]}</div>', unsafe_allow_html=True)
    
    # Key metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown('<div class="metric-label">Affected Inventory</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="metric-value">{results["metrics"]["affected_inventory"]} units</div>', unsafe_allow_html=True)
        
        st.markdown('<div class="metric-label">Expected Recovery</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="metric-value">{results["metrics"]["expected_units_recovered"]:.0f} units ({results["metrics"]["recovery_rate"]}%)</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="metric-label">Total Rework Cost</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="metric-value">${results["financial"]["total_rework_cost"]:.2f}</div>', unsafe_allow_html=True)
        
        st.markdown('<div class="metric-label">Salvage Revenue</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="metric-value">${results["financial"]["salvage_revenue"]:.2f}</div>', unsafe_allow_html=True)
    
    with col3:
        st.markdown('<div class="metric-label">Complete Write-off Cost</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="metric-value">${results["financial"]["complete_writeoff_cost"]:.2f}</div>', unsafe_allow_html=True)
        
        st.markdown('<div class="metric-label">Salvage Net Profit/Loss</div>', unsafe_allow_html=True)
        profit_color = "#10B981" if results["financial"]["salvage_profit"] >= 0 else "#EF4444"
        st.markdown(f'<div class="metric-value" style="color:{profit_color}">${results["financial"]["salvage_profit"]:.2f}</div>', unsafe_allow_html=True)
    
    # Recommendation
    st.markdown('<div class="metric-label">Recommendation</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="{results["recommendation_class"]}">{results["recommendation"]}</div>', unsafe_allow_html=True)
    
    # Unit economics
    st.markdown('<div class="sub-header">Unit Economics</div>', unsafe_allow_html=True)
    
    col4, col5 = st.columns(2)
    
    with col4:
        st.markdown(f"""
        <div class="metric-card">
            <p class="metric-label">Original Product</p>
            <p class="metric-value">Original Profit: ${results["metrics"]["original_profit_per_unit"]:.2f}/unit</p>
            <p class="metric-label">MSRP: ${results["metrics"]["regular_price"]:.2f}</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col5:
        profit_color = "#10B981" if results["metrics"]["salvage_profit_per_unit"] >= 0 else "#EF4444"
        st.markdown(f"""
        <div class="metric-card">
            <p class="metric-label">Salvaged Product</p>
            <p class="metric-value" style="color:{profit_color}">Salvage Profit: ${results["metrics"]["salvage_profit_per_unit"]:.2f}/unit</p>
            <p class="metric-label">Discounted Price: ${results["metrics"]["discounted_price"]:.2f} ({results["metrics"]["discount_percentage"]}% off)</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Comparison chart
    fig = go.Figure()
    
    # Scenario comparison
    fig.add_trace(go.Bar(
        x=['Complete Write-off', 'Salvage Operation'],
        y=[-results["financial"]["complete_writeoff_cost"], results["financial"]["salvage_profit"]],
        name='Financial Impact',
        marker_color=['#EF4444', '#10B981' if results["financial"]["salvage_profit"] >= 0 else '#EF4444']
    ))
    
    fig.update_layout(
        title='Financial Comparison: Write-off vs. Salvage',
        xaxis_title='Option',
        yaxis_title='Profit/Loss ($)',
        height=300,
        margin=dict(l=20, r=20, t=40, b=20),
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Create data for export
    if 'salvage_analysis_data' not in st.session_state:
        st.session_state.salvage_analysis_data = []
    
    salvage_data = {
        "SKU": results["sku"],
        "Affected Inventory": results["metrics"]["affected_inventory"],
        "Recovery Rate": results["metrics"]["recovery_rate"],
        "Total Rework Cost": results["financial"]["total_rework_cost"],
        "Salvage Revenue": results["financial"]["salvage_revenue"],
        "Write-off Loss": results["financial"]["write_off_loss"],
        "Salvage Profit/Loss": results["financial"]["salvage_profit"],
        "Complete Write-off Cost": results["financial"]["complete_writeoff_cost"],
        "ROI Percentage": results["financial"]["roi_percent"],
        "Regular Price": results["metrics"]["regular_price"],
        "Discounted Price": results["metrics"]["discounted_price"],
        "Discount Percentage": results["metrics"]["discount_percentage"],
        "Recommendation": results["recommendation"],
        "Analysis Date": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    
    st.session_state.salvage_analysis_data.append(salvage_data)
    
    # Export button
    if len(st.session_state.salvage_analysis_data) > 0:
        export_df = pd.DataFrame(st.session_state.salvage_analysis_data)
        st.markdown(get_csv_download_link(export_df, "salvage_analysis_results.csv", "Export Salvage Analysis to CSV"), unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

def chat_with_ai(results, issue_description, chat_history=None):
    """Initialize or continue chat with AI about quality issues"""
    if chat_history is None:
        chat_history = []
        
        # Generate initial AI analysis
        system_prompt = f"""
        You are a Quality Management expert for a product line. You analyze quality issues, provide insights on cost-benefit analyses, and suggest solutions.
        
        Product details:
        - SKU: {results["sku"]}
        - Type: {results["product_type"]}
        - MSRP: ${results["msrp"]:.2f}
        - Issue: {issue_description}
        
        Metrics:
        - Return Rate (30 days): {results["current_metrics"]["return_rate_30d"]:.2f}%
        - Current Profit Margin: {results["current_metrics"]["margin_percentage"]:.2f}%
        - Future Profit Margin: {results["future_metrics"]["margin_percentage"]:.2f}%
        - Monthly Return Cost: ${results["financial_impact"]["monthly_return_cost"]:.2f}
        - Estimated Savings: ${results["financial_impact"]["estimated_monthly_savings"]:.2f}/month
        - Payback Period: {results["financial_impact"]["payback_months"]:.1f} months
        
        Recommendation: {results["recommendation"]}
        
        Your task is to analyze this quality issue and provide expert insights and recommendations.
        Focus on the financial impact, profitability, and strategic considerations.
        Give specific advice about how to implement the fix effectively or alternatives if appropriate.
        """
        
        initial_analysis = get_ai_analysis(
            client,
            system_prompt,
            "Based on the product information and quality metrics, provide your initial analysis of the issue and suggested next steps. Be specific about potential fixes and quality improvements.",
            model="gpt-4o"
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
    Create an empty sales data template
    """
    # Create minimal template with columns
    data = {
        "date": [],
        "sales": [],
        "channel": [],
        "product_category": [],
    }
    return pd.DataFrame(data)

def analyze_sales_trends(df):
    """Analyze sales trends from the dataframe"""
    if df.empty:
        return None
    
    # Group by date and calculate daily sales
    daily_sales = df.groupby("date")["sales"].sum().reset_index()
    
    # Calculate 7-day moving average
    daily_sales["7d_moving_avg"] = daily_sales["sales"].rolling(window=7).mean()
    
    # Calculate month-to-date sales
    daily_sales["month"] = daily_sales["date"].dt.month
    daily_sales["year"] = daily_sales["date"].dt.year
    monthly_sales = daily_sales.groupby(["year", "month"])["sales"].sum().reset_index()
    
    # Calculate sales by channel
    channel_sales = df.groupby("channel")["sales"].sum().reset_index()
    
    # Calculate sales by product category
    category_sales = df.groupby("product_category")["sales"].sum().reset_index()
    
    return {
        "daily_sales": daily_sales,
        "monthly_sales": monthly_sales,
        "channel_sales": channel_sales,
        "category_sales": category_sales
    }

# --- INVENTORY MANAGEMENT FUNCTIONS ---

def load_inventory_data():
    """
    Create an empty inventory data template
    """
    # Create minimal template with columns
    data = {
        "sku": [],
        "product_name": [],
        "category": [],
        "in_stock": [],
        "reorder_point": [],
        "lead_time_days": [],
        "cost": [],
        "last_ordered": []
    }
    return pd.DataFrame(data)

def analyze_inventory_status(df):
    """Analyze inventory status from the dataframe"""
    if df.empty:
        return None
    
    # Calculate inventory metrics
    # Add status column if it doesn't exist
    if "status" not in df.columns:
        df["status"] = df.apply(lambda x: "Low Stock" if x["in_stock"] <= x["reorder_point"] else "OK", axis=1)
        
    low_stock_items = df[df["status"] == "Low Stock"]
    stock_out_risk = len(low_stock_items) / len(df) * 100 if len(df) > 0 else 0
    
    # Calculate inventory value
    df["inventory_value"] = df["in_stock"] * df["cost"]
    total_inventory_value = df["inventory_value"].sum()
    
    # Calculate inventory by category
    category_inventory = df.groupby("category")["inventory_value"].sum().reset_index()
    
    return {
        "low_stock_items": low_stock_items,
        "stock_out_risk": stock_out_risk,
        "total_inventory_value": total_inventory_value,
        "category_inventory": category_inventory
    }

# --- DASHBOARD FUNCTIONS ---

def load_dashboard_data():
    """
    Create an empty dashboard template
    """
    # Return a simple placeholder with empty data
    return {
        "monthly_revenue": 0,
        "monthly_profit": 0,
        "inventory_turnover": 0,
        "stock_out_risk": 0,
        "low_stock_items": pd.DataFrame(),
        "sales_trend": pd.DataFrame({"date": [], "sales": [], "7d_moving_avg": []}),
        "sales_by_channel": pd.DataFrame({"channel": [], "sales": []}),
        "sales_by_category": pd.DataFrame({"product_category": [], "sales": []}),
        "inventory_by_category": pd.DataFrame({"category": [], "inventory_value": []})
    }

# --- PAGE DISPLAY FUNCTIONS ---

def display_quality_manager():
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
                st.markdown('<div class="sub-header">Product Information</div>', unsafe_allow_html=True)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # Required inputs
                    st.markdown('<span class="required-field">SKU</span>', unsafe_allow_html=True)
                    sku = st.text_input(
                        "SKU",
                        label_visibility="collapsed",
                        help="Required: Product SKU"
                    )
                    
                    st.markdown('<span class="required-field">Product Type</span>', unsafe_allow_html=True)
                    product_type = st.selectbox(
                        "Product Type",
                        ["B2C", "B2B", "Both"],
                        label_visibility="collapsed",
                        help="Required: Distribution channel for this product"
                    )
                    
                    st.markdown('<span class="required-field">Total Sales Last 30 Days</span>', unsafe_allow_html=True)
                    sales_30d = st.number_input(
                        "Total Sales Last 30 Days",
                        min_value=0.0,
                        label_visibility="collapsed",
                        help="Required: Units sold in the last 30 days"
                    )
                    
                    st.markdown('<span class="required-field">Total Returns Last 30 Days</span>', unsafe_allow_html=True)
                    returns_30d = st.number_input(
                        "Total Returns Last 30 Days",
                        min_value=0.0,
                        label_visibility="collapsed",
                        help="Required: Units returned in the last 30 days"
                    )
                    
                with col2:
                    st.markdown('<span class="required-field">MSRP (Retail Price)</span>', unsafe_allow_html=True)
                    msrp = st.number_input(
                        "MSRP (Retail Price)",
                        min_value=0.0,
                        label_visibility="collapsed",
                        help="Required: Manufacturer's Suggested Retail Price"
                    )
                    
                    st.markdown('<span class="required-field">Current Unit Cost (Landed Cost)</span>', unsafe_allow_html=True)
                    current_unit_cost = st.number_input(
                        "Current Unit Cost (Landed Cost)",
                        min_value=0.0,
                        label_visibility="collapsed",
                        help="Required: Current per-unit cost to produce"
                    )
                    
                    st.markdown('<span class="required-field">Fix Cost Upfront</span>', unsafe_allow_html=True)
                    fix_cost_upfront = st.number_input(
                        "Fix Cost Upfront",
                        min_value=0.0,
                        label_visibility="collapsed",
                        help="Required: One-time cost to implement the quality fix"
                    )
                    
                    st.markdown('<span class="required-field">Additional Cost Per Unit</span>', unsafe_allow_html=True)
                    fix_cost_per_unit = st.number_input(
                        "Additional Cost Per Unit",
                        min_value=0.0,
                        label_visibility="collapsed",
                        help="Required: Additional cost per unit after implementing the fix"
                    )
                
                # Description of issue
                st.markdown('<span class="required-field">Description of Quality Issue</span>', unsafe_allow_html=True)
                issue_description = st.text_area(
                    "Description of Quality Issue",
                    label_visibility="collapsed",
                    help="Required: Detailed description of the quality problem"
                )
                
                # Expandable section for optional metrics
                with st.expander("Additional Metrics (Optional)"):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        asin = st.text_input("ASIN", help="Amazon Standard Identification Number")
                        ncx_rate = st.number_input(
                            "Negative Customer Experience Rate (%)", 
                            min_value=0.0, 
                            max_value=100.0,
                            help="Composite of bad reviews and returns divided by total sales"
                        )
                        sales_365d = st.number_input(
                            "Total Sales Last 365 Days", 
                            min_value=0.0,
                            help="Units sold in the last 365 days"
                        )
                        returns_365d = st.number_input(
                            "Total Returns Last 365 Days", 
                            min_value=0.0,
                            help="Units returned in the last 365 days"
                        )
                    
                    with col2:
                        star_rating = st.number_input(
                            "Current Star Rating", 
                            min_value=1.0, 
                            max_value=5.0,
                            help="Current average star rating on Amazon"
                        )
                        total_reviews = st.number_input(
                            "Total Reviews on Amazon", 
                            min_value=0,
                            help="Total number of reviews on Amazon"
                        )
                        fba_fee = st.number_input(
                            "FBA Fee", 
                            min_value=0.0,
                            help="Amazon FBA fee per unit"
                        )
                
                # Form submission
                submit_button = st.form_submit_button("Analyze Quality Issue")
                
                if submit_button:
                    # Validate required fields
                    if not all([sku, sales_30d > 0, current_unit_cost > 0, msrp > 0, issue_description]):
                        st.error("Please fill in all required fields marked with *")
                    else:
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
                            msrp=msrp,  # Added MSRP
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
                            issue_description
                        )
                        
                        # Rerun to show results
                        st.experimental_rerun()
        
        # Display results if available
        if st.session_state.analysis_submitted and st.session_state.quality_analysis_results:
            # Display analysis results
            display_quality_issue_results(st.session_state.quality_analysis_results)
            
            # Display AI chat interface
            st.markdown('<div class="sub-header">AI Quality Consultant</div>', unsafe_allow_html=True)
            
            # Chat container
            chat_container = st.container()
            
            with chat_container:
                # Display chat history
                for message in st.session_state.chat_history:
                    if message["role"] == "user":
                        st.markdown(f'<div class="chat-message user-message">{message["content"]}</div>', unsafe_allow_html=True)
                    else:
                        st.markdown(f'<div class="chat-message ai-message">{message["content"]}</div>', unsafe_allow_html=True)
            
            # Input for new messages
            user_input = st.text_input("Ask about the quality issue or potential solutions:", key="user_message")
            
            if st.button("Send", key="send_button"):
                if user_input:
                    # Add user message to chat history
                    st.session_state.chat_history.append({
                        "role": "user",
                        "content": user_input
                    })
                    
                    # Get AI response
                    system_prompt = f"""
                    You are a Quality Management expert for a product line. You analyze quality issues, provide insights, and suggest solutions.
                    
                    Product details:
                    - SKU: {st.session_state.quality_analysis_results["sku"]}
                    - Type: {st.session_state.quality_analysis_results["product_type"]}
                    - MSRP: ${st.session_state.quality_analysis_results["msrp"]:.2f}
                    - Issue: {st.session_state.quality_analysis_results["issue_description"]}
                    
                    Metrics:
                    - Return Rate (30 days): {st.session_state.quality_analysis_results["current_metrics"]["return_rate_30d"]:.2f}%
                    - Current Profit Margin: {st.session_state.quality_analysis_results["current_metrics"]["margin_percentage"]:.2f}%
                    - Future Profit Margin: {st.session_state.quality_analysis_results["future_metrics"]["margin_percentage"]:.2f}%
                    - Monthly Return Cost: ${st.session_state.quality_analysis_results["financial_impact"]["monthly_return_cost"]:.2f}
                    - Estimated Savings: ${st.session_state.quality_analysis_results["financial_impact"]["estimated_monthly_savings"]:.2f}/month
                    - Payback Period: {st.session_state.quality_analysis_results["financial_impact"]["payback_months"]:.1f} months
                    
                    Recommendation: {st.session_state.quality_analysis_results["recommendation"]}
                    """
                    
                    # Get the full chat history for context
                    messages_history = []
                    for msg in st.session_state.chat_history:
                        messages_history.append({"role": msg["role"], "content": msg["content"]})
                    
                    ai_response = get_ai_analysis(
                        client,
                        system_prompt,
                        user_input,
                        model="gpt-4o",
                        messages=messages_history[:-1]  # Exclude the latest user message as it's passed separately
                    )
                    
                    # Add AI response to chat history
                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": ai_response
                    })
                    
                    # Clear input and rerun to show updated chat
                    st.experimental_rerun()
            
            # Reset button
            if st.button("Start New Analysis", key="reset_button"):
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
                st.markdown('<div class="sub-header">Salvage Operation Details</div>', unsafe_allow_html=True)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown('<span class="required-field">SKU</span>', unsafe_allow_html=True)
                    sku = st.text_input(
                        "SKU",
                        key="salvage_sku",
                        label_visibility="collapsed",
                        help="Required: Product SKU"
                    )
                    
                    st.markdown('<span class="required-field">Affected Inventory Units</span>', unsafe_allow_html=True)
                    affected_inventory = st.number_input(
                        "Affected Inventory Units",
                        min_value=1,
                        key="affected_inventory",
                        label_visibility="collapsed",
                        help="Required: Number of units affected by quality issue"
                    )
                    
                    st.markdown('<span class="required-field">Current Unit Cost (Landed Cost)</span>', unsafe_allow_html=True)
                    current_unit_cost = st.number_input(
                        "Current Unit Cost (Landed Cost)",
                        min_value=0.01,
                        key="salvage_unit_cost",
                        label_visibility="collapsed",
                        help="Required: Current per-unit cost to produce"
                    )
                    
                    st.markdown('<span class="required-field">MSRP (Retail Price)</span>', unsafe_allow_html=True)
                    msrp = st.number_input(
                        "MSRP (Retail Price)",
                        min_value=0.01,
                        key="salvage_msrp",
                        label_visibility="collapsed",
                        help="Required: Original retail price before discount"
                    )
                
                with col2:
                    st.markdown('<span class="required-field">Rework Setup Cost</span>', unsafe_allow_html=True)
                    rework_cost_upfront = st.number_input(
                        "Rework Setup Cost",
                        min_value=0.0,
                        key="rework_upfront",
                        label_visibility="collapsed",
                        help="Required: One-time cost to set up the rework operation"
                    )
                    
                    st.markdown('<span class="required-field">Rework Cost Per Unit</span>', unsafe_allow_html=True)
                    rework_cost_per_unit = st.number_input(
                        "Rework Cost Per Unit",
                        min_value=0.0,
                        key="rework_per_unit",
                        label_visibility="collapsed",
                        help="Required: Cost to rework each affected unit"
                    )
                    
                    st.markdown('<span class="required-field">Expected Recovery Percentage</span>', unsafe_allow_html=True)
                    expected_recovery_pct = st.slider(
                        "Expected Recovery Percentage",
                        min_value=0.0,
                        max_value=100.0,
                        value=80.0,
                        label_visibility="collapsed",
                        help="Required: Percentage of affected units expected to be successfully reworked"
                    )
                    
                    st.markdown('<span class="required-field">Expected Discount Percentage</span>', unsafe_allow_html=True)
                    expected_discount_pct = st.slider(
                        "Expected Discount Percentage",
                        min_value=0.0,
                        max_value=100.0,
                        value=30.0,
                        label_visibility="collapsed",
                        help="Required: Discount percentage for selling reworked units"
                    )
                
                # Form submission
                submit_button = st.form_submit_button("Analyze Salvage Operation")
                
                if submit_button:
                    # Validate required fields
                    if not all([sku, affected_inventory > 0, current_unit_cost > 0, msrp > 0]):
                        st.error("Please fill in all required fields marked with *")
                    else:
                        # Perform analysis
                        results = analyze_salvage_operation(
                            sku=sku,
                            affected_inventory=affected_inventory,
                            current_unit_cost=current_unit_cost,
                            rework_cost_upfront=rework_cost_upfront,
                            rework_cost_per_unit=rework_cost_per_unit,
                            expected_recovery_pct=expected_recovery_pct,
                            expected_discount_pct=expected_discount_pct,
                            msrp=msrp  # Added MSRP
                        )
                        
                        # Store results in session state
                        st.session_state.salvage_results = results
                        st.session_state.salvage_submitted = True
                        
                        # Rerun to show results
                        st.experimental_rerun()
        
        # Display results if available
        if st.session_state.salvage_submitted and st.session_state.salvage_results:
            # Display analysis results
            display_salvage_results(st.session_state.salvage_results)
            
            # Scenario modeling
            st.markdown('<div class="sub-header">Scenario Modeling</div>', unsafe_allow_html=True)
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                recovery_adjustment = st.slider(
                    "Adjust Recovery Rate",
                    min_value=0.0,
                    max_value=100.0,
                    value=float(st.session_state.salvage_results["metrics"]["recovery_rate"]),
                    step=5.0,
                    key="recovery_slider"
                )
            
            with col2:
                discount_adjustment = st.slider(
                    "Adjust Discount Percentage",
                    min_value=0.0,
                    max_value=100.0,
                    value=float(st.session_state.salvage_results["metrics"]["discount_percentage"]),
                    step=5.0,
                    key="discount_slider"
                )
            
            with col3:
                rework_adjustment = st.slider(
                    "Adjust Per-Unit Rework Cost",
                    min_value=0.0,
                    max_value=float(st.session_state.salvage_results["financial"]["total_rework_cost"]/st.session_state.salvage_results["metrics"]["affected_inventory"])*2,
                    value=float(st.session_state.salvage_results["financial"]["total_rework_cost"]/st.session_state.salvage_results["metrics"]["affected_inventory"] - st.session_state.salvage_results["financial"]["total_rework_cost"]/st.session_state.salvage_results["metrics"]["affected_inventory"]/10),
                    step=0.5,
                    key="rework_slider"
                )
            
            if st.button("Run Scenario", key="scenario_button"):
                # Calculate new scenario
                new_results = analyze_salvage_operation(
                    sku=st.session_state.salvage_results["sku"],
                    affected_inventory=st.session_state.salvage_results["metrics"]["affected_inventory"],
                    current_unit_cost=st.session_state.salvage_results["financial"]["complete_writeoff_cost"]/st.session_state.salvage_results["metrics"]["affected_inventory"],
                    rework_cost_upfront=st.session_state.salvage_results["financial"]["total_rework_cost"] - (st.session_state.salvage_results["metrics"]["affected_inventory"] * (st.session_state.salvage_results["financial"]["total_rework_cost"]/st.session_state.salvage_results["metrics"]["affected_inventory"] - st.session_state.salvage_results["financial"]["total_rework_cost"]/st.session_state.salvage_results["metrics"]["affected_inventory"]/10)),
                    rework_cost_per_unit=rework_adjustment,
                    expected_recovery_pct=recovery_adjustment,
                    expected_discount_pct=discount_adjustment,
                    msrp=st.session_state.salvage_results["metrics"]["regular_price"]  # Use original MSRP
                )
                
                # Compare current vs new scenario
                st.markdown('<div class="sub-header">Scenario Comparison</div>', unsafe_allow_html=True)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown('<div class="metric-label">Current Scenario</div>', unsafe_allow_html=True)
                    st.markdown(f'<div class="metric-value">Net Profit/Loss: ${st.session_state.salvage_results["financial"]["salvage_profit"]:.2f}</div>', unsafe_allow_html=True)
                    st.markdown(f'<div class="{st.session_state.salvage_results["recommendation_class"]}">{st.session_state.salvage_results["recommendation"]}</div>', unsafe_allow_html=True)
                
                with col2:
                    st.markdown('<div class="metric-label">New Scenario</div>', unsafe_allow_html=True)
                    st.markdown(f'<div class="metric-value">Net Profit/Loss: ${new_results["financial"]["salvage_profit"]:.2f}</div>', unsafe_allow_html=True)
                    st.markdown(f'<div class="{new_results["recommendation_class"]}">{new_results["recommendation"]}</div>', unsafe_allow_html=True)
                
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
                    marker_color='#3B82F6'
                ))
                
                fig.add_trace(go.Bar(
                    x=metrics,
                    y=new_values,
                    name='New Scenario',
                    marker_color='#10B981'
                ))
                
                fig.update_layout(
                    title='Scenario Comparison',
                    xaxis_title='Metric',
                    yaxis_title='Amount ($)',
                    barmode='group',
                    height=400,
                    margin=dict(l=20, r=20, t=40, b=20),
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            # Reset button
            if st.button("Start New Analysis", key="salvage_reset_button"):
                st.session_state.salvage_results = None
                st.session_state.salvage_submitted = False
                st.experimental_rerun()

def display_dashboard():
    st.markdown('<div class="main-header">Business Overview Dashboard</div>', unsafe_allow_html=True)
    
    # Data upload option
    st.markdown('<div class="data-warning">No data available. Please upload or enter data in the Sales Analysis and Inventory Management sections.</div>', unsafe_allow_html=True)
    
    # Placeholder for dashboard widgets
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="metric-label">Monthly Revenue</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="metric-value">$0.00</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="metric-label">Monthly Profit</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="metric-value">$0.00</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col3:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="metric-label">Inventory Turnover</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="metric-value">0.00x</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col4:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="metric-label">Stock-Out Risk</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="metric-value">0.0%</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Placeholder charts
    st.markdown('<div class="sub-header">Sales Trend</div>', unsafe_allow_html=True)
    
    # Create empty chart
    placeholder_fig = go.Figure()
    placeholder_fig.add_trace(go.Scatter(x=[], y=[], mode='lines', name='Sales'))
    placeholder_fig.update_layout(
        xaxis_title="Date",
        yaxis_title="Sales ($)",
        height=400,
        margin=dict(l=20, r=20, t=20, b=20),
    )
    
    st.plotly_chart(placeholder_fig, use_container_width=True)
    
    # Sample quality issues
    st.markdown('<div class="sub-header">Recent Quality Issues</div>', unsafe_allow_html=True)
    
    if 'quality_analysis_data' in st.session_state and len(st.session_state.quality_analysis_data) > 0:
        # Display recent quality analyses
        quality_df = pd.DataFrame(st.session_state.quality_analysis_data)
        # Sort by date descending
        if 'Analysis Date' in quality_df.columns:
            quality_df = quality_df.sort_values('Analysis Date', ascending=False)
        
        st.dataframe(quality_df[['SKU', 'MSRP', 'Current Return Rate', 'Annual Profit Improvement', 'Recommendation', 'Analysis Date']], 
                   use_container_width=True, 
                   hide_index=True)
    else:
        st.markdown('<div class="data-warning">No quality issues have been analyzed yet. Use the Quality Manager to analyze quality issues.</div>', unsafe_allow_html=True)

def display_sales_analysis():
    st.markdown('<div class="main-header">Sales Analysis</div>', unsafe_allow_html=True)
    
    # Initialize sales data if not exist
    if 'sales_data' not in st.session_state:
        st.session_state.sales_data = load_sales_data()
    
    # Upload sales data
    st.markdown('<div class="sub-header">Upload Sales Data</div>', unsafe_allow_html=True)
    
    upload_col1, upload_col2 = st.columns([2, 1])
    
    with upload_col1:
        uploaded_file = st.file_uploader("Upload sales data CSV (Columns: date, sales, channel, product_category)", type="csv")
        if uploaded_file is not None:
            try:
                sales_df = pd.read_csv(uploaded_file)
                # Convert date column to datetime
                if 'date' in sales_df.columns:
                    sales_df['date'] = pd.to_datetime(sales_df['date'])
                # Store in session state
                st.session_state.sales_data = sales_df
                st.success("Sales data uploaded successfully!")
            except Exception as e:
                st.error(f"Error uploading file: {e}")
    
    with upload_col2:
        if st.button("Add Example Data", key="add_example_sales"):
            # Create example data
            example_dates = pd.date_range(start="2023-01-01", periods=90, freq="D")
            example_sales = np.random.randint(100, 1000, 90)
            example_channels = np.random.choice(["Amazon", "Website", "Retail"], 90)
            example_categories = np.random.choice(["Health", "Wellness", "Medical", "Fitness"], 90)
            
            example_df = pd.DataFrame({
                "date": example_dates,
                "sales": example_sales,
                "channel": example_channels,
                "product_category": example_categories
            })
            
            st.session_state.sales_data = example_df
            st.success("Example sales data added successfully!")
    
    # Manual data entry option
    with st.expander("Manual Data Entry"):
        st.markdown("Add sales data manually:")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            entry_date = st.date_input("Date", value=datetime.now().date())
        
        with col2:
            entry_sales = st.number_input("Sales Amount", min_value=0)
        
        with col3:
            entry_channel = st.selectbox("Channel", options=["Amazon", "Website", "Retail", "Wholesale", "Other"])
        
        with col4:
            entry_category = st.selectbox("Product Category", options=["Health", "Wellness", "Medical", "Fitness", "Other"])
        
        if st.button("Add Entry"):
            new_row = pd.DataFrame({
                "date": [pd.Timestamp(entry_date)],
                "sales": [entry_sales],
                "channel": [entry_channel],
                "product_category": [entry_category]
            })
            
            st.session_state.sales_data = pd.concat([st.session_state.sales_data, new_row], ignore_index=True)
            st.success("Sales entry added successfully!")
    
    # Display and analyze data if available
    if not st.session_state.sales_data.empty:
        # Data preview
        st.markdown('<div class="sub-header">Sales Data Preview</div>', unsafe_allow_html=True)
        st.dataframe(st.session_state.sales_data.head(10), use_container_width=True)
        
        # Export data
        st.markdown(get_csv_download_link(st.session_state.sales_data, "sales_data.csv", "Export Sales Data to CSV"), unsafe_allow_html=True)
        
        # Date filter
        st.markdown('<div class="sub-header">Analysis Period</div>', unsafe_allow_html=True)
        
        date_col1, date_col2 = st.columns(2)
        
        with date_col1:
            min_date = st.session_state.sales_data["date"].min().date() if not st.session_state.sales_data.empty else datetime.now().date()
            max_date = st.session_state.sales_data["date"].max().date() if not st.session_state.sales_data.empty else datetime.now().date()
            
            start_date = st.date_input(
                "Start Date",
                min_date,
                min_value=min_date,
                max_value=max_date
            )
        
        with date_col2:
            end_date = st.date_input(
                "End Date",
                max_date,
                min_value=min_date,
                max_value=max_date
            )
        
        # Filter data based on date range
        filtered_df = st.session_state.sales_data[
            (st.session_state.sales_data["date"].dt.date >= start_date) & 
            (st.session_state.sales_data["date"].dt.date <= end_date)
        ]
        
        # Channel and category filters
        filter_col1, filter_col2 = st.columns(2)
        
        with filter_col1:
            available_channels = sorted(st.session_state.sales_data["channel"].unique())
            selected_channels = st.multiselect(
                "Filter by Channel",
                options=available_channels,
                default=available_channels
            )
        
        with filter_col2:
            available_categories = sorted(st.session_state.sales_data["product_category"].unique())
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
            analysis_results = analyze_sales_trends(filtered_df)
            
            # Display summary metrics
            st.markdown('<div class="sub-header">Sales Summary</div>', unsafe_allow_html=True)
            
            summary_col1, summary_col2, summary_col3 = st.columns(3)
            
            with summary_col1:
                total_sales = filtered_df["sales"].sum()
                st.markdown('<div class="card">', unsafe_allow_html=True)
                st.markdown('<div class="metric-label">Total Sales</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="metric-value">${total_sales:,.2f}</div>', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
            
            with summary_col2:
                avg_daily_sales = filtered_df.groupby("date")["sales"].sum().mean()
                st.markdown('<div class="card">', unsafe_allow_html=True)
                st.markdown('<div class="metric-label">Average Daily Sales</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="metric-value">${avg_daily_sales:,.2f}</div>', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
            
            with summary_col3:
                sales_growth = 0
                # Calculate growth if we have enough data
                if len(analysis_results["monthly_sales"]) > 1:
                    first_month = analysis_results["monthly_sales"].iloc[0]["sales"]
                    last_month = analysis_results["monthly_sales"].iloc[-1]["sales"]
                    if first_month > 0:
                        sales_growth = (last_month - first_month) / first_month * 100
                
                st.markdown('<div class="card">', unsafe_allow_html=True)
                st.markdown('<div class="metric-label">Sales Growth</div>', unsafe_allow_html=True)
                growth_color = "#10B981" if sales_growth >= 0 else "#EF4444"
                st.markdown(f'<div class="metric-value" style="color:{growth_color}">{sales_growth:.1f}%</div>', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
            
            # Sales trend chart
            st.markdown('<div class="sub-header">Sales Trend</div>', unsafe_allow_html=True)
            
            trend_fig = px.line(
                analysis_results["daily_sales"], 
                x="date", 
                y=["sales", "7d_moving_avg"],
                labels={"value": "Amount ($)", "variable": "Metric"},
                color_discrete_map={"sales": "#3B82F6", "7d_moving_avg": "#10B981"}
            )
            
            trend_fig.update_layout(
                xaxis_title="Date",
                yaxis_title="Sales ($)",
                legend_title="",
                height=400,
                margin=dict(l=20, r=20, t=20, b=20),
            )
            
            st.plotly_chart(trend_fig, use_container_width=True)
            
            # Sales breakdown
            breakdown_col1, breakdown_col2 = st.columns(2)
            
            with breakdown_col1:
                st.markdown('<div class="sub-header">Sales by Channel</div>', unsafe_allow_html=True)
                
                if not analysis_results["channel_sales"].empty:
                    channel_fig = px.pie(
                        analysis_results["channel_sales"],
                        values="sales",
                        names="channel",
                        hole=0.4
                    )
                    
                    channel_fig.update_layout(
                        height=300,
                        margin=dict(l=20, r=20, t=20, b=20),
                    )
                    
                    st.plotly_chart(channel_fig, use_container_width=True)
            
            with breakdown_col2:
                st.markdown('<div class="sub-header">Sales by Category</div>', unsafe_allow_html=True)
                
                if not analysis_results["category_sales"].empty:
                    category_fig = px.bar(
                        analysis_results["category_sales"],
                        x="product_category",
                        y="sales",
                        color="product_category",
                        labels={"product_category": "Category", "sales": "Sales ($)"}
                    )
                    
                    category_fig.update_layout(
                        xaxis_title="",
                        showlegend=False,
                        height=300,
                        margin=dict(l=20, r=20, t=20, b=20),
                    )
                    
                    st.plotly_chart(category_fig, use_container_width=True)
            
            # Raw data table
            with st.expander("Show Raw Data"):
                st.dataframe(
                    filtered_df.sort_values("date", ascending=False),
                    hide_index=True,
                    use_container_width=True
                )
        else:
            st.warning("No data available for the selected filters. Please adjust your selection.")
    else:
        st.info("No sales data available. Please upload data or add entries manually.")

def display_inventory_management():
    st.markdown('<div class="main-header">Inventory Management</div>', unsafe_allow_html=True)
    
    # Initialize inventory data if not exist
    if 'inventory_data' not in st.session_state:
        st.session_state.inventory_data = load_inventory_data()
    
    # Upload inventory data
    st.markdown('<div class="sub-header">Upload Inventory Data</div>', unsafe_allow_html=True)
    
    upload_col1, upload_col2 = st.columns([2, 1])
    
    with upload_col1:
        uploaded_file = st.file_uploader("Upload inventory data CSV (Columns: sku, product_name, category, in_stock, reorder_point, lead_time_days, cost, last_ordered)", type="csv")
        if uploaded_file is not None:
            try:
                inventory_df = pd.read_csv(uploaded_file)
                # Convert date column to datetime if exists
                if 'last_ordered' in inventory_df.columns:
                    inventory_df['last_ordered'] = pd.to_datetime(inventory_df['last_ordered'])
                # Store in session state
                st.session_state.inventory_data = inventory_df
                st.success("Inventory data uploaded successfully!")
            except Exception as e:
                st.error(f"Error uploading file: {e}")
    
    with upload_col2:
        if st.button("Add Example Data", key="add_example_inventory"):
            # Create example data
            skus = [f"VH-{i:04d}" for i in range(1, 11)]
            product_names = [f"Product {i}" for i in range(1, 11)]
            categories = np.random.choice(["Health", "Wellness", "Medical", "Fitness"], 10)
            in_stock = np.random.randint(0, 100, 10)
            reorder_points = np.random.randint(10, 30, 10)
            lead_times = np.random.randint(7, 45, 10)
            costs = np.random.uniform(10, 100, 10).round(2)
            last_ordered = pd.date_range(end=pd.Timestamp.now(), periods=10, freq="D")
            
            example_df = pd.DataFrame({
                "sku": skus,
                "product_name": product_names,
                "category": categories,
                "in_stock": in_stock,
                "reorder_point": reorder_points,
                "lead_time_days": lead_times,
                "cost": costs,
                "last_ordered": last_ordered
            })
            
            # Add status column
            example_df["status"] = example_df.apply(lambda x: "Low Stock" if x["in_stock"] <= x["reorder_point"] else "OK", axis=1)
            
            st.session_state.inventory_data = example_df
            st.success("Example inventory data added successfully!")
    
    # Manual data entry option
    with st.expander("Manual Data Entry"):
        st.markdown("Add inventory data manually:")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            entry_sku = st.text_input("SKU")
            entry_product_name = st.text_input("Product Name")
            entry_category = st.selectbox("Category", options=["Health", "Wellness", "Medical", "Fitness", "Other"])
        
        with col2:
            entry_in_stock = st.number_input("In Stock", min_value=0)
            entry_reorder_point = st.number_input("Reorder Point", min_value=0)
            entry_lead_time = st.number_input("Lead Time (Days)", min_value=1)
        
        with col3:
            entry_cost = st.number_input("Cost", min_value=0.01)
            entry_last_ordered = st.date_input("Last Ordered", value=datetime.now().date())
        
        if st.button("Add Inventory Item"):
            # Check if required fields are provided
            if entry_sku and entry_product_name:
                new_row = pd.DataFrame({
                    "sku": [entry_sku],
                    "product_name": [entry_product_name],
                    "category": [entry_category],
                    "in_stock": [entry_in_stock],
                    "reorder_point": [entry_reorder_point],
                    "lead_time_days": [entry_lead_time],
                    "cost": [entry_cost],
                    "last_ordered": [pd.Timestamp(entry_last_ordered)]
                })
                
                # Add status column
                new_row["status"] = new_row.apply(lambda x: "Low Stock" if x["in_stock"] <= x["reorder_point"] else "OK", axis=1)
                
                st.session_state.inventory_data = pd.concat([st.session_state.inventory_data, new_row], ignore_index=True)
                st.success("Inventory item added successfully!")
            else:
                st.error("SKU and Product Name are required.")
    
    # Display and analyze data if available
    if not st.session_state.inventory_data.empty:
        # Data preview
        st.markdown('<div class="sub-header">Inventory Data Preview</div>', unsafe_allow_html=True)
        st.dataframe(st.session_state.inventory_data.head(10), use_container_width=True)
        
        # Export data
        st.markdown(get_csv_download_link(st.session_state.inventory_data, "inventory_data.csv", "Export Inventory Data to CSV"), unsafe_allow_html=True)
        
        # Filter options
        st.markdown('<div class="sub-header">Filter Inventory</div>', unsafe_allow_html=True)
        
        filter_col1, filter_col2, filter_col3 = st.columns(3)
        
        with filter_col1:
            if 'category' in st.session_state.inventory_data.columns:
                available_categories = sorted(st.session_state.inventory_data["category"].unique())
                selected_categories = st.multiselect(
                    "Filter by Category",
                    options=available_categories,
                    default=available_categories
                )
            else:
                selected_categories = []
        
        with filter_col2:
            if 'status' in st.session_state.inventory_data.columns:
                available_statuses = sorted(st.session_state.inventory_data["status"].unique())
                selected_status = st.multiselect(
                    "Filter by Status",
                    options=available_statuses,
                    default=available_statuses
                )
            else:
                selected_status = []
        
        with filter_col3:
            sku_search = st.text_input("Search by SKU or Product Name")
        
        # Apply filters
        filtered_df = st.session_state.inventory_data.copy()
        
        # Ensure status column exists
        if 'status' not in filtered_df.columns and 'in_stock' in filtered_df.columns and 'reorder_point' in filtered_df.columns:
            filtered_df["status"] = filtered_df.apply(lambda x: "Low Stock" if x["in_stock"] <= x["reorder_point"] else "OK", axis=1)
        
        if selected_categories:
            filtered_df = filtered_df[filtered_df["category"].isin(selected_categories)]
        
        if selected_status:
            filtered_df = filtered_df[filtered_df["status"].isin(selected_status)]
        
        if sku_search:
            filtered_df = filtered_df[
                (filtered_df["sku"].str.contains(sku_search, case=False, na=False)) |
                (filtered_df["product_name"].str.contains(sku_search, case=False, na=False))
            ]
        
        # Check if we have data after filtering
        if not filtered_df.empty:
            # Analyze the filtered data
            analysis_results = analyze_inventory_status(filtered_df)
            
            # Display summary metrics
            st.markdown('<div class="sub-header">Inventory Summary</div>', unsafe_allow_html=True)
            
            summary_col1, summary_col2, summary_col3 = st.columns(3)
            
            with summary_col1:
                total_inventory = filtered_df["in_stock"].sum()
                st.markdown('<div class="card">', unsafe_allow_html=True)
                st.markdown('<div class="metric-label">Total Inventory Units</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="metric-value">{total_inventory:,}</div>', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
            
            with summary_col2:
                total_value = analysis_results["total_inventory_value"]
                st.markdown('<div class="card">', unsafe_allow_html=True)
                st.markdown('<div class="metric-label">Total Inventory Value</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="metric-value">${total_value:,.2f}</div>', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
            
            with summary_col3:
                low_stock_count = len(analysis_results["low_stock_items"])
                low_stock_percent = low_stock_count / len(filtered_df) * 100 if len(filtered_df) > 0 else 0
                st.markdown('<div class="card">', unsafe_allow_html=True)
                st.markdown('<div class="metric-label">Low Stock Items</div>', unsafe_allow_html=True)
                stock_color = "#EF4444" if low_stock_percent > 20 else "#10B981"
                st.markdown(f'<div class="metric-value" style="color:{stock_color}">{low_stock_count} ({low_stock_percent:.1f}%)</div>', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
            
            # Inventory by category chart
            st.markdown('<div class="sub-header">Inventory Value by Category</div>', unsafe_allow_html=True)
            
            if not analysis_results["category_inventory"].empty:
                category_fig = px.pie(
                    analysis_results["category_inventory"],
                    values="inventory_value",
                    names="category",
                    hole=0.4
                )
                
                category_fig.update_layout(
                    height=350,
                    margin=dict(l=20, r=20, t=20, b=20),
                )
                
                st.plotly_chart(category_fig, use_container_width=True)
            
            # Inventory Status Table
            st.markdown('<div class="sub-header">Inventory Status</div>', unsafe_allow_html=True)
            
            # Create a styled dataframe
            st.dataframe(
                filtered_df.sort_values(["status", "category", "sku"]),
                column_config={
                    "sku": "SKU",
                    "product_name": "Product Name",
                    "category": "Category",
                    "in_stock": "In Stock",
                    "reorder_point": "Reorder Point",
                    "lead_time_days": "Lead Time (Days)",
                    "cost": st.column_config.NumberColumn(
                        "Cost",
                        format="$%.2f"
                    ),
                    "last_ordered": "Last Ordered",
                    "status": st.column_config.TextColumn(
                        "Status",
                        help="Inventory status based on reorder point"
                    ),
                    "inventory_value": st.column_config.NumberColumn(
                        "Inventory Value",
                        format="$%.2f"
                    )
                },
                hide_index=True,
                use_container_width=True
            )
            
            # Low stock items alert
            if low_stock_count > 0:
                st.markdown('<div class="sub-header">Low Stock Alert</div>', unsafe_allow_html=True)
                
                low_stock_df = analysis_results["low_stock_items"].sort_values("in_stock")
                
                st.warning(f"{low_stock_count} items are below reorder point and may need to be reordered soon.")
                
                st.dataframe(
                    low_stock_df[["sku", "product_name", "in_stock", "reorder_point", "lead_time_days", "last_ordered"]],
                    hide_index=True,
                    use_container_width=True
                )
        else:
            st.warning("No inventory data available for the selected filters. Please adjust your selection.")
    else:
        st.info("No inventory data available. Please upload data or add entries manually.")
