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

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="QualityROI - Medical Device CBA Tool",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- THEME COLORS ---
VIVE_TEAL = "#41BEB6"
VIVE_TEAL_LIGHT = "#9BE4DF"
VIVE_TEAL_DARK = "#2D8C86"
VIVE_NAVY = "#1A3A4F"
VIVE_GREY_LIGHT = "#E5E7EB"
VIVE_GREY = "#6B7280"
VIVE_RED = "#E53E3E"
VIVE_GREEN = "#10B981"
VIVE_AMBER = "#FBBF24"
VIVE_BLUE = "#3B82F6"

# --- CUSTOM STYLING ---
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
    
    /* Buttons */
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
    
    /* Form elements */
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
    
    /* Headers */
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
    
    /* Cards */
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
    
    /* Metrics */
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
    
    /* Recommendations */
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
    
    /* Chat styling */
    .chat-container {{
        background-color: #F9FAFB;
        border-radius: 0.75rem;
        padding: 1rem;
        margin-bottom: 1rem;
        max-height: 400px;
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
    
    /* Header bar */
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
    
    /* Form sections */
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
        color: var(--vive-navy);
        margin-bottom: 1.5rem;
        text-align: center;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid var(--vive-teal);
    }}
    
    /* Charts */
    .chart-container {{
        background-color: white;
        border-radius: 0.5rem;
        padding: 1rem;
        box-shadow: 0 2px 6px rgba(0, 0, 0, 0.05);
        margin-bottom: 1.5rem;
    }}
</style>
""", unsafe_allow_html=True)

# --- SESSION STATE INITIALIZATION ---
if 'quality_analysis_results' not in st.session_state:
    st.session_state.quality_analysis_results = None
    
if 'analysis_submitted' not in st.session_state:
    st.session_state.analysis_submitted = False
    
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = None

# --- UTILITY FUNCTIONS ---

def vive_header():
    """Displays a Vive Health branded header"""
    st.markdown(f"""
    <div class="vive-health-header">
        <div class="vive-logo">VIVE HEALTH</div>
        <div>QualityROI Analysis Tool</div>
    </div>
    """, unsafe_allow_html=True)

def format_currency(value: float) -> str:
    """Format a value as currency with $ symbol"""
    return f"${value:,.2f}"

def format_percentage(value: float) -> str:
    """Format a value as percentage with % symbol"""
    return f"{value:.1f}%"

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
    Analyze quality issue and determine cost-effectiveness of fixes
    """
    try:
        # Calculate basic metrics with error handling
        return_rate_30d = (returns_30d / sales_30d) * 100 if sales_30d > 0 else 0
        
        # Include 365-day data if available
        return_rate_365d = None
        if sales_365d is not None and returns_365d is not None and sales_365d > 0:
            return_rate_365d = (returns_365d / sales_365d) * 100
        
        # Calculate financial impact
        monthly_return_cost = returns_30d * current_unit_cost
        
        # Calculate reduction factor based on return rate and risk level
        if risk_level == "High":
            base_reduction = 0.85  # 85% reduction for high risk
        elif risk_level == "Medium":
            base_reduction = 0.80  # 80% reduction for medium risk
        else:
            base_reduction = 0.75  # 75% reduction for low risk
            
        # Adjust based on return rate
        if return_rate_30d > 20:
            reduction_factor = min(0.95, base_reduction + 0.10)  # Up to 95% reduction for high return rates
        elif return_rate_30d > 10:
            reduction_factor = min(0.90, base_reduction + 0.05)  # Up to 90% reduction for medium return rates
        else:
            reduction_factor = base_reduction  # Base reduction for low return rates
            
        estimated_monthly_savings = monthly_return_cost * reduction_factor
        
        # Annual projections
        annual_return_cost = monthly_return_cost * 12
        annual_savings = estimated_monthly_savings * 12
        
        # Simple payback period (months) with error handling
        if estimated_monthly_savings > 0:
            payback_months = fix_cost_upfront / estimated_monthly_savings
        else:
            payback_months = float('inf')
        
        # Calculate 3-year ROI
        projected_sales_36m = sales_30d * 36  # 36 months projection
        total_investment = fix_cost_upfront + (fix_cost_per_unit * projected_sales_36m)
        total_savings = annual_savings * 3
        
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
            potential_review_improvement = int(total_reviews * negative_reviews_ratio * reduction_factor * 0.5)
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
                "fba_fee": fba_fee,
                "projected_sales_36m": projected_sales_36m,
                "total_investment": total_investment,
                "total_savings": total_savings
            },
            "recommendation": recommendation,
            "recommendation_class": recommendation_class,
            "brand_impact": brand_impact,
            "issue_description": issue_description,
            "reduction_factor": reduction_factor * 100,  # Convert to percentage
            "customer_impact_metrics": customer_impact_metrics,
            "medical_device_impact": medical_device_impact  # Use medical_device_impact instead of medical_impact
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

def display_quality_issue_results(results):
    """Display the quality issue analysis results in a visually appealing way"""
    st.markdown(f'<div class="card">', unsafe_allow_html=True)
    
    # Results header
    st.markdown(f'<div class="sub-header">Analysis Results for SKU: {results["sku"]}</div>', unsafe_allow_html=True)
    
    # Check for error in results
    if "error" in results:
        st.error(f"Analysis Error: {results['error']}")
        st.info("Please check your inputs and try again.")
        st.markdown('</div>', unsafe_allow_html=True)
        return
    
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
        
        st.markdown('<div class="metric-label">Medical Risk Level</div>', unsafe_allow_html=True)
        risk_level = results["medical_device_impact"]["risk_level"]  # Changed from medical_impact to medical_device_impact
        risk_color = VIVE_RED if risk_level == "High" else (VIVE_AMBER if risk_level == "Medium" else VIVE_GREEN)
        st.markdown(f'<div class="metric-value" style="color:{risk_color}">{risk_level}</div>', unsafe_allow_html=True)
        
        if results["medical_device_impact"]["regulatory_impact"] != "None":  # Changed from medical_impact to medical_device_impact
            st.markdown('<div class="metric-label">Regulatory Impact</div>', unsafe_allow_html=True)
            reg_impact = results["medical_device_impact"]["regulatory_impact"]  # Changed from medical_impact to medical_device_impact
            reg_color = VIVE_RED if reg_impact == "Significant" else VIVE_AMBER
            st.markdown(f'<div class="metric-value" style="color:{reg_color}">{reg_impact}</div>', unsafe_allow_html=True)
        
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
    per_unit_cost_total = results["financial_impact"]["fix_cost_per_unit"] * results["financial_impact"]["projected_sales_36m"]
    
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

# --- MAIN APPLICATION ---

def main():
    """Main application function"""
    
    # Set sidebar style
    st.sidebar.markdown('<div class="sidebar-title">QualityROI CBA Tool</div>', unsafe_allow_html=True)
    
    # Authenticate with simple password (in production you'd use more secure methods)
    password = "MPFvive8955@#@"  # This would normally be stored securely
    
    # Simple password check just for demo
    if 'authenticated' not in st.session_state:
        st.session_state.authenticated = False
    
    if not st.session_state.authenticated:
        entered_password = st.sidebar.text_input("Enter password:", type="password")
        if st.sidebar.button("Login"):
            if entered_password == password:
                st.session_state.authenticated = True
                st.rerun()
            else:
                st.sidebar.error("Incorrect password")
    
    if not st.session_state.authenticated:
        st.title("QualityROI CBA Tool")
        st.write("Please login to access the tool.")
        return
    
    # Display main application
    vive_header()
    st.markdown('<div class="main-header">Medical Device Quality ROI Analysis</div>', unsafe_allow_html=True)
    
    # Check if analysis results exist
    if not st.session_state.analysis_submitted:
        # New Analysis button
        if st.button("Start New Analysis"):
            st.session_state.quality_analysis_results = None
            st.session_state.analysis_submitted = False
            st.session_state.chat_history = None
            
        with st.form("quality_issue_form"):
            st.markdown('<div class="form-section-header">Product Information</div>', unsafe_allow_html=True)
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Required inputs
                sku = st.text_input(
                    "SKU*", 
                    help="Required: Product SKU (e.g., MPF-1234)"
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
                
                risk_level = st.select_slider(
                    "Medical Risk Level*",
                    options=["Low", "Medium", "High"],
                    value="Medium",
                    help="Required: Risk level assessment for this medical device issue"
                )
                
                regulatory_impact = st.selectbox(
                    "Regulatory Impact*",
                    ["None", "Possible", "Significant"],
                    help="Required: Potential regulatory impact of this issue"
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
        if st.button("Start New Analysis", key="reset_top_button"):
            st.session_state.quality_analysis_results = None
            st.session_state.analysis_submitted = False
            st.session_state.chat_history = None
            st.rerun()
            
        # Display analysis results
        display_quality_issue_results(st.session_state.quality_analysis_results)
        
        # Display AI chat interface if enabled
        with st.expander("AI Quality Consultant", expanded=True):
            # Chat container
            st.markdown('<div class="chat-container" id="chat-container">', unsafe_allow_html=True)
            
            if st.session_state.chat_history:
                for message in st.session_state.chat_history:
                    if message["role"] == "user":
                        st.markdown(f'<div class="chat-message user-message">{message["content"]}</div>', unsafe_allow_html=True)
                    else:
                        st.markdown(f'<div class="chat-message ai-message">{message["content"]}</div>', unsafe_allow_html=True)
            else:
                st.info("AI consultant not available or initialization failed.")
                
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Input for new messages
            user_input = st.text_area(
                "Ask about the quality issue, potential solutions, or regulatory implications:",
                placeholder="E.g., What could be causing this issue? What fixes do you recommend? What are the regulatory considerations?"
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
                            You are a Quality Management expert specializing in medical devices for Vive Health. You analyze quality issues, provide insights on cost-benefit analyses, and suggest solutions.
                            
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
                results_df = pd.DataFrame({
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
                        f"${st.session_state.quality_analysis_results['financial_impact']['fix_cost_upfront']:.2f}",
                        f"${st.session_state.quality_analysis_results['financial_impact']['fix_cost_per_unit']:.2f}",
                        f"${st.session_state.quality_analysis_results['financial_impact']['current_unit_cost']:.2f}",
                        st.session_state.quality_analysis_results['recommendation'],
                        st.session_state.quality_analysis_results['medical_device_impact']['risk_level'],
                        st.session_state.quality_analysis_results['medical_device_impact']['regulatory_impact']
                    ]
                })
                
                csv = results_df.to_csv(index=False).encode('utf-8')
                
                st.download_button(
                    "Download Analysis as CSV",
                    csv,
                    f"qualityroi_analysis_{st.session_state.quality_analysis_results['sku']}_{datetime.now().strftime('%Y%m%d')}.csv",
                    "text/csv",
                    key="download_csv"
                )
            
            with col2:
                # Export chat history
                if st.session_state.chat_history:
                    chat_export = ""
                    for msg in st.session_state.chat_history:
                        role_prefix = "Question: " if msg["role"] == "user" else "Answer: "
                        chat_export += f"{role_prefix}{msg['content']}\n\n"
                    
                    st.download_button(
                        "Download AI Consultation",
                        chat_export,
                        f"qualityroi_consultation_{st.session_state.quality_analysis_results['sku']}_{datetime.now().strftime('%Y%m%d')}.txt",
                        "text/plain",
                        key="download_chat"
                    )
        
        # Add a second reset button at the bottom
        if st.button("Start New Analysis", key="reset_bottom_button"):
            st.session_state.quality_analysis_results = None
            st.session_state.analysis_submitted = False
            st.session_state.chat_history = None
            st.rerun()

# Run the application
if __name__ == "__main__":
    main()
