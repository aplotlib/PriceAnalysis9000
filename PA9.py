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
    model: str = "ft:gpt-4o-2024-08-06:vive-health-quality-department:1vive-quality-training-data:BQqHZoPo",
    messages: Optional[List[Dict[str, str]]] = None
) -> str:
    """
    Get AI analysis for quality issues using the OpenAI API
    
    Args:
        client: OpenAI client instance
        system_prompt: System prompt providing context to the AI
        user_message: User message or query
        model: Model ID to use (default is the fine-tuned model)
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
    # Calculate basic metrics
    return_rate_30d = (returns_30d / sales_30d) * 100 if sales_30d > 0 else 0
    
    # Include 365-day data if available
    return_rate_365d = None
    if sales_365d is not None and returns_365d is not None and sales_365d > 0:
        return_rate_365d = (returns_365d / sales_365d) * 100
    
    # Calculate financial impact
    monthly_return_cost = returns_30d * current_unit_cost
    estimated_monthly_savings = monthly_return_cost * 0.80  # Assuming 80% reduction in returns
    
    # Annual projections
    annual_return_cost = monthly_return_cost * 12
    annual_savings = estimated_monthly_savings * 12
    
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
    expected_discount_pct: float
) -> Dict[str, Any]:
    """
    Analyze potential salvage operation for affected inventory
    """
    # Calculate salvage metrics
    expected_units_recovered = affected_inventory * (expected_recovery_pct / 100)
    regular_price = current_unit_cost * 2.5  # Assuming standard markup
    discounted_price = regular_price * (1 - expected_discount_pct / 100)
    
    total_rework_cost = rework_cost_upfront + (rework_cost_per_unit * affected_inventory)
    salvage_revenue = expected_units_recovered * discounted_price
    write_off_loss = (affected_inventory - expected_units_recovered) * current_unit_cost
    
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
            "discount_percentage": expected_discount_pct
        },
        "financial": {
            "total_rework_cost": total_rework_cost,
            "salvage_revenue": salvage_revenue,
            "write_off_loss": write_off_loss,
            "salvage_profit": salvage_profit,
            "complete_writeoff_cost": writeoff_cost,
            "roi_percent": roi_percent,
            "profit_per_unit": salvage_profit / affected_inventory if affected_inventory > 0 else 0
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
    
    with col2:
        st.markdown('<div class="metric-label">Monthly Return Cost</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="metric-value">${results["financial_impact"]["monthly_return_cost"]:.2f}</div>', unsafe_allow_html=True)
        
        st.markdown('<div class="metric-label">Annual Return Cost</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="metric-value">${results["financial_impact"]["annual_return_cost"]:.2f}</div>', unsafe_allow_html=True)
    
    with col3:
        st.markdown('<div class="metric-label">Est. Monthly Savings</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="metric-value">${results["financial_impact"]["estimated_monthly_savings"]:.2f}</div>', unsafe_allow_html=True)
        
        st.markdown('<div class="metric-label">Payback Period</div>', unsafe_allow_html=True)
        if results["financial_impact"]["payback_months"] == float('inf'):
            payback_text = "N/A"
        else:
            payback_text = f"{results['financial_impact']['payback_months']:.1f} months"
        st.markdown(f'<div class="metric-value">{payback_text}</div>', unsafe_allow_html=True)
    
    # Recommendation
    st.markdown('<div class="metric-label">Recommendation</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="{results["recommendation_class"]}">{results["recommendation"]}</div>', unsafe_allow_html=True)
    
    # Brand impact if available
    if results["brand_impact"]:
        st.markdown('<div class="metric-label">Brand Impact Assessment</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="metric-value">{results["brand_impact"]}</div>', unsafe_allow_html=True)
    
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
    
    fig.update_layout(
        title='Projected Returns Over Time',
        xaxis_title='Timeline',
        yaxis_title='Amount ($)',
        barmode='group',
        height=300,
        margin=dict(l=20, r=20, t=40, b=20),
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
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
    
    # Price breakdown
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<div class="metric-label">Regular Price</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="metric-value">${results["metrics"]["regular_price"]:.2f}</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="metric-label">Discounted Price</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="metric-value">${results["metrics"]["discounted_price"]:.2f} ({results["metrics"]["discount_percentage"]}% off)</div>', unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

def chat_with_ai(results, issue_description, chat_history=None):
    """Initialize or continue chat with AI about quality issues"""
    if chat_history is None:
        chat_history = []
        
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
        """
        
        initial_analysis = get_ai_analysis(
            client,
            system_prompt,
            "Based on the product information and quality metrics, provide your initial analysis of the issue and suggested next steps. Be specific about potential fixes and quality improvements.",
            model="ft:gpt-4o-2024-08-06:vive-health-quality-department:1vive-quality-training-data:BQqHZoPo"
        )
        
        # Add initial AI message
        chat_history.append({
            "role": "assistant",
            "content": initial_analysis
        })
    
    return chat_history

# --- PASSWORD VERIFICATION ---
def verify_password(password):
    """Temporary simplified password check"""
    return password == "MPFvive8955@#@"  # Just check direct equality

# --- API KEY HANDLING ---
try:
    api_key = st.secrets["openai_api_key"]
except (FileNotFoundError, KeyError):
    api_key = os.environ.get("OPENAI_API_KEY", "")

# Initialize OpenAI client
client = initialize_openai_client(api_key)

def display_dashboard():
    st.markdown('<div class="main-header">Business Overview Dashboard</div>', unsafe_allow_html=True)
    # Your existing dashboard code here

def display_sales_analysis():
    st.markdown('<div class="main-header">Sales Analysis</div>', unsafe_allow_html=True)
    # Your existing sales analysis code here

def display_inventory_management():
    st.markdown('<div class="main-header">Inventory Management</div>', unsafe_allow_html=True)
    # Your existing inventory management code here

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
                    sku = st.text_input("SKU*", help="Required: Product SKU")
                    product_type = st.selectbox(
                        "Product Type*", 
                        ["B2C", "B2B", "Both"],
                        help="Required: Distribution channel for this product"
                    )
                    sales_30d = st.number_input(
                        "Total Sales Last 30 Days*", 
                        min_value=0.0,
                        help="Required: Units sold in the last 30 days"
                    )
                    returns_30d = st.number_input(
                        "Total Returns Last 30 Days*", 
                        min_value=0.0,
                        help="Required: Units returned in the last 30 days"
                    )
                    
                with col2:
                    current_unit_cost = st.number_input(
                        "Current Unit Cost* (Landed Cost)", 
                        min_value=0.0,
                        help="Required: Current per-unit cost to produce"
                    )
                    fix_cost_upfront = st.number_input(
                        "Fix Cost Upfront*", 
                        min_value=0.0,
                        help="Required: One-time cost to implement the quality fix"
                    )
                    fix_cost_per_unit = st.number_input(
                        "Additional Cost Per Unit*", 
                        min_value=0.0,
                        help="Required: Additional cost per unit after implementing the fix"
                    )
                
                # Description of issue
                issue_description = st.text_area(
                    "Description of Quality Issue*",
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
                    if not all([sku, sales_30d > 0, current_unit_cost > 0, issue_description]):
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
                    """
                    
                    # Get the full chat history for context
                    messages_history = []
                    for msg in st.session_state.chat_history:
                        messages_history.append({"role": msg["role"], "content": msg["content"]})
                    
                    ai_response = get_ai_analysis(
                        client,
                        system_prompt,
                        user_input,
                        model="ft:gpt-4o-2024-08-06:vive-health-quality-department:1vive-quality-training-data:BQqHZoPo",
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
                    sku = st.text_input("SKU*", key="salvage_sku", help="Required: Product SKU")
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
                        help="Required: Current per-unit cost to produce"
                    )
                    rework_cost_upfront = st.number_input(
                        "Rework Setup Cost*", 
                        min_value=0.0,
                        key="rework_upfront",
                        help="Required: One-time cost to set up the rework operation"
                    )
                
                with col2:
                    rework_cost_per_unit = st.number_input(
                        "Rework Cost Per Unit*", 
                        min_value=0.0,
                        key="rework_per_unit",
                        help="Required: Cost to rework each affected unit"
                    )
                    expected_recovery_pct = st.slider(
                        "Expected Recovery Percentage*", 
                        min_value=0.0,
                        max_value=100.0,
                        value=80.0,
                        help="Required: Percentage of affected units expected to be successfully reworked"
                    )
                    expected_discount_pct = st.slider(
                        "Expected Discount Percentage*", 
                        min_value=0.0,
                        max_value=100.0,
                        value=30.0,
                        help="Required: Discount percentage for selling reworked units"
                    )
                
                # Form submission
                submit_button = st.form_submit_button("Analyze Salvage Operation")
                
                if submit_button:
                    # Validate required fields
                    if not all([sku, affected_inventory > 0, current_unit_cost > 0]):
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
                            expected_discount_pct=expected_discount_pct
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
                    expected_discount_pct=discount_adjustment
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

def main():
    # Sidebar navigation
    st.sidebar.title("QualityROI Dashboard")
    app_mode = st.sidebar.selectbox(
        "Choose the app mode",
        ["Quality Manager", "Dashboard", "Sales Analysis", "Inventory Management", "Login"]
    )
    
    # Login Screen
    if app_mode == "Login":
        st.title("QualityROI - Cost-Benefit Analysis Tool")
        st.subheader("Login")
        password = st.text_input("Enter password", type="password")
        
        if st.button("Login"):
            if verify_password(password):
                st.session_state["authenticated"] = True
                st.success("Login successful!")
                st.experimental_rerun()
            else:
                st.error("Incorrect password")
    
    # Protected pages - only shown if authenticated
    elif "authenticated" in st.session_state and st.session_state["authenticated"]:
        if app_mode == "Dashboard":
            display_dashboard()
        elif app_mode == "Sales Analysis":
            display_sales_analysis()
        elif app_mode == "Inventory Management":
            display_inventory_management()
        elif app_mode == "Quality Manager":
            display_quality_manager()
    else:
        # If quality mode was explicitly chosen but not logged in, show login with explanation
        if app_mode == "Quality Manager":
            st.title("QualityROI - Cost-Benefit Analysis Tool")
            st.warning("Please login to access the Quality Manager")
            
            password = st.text_input("Enter password", type="password")
            
            if st.button("Login"):
                if verify_password(password):
                    st.session_state["authenticated"] = True
                    st.success("Login successful!")
                    st.experimental_rerun()
                else:
                    st.error("Incorrect password")
        else:
            st.warning("Please login to access this page")
            st.sidebar.info("Please login to access the application")

if __name__ == "__main__":
    main()
