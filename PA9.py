import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

# App configuration
st.set_page_config(
    page_title="TariffSight Pro: Import & Marketing ROI Calculator (use light mode browser, display issues for dark mode)",
    page_icon="🧮",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E88E5;
        font-weight: bold;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #0D47A1;
    }
    .info-box {
        background-color: #E3F2FD;
        padding: 15px;
        border-radius: 5px;
        border-left: 5px solid #1E88E5;
        margin-bottom: 15px;
    }
    .result-box {
        background-color: #E8F5E9;
        padding: 15px;
        border-radius: 5px;
        border-left: 5px solid #43A047;
        margin: 15px 0;
    }
    .warning-box {
        background-color: #FFF3E0;
        padding: 15px;
        border-radius: 5px;
        border-left: 5px solid #FF9800;
        margin-bottom: 15px;
    }
    .metric-card {
        background-color: white;
        padding: 15px;
        border-radius: 5px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        text-align: center;
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: bold;
    }
    .metric-label {
        font-size: 0.9rem;
        color: #616161;
    }
    .tooltip-header {
        font-weight: bold;
        margin-bottom: 5px;
    }
    .tooltip-example {
        font-style: italic;
        color: #1976D2;
    }
    .positive-value {
        color: #4CAF50;
        font-weight: bold;
    }
    .negative-value {
        color: #F44336;
        font-weight: bold;
    }
    .sidebar .sidebar-content {
        background-color: #f5f5f5;
    }
    .section-divider {
        height: 3px;
        background-color: #f0f2f6;
        margin: 20px 0;
    }
</style>
""", unsafe_allow_html=True)

# Utility functions
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

# Main app function
def main():
    # Display header
    st.markdown("<h1 class='main-header'>TariffSight Pro: Import & Marketing ROI Calculator</h1>", unsafe_allow_html=True)
    st.markdown("""
    <div class='info-box'>
    Calculate how tariffs and advertising spend impact your product profitability. Model different scenarios to optimize your pricing and marketing strategy.
    </div>
    """, unsafe_allow_html=True)
    
    # Create main tabs
    tabs = st.tabs(["Calculator", "Scenario Modeling", "Marketing ROI", "Tariff Resources"])
    
    # Session state for saving calculations
    if 'calculations' not in st.session_state:
        st.session_state.calculations = []
    
    # Calculator Tab
    with tabs[0]:
        st.markdown("<h2 class='sub-header'>Product Details</h2>", unsafe_allow_html=True)
        
        # Create two columns for basic product info
        col1, col2 = st.columns(2)
        
        with col1:
            product_name = st.text_input(
                "Product Name", 
                value="",
                help="Enter the full product name or description. Example: 'Premium Leather Wallet'"
            )
            
            sku = st.text_input(
                "Product SKU", 
                value="",
                help="Enter the Stock Keeping Unit (unique identifier). Example: 'LW-2025-BLK'"
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
                max_value=500, 
                value=25, 
                step=1,
                help="Import duty rate as a percentage of the manufacturing cost. Varies by product category and country of origin. Slide to model different rates."
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
        
        # Optional sales and marketing inputs
        with st.expander("Sales & Marketing Data (Optional)", expanded=False):
            st.info("Enter these fields if you want to include sales and marketing metrics in your analysis.")
            
            col5, col6 = st.columns(2)
            
            with col5:
                monthly_sales_qty = st.number_input(
                    "Monthly Sales Quantity", 
                    min_value=0, 
                    value=500, 
                    step=10,
                    help="Average number of units sold per month."
                )
                
                return_rate = st.slider(
                    "Return Rate (%)", 
                    min_value=0.0, 
                    max_value=50.0, 
                    value=3.0, 
                    step=0.1,
                    help="Percentage of sold units that are returned by customers."
                )
            
            with col6:
                monthly_ad_spend = st.number_input(
                    "Monthly Ad Spend ($)", 
                    min_value=0.0, 
                    value=2000.0, 
                    step=100.0,
                    help="Total amount spent on advertising per month."
                )
                
                customer_acquisition_cost = st.number_input(
                    "Customer Acquisition Cost ($)", 
                    min_value=0.0, 
                    value=0.0, 
                    step=1.0,
                    help="Average cost to acquire a new customer. Leave at 0 if unknown."
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
                    profit_color = "green" if result['profit'] > 0 else "red"
                    st.markdown(f"""
                    <div class='metric-card'>
                        <p class='metric-label'>Profit per Unit</p>
                        <p class='metric-value' style='color: {profit_color}'>${result['profit']:.2f}</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col10:
                    margin_color = "green" if result['margin_percentage'] > 15 else ("orange" if result['margin_percentage'] > 0 else "red")
                    st.markdown(f"""
                    <div class='metric-card'>
                        <p class='metric-label'>Profit Margin</p>
                        <p class='metric-value' style='color: {margin_color}'>{result['margin_percentage']:.1f}%</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Display breakeven and profitability
                st.markdown("<br>", unsafe_allow_html=True)
                col11, col12 = st.columns(2)
                
                with col11:
                    st.markdown(f"""
                    <div class='result-box'>
                        <h3>Breakeven Price: ${result['breakeven_price']:.2f}</h3>
                        <p>At this selling price, you will neither make a profit nor a loss after all import costs.</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col12:
                    st.markdown(f"""
                    <div class='result-box'>
                        <h3>Minimum Profitable Price: ${result['min_profitable_msrp']:.2f}</h3>
                        <p>We recommend a minimum price point of ${result['min_profitable_msrp']:.2f} to ensure profitability.</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Cost breakdown visualization
                st.markdown("<h3>Cost Breakdown</h3>", unsafe_allow_html=True)
                
                # Extract cost components
                cost_items = list(result["cost_breakdown"].keys())
                cost_values = list(result["cost_breakdown"].values())
                
                # Create pie chart
                fig = px.pie(
                    names=cost_items,
                    values=cost_values,
                    title="Cost Breakdown per Unit",
                    color_discrete_sequence=px.colors.qualitative.Safe,
                )
                
                # Update layout
                fig.update_traces(textposition='inside', textinfo='percent+label')
                fig.update_layout(
                    legend=dict(orientation="h", yanchor="bottom", y=-0.15),
                    margin=dict(t=50, b=100)
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Display additional sales metrics if provided
                if monthly_sales_qty > 0:
                    st.markdown("<h3>Sales & Marketing Analysis</h3>", unsafe_allow_html=True)
                    
                    # Calculate sales metrics
                    monthly_revenue = monthly_sales_qty * msrp
                    monthly_returns = monthly_sales_qty * (return_rate / 100)
                    net_units_sold = monthly_sales_qty - monthly_returns
                    net_revenue = net_units_sold * msrp
                    
                    # Calculate profit metrics
                    unit_profit = result['profit']
                    monthly_profit_before_ads = unit_profit * net_units_sold
                    monthly_profit_after_ads = monthly_profit_before_ads - monthly_ad_spend
                    
                    # Display sales metrics
                    col13, col14, col15, col16 = st.columns(4)
                    
                    with col13:
                        st.markdown(f"""
                        <div class='metric-card'>
                            <p class='metric-label'>Monthly Revenue</p>
                            <p class='metric-value'>${monthly_revenue:.2f}</p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col14:
                        st.markdown(f"""
                        <div class='metric-card'>
                            <p class='metric-label'>Net Units Sold</p>
                            <p class='metric-value'>{net_units_sold:.0f}</p>
                            <p class='metric-label'>(After Returns)</p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col15:
                        profit_color = "green" if monthly_profit_before_ads > 0 else "red"
                        st.markdown(f"""
                        <div class='metric-card'>
                            <p class='metric-label'>Monthly Profit</p>
                            <p class='metric-value' style='color: {profit_color}'>${monthly_profit_before_ads:.2f}</p>
                            <p class='metric-label'>(Before Ad Spend)</p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col16:
                        profit_color = "green" if monthly_profit_after_ads > 0 else "red"
                        st.markdown(f"""
                        <div class='metric-card'>
                            <p class='metric-label'>Net Profit</p>
                            <p class='metric-value' style='color: {profit_color}'>${monthly_profit_after_ads:.2f}</p>
                            <p class='metric-label'>(After Ad Spend)</p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # Display marketing metrics if ad spend is provided
                    if monthly_ad_spend > 0:
                        st.markdown("<br>", unsafe_allow_html=True)
                        col17, col18, col19 = st.columns(3)
                        
                        with col17:
                            ad_to_revenue = (monthly_ad_spend / monthly_revenue * 100) if monthly_revenue > 0 else 0
                            st.markdown(f"""
                            <div class='metric-card'>
                                <p class='metric-label'>Ad Spend to Revenue</p>
                                <p class='metric-value'>{ad_to_revenue:.1f}%</p>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        with col18:
                            roas = monthly_revenue / monthly_ad_spend if monthly_ad_spend > 0 else 0
                            st.markdown(f"""
                            <div class='metric-card'>
                                <p class='metric-label'>ROAS</p>
                                <p class='metric-value'>{roas:.2f}x</p>
                                <p class='metric-label'>(Return on Ad Spend)</p>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        with col19:
                            ad_cost_per_unit = monthly_ad_spend / monthly_sales_qty if monthly_sales_qty > 0 else 0
                            st.markdown(f"""
                            <div class='metric-card'>
                                <p class='metric-label'>Ad Cost per Unit</p>
                                <p class='metric-value'>${ad_cost_per_unit:.2f}</p>
                            </div>
                            """, unsafe_allow_html=True)
                
                # Add to saved calculations
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
                
                st.session_state.calculations.append(calculation_entry)
                
                # Display recommendation based on margin
                if result['margin_percentage'] < 0:
                    st.markdown("""
                    <div class='warning-box'>
                        <h3>⚠️ Warning: Negative Margin</h3>
                        <p>This product is not profitable at the current price and tariff rate. Consider increasing your selling price or finding ways to reduce costs.</p>
                    </div>
                    """, unsafe_allow_html=True)
                elif result['margin_percentage'] < 15:
                    st.markdown(f"""
                    <div class='warning-box'>
                        <h3>⚠️ Low Profit Margin</h3>
                        <p>Your profit margin is below 15%, which may be risky. Consider adjusting your pricing strategy or finding ways to reduce costs.</p>
                        <p>To achieve a 20% profit margin, your selling price should be at least ${result['landed_cost'] / 0.8:.2f}.</p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class='result-box'>
                        <h3>✅ Healthy Profit Margin</h3>
                        <p>Your profit margin of {result['margin_percentage']:.1f}% is healthy. This product should be profitable at the current price and tariff rate.</p>
                    </div>
                    """, unsafe_allow_html=True)
    
    # Scenario Modeling Tab
    with tabs[1]:
        st.markdown("<h2 class='sub-header'>Scenario Modeling</h2>", unsafe_allow_html=True)
        
        st.info("""
        Use this tab to model multiple scenarios. You can either:
        1. Keep your price fixed and see how different tariff rates affect profitability, or
        2. Keep the tariff rate fixed and see how different price points affect profitability
        
        This is helpful for planning ahead and making strategic pricing decisions.
        """)
        
        scenario_type = st.radio(
            "Choose scenario type:",
            ["Varying Tariff Rates", "Varying Price Points"],
            help="Select whether you want to model different tariff rates at a fixed price, or different price points at a fixed tariff rate."
        )
        
        # Input fields for scenario modeling
        col1, col2 = st.columns(2)
        
        if scenario_type == "Varying Tariff Rates":
            with col1:
                base_msrp = st.number_input(
                    "Fixed MSRP ($)", 
                    min_value=0.01, 
                    value=100.00, 
                    step=0.01, 
                    key="scen_msrp",
                    help="The retail price that will remain constant across all tariff rate scenarios."
                )
                
                base_cost = st.number_input(
                    "Manufacturing Cost per Unit ($)", 
                    min_value=0.01, 
                    value=50.00, 
                    step=0.01, 
                    key="scen_cost",
                    help="The direct cost to produce or acquire each unit before any import costs or tariffs."
                )
            
            with col2:
                min_tariff = st.number_input(
                    "Minimum Tariff Rate (%)", 
                    min_value=0, 
                    value=0, 
                    step=5,
                    help="The lowest tariff rate to include in your scenario analysis. Enter 0 to start from no tariffs."
                )
                
                max_tariff = st.number_input(
                    "Maximum Tariff Rate (%)", 
                    min_value=1, 
                    value=100, 
                    step=5,
                    help="The highest tariff rate to include in your scenario analysis. For worst-case scenario planning, you may want to set this higher than current rates."
                )
                
                steps = st.slider(
                    "Number of Scenarios", 
                    min_value=5, 
                    max_value=50, 
                    value=10,
                    help="How many data points to generate between the minimum and maximum tariff rates. More steps give a smoother curve but can make the table longer."
                )
        
        else:  # Varying Price Points
            with col1:
                fixed_tariff = st.number_input(
                    "Fixed Tariff Rate (%)", 
                    min_value=0, 
                    value=25, 
                    step=5,
                    help="The tariff rate that will remain constant while you model different price points. Use your expected or current tariff rate."
                )
                
                base_cost = st.number_input(
                    "Manufacturing Cost per Unit ($)", 
                    min_value=0.01, 
                    value=50.00, 
                    step=0.01, 
                    key="scen_cost2",
                    help="The direct cost to produce or acquire each unit before any import costs or tariffs."
                )
            
            with col2:
                min_price_factor = st.slider(
                    "Minimum Price Factor", 
                    min_value=0.5, 
                    max_value=0.99, 
                    value=0.8, 
                    step=0.01, 
                    help="The lowest price point as a factor of your landed cost. Example: 0.8 means the lowest price is 80% of your landed cost (unprofitable)."
                )
                
                max_price_factor = st.slider(
                    "Maximum Price Factor", 
                    min_value=1.01, 
                    max_value=5.0, 
                    value=2.0, 
                    step=0.1,
                    help="The highest price point as a factor of your landed cost. Example: 2.0 means the highest price is double your landed cost."
                )
                
                steps = st.slider(
                    "Number of Price Points", 
                    min_value=5, 
                    max_value=50, 
                    value=10,
                    help="How many price points to generate between the minimum and maximum price factors. More steps give a smoother curve but can make the table longer."
                )
        
        # Optional import costs
        with st.expander("Additional Import Costs (Optional)", expanded=False):
            st.info("These costs will be distributed across the number of units in your shipment. All fields are optional.")
            
            col3, col4 = st.columns(2)
            
            with col3:
                shipping_cost_scen = st.number_input(
                    "Shipping Cost per Shipment ($)", 
                    min_value=0.0, 
                    value=1000.0, 
                    step=10.0, 
                    key="scen_ship",
                    help="Total cost to ship an entire container or shipment."
                )
                
                storage_cost_scen = st.number_input(
                    "Storage/Warehousing Cost ($)", 
                    min_value=0.0, 
                    value=0.0, 
                    step=10.0, 
                    key="scen_store",
                    help="Costs for warehousing or storage associated with this shipment."
                )
                
                customs_fee_scen = st.number_input(
                    "Customs Processing Fee ($)", 
                    min_value=0.0, 
                    value=250.0, 
                    step=10.0, 
                    key="scen_customs",
                    help="Fees charged by customs for processing your shipment."
                )
            
            with col4:
                broker_fee_scen = st.number_input(
                    "Customs Broker Fee ($)", 
                    min_value=0.0, 
                    value=150.0, 
                    step=10.0, 
                    key="scen_broker",
                    help="Fees paid to customs brokers for handling import documentation."
                )
                
                other_costs_scen = st.number_input(
                    "Other Import Costs ($)", 
                    min_value=0.0, 
                    value=0.0, 
                    step=10.0, 
                    key="scen_other",
                    help="Any other import-related costs not covered by the categories above."
                )
                
                units_scen = st.number_input(
                    "Units per Shipment", 
                    min_value=1, 
                    value=1000, 
                    step=10, 
                    key="scen_units",
                    help="The total number of product units in this shipment. Used to calculate per-unit costs from total shipment costs."
                )
        
        # Generate scenarios button
        if st.button(
            "Generate Scenarios",
            help="Click to generate multiple scenarios based on your selected parameters."
        ):
            with st.spinner("Generating scenarios..."):
                if scenario_type == "Varying Tariff Rates":
                    # Generate tariff scenarios
                    scenarios_df = generate_tariff_scenarios(
                        base_msrp, base_cost, min_tariff, max_tariff, steps,
                        shipping_cost_scen, storage_cost_scen, customs_fee_scen,
                        broker_fee_scen, other_costs_scen, units_scen
                    )
                    
                    # Format the dataframe for display
                    display_df = scenarios_df.copy()
                    display_df["tariff_rate"] = display_df["tariff_rate"].round(1).astype(str) + "%"
                    display_df["landed_cost"] = display_df["landed_cost"].round(2).map("${:.2f}".format)
                    display_df["profit"] = display_df["profit"].round(2).map("${:.2f}".format)
                    display_df["margin"] = display_df["margin"].round(1).astype(str) + "%"
                    display_df["breakeven_price"] = display_df["breakeven_price"].round(2).map("${:.2f}".format)
                    
                    display_df.columns = ["Tariff Rate", "Landed Cost", "Profit", "Margin", "Breakeven Price"]
                    
                    st.dataframe(display_df, use_container_width=True)
                    
                    # Create visualization
                    fig = go.Figure()
                    
                    # Add profit line
                    fig.add_trace(go.Scatter(
                        x=scenarios_df["tariff_rate"],
                        y=scenarios_df["profit"],
                        mode='lines+markers',
                        name='Profit per Unit',
                        line=dict(color='#4CAF50', width=3),
                        yaxis='y1'
                    ))
                    
                    # Add margin line
                    fig.add_trace(go.Scatter(
                        x=scenarios_df["tariff_rate"],
                        y=scenarios_df["margin"],
                        mode='lines+markers',
                        name='Profit Margin (%)',
                        line=dict(color='#2196F3', width=3, dash='dot'),
                        yaxis='y2'
                    ))
                    
                    # Update layout with dual y-axes
                    fig.update_layout(
                        title='Profitability at Different Tariff Rates',
                        xaxis=dict(title='Tariff Rate (%)'),
                        yaxis=dict(
                            title='Profit per Unit ($)',
                            titlefont=dict(color='#4CAF50'),
                            tickfont=dict(color='#4CAF50')
                        ),
                        yaxis2=dict(
                            title='Profit Margin (%)',
                            titlefont=dict(color='#2196F3'),
                            tickfont=dict(color='#2196F3'),
                            anchor='x',
                            overlaying='y',
                            side='right'
                        ),
                        legend=dict(x=0.01, y=0.99),
                        margin=dict(t=50, b=50, l=50, r=50),
                        hovermode='x unified'
                    )
                    
                    # Add zero line for profit reference
                    fig.add_shape(
                        type="line",
                        x0=min_tariff,
                        y0=0,
                        x1=max_tariff,
                        y1=0,
                        line=dict(color="red", width=2, dash="dot"),
                        yref='y1'
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Find breakeven tariff rate
                    breakeven_tariff = None
                    for i in range(len(scenarios_df)-1):
                        if (scenarios_df.iloc[i]["profit"] >= 0 and scenarios_df.iloc[i+1]["profit"] < 0) or \
                           (scenarios_df.iloc[i]["profit"] <= 0 and scenarios_df.iloc[i+1]["profit"] > 0):
                            # Simple linear interpolation
                            rate1 = scenarios_df.iloc[i]["tariff_rate"]
                            rate2 = scenarios_df.iloc[i+1]["tariff_rate"]
                            profit1 = scenarios_df.iloc[i]["profit"]
                            profit2 = scenarios_df.iloc[i+1]["profit"]
                            
                            if profit1 == profit2:
                                breakeven_tariff = rate1
                            else:
                                breakeven_tariff = rate1 + (0 - profit1) * (rate2 - rate1) / (profit2 - profit1)
                            break
                    
                    if breakeven_tariff is not None:
                        st.markdown(f"""
                        <div class='result-box'>
                            <h3>Breakeven Tariff Rate: {breakeven_tariff:.1f}%</h3>
                            <p>At this tariff rate, your product will break even at the current MSRP of ${base_msrp:.2f}.</p>
                            <p>To remain profitable with higher tariff rates, you'll need to increase your selling price or reduce other costs.</p>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        if scenarios_df["profit"].min() > 0:
                            st.markdown(f"""
                            <div class='result-box'>
                                <h3>Profitable Across All Scenarios</h3>
                                <p>Your product remains profitable at all tariff rates from {min_tariff}% to {max_tariff}% at the current MSRP of ${base_msrp:.2f}.</p>
                            </div>
                            """, unsafe_allow_html=True)
                        elif scenarios_df["profit"].max() < 0:
                            st.markdown(f"""
                            <div class='warning-box'>
                                <h3>Unprofitable Across All Scenarios</h3>
                                <p>Your product is not profitable at any tariff rate from {min_tariff}% to {max_tariff}% at the current MSRP of ${base_msrp:.2f}.</p>
                                <p>You need to increase your selling price or reduce manufacturing costs to achieve profitability.</p>
                            </div>
                            """, unsafe_allow_html=True)
                
                else:  # Varying Price Points
                    # Generate price scenarios
                    scenarios_df = generate_price_scenarios(
                        fixed_tariff, base_cost, min_price_factor, max_price_factor, steps,
                        shipping_cost_scen, storage_cost_scen, customs_fee_scen,
                        broker_fee_scen, other_costs_scen, units_scen
                    )
                    
                    # Format the dataframe for display
                    display_df = scenarios_df.copy()
                    display_df["msrp"] = display_df["msrp"].round(2).map("${:.2f}".format)
                    display_df["profit"] = display_df["profit"].round(2).map("${:.2f}".format)
                    display_df["margin"] = display_df["margin"].round(1).astype(str) + "%"
                    display_df["landed_cost"] = display_df["landed_cost"].round(2).map("${:.2f}".format)
                    
                    display_df.columns = ["Selling Price", "Profit", "Margin", "Landed Cost"]
                    
                    st.dataframe(display_df, use_container_width=True)
                    
                    # Create visualization
                    fig = go.Figure()
                    
                    # Add profit line
                    fig.add_trace(go.Scatter(
                        x=scenarios_df["msrp"],
                        y=scenarios_df["profit"],
                        mode='lines+markers',
                        name='Profit per Unit',
                        line=dict(color='#4CAF50', width=3),
                        yaxis='y1'
                    ))
                    
                    # Add margin line
                    fig.add_trace(go.Scatter(
                        x=scenarios_df["msrp"],
                        y=scenarios_df["margin"],
                        mode='lines+markers',
                        name='Profit Margin (%)',
                        line=dict(color='#2196F3', width=3, dash='dot'),
                        yaxis='y2'
                    ))
                    
                    # Update layout with dual y-axes
                    fig.update_layout(
                        title=f'Profitability at Different Price Points ({fixed_tariff}% Tariff)',
                        xaxis=dict(title='Selling Price ($)'),
                        yaxis=dict(
                            title='Profit per Unit ($)',
                            titlefont=dict(color='#4CAF50'),
                            tickfont=dict(color='#4CAF50')
                        ),
                        yaxis2=dict(
                            title='Profit Margin (%)',
                            titlefont=dict(color='#2196F3'),
                            tickfont=dict(color='#2196F3'),
                            anchor='x',
                            overlaying='y',
                            side='right'
                        ),
                        legend=dict(x=0.01, y=0.99),
                        margin=dict(t=50, b=50, l=50, r=50),
                        hovermode='x unified'
                    )
                    
                    # Add zero line for profit reference
                    fig.add_shape(
                        type="line",
                        x0=scenarios_df["msrp"].min(),
                        y0=0,
                        x1=scenarios_df["msrp"].max(),
                        y1=0,
                        line=dict(color="red", width=2, dash="dot"),
                        yref='y1'
                    )
                    
                    # Add breakeven price marker
                    breakeven_price = scenarios_df.iloc[0]["landed_cost"]
                    fig.add_trace(go.Scatter(
                        x=[breakeven_price],
                        y=[0],
                        mode='markers',
                        marker=dict(size=12, color='red', symbol='star'),
                        name='Breakeven Price',
                        hoverinfo='text',
                        hovertext=f'Breakeven: ${breakeven_price:.2f}'
                    ))
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Find the price needed for 20% margin
                    target_margin = 20.0
                    target_price = None
                    
                    for i in range(len(scenarios_df)-1):
                        if (scenarios_df.iloc[i]["margin"] <= target_margin and scenarios_df.iloc[i+1]["margin"] > target_margin):
                            # Simple linear interpolation
                            price1 = scenarios_df.iloc[i]["msrp"]
                            price2 = scenarios_df.iloc[i+1]["msrp"]
                            margin1 = scenarios_df.iloc[i]["margin"]
                            margin2 = scenarios_df.iloc[i+1]["margin"]
                            
                            target_price = price1 + (target_margin - margin1) * (price2 - price1) / (margin2 - margin1)
                            break
                    
                    landed_cost = scenarios_df.iloc[0]["landed_cost"]
                    
                    if target_price is not None:
                        st.markdown(f"""
                        <div class='result-box'>
                            <h3>Pricing Recommendations</h3>
                            <ul>
                                <li><strong>Breakeven Price:</strong> ${landed_cost:.2f}</li>
                                <li><strong>Minimum Recommended Price:</strong> ${landed_cost * 1.05:.2f} (5% margin)</li>
                                <li><strong>Price for 20% Margin:</strong> ${target_price:.2f}</li>
                            </ul>
                            <p>With a {fixed_tariff}% tariff rate and your current cost structure, these are the key price points to consider.</p>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div class='result-box'>
                            <h3>Pricing Recommendations</h3>
                            <ul>
                                <li><strong>Breakeven Price:</strong> ${landed_cost:.2f}</li>
                                <li><strong>Minimum Recommended Price:</strong> ${landed_cost * 1.05:.2f} (5% margin)</li>
                            </ul>
                            <p>With a {fixed_tariff}% tariff rate and your current cost structure, these are the key price points to consider.</p>
                        </div>
                        """, unsafe_allow_html=True)
    
    # Marketing ROI Tab
    with tabs[2]:
        st.markdown("<h2 class='sub-header'>Marketing ROI Calculator</h2>", unsafe_allow_html=True)
        
        st.info("""
        This calculator helps you estimate the impact of price changes and advertising spend on your overall profitability.
        Enter your current metrics and proposed changes to see how they affect your bottom line.
        """)
        
        # Basic product info
        col1, col2, col3 = st.columns(3)
        
        with col1:
            product_sku = st.text_input(
                "Product SKU",
                value="",
                key="roi_sku",
                help="Enter your product's SKU or identifier. Example: 'MOB1027'"
            )
        
        with col2:
            product_name = st.text_input(
                "Product Name",
                value="",
                key="roi_name",
                help="Enter the product name. Example: '4-Wheel Mobility Scooter'"
            )
        
        with col3:
            product_asin = st.text_input(
                "ASIN/UPC/EAN (Optional)",
                value="",
                key="roi_asin",
                help="Enter the product's marketplace identifier if applicable. Example: 'B07XYZABC1'"
            )
        
        # Current metrics
        st.markdown("<h3>Current Metrics</h3>", unsafe_allow_html=True)
        
        col4, col5, col6 = st.columns(3)
        
        with col4:
            current_price = st.number_input(
                "Current Sales Price ($)",
                min_value=0.01,
                value=100.00,
                step=0.01,
                key="roi_current_price",
                help="The current selling price of your product."
            )
            
            current_sales_qty = st.number_input(
                "Current Monthly Sales (Units)",
                min_value=1,
                value=750,
                step=10,
                key="roi_current_sales",
                help="Average number of units sold per month at the current price."
            )
        
        with col5:
            current_ad_spend = st.number_input(
                "Current Monthly Ad Spend ($)",
                min_value=0.0,
                value=15000.0,
                step=100.0,
                key="roi_current_ad",
                help="Your current monthly advertising budget for this product."
            )
            
            current_return_rate = st.slider(
                "Current Return Rate (%)",
                min_value=0.0,
                max_value=50.0,
                value=3.0,
                step=0.1,
                key="roi_current_return",
                help="Percentage of products currently being returned by customers."
            )
        
        with col6:
            cost_to_produce = st.number_input(
                "Manufacturing Cost per Unit ($)",
                min_value=0.01,
                value=40.00,
                step=0.01,
                key="roi_cost",
                help="Cost to produce or acquire each unit, excluding import costs."
            )
            
            tariff_rate_roi = st.number_input(
                "Import Tariff Rate (%)",
                min_value=0.0,
                value=25.0,
                step=0.5,
                key="roi_tariff",
                help="The tariff rate applied to the product's manufacturing cost."
            )
        
        # Display current sales dollars
        current_sales_dollars = current_price * current_sales_qty
        st.markdown(f"""
        <div class='result-box'>
            <h4>Current Monthly Sales Revenue: <span class='positive-value'>${current_sales_dollars:,.2f}</span></h4>
        </div>
        """, unsafe_allow_html=True)
        
        # Divider
        st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)
        
        # Proposed changes
        st.markdown("<h3>Proposed Changes</h3>", unsafe_allow_html=True)
        
        col7, col8, col9 = st.columns(3)
        
        with col7:
            proposed_price = st.number_input(
                "Proposed Sales Price ($)",
                min_value=0.01,
                value=current_price,
                step=0.01,
                key="roi_proposed_price",
                help="The new selling price you want to evaluate."
            )
            
            estimated_sales_change = st.slider(
                "Estimated Sales Volume Change (%)",
                min_value=-50.0,
                max_value=200.0,
                value=-10.0,
                step=1.0,
                key="roi_sales_change",
                help="Expected percentage change in sales volume after price and/or ad spend changes. Negative for decreased volume, positive for increased volume."
            )
        
        with col8:
            proposed_ad_spend = st.number_input(
                "Proposed Monthly Ad Spend ($)",
                min_value=0.0,
                value=7500.0,
                step=100.0,
                key="roi_proposed_ad",
                help="Your planned new monthly advertising budget for this product."
            )
            
            expected_return_rate = st.slider(
                "Expected Return Rate (%)",
                min_value=0.0,
                max_value=50.0,
                value=current_return_rate,
                step=0.1,
                key="roi_expected_return",
                help="Expected percentage of returns after the proposed changes."
            )
            
        with col9:
            sensitivity_factor = st.slider(
                "Ad Spend Sensitivity",
                min_value=0.1,
                max_value=5.0,
                value=1.0,
                step=0.1,
                key="roi_sensitivity",
                help="How responsive are sales to ad spend changes? Higher values mean more sales impact per ad dollar (1.0 is neutral)."
            )
            
            price_elasticity = st.slider(
                "Price Elasticity",
                min_value=0.1,
                max_value=5.0,
                value=1.0,
                step=0.1,
                key="roi_elasticity",
                help="How responsive are sales to price changes? Higher values mean more sales impact from price changes (1.0 is neutral)."
            )
        
        # Calculate button
        if st.button(
            "Calculate Marketing ROI",
            key="calc_roi_button",
            help="Calculate the projected impact of your proposed changes."
        ):
            with st.spinner("Calculating marketing ROI..."):
                # Convert percentage to decimal for calculation
                estimated_sales_change_decimal = estimated_sales_change / 100
                current_return_rate_decimal = current_return_rate / 100
                expected_return_rate_decimal = expected_return_rate / 100
                
                # Calculate ROI
                roi_result = calculate_ad_roi(
                    current_price, proposed_price, current_ad_spend, proposed_ad_spend,
                    current_sales_qty, estimated_sales_change_decimal, current_return_rate_decimal,
                    expected_return_rate_decimal, cost_to_produce, tariff_rate_roi
                )
                
                # Display results
                st.markdown("<h3>ROI Analysis Results</h3>", unsafe_allow_html=True)
                
                # Key metrics comparison
                col10, col11, col12 = st.columns(3)
                
                with col10:
                    st.markdown("<p><strong>Sales Volume</strong></p>", unsafe_allow_html=True)
                    
                    sales_change = roi_result["sales_qty_change"]
                    sales_change_pct = (sales_change / roi_result["current_sales_qty"]) * 100 if roi_result["current_sales_qty"] > 0 else 0
                    
                    st.markdown(f"""
                    <div style="display: flex; justify-content: space-between; margin-bottom: 10px;">
                        <div>Current:</div>
                        <div>{roi_result["current_sales_qty"]:.0f} units</div>
                    </div>
                    <div style="display: flex; justify-content: space-between; margin-bottom: 10px;">
                        <div>New:</div>
                        <div>{roi_result["new_sales_qty"]:.0f} units</div>
                    </div>
                    <div style="display: flex; justify-content: space-between; margin-bottom: 10px;">
                        <div>Change:</div>
                        <div class="{sales_change >= 0 and 'positive-value' or 'negative-value'}">{sales_change:.0f} units ({sales_change_pct:.1f}%)</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col11:
                    st.markdown("<p><strong>Monthly Revenue</strong></p>", unsafe_allow_html=True)
                    
                    revenue_change = roi_result["sales_dollars_change"]
                    revenue_change_pct = (revenue_change / roi_result["current_sales_dollars"]) * 100 if roi_result["current_sales_dollars"] > 0 else 0
                    
                    st.markdown(f"""
                    <div style="display: flex; justify-content: space-between; margin-bottom: 10px;">
                        <div>Current:</div>
                        <div>${roi_result["current_sales_dollars"]:,.2f}</div>
                    </div>
                    <div style="display: flex; justify-content: space-between; margin-bottom: 10px;">
                        <div>New:</div>
                        <div>${roi_result["new_sales_dollars"]:,.2f}</div>
                    </div>
                    <div style="display: flex; justify-content: space-between; margin-bottom: 10px;">
                        <div>Change:</div>
                        <div class="{revenue_change >= 0 and 'positive-value' or 'negative-value'}">${revenue_change:,.2f} ({revenue_change_pct:.1f}%)</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col12:
                    st.markdown("<p><strong>Monthly Profit (After Ad Spend)</strong></p>", unsafe_allow_html=True)
                    
                    profit_change = roi_result["net_profit_change"]
                    profit_change_pct = (profit_change / roi_result["current_profit_after_ads"]) * 100 if roi_result["current_profit_after_ads"] > 0 else 0
                    
                    st.markdown(f"""
                    <div style="display: flex; justify-content: space-between; margin-bottom: 10px;">
                        <div>Current:</div>
                        <div>${roi_result["current_profit_after_ads"]:,.2f}</div>
                    </div>
                    <div style="display: flex; justify-content: space-between; margin-bottom: 10px;">
                        <div>New:</div>
                        <div>${roi_result["new_profit_after_ads"]:,.2f}</div>
                    </div>
                    <div style="display: flex; justify-content: space-between; margin-bottom: 10px;">
                        <div>Change:</div>
                        <div class="{profit_change >= 0 and 'positive-value' or 'negative-value'}">${profit_change:,.2f} ({profit_change_pct:.1f}%)</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Monthly ad spend comparison
                st.markdown("<br>", unsafe_allow_html=True)
                col13, col14 = st.columns(2)
                
                with col13:
                    ad_change = roi_result["ad_spend_change"]
                    ad_change_pct = (ad_change / roi_result["current_ad_spend"]) * 100 if roi_result["current_ad_spend"] > 0 else 0
                    
                    st.markdown(f"""
                    <div class="metric-card">
                        <p class="metric-label">Ad Spend Change</p>
                        <p class="metric-value" style="color: {ad_change <= 0 and '#4CAF50' or '#F44336'}">
                            ${ad_change:,.2f} ({ad_change_pct:.1f}%)
                        </p>
                        <p class="metric-label">From ${roi_result["current_ad_spend"]:,.2f} to ${roi_result["proposed_ad_spend"]:,.2f}</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col14:
                    if roi_result["ad_spend_change"] > 0 and roi_result["roi_percentage"] > 0:
                        roi_color = "#4CAF50"  # Green for positive ROI
                    elif roi_result["ad_spend_change"] > 0:
                        roi_color = "#F44336"  # Red for negative ROI
                    else:
                        roi_color = "#2196F3"  # Blue for ad spend reduction
                    
                    roi_text = f"{roi_result['roi_percentage']:.1f}%" if roi_result["ad_spend_change"] > 0 else "N/A (Ad spend decreased)"
                    
                    st.markdown(f"""
                    <div class="metric-card">
                        <p class="metric-label">Return on Ad Spend Investment (ROI)</p>
                        <p class="metric-value" style="color: {roi_color}">
                            {roi_text}
                        </p>
                        <p class="metric-label">Net profit change per additional ad dollar</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Net benefit calculation
                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown(f"""
                <div class="result-box">
                    <h3>Net Benefit Analysis</h3>
                    <p>Based on your inputs, the proposed changes are projected to result in a <span class="{profit_change >= 0 and 'positive-value' or 'negative-value'}">${profit_change:,.2f}</span> change in monthly profit.</p>
                    <p>This equals <span class="{profit_change*12 >= 0 and 'positive-value' or 'negative-value'}">${profit_change*12:,.2f}</span> in annual net benefit.</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Breakeven analysis
                if roi_result["ad_spend_change"] > 0:
                    st.markdown("<h3>Breakeven Analysis</h3>", unsafe_allow_html=True)
                    
                    breakeven_units = roi_result["breakeven_additional_units"]
                    breakeven_pct = roi_result["breakeven_sales_change"] * 100
                    
                    st.markdown(f"""
                    <p>To break even on the additional ${roi_result["ad_spend_change"]:,.2f} in ad spend, you need:</p>
                    <ul>
                        <li>Additional <strong>{breakeven_units:.0f} units sold</strong></li>
                        <li>A sales volume increase of <strong>{breakeven_pct:.1f}%</strong></li>
                    </ul>
                    
                    <div style="margin-top: 15px;">
                        <p>Your projected change is <strong>{estimated_sales_change:.1f}%</strong> ({roi_result["sales_qty_change"]:.0f} units), which is 
                        <span class="{estimated_sales_change >= breakeven_pct and 'positive-value' or 'negative-value'}">
                            {estimated_sales_change >= breakeven_pct and 'above' or 'below'} the breakeven point
                        </span>.</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Create visualization of different ad spend levels
                st.markdown("<h3>Ad Spend Scenario Analysis</h3>", unsafe_allow_html=True)
                
                # Generate ad spend scenarios
                est_change_per_ad_dollar = estimated_sales_change_decimal / ((proposed_ad_spend / current_ad_spend) - 1) if current_ad_spend > 0 and proposed_ad_spend > current_ad_spend else 0.1
                
                ad_scenarios = generate_ad_spend_scenarios(
                    current_price, proposed_price, current_ad_spend,
                    current_sales_qty, est_change_per_ad_dollar,
                    min_spend_factor=0.5, max_spend_factor=2.0, steps=10,
                    current_return_rate=current_return_rate_decimal,
                    expected_return_rate=expected_return_rate_decimal,
                    cost_to_produce=cost_to_produce, tariff_rate=tariff_rate_roi
                )
                
                # Create visualization
                fig = go.Figure()
                
                # Add profit line
                fig.add_trace(go.Scatter(
                    x=ad_scenarios["ad_spend"],
                    y=ad_scenarios["profit"],
                    mode='lines+markers',
                    name='Monthly Profit',
                    line=dict(color='#4CAF50', width=3)
                ))
                
                # Mark current and proposed points
                fig.add_trace(go.Scatter(
                    x=[current_ad_spend],
                    y=[roi_result["current_profit_after_ads"]],
                    mode='markers',
                    marker=dict(size=12, color='blue', symbol='circle'),
                    name='Current Ad Spend',
                    hoverinfo='text',
                    hovertext=f'Current: ${current_ad_spend:,.2f} → ${roi_result["current_profit_after_ads"]:,.2f} profit'
                ))
                
                fig.add_trace(go.Scatter(
                    x=[proposed_ad_spend],
                    y=[roi_result["new_profit_after_ads"]],
                    mode='markers',
                    marker=dict(size=12, color='red', symbol='star'),
                    name='Proposed Ad Spend',
                    hoverinfo='text',
                    hovertext=f'Proposed: ${proposed_ad_spend:,.2f} → ${roi_result["new_profit_after_ads"]:,.2f} profit'
                ))
                
                # Update layout
                fig.update_layout(
                    title='Monthly Profit at Different Ad Spend Levels',
                    xaxis=dict(title='Monthly Ad Spend ($)'),
                    yaxis=dict(title='Monthly Profit ($)'),
                    legend=dict(x=0.01, y=0.99),
                    margin=dict(t=50, b=50, l=50, r=50),
                    hovermode='closest'
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Find optimal ad spend (max profit point)
                optimal_idx = ad_scenarios["profit"].idxmax()
                optimal_ad_spend = ad_scenarios.iloc[optimal_idx]["ad_spend"]
                optimal_profit = ad_scenarios.iloc[optimal_idx]["profit"]
                
                st.markdown(f"""
                <div class="result-box">
                    <h3>Optimal Ad Spend</h3>
                    <p>Based on the modeled relationship between ad spend and sales, the optimal monthly ad spend appears to be around <strong>${optimal_ad_spend:,.2f}</strong>, which would result in an estimated monthly profit of <strong>${optimal_profit:,.2f}</strong>.</p>
                    <p><em>Note: This is a simplified model and actual results may vary based on market conditions and advertising effectiveness.</em></p>
                </div>
                """, unsafe_allow_html=True)
        
        # Ad spend vs. sales sensitivity analysis
        with st.expander("Ad Spend Sensitivity Analysis", expanded=False):
            st.markdown("""
            ### Understanding Ad Spend Sensitivity
            
            The relationship between advertising spend and sales can vary greatly between products, markets, and channels. Use this section to model different sensitivity levels.
            
            **What affects ad spend sensitivity:**
            - Market saturation
            - Ad channel effectiveness
            - Product category competitiveness
            - Current brand awareness
            - Price point
            """)
            
            col_sens1, col_sens2 = st.columns(2)
            
            with col_sens1:
                base_ad_spend = st.number_input(
                    "Base Monthly Ad Spend ($)",
                    min_value=100.0,
                    value=10000.0,
                    step=1000.0,
                    key="sens_base_ad",
                    help="Starting point for ad spend analysis."
                )
                
                base_sales = st.number_input(
                    "Base Monthly Sales (Units)",
                    min_value=10,
                    value=500,
                    step=50,
                    key="sens_base_sales",
                    help="Monthly sales at the base ad spend level."
                )
            
            with col_sens2:
                low_sensitivity = st.slider(
                    "Low Sensitivity (% sales increase per 10% ad increase)",
                    min_value=0.1,
                    max_value=10.0,
                    value=1.0,
                    step=0.1,
                    key="sens_low",
                    help="Conservative estimate: how much sales increase with a 10% ad spend increase."
                )
                
                high_sensitivity = st.slider(
                    "High Sensitivity (% sales increase per 10% ad increase)",
                    min_value=0.1,
                    max_value=20.0,
                    value=5.0,
                    step=0.1,
                    key="sens_high",
                    help="Optimistic estimate: how much sales increase with a 10% ad spend increase."
                )
            
            if st.button("Generate Sensitivity Analysis", key="sens_button"):
                # Generate ad spend range (50% to 200% of base)
                ad_factors = np.linspace(0.5, 2.0, 10)
                ad_spend_levels = base_ad_spend * ad_factors
                
                # Calculate sales at different sensitivity levels
                low_sens_sales = []
                high_sens_sales = []
                
                for factor in ad_factors:
                    # Calculate relative change from base
                    pct_change = (factor - 1) * 100  # percentage change from base
                    
                    # Calculate sales change at different sensitivities
                    # For each 10% change in ad spend, sales change by sensitivity %
                    low_change_pct = pct_change * (low_sensitivity / 10)
                    high_change_pct = pct_change * (high_sensitivity / 10)
                    
                    # Calculate resulting sales
                    low_sens_sales.append(base_sales * (1 + low_change_pct/100))
                    high_sens_sales.append(base_sales * (1 + high_change_pct/100))
                
                # Create dataframe
                sens_df = pd.DataFrame({
                    "Ad Spend": ad_spend_levels,
                    "Low Sensitivity Sales": low_sens_sales,
                    "High Sensitivity Sales": high_sens_sales
                })
                
                # Create visualization
                fig_sens = go.Figure()
                
                # Add low sensitivity line
                fig_sens.add_trace(go.Scatter(
                    x=sens_df["Ad Spend"],
                    y=sens_df["Low Sensitivity Sales"],
                    mode='lines+markers',
                    name=f'Low Sensitivity ({low_sensitivity}%)',
                    line=dict(color='#FFA000', width=3)
                ))
                
                # Add high sensitivity line
                fig_sens.add_trace(go.Scatter(
                    x=sens_df["Ad Spend"],
                    y=sens_df["High Sensitivity Sales"],
                    mode='lines+markers',
                    name=f'High Sensitivity ({high_sensitivity}%)',
                    line=dict(color='#4CAF50', width=3)
                ))
                
                # Mark base point
                fig_sens.add_trace(go.Scatter(
                    x=[base_ad_spend],
                    y=[base_sales],
                    mode='markers',
                    marker=dict(size=12, color='blue', symbol='circle'),
                    name='Base Point',
                    hoverinfo='text',
                    hovertext=f'Base: ${base_ad_spend:,.2f} → {base_sales} units'
                ))
                
                # Update layout
                fig_sens.update_layout(
                    title='Impact of Ad Spend on Sales at Different Sensitivity Levels',
                    xaxis=dict(title='Monthly Ad Spend ($)'),
                    yaxis=dict(title='Monthly Sales (Units)'),
                    legend=dict(x=0.01, y=0.99),
                    margin=dict(t=50, b=50, l=50, r=50),
                    hovermode='closest'
                )
                
                st.plotly_chart(fig_sens, use_container_width=True)
                
                # Display interpretation
                st.markdown(f"""
                <div class="info-box">
                    <h4>Interpretation</h4>
                    <p>This chart shows how different ad sensitivity levels affect sales projections:</p>
                    <ul>
                        <li><strong>Low Sensitivity ({low_sensitivity}%):</strong> For every 10% increase in ad spend, sales increase by {low_sensitivity}%</li>
                        <li><strong>High Sensitivity ({high_sensitivity}%):</strong> For every 10% increase in ad spend, sales increase by {high_sensitivity}%</li>
                    </ul>
                    <p>If your product has low ad sensitivity, increasing ad spend may not be cost-effective. With high sensitivity, increased advertising could significantly boost sales.</p>
                </div>
                """, unsafe_allow_html=True)
    
    # Tariff Resources Tab
    with tabs[3]:
        st.markdown("<h2 class='sub-header'>Tariff Resources</h2>", unsafe_allow_html=True)
        
        # Information about tariffs
        st.markdown("""
        <div class='info-box'>
            <h3>Understanding Import Tariffs</h3>
            <p>Tariffs are taxes imposed on imported goods and services. They are typically calculated as a percentage of the import's value.</p>
            <p>Tariff rates vary widely based on:</p>
            <ul>
                <li>Product category and harmonized system (HS) code</li>
                <li>Country of origin</li>
                <li>Trade agreements between countries</li>
                <li>Special trade statuses or exceptions</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        # Create two columns for resources
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("<h3>Official Tariff Resources</h3>", unsafe_allow_html=True)
            st.markdown("""
            * [U.S. International Trade Commission Tariff Database](https://hts.usitc.gov/) - Official HTSUS tariff information
            * [U.S. Customs and Border Protection](https://www.cbp.gov/trade/programs-administration/entry-summary/tariff-resources)
            * [European Union Trade Tariff Database](https://taxation-customs.ec.europa.eu/eu-customs-tariff_en)
            * [World Trade Organization Tariff Database](https://tao.wto.org/)
            """)
            
            st.markdown("<h3>Product Classification</h3>", unsafe_allow_html=True)
            st.markdown("""
            * [Harmonized System (HS) Classification](https://www.trade.gov/harmonized-system-hs-codes)
            * [Schedule B Search Engine](https://uscensus.prod.3ceonline.com/) - Find export codes for your products
            * [CROSS Rulings Database](https://rulings.cbp.gov/home) - Search customs rulings
            """)
        
        with col2:
            st.markdown("<h3>Trade Agreements & Resources</h3>", unsafe_allow_html=True)
            st.markdown("""
            * [USTR Free Trade Agreements](https://ustr.gov/trade-agreements/free-trade-agreements)
            * [International Trade Administration](https://www.trade.gov/)
            * [Global Trade Helpdesk](https://globaltradehelpdesk.org/en)
            * [CBP Trade Relief Programs](https://www.cbp.gov/trade/programs-administration/trade-remedies)
            """)
            
            st.markdown("<h3>Tariff Calculators & Tools</h3>", unsafe_allow_html=True)
            st.markdown("""
            * [CBP Duty Calculator](https://dataweb.usitc.gov/tariff/calculate)
            * [Shipping Solutions Trade Wizards](https://www.shippingsolutions.com/trade-wizards)
            * [Flexport Duty Calculator](https://www.flexport.com/tools/duty-calculator/)
            * [DHL Customs Duty Calculator](https://dhlguide.co.uk/tools-and-services/customs-duty-calculator/)
            """)
        
        # Recent tariff news
        st.markdown("<h3>Finding Current Tariff Rates</h3>", unsafe_allow_html=True)
        st.markdown("""
        To find the most current tariff rates for your specific product:
        
        1. **Determine your product's HS code** - This 6-10 digit classification code determines which tariff rates apply
        2. **Check the official tariff database** for your importing country (links above)
        3. **Consider country of origin** - Rates vary based on trade relationships
        4. **Check for special programs** - Various duty reduction or elimination programs may apply
        5. **Consult with a customs broker** - For complex products or situations, professional advice is recommended
        
        Remember that tariff rates can change due to policy changes, trade disputes, or new trade agreements.
        """)
        
        st.markdown("""
        <div class='warning-box'>
            <h3>Disclaimer</h3>
            <p>This calculator provides estimates only and should not be considered tax, legal, or customs advice. Actual duties, taxes, and fees may vary. Always consult with qualified customs professionals for official guidance.</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Display saved calculations
        if len(st.session_state.calculations) > 0:
            st.markdown("<h3>Your Recent Calculations</h3>", unsafe_allow_html=True)
            
            # Create a dataframe of saved calculations
            saved_df = pd.DataFrame(st.session_state.calculations)
            
            # Format for display
            display_saved = saved_df.copy()
            display_saved["msrp"] = display_saved["msrp"].map("${:.2f}".format)
            display_saved["cost"] = display_saved["cost"].map("${:.2f}".format)
            display_saved["tariff_rate"] = display_saved["tariff_rate"].astype(str) + "%"
            display_saved["landed_cost"] = display_saved["landed_cost"].map("${:.2f}".format)
            display_saved["profit"] = display_saved["profit"].map("${:.2f}".format)
            display_saved["margin"] = display_saved["margin"].round(1).astype(str) + "%"
            
            # Rename columns
            display_saved.columns = ["Timestamp", "Product", "SKU", "MSRP", "Manufacturing Cost", 
                                    "Tariff Rate", "Landed Cost", "Profit", "Margin"]
            
            st.dataframe(display_saved, use_container_width=True)
            
            if st.button(
                "Clear History", 
                help="Click to remove all saved calculation history. This cannot be undone."
            ):
                st.session_state.calculations = []
                st.rerun()

if __name__ == "__main__":
    main()
