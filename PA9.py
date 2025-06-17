"""
FDA Medical Device Return Analyzer
Version: 8.0 - Reportable Event Detection System
Purpose: Identify potential FDA MDR (Medical Device Reporting) events
"""

import streamlit as st
import pandas as pd
import numpy as np
import logging
from datetime import datetime
import io
from typing import Dict, List, Any, Optional
import plotly.express as px
import plotly.graph_objects as go
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configure page - MUST BE FIRST STREAMLIT COMMAND
st.set_page_config(
    page_title="FDA Medical Device Return Analyzer",
    page_icon="🚨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Import enhanced AI analysis module
try:
    from enhanced_ai_analysis import (
        EnhancedAIAnalyzer, AIProvider, MEDICAL_DEVICE_CATEGORIES,
        FDA_MDR_TRIGGERS, detect_fda_reportable_event, FileProcessor
    )
    AI_AVAILABLE = True
except ImportError as e:
    AI_AVAILABLE = False
    logger.error(f"Enhanced AI analysis not available: {e}")
    st.error("Critical module missing. Please check installation.")

# App styling
def apply_custom_css():
    """Apply custom CSS for professional appearance"""
    st.markdown("""
    <style>
    /* Main theme */
    .stApp {
        background-color: #f5f5f5;
    }
    
    /* Headers */
    h1, h2, h3 {
        color: #1e3a8a;
    }
    
    /* FDA Alert styling */
    .fda-alert {
        background-color: #dc2626;
        color: white;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
        font-weight: bold;
    }
    
    /* Reportable event card */
    .reportable-event {
        background-color: #fee2e2;
        border: 2px solid #dc2626;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    
    /* Metrics */
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 0.5rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.12);
        text-align: center;
    }
    
    .metric-value {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1e3a8a;
    }
    
    .metric-label {
        color: #6b7280;
        font-size: 0.875rem;
        text-transform: uppercase;
    }
    
    /* Critical events */
    .critical {
        color: #dc2626;
        font-weight: bold;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    </style>
    """, unsafe_allow_html=True)

def initialize_session_state():
    """Initialize session state variables"""
    defaults = {
        'data': None,
        'processed_data': None,
        'fda_report': None,
        'file_uploaded': False,
        'processing_complete': False,
        'ai_analyzer': None,
        'reportable_events': 0,
        'critical_events': 0
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

def display_header():
    """Display app header with FDA focus"""
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.title("🚨 FDA Medical Device Return Analyzer")
        st.markdown("""
        **Purpose**: Identify potential FDA reportable events in product returns
        
        **Detects**: Deaths, serious injuries, falls, malfunctions, allergic reactions, and other adverse events
        """)
    
    with col2:
        st.info("""
        **FDA MDR Requirements**
        - Report deaths immediately
        - Report serious injuries within 30 days
        - Document all adverse events
        """)

def process_file(uploaded_file) -> pd.DataFrame:
    """Process uploaded file of any supported format"""
    try:
        file_type = uploaded_file.type.split('/')[-1]
        
        # Use FileProcessor for universal file handling
        if AI_AVAILABLE:
            df = FileProcessor.read_file(uploaded_file, uploaded_file.type)
        else:
            # Fallback to basic CSV reading
            df = pd.read_csv(uploaded_file)
        
        # Standardize column names
        df.columns = df.columns.str.lower().str.strip()
        
        # Check for required columns
        required_cols = ['reason', 'customer-comments']
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            # Try alternative column names
            alt_names = {
                'reason': ['return reason', 'return_reason', 'reason_code'],
                'customer-comments': ['comments', 'customer comments', 'customer_comments', 'notes']
            }
            
            for missing, alternatives in alt_names.items():
                if missing in missing_cols:
                    for alt in alternatives:
                        if alt in df.columns:
                            df.rename(columns={alt: missing}, inplace=True)
                            missing_cols.remove(missing)
                            break
        
        return df
        
    except Exception as e:
        st.error(f"Error processing file: {str(e)}")
        return None

def display_fda_summary(report: Dict[str, Any]):
    """Display FDA-focused summary with alerts"""
    if report['summary']['reportable_events'] > 0:
        st.markdown("""
        <div class="fda-alert">
        ⚠️ FDA REPORTABLE EVENTS DETECTED - IMMEDIATE REVIEW REQUIRED
        </div>
        """, unsafe_allow_html=True)
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{report['summary']['total_returns']}</div>
            <div class="metric-label">Total Returns</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value critical">{report['summary']['reportable_events']}</div>
            <div class="metric-label">Reportable Events</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value critical">{report['summary']['critical_events']}</div>
            <div class="metric-label">Critical Severity</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{report['summary']['mdr_required']}</div>
            <div class="metric-label">MDR Required</div>
        </div>
        """, unsafe_allow_html=True)
    
    # Event type breakdown
    if report['by_event_type']:
        st.subheader("📊 Events by Type")
        
        # Create bar chart
        event_df = pd.DataFrame(
            list(report['by_event_type'].items()),
            columns=['Event Type', 'Count']
        )
        
        fig = px.bar(
            event_df, 
            x='Event Type', 
            y='Count',
            title='FDA Reportable Events by Type',
            color='Count',
            color_continuous_scale=['#fee2e2', '#dc2626']
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Affected products
    if report['affected_products']:
        st.subheader("🏥 Most Affected Products")
        
        products_df = pd.DataFrame.from_dict(
            report['affected_products'], 
            orient='index'
        ).reset_index()
        products_df.columns = ['Product', 'ASIN', 'Events', 'MDR Required']
        
        st.dataframe(
            products_df,
            use_container_width=True,
            hide_index=True
        )
    
    # Recommendations
    if report['recommendations']:
        st.subheader("⚡ Immediate Actions Required")
        for rec in report['recommendations']:
            st.warning(f"• {rec}")

def main():
    """Main application flow"""
    # Initialize
    initialize_session_state()
    apply_custom_css()
    display_header()
    
    # Sidebar
    with st.sidebar:
        st.header("📁 File Upload")
        
        uploaded_file = st.file_uploader(
            "Choose a file",
            type=['csv', 'xlsx', 'xls', 'tsv', 'txt', 'pdf'],
            help="Upload returns data from Amazon Seller Central"
        )
        
        if uploaded_file:
            st.success(f"File uploaded: {uploaded_file.name}")
            st.session_state.file_uploaded = True
        
        st.divider()
        
        # AI Provider selection
        if AI_AVAILABLE:
            st.subheader("🤖 AI Settings")
            provider = st.selectbox(
                "AI Provider",
                options=[AIProvider.FASTEST, AIProvider.QUALITY, AIProvider.OPENAI, AIProvider.CLAUDE],
                help="Select AI provider for analysis"
            )
        
        st.divider()
        
        # Info
        st.info("""
        **Supported Formats:**
        - CSV, TSV, TXT
        - Excel (XLSX, XLS)
        - PDF (with tables)
        
        **Required Columns:**
        - reason / return_reason
        - customer-comments / comments
        """)
    
    # Main content
    if st.session_state.file_uploaded and uploaded_file:
        # Process file
        if st.button("🔍 Analyze for FDA Reportable Events", type="primary", use_container_width=True):
            with st.spinner("Processing file and detecting reportable events..."):
                # Read file
                df = process_file(uploaded_file)
                
                if df is not None:
                    st.session_state.data = df
                    
                    # Initialize AI analyzer
                    if AI_AVAILABLE and not st.session_state.ai_analyzer:
                        st.session_state.ai_analyzer = EnhancedAIAnalyzer(provider)
                    
                    # Analyze returns
                    if st.session_state.ai_analyzer:
                        processed_df = st.session_state.ai_analyzer.batch_categorize(df)
                        st.session_state.processed_data = processed_df
                        
                        # Generate FDA report
                        fda_report = st.session_state.ai_analyzer.generate_fda_report(processed_df)
                        st.session_state.fda_report = fda_report
                        st.session_state.processing_complete = True
                    else:
                        st.error("AI analyzer not available")
    
    # Display results
    if st.session_state.processing_complete:
        st.divider()
        
        # FDA Summary
        display_fda_summary(st.session_state.fda_report)
        
        # Detailed reportable events
        if st.session_state.fda_report['summary']['reportable_events'] > 0:
            st.divider()
            st.subheader("🚨 Detailed Reportable Events")
            
            reportable_df = st.session_state.processed_data[
                st.session_state.processed_data['fda_reportable'] == True
            ]
            
            # Add severity color coding
            def severity_color(severity):
                colors = {
                    'CRITICAL': '#dc2626',
                    'HIGH': '#f59e0b',
                    'MODERATE': '#3b82f6',
                    'LOW': '#10b981'
                }
                return f'background-color: {colors.get(severity, "#e5e7eb")}'
            
            # Display with styling
            st.dataframe(
                reportable_df[[
                    'order-id', 'asin', 'product-name', 'reason',
                    'customer-comments', 'severity', 'event_types', 'requires_mdr'
                ]].style.applymap(
                    lambda x: severity_color(x) if isinstance(x, str) and x in ['CRITICAL', 'HIGH', 'MODERATE', 'LOW'] else '',
                    subset=['severity']
                ),
                use_container_width=True,
                height=400
            )
        
        # Export options
        st.divider()
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # Export FDA summary
            if st.button("📄 Export FDA Summary", use_container_width=True):
                if st.session_state.ai_analyzer:
                    summary_df = st.session_state.ai_analyzer.export_fda_summary(
                        st.session_state.processed_data
                    )
                    
                    if not summary_df.empty:
                        csv = summary_df.to_csv(index=False)
                        st.download_button(
                            label="Download FDA Summary CSV",
                            data=csv,
                            file_name=f"fda_reportable_events_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                            mime="text/csv"
                        )
        
        with col2:
            # Export full analysis
            if st.button("📊 Export Full Analysis", use_container_width=True):
                csv = st.session_state.processed_data.to_csv(index=False)
                st.download_button(
                    label="Download Full Analysis CSV",
                    data=csv,
                    file_name=f"return_analysis_full_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
        
        with col3:
            # Generate MDR template
            if st.session_state.fda_report['summary']['mdr_required'] > 0:
                if st.button("📋 Generate MDR Template", use_container_width=True):
                    st.info("MDR form template generation would be implemented here")
    
    else:
        # Instructions
        st.markdown("""
        ### 📋 How to Use This Tool
        
        1. **Export your returns data** from Amazon Seller Central
        2. **Upload the file** using the sidebar (supports CSV, Excel, PDF, etc.)
        3. **Click Analyze** to detect FDA reportable events
        4. **Review results** and take immediate action on critical events
        5. **Export summaries** for FDA MDR submission
        
        ### 🎯 What We Detect
        
        - **Deaths** - Immediate reporting required
        - **Serious Injuries** - Including hospitalizations, permanent impairments
        - **Falls** - Any fall, slip, or collapse related to product use
        - **Malfunctions** - Product failures that could cause harm
        - **Allergic Reactions** - Rashes, swelling, breathing issues
        - **Infections** - Contamination or infection risks
        
        ### ⚡ Quick Actions
        
        For any detected reportable events:
        1. Document all details immediately
        2. Initiate FDA MDR process
        3. Contact legal/regulatory team
        4. Preserve product samples if available
        """)

if __name__ == "__main__":
    main()
