"""
Medical Device Return PDF Analyzer - Injury Detection System
Version: 4.0 - Focused on Safety and Liability Detection
Designed for Quality Managers to identify potential injury cases from Amazon return PDFs
"""

import streamlit as st
import pandas as pd
import numpy as np
import logging
from datetime import datetime
import io
from typing import Dict, List, Any, Optional, Tuple
import re
import json
from collections import Counter, defaultdict
import plotly.express as px
import plotly.graph_objects as go
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import modules
from pdf_analyzer import PDFAnalyzer, InjuryAnalysis
from injury_detector import InjuryDetector, INJURY_KEYWORDS, SEVERITY_LEVELS

# App Configuration
st.set_page_config(
    page_title="Medical Device Return Analyzer",
    page_icon="⚠️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Professional color scheme with safety focus
COLORS = {
    'critical': '#DC143C',     # Crimson for injuries
    'high': '#FF6B35',         # Orange for high risk
    'medium': '#F39C12',       # Amber for medium risk
    'low': '#3498DB',          # Blue for low risk
    'safe': '#27AE60',         # Green for safe
    'neutral': '#95A5A6',      # Gray
    'dark': '#2C3E50',         # Dark blue
    'light': '#ECF0F1'         # Light gray
}

def inject_safety_css():
    """Safety-focused CSS styling"""
    st.markdown(f"""
    <style>
    /* Professional medical/safety styling */
    .main {{
        padding: 0;
        background-color: #FAFAFA;
    }}
    
    /* Header */
    .main-header {{
        background: linear-gradient(135deg, {COLORS['dark']} 0%, {COLORS['critical']} 100%);
        color: white;
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
    }}
    
    .main-header h1 {{
        margin: 0;
        font-size: 2.5rem;
        font-weight: 700;
    }}
    
    /* Critical alert box */
    .critical-alert {{
        background: linear-gradient(135deg, rgba(220, 20, 60, 0.1), rgba(220, 20, 60, 0.2));
        border: 2px solid {COLORS['critical']};
        border-radius: 10px;
        padding: 1.5rem;
        margin: 1rem 0;
        animation: pulse 2s infinite;
    }}
    
    @keyframes pulse {{
        0% {{ box-shadow: 0 0 0 0 rgba(220, 20, 60, 0.4); }}
        70% {{ box-shadow: 0 0 0 20px rgba(220, 20, 60, 0); }}
        100% {{ box-shadow: 0 0 0 0 rgba(220, 20, 60, 0); }}
    }}
    
    /* Severity badges */
    .severity-critical {{
        background: {COLORS['critical']};
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-weight: bold;
        display: inline-block;
    }}
    
    .severity-high {{
        background: {COLORS['high']};
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-weight: bold;
        display: inline-block;
    }}
    
    .severity-medium {{
        background: {COLORS['medium']};
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-weight: bold;
        display: inline-block;
    }}
    
    /* Info cards */
    .injury-card {{
        background: white;
        border-left: 5px solid {COLORS['critical']};
        border-radius: 8px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
    }}
    
    .metric-card {{
        background: white;
        border-radius: 8px;
        padding: 1.5rem;
        text-align: center;
        box-shadow: 0 2px 10px rgba(0,0,0,0.05);
        border: 1px solid #E0E0E0;
    }}
    
    .metric-value {{
        font-size: 2.5rem;
        font-weight: 700;
        margin: 0.5rem 0;
    }}
    
    .metric-label {{
        color: {COLORS['neutral']};
        font-size: 0.9rem;
        text-transform: uppercase;
        letter-spacing: 1px;
    }}
    
    /* Buttons */
    .stButton > button {{
        background: {COLORS['dark']};
        color: white;
        border: none;
        padding: 0.75rem 2rem;
        border-radius: 5px;
        font-weight: 600;
        transition: all 0.3s;
    }}
    
    .stButton > button:hover {{
        background: {COLORS['critical']};
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(0,0,0,0.2);
    }}
    
    /* Hide Streamlit branding */
    #MainMenu {{visibility: hidden;}}
    footer {{visibility: hidden;}}
    </style>
    """, unsafe_allow_html=True)

def initialize_session_state():
    """Initialize session state variables"""
    defaults = {
        # Data
        'pdf_data': None,
        'injury_analysis': None,
        'processed_returns': [],
        'critical_cases': [],
        
        # Processing state
        'processing_complete': False,
        'pdf_uploaded': False,
        'analysis_ready': False,
        
        # Filters
        'severity_filter': 'All',
        'date_filter': 'All Time',
        
        # AI components
        'pdf_analyzer': None,
        'injury_detector': None,
        
        # Tracking
        'total_returns_processed': 0,
        'injuries_found': 0,
        'processing_time': 0.0,
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

def display_header():
    """Display application header with safety focus"""
    st.markdown("""
    <div class="main-header">
        <h1>⚠️ Medical Device Return Analyzer</h1>
        <p style="font-size: 1.2rem; margin-top: 0.5rem;">
            AI-Powered Injury Detection for Amazon Return PDFs
        </p>
        <p style="font-size: 1rem; opacity: 0.9;">
            Identify potential liability cases • Extract critical return data • Ensure patient safety
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Show critical alerts if injuries found
    if st.session_state.injuries_found > 0:
        st.markdown(f"""
        <div class="critical-alert">
            <h2 style="color: {COLORS['critical']}; margin: 0;">
                🚨 {st.session_state.injuries_found} Potential Injury Cases Detected
            </h2>
            <p style="margin: 0.5rem 0 0 0;">Immediate review recommended for customer safety</p>
        </div>
        """, unsafe_allow_html=True)

def upload_pdf_section():
    """PDF upload section"""
    st.markdown("### 📄 Upload Amazon Return PDF")
    
    with st.expander("ℹ️ Instructions", expanded=True):
        st.markdown("""
        **How to export from Amazon Seller Central:**
        1. Go to **Manage Returns** in Seller Central
        2. Filter by date range and/or ASIN
        3. Click **Print** or **Export as PDF**
        4. Upload the PDF file here
        
        **What this tool analyzes:**
        - 🚨 **Injury Keywords**: hurt, injured, hospital, emergency, pain, bleeding, etc.
        - 📋 **Return Details**: Order ID, ASIN, SKU, dates
        - 💬 **Customer Comments**: Full text analysis
        - ⚠️ **Risk Assessment**: Severity scoring for each return
        """)
    
    uploaded_file = st.file_uploader(
        "Choose PDF file",
        type=['pdf'],
        help="Upload return report PDF from Amazon Seller Central"
    )
    
    if uploaded_file:
        process_pdf(uploaded_file)

def process_pdf(uploaded_file):
    """Process uploaded PDF file"""
    try:
        # Initialize analyzers if needed
        if st.session_state.pdf_analyzer is None:
            st.session_state.pdf_analyzer = PDFAnalyzer()
        if st.session_state.injury_detector is None:
            st.session_state.injury_detector = InjuryDetector()
        
        # Read PDF content
        pdf_content = uploaded_file.read()
        
        with st.spinner("🔍 Analyzing PDF for return data and potential injuries..."):
            start_time = time.time()
            
            # Extract return data from PDF
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Step 1: Extract text from PDF
            status_text.text("Extracting text from PDF...")
            progress_bar.progress(0.2)
            
            extracted_data = st.session_state.pdf_analyzer.extract_returns_from_pdf(
                pdf_content, 
                uploaded_file.name
            )
            
            if not extracted_data or not extracted_data.get('returns'):
                st.error("❌ No return data found in PDF. Please ensure this is a valid Amazon return report.")
                return
            
            # Step 2: Analyze for injuries
            status_text.text("Analyzing for potential injury cases...")
            progress_bar.progress(0.5)
            
            returns_data = extracted_data['returns']
            injury_analysis = st.session_state.injury_detector.analyze_returns_for_injuries(returns_data)
            
            # Step 3: Process results
            status_text.text("Processing results...")
            progress_bar.progress(0.8)
            
            # Store results
            st.session_state.pdf_data = extracted_data
            st.session_state.injury_analysis = injury_analysis
            st.session_state.processed_returns = returns_data
            st.session_state.critical_cases = injury_analysis['critical_cases']
            st.session_state.injuries_found = injury_analysis['total_injuries']
            st.session_state.total_returns_processed = len(returns_data)
            st.session_state.processing_time = time.time() - start_time
            st.session_state.processing_complete = True
            st.session_state.analysis_ready = True
            
            progress_bar.progress(1.0)
            status_text.empty()
            progress_bar.empty()
            
            # Success message
            st.success(f"""
            ✅ PDF Analysis Complete!
            - Total returns processed: {st.session_state.total_returns_processed}
            - Potential injury cases: {st.session_state.injuries_found}
            - Processing time: {st.session_state.processing_time:.1f} seconds
            """)
            
            if st.session_state.injuries_found > 0:
                st.balloons()
            
            # Refresh to show results
            st.rerun()
            
    except Exception as e:
        st.error(f"❌ Error processing PDF: {str(e)}")
        logger.error(f"PDF processing error: {e}", exc_info=True)

def display_injury_summary():
    """Display summary of injury findings"""
    if not st.session_state.analysis_ready:
        return
    
    st.markdown("### 🚨 Injury Analysis Summary")
    
    analysis = st.session_state.injury_analysis
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Total Returns</div>
            <div class="metric-value">{st.session_state.total_returns_processed}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        injury_rate = (analysis['total_injuries'] / st.session_state.total_returns_processed * 100) if st.session_state.total_returns_processed > 0 else 0
        color = COLORS['critical'] if injury_rate > 5 else COLORS['high'] if injury_rate > 2 else COLORS['medium']
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Injury Cases</div>
            <div class="metric-value" style="color: {color};">{analysis['total_injuries']}</div>
            <div style="color: {color}; font-size: 0.9rem;">{injury_rate:.1f}% of returns</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Critical Severity</div>
            <div class="metric-value" style="color: {COLORS['critical']};">{analysis['severity_breakdown']['critical']}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">High Severity</div>
            <div class="metric-value" style="color: {COLORS['high']};">{analysis['severity_breakdown']['high']}</div>
        </div>
        """, unsafe_allow_html=True)
    
    # Severity breakdown chart
    if analysis['total_injuries'] > 0:
        st.markdown("#### Severity Distribution")
        
        severity_data = pd.DataFrame([
            {'Severity': 'Critical', 'Count': analysis['severity_breakdown']['critical'], 'Color': COLORS['critical']},
            {'Severity': 'High', 'Count': analysis['severity_breakdown']['high'], 'Color': COLORS['high']},
            {'Severity': 'Medium', 'Count': analysis['severity_breakdown']['medium'], 'Color': COLORS['medium']},
            {'Severity': 'Low', 'Count': analysis['severity_breakdown']['low'], 'Color': COLORS['low']}
        ])
        
        fig = px.bar(
            severity_data,
            x='Severity',
            y='Count',
            color='Severity',
            color_discrete_map={
                'Critical': COLORS['critical'],
                'High': COLORS['high'],
                'Medium': COLORS['medium'],
                'Low': COLORS['low']
            },
            title="Injury Cases by Severity"
        )
        fig.update_layout(showlegend=False, height=300)
        st.plotly_chart(fig, use_container_width=True)
    
    # Most common injury types
    if analysis['injury_types']:
        st.markdown("#### Most Common Injury Types")
        injury_types_df = pd.DataFrame(
            analysis['injury_types'].most_common(10),
            columns=['Injury Type', 'Count']
        )
        
        col1, col2 = st.columns([2, 1])
        with col1:
            fig = px.bar(
                injury_types_df,
                x='Count',
                y='Injury Type',
                orientation='h',
                color='Count',
                color_continuous_scale='Reds'
            )
            fig.update_layout(height=400, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.dataframe(injury_types_df, use_container_width=True, hide_index=True)

def display_critical_cases():
    """Display critical injury cases"""
    if not st.session_state.critical_cases:
        return
    
    st.markdown("### 🚨 Critical Cases Requiring Immediate Review")
    
    # Filters
    col1, col2, col3 = st.columns([2, 2, 1])
    
    with col1:
        severity_filter = st.selectbox(
            "Filter by Severity",
            ["All", "Critical Only", "High & Critical", "Medium & Above"],
            key="severity_select"
        )
    
    with col2:
        search_term = st.text_input(
            "Search in comments",
            placeholder="e.g., hospital, bleeding, emergency"
        )
    
    with col3:
        if st.button("📥 Export Critical Cases", use_container_width=True):
            export_critical_cases()
    
    # Filter cases
    filtered_cases = filter_cases(st.session_state.critical_cases, severity_filter, search_term)
    
    # Display cases
    st.markdown(f"**Showing {len(filtered_cases)} cases**")
    
    for idx, case in enumerate(filtered_cases):
        severity = case['severity']
        severity_class = f"severity-{severity.lower()}"
        
        st.markdown(f"""
        <div class="injury-card">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 1rem;">
                <div>
                    <span class="{severity_class}">{severity.upper()}</span>
                    <span style="margin-left: 1rem; color: {COLORS['neutral']};">
                        {case['return_date']}
                    </span>
                </div>
                <div style="font-weight: bold;">
                    Order: {case['order_id']}
                </div>
            </div>
            
            <div style="margin: 1rem 0;">
                <strong>ASIN:</strong> {case.get('asin', 'N/A')} | 
                <strong>SKU:</strong> {case.get('sku', 'N/A')}
            </div>
            
            <div style="background: #F5F5F5; padding: 1rem; border-radius: 5px; margin: 1rem 0;">
                <strong>Customer Comment:</strong><br>
                {case['customer_comment']}
            </div>
            
            <div style="margin-top: 1rem;">
                <strong>Detected Issues:</strong> {', '.join(case['injury_keywords'])}
            </div>
        </div>
        """, unsafe_allow_html=True)

def filter_cases(cases, severity_filter, search_term):
    """Filter injury cases"""
    filtered = cases
    
    # Apply severity filter
    if severity_filter == "Critical Only":
        filtered = [c for c in filtered if c['severity'] == 'critical']
    elif severity_filter == "High & Critical":
        filtered = [c for c in filtered if c['severity'] in ['critical', 'high']]
    elif severity_filter == "Medium & Above":
        filtered = [c for c in filtered if c['severity'] in ['critical', 'high', 'medium']]
    
    # Apply search filter
    if search_term:
        search_lower = search_term.lower()
        filtered = [c for c in filtered if search_lower in c['customer_comment'].lower()]
    
    return filtered

def display_analysis_insights():
    """Display analysis insights and recommendations"""
    if not st.session_state.analysis_ready:
        return
    
    st.markdown("### 💡 Analysis Insights & Recommendations")
    
    analysis = st.session_state.injury_analysis
    
    # Risk assessment
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("#### 🎯 Risk Assessment")
        
        injury_rate = (analysis['total_injuries'] / st.session_state.total_returns_processed * 100) if st.session_state.total_returns_processed > 0 else 0
        
        if injury_rate > 5:
            risk_level = "CRITICAL"
            risk_color = COLORS['critical']
            risk_message = "Immediate action required. Consider product recall evaluation."
        elif injury_rate > 2:
            risk_level = "HIGH"
            risk_color = COLORS['high']
            risk_message = "Significant safety concerns. Investigate root causes immediately."
        elif injury_rate > 1:
            risk_level = "MEDIUM"
            risk_color = COLORS['medium']
            risk_message = "Monitor closely and implement preventive measures."
        else:
            risk_level = "LOW"
            risk_color = COLORS['safe']
            risk_message = "Within acceptable range, continue monitoring."
        
        st.markdown(f"""
        <div style="background: {risk_color}20; border: 2px solid {risk_color}; 
                    border-radius: 10px; padding: 1.5rem; text-align: center;">
            <h3 style="color: {risk_color}; margin: 0;">Risk Level: {risk_level}</h3>
            <p style="margin: 0.5rem 0 0 0;">{risk_message}</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("#### 📋 Recommended Actions")
        
        recommendations = generate_recommendations(analysis, st.session_state.total_returns_processed)
        for rec in recommendations:
            st.markdown(f"• {rec}")
    
    # Product impact analysis
    if st.session_state.critical_cases:
        st.markdown("#### 📊 Product Impact Analysis")
        
        # Group by ASIN
        asin_injuries = defaultdict(list)
        for case in st.session_state.critical_cases:
            asin = case.get('asin', 'Unknown')
            asin_injuries[asin].append(case)
        
        # Create product impact table
        product_data = []
        for asin, cases in asin_injuries.items():
            product_data.append({
                'ASIN': asin,
                'Injury Cases': len(cases),
                'Critical': sum(1 for c in cases if c['severity'] == 'critical'),
                'High': sum(1 for c in cases if c['severity'] == 'high'),
                'Most Common Issue': Counter([kw for c in cases for kw in c['injury_keywords']]).most_common(1)[0][0] if cases else 'N/A'
            })
        
        product_df = pd.DataFrame(product_data).sort_values('Injury Cases', ascending=False)
        st.dataframe(product_df, use_container_width=True, hide_index=True)

def generate_recommendations(analysis, total_returns):
    """Generate safety recommendations based on analysis"""
    recommendations = []
    
    injury_rate = (analysis['total_injuries'] / total_returns * 100) if total_returns > 0 else 0
    
    # Critical recommendations
    if analysis['severity_breakdown']['critical'] > 0:
        recommendations.append("🚨 **Immediate**: Contact customers with critical injury cases for follow-up")
        recommendations.append("🚨 **Immediate**: Document all injury cases for potential regulatory reporting")
    
    # High injury rate recommendations
    if injury_rate > 5:
        recommendations.append("⚠️ **Urgent**: Conduct emergency quality review with manufacturing")
        recommendations.append("⚠️ **Urgent**: Consider temporary sales suspension pending investigation")
    elif injury_rate > 2:
        recommendations.append("⚠️ **High Priority**: Review product design and safety features")
        recommendations.append("⚠️ **High Priority**: Update product warnings and instructions")
    
    # General recommendations
    recommendations.append("📋 **Standard**: File safety report with quality assurance team")
    recommendations.append("📋 **Standard**: Monitor future returns for similar patterns")
    
    # Specific injury type recommendations
    if 'bleeding' in str(analysis['injury_types']).lower():
        recommendations.append("🩹 **Safety**: Review sharp edges and protective features")
    if 'fall' in str(analysis['injury_types']).lower():
        recommendations.append("🦺 **Safety**: Evaluate product stability and anti-slip features")
    
    return recommendations

def export_critical_cases():
    """Export critical cases to Excel"""
    if not st.session_state.critical_cases:
        st.warning("No critical cases to export")
        return
    
    try:
        # Create DataFrame
        export_df = pd.DataFrame(st.session_state.critical_cases)
        
        # Reorder columns
        column_order = ['severity', 'return_date', 'order_id', 'asin', 'sku', 
                       'customer_comment', 'injury_keywords', 'risk_score']
        
        # Only include columns that exist
        columns_to_export = [col for col in column_order if col in export_df.columns]
        export_df = export_df[columns_to_export]
        
        # Convert injury keywords list to string
        if 'injury_keywords' in export_df.columns:
            export_df['injury_keywords'] = export_df['injury_keywords'].apply(lambda x: ', '.join(x) if isinstance(x, list) else x)
        
        # Create Excel file
        output = io.BytesIO()
        
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            export_df.to_excel(writer, sheet_name='Critical_Injury_Cases', index=False)
            
            # Get workbook and worksheet
            workbook = writer.book
            worksheet = writer.sheets['Critical_Injury_Cases']
            
            # Add formatting
            header_format = workbook.add_format({
                'bold': True,
                'bg_color': '#2C3E50',
                'font_color': 'white',
                'border': 1
            })
            
            critical_format = workbook.add_format({
                'bg_color': '#FFE5E5',
                'font_color': '#8B0000'
            })
            
            # Apply header format
            for col_num, value in enumerate(export_df.columns.values):
                worksheet.write(0, col_num, value, header_format)
            
            # Apply conditional formatting for critical severity
            if 'severity' in export_df.columns:
                severity_col_idx = export_df.columns.get_loc('severity')
                worksheet.conditional_format(
                    1, severity_col_idx, len(export_df), severity_col_idx,
                    {'type': 'text', 'criteria': 'containing', 'value': 'critical', 'format': critical_format}
                )
            
            # Adjust column widths
            worksheet.set_column('A:A', 12)  # Severity
            worksheet.set_column('B:B', 12)  # Date
            worksheet.set_column('C:C', 20)  # Order ID
            worksheet.set_column('D:E', 15)  # ASIN, SKU
            worksheet.set_column('F:F', 60)  # Comment
            worksheet.set_column('G:G', 30)  # Keywords
        
        output.seek(0)
        
        # Download button
        st.download_button(
            label="📥 Download Excel Report",
            data=output.getvalue(),
            file_name=f"critical_injury_cases_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
        
    except Exception as e:
        st.error(f"Export error: {str(e)}")
        logger.error(f"Export error: {e}", exc_info=True)

def main():
    """Main application"""
    initialize_session_state()
    inject_safety_css()
    
    # Header
    display_header()
    
    # Sidebar
    with st.sidebar:
        st.markdown("### ⚙️ Analysis Settings")
        
        # Severity thresholds
        st.markdown("#### 🎯 Severity Thresholds")
        with st.expander("Customize detection", expanded=False):
            st.slider("Critical threshold", 0.7, 1.0, 0.9, 0.05,
                     help="Confidence threshold for critical severity")
            st.slider("High threshold", 0.5, 0.9, 0.7, 0.05,
                     help="Confidence threshold for high severity")
        
        st.markdown("---")
        
        # Injury keywords info
        st.markdown("#### 🔍 Monitored Keywords")
        with st.expander("View injury keywords"):
            st.markdown("**Critical Keywords:**")
            st.text(", ".join(INJURY_KEYWORDS['critical'][:5]) + "...")
            
            st.markdown("**High Priority:**")
            st.text(", ".join(INJURY_KEYWORDS['high'][:5]) + "...")
            
            st.markdown("**Medium Priority:**")
            st.text(", ".join(INJURY_KEYWORDS['medium'][:5]) + "...")
        
        st.markdown("---")
        
        # Help section
        with st.expander("📖 How to Use"):
            st.markdown("""
            1. **Export PDF** from Amazon Seller Central
            2. **Upload** the PDF file
            3. **Review** injury cases detected
            4. **Export** critical cases for follow-up
            
            **Severity Levels:**
            - 🔴 **Critical**: Immediate action required
            - 🟠 **High**: Urgent review needed
            - 🟡 **Medium**: Monitor closely
            - 🔵 **Low**: Standard review
            """)
    
    # Main content
    if not st.session_state.analysis_ready:
        # Upload section
        upload_pdf_section()
    else:
        # Results section
        tab1, tab2, tab3, tab4 = st.tabs([
            "🚨 Injury Summary", 
            "📋 Critical Cases", 
            "💡 Insights & Actions",
            "📄 New Analysis"
        ])
        
        with tab1:
            display_injury_summary()
        
        with tab2:
            display_critical_cases()
        
        with tab3:
            display_analysis_insights()
        
        with tab4:
            if st.button("🔄 Start New Analysis", type="primary", use_container_width=True):
                # Reset state
                for key in ['pdf_data', 'injury_analysis', 'processed_returns', 
                           'critical_cases', 'processing_complete', 'analysis_ready']:
                    st.session_state[key] = None if 'complete' in key or 'ready' in key else []
                st.session_state.injuries_found = 0
                st.session_state.total_returns_processed = 0
                st.rerun()

if __name__ == "__main__":
    main()
