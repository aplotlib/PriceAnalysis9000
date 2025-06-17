"""
Amazon Returns Quality Analyzer - PDF & FBA Returns Processing
Version: 6.0 - Enhanced Medical Device Safety Focus
Critical: Immediate injury case notifications for quality managers
"""

import streamlit as st
import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
import io
from typing import Dict, List, Any, Optional, Tuple
import re
import json
from collections import Counter, defaultdict
import plotly.express as px
import plotly.graph_objects as go
import time
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configure page
st.set_page_config(
    page_title="Amazon Returns Quality Analyzer - Medical Device Safety",
    page_icon="🚨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Optional imports
try:
    import chardet
    CHARDET_AVAILABLE = True
except ImportError:
    CHARDET_AVAILABLE = False
    logger.warning("chardet not available")

# Import modules with error handling
try:
    from pdf_analyzer import PDFAnalyzer
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False
    logger.warning("pdf_analyzer not available")

try:
    from injury_detector import InjuryDetector
    INJURY_AVAILABLE = True
except ImportError:
    INJURY_AVAILABLE = False
    logger.warning("injury_detector not available")

try:
    from universal_file_detector import UniversalFileDetector
    FILE_DETECTOR_AVAILABLE = True
except ImportError:
    FILE_DETECTOR_AVAILABLE = False
    logger.warning("universal_file_detector not available")

try:
    from enhanced_ai_analysis import (
        EnhancedAIAnalyzer, AIProvider, MEDICAL_DEVICE_CATEGORIES, 
        FBA_REASON_MAP, INJURY_RISK_CATEGORIES, detect_injury_severity
    )
    AI_AVAILABLE = True
except ImportError:
    AI_AVAILABLE = False
    logger.warning("enhanced_ai_analysis not available")
    # Define fallbacks
    MEDICAL_DEVICE_CATEGORIES = [
        'Size/Fit Issues',
        'Comfort Issues',
        'Product Defects/Quality',
        'Performance/Effectiveness',
        'Stability/Positioning Issues',
        'Equipment Compatibility',
        'Design/Material Issues',
        'Wrong Product/Misunderstanding',
        'Missing Components',
        'Customer Error/Changed Mind',
        'Shipping/Fulfillment Issues',
        'Assembly/Usage Difficulty',
        'Medical/Health Concerns',
        'Price/Value',
        'Other/Miscellaneous'
    ]
    FBA_REASON_MAP = {}
    INJURY_RISK_CATEGORIES = {}
    
    def detect_injury_severity(text):
        return 'none', []
    
    class AIProvider:
        FASTEST = "fastest"
        OPENAI = "openai"
        CLAUDE = "claude"
        BOTH = "both"
    
    class EnhancedAIAnalyzer:
        def __init__(self, provider):
            self.provider = provider
            self.injury_cases = []
        
        def categorize_return(self, complaint, fba_reason=None, return_reason=None, return_data=None):
            return 'Other/Miscellaneous', 0.1, 'none', 'en'
        
        def get_cost_summary(self):
            return {'total_cost': 0.0, 'api_calls': 0}
        
        def get_injury_summary(self):
            return {'total_injuries': 0, 'critical': 0, 'high': 0, 'medium': 0, 'cases': []}
        
        def export_injury_report(self):
            return "No injury cases detected"

try:
    from smart_column_mapper import SmartColumnMapper
    MAPPER_AVAILABLE = True
except ImportError:
    MAPPER_AVAILABLE = False
    logger.warning("smart_column_mapper not available")
    
    class SmartColumnMapper:
        def __init__(self, ai_analyzer=None):
            self.ai_analyzer = ai_analyzer
        
        def detect_columns(self, df):
            column_mapping = {}
            for col in df.columns:
                col_lower = str(col).lower()
                if 'date' in col_lower:
                    column_mapping['date'] = col
                elif 'complaint' in col_lower or 'comment' in col_lower:
                    column_mapping['complaint'] = col
                elif 'product' in col_lower or 'sku' in col_lower:
                    column_mapping['product_id'] = col
                elif 'asin' in col_lower:
                    column_mapping['asin'] = col
                elif 'order' in col_lower:
                    column_mapping['order_id'] = col
            return column_mapping
        
        def validate_mapping(self, df, mapping):
            return {'is_valid': True, 'missing_required': [], 'warnings': []}
        
        def map_dataframe(self, df, column_mapping):
            return df

MODULES_AVAILABLE = all([PDF_AVAILABLE, AI_AVAILABLE])

# Professional color scheme with safety focus
COLORS = {
    'primary': '#00D9FF',
    'secondary': '#FF006E',
    'accent': '#FFB700',
    'success': '#00F5A0',
    'warning': '#FF6B35',
    'danger': '#FF0054',
    'critical': '#DC143C',  # For critical injuries
    'injury': '#FF1744',    # For injury alerts
    'dark': '#2C3E50',
    'light': '#ECF0F1',
    'quality': '#9B59B6'
}

# Quality-focused categories
QUALITY_CATEGORIES = [
    'Product Defects/Quality',
    'Performance/Effectiveness',
    'Design/Material Issues',
    'Medical/Health Concerns',
    'Missing Components'
]

def inject_professional_css():
    """Professional CSS styling with injury alert focus"""
    st.markdown(f"""
    <style>
    /* Professional styling */
    .main {{
        padding: 0;
        background-color: #FAFAFA;
    }}
    
    /* Header */
    .main-header {{
        background: linear-gradient(135deg, {COLORS['dark']} 0%, {COLORS['primary']} 100%);
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
    
    /* Critical injury alert */
    .injury-alert {{
        background: linear-gradient(135deg, {COLORS['injury']} 0%, {COLORS['critical']} 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        box-shadow: 0 0 20px rgba(255, 23, 68, 0.5);
        animation: pulse 2s infinite;
    }}
    
    @keyframes pulse {{
        0% {{ box-shadow: 0 0 20px rgba(255, 23, 68, 0.5); }}
        50% {{ box-shadow: 0 0 30px rgba(255, 23, 68, 0.8); }}
        100% {{ box-shadow: 0 0 20px rgba(255, 23, 68, 0.5); }}
    }}
    
    /* Injury case card */
    .injury-case-card {{
        background: white;
        border-left: 5px solid {COLORS['injury']};
        border-radius: 8px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 2px 10px rgba(255, 23, 68, 0.2);
    }}
    
    .injury-severity-critical {{
        background: {COLORS['critical']};
        color: white;
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        font-weight: bold;
        display: inline-block;
    }}
    
    .injury-severity-high {{
        background: {COLORS['danger']};
        color: white;
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        font-weight: bold;
        display: inline-block;
    }}
    
    .injury-severity-medium {{
        background: {COLORS['warning']};
        color: white;
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        font-weight: bold;
        display: inline-block;
    }}
    
    /* Quality alert box */
    .quality-alert {{
        background: linear-gradient(135deg, rgba(155, 89, 182, 0.1), rgba(155, 89, 182, 0.2));
        border: 2px solid {COLORS['quality']};
        border-radius: 10px;
        padding: 1.5rem;
        margin: 1rem 0;
    }}
    
    /* Category cards */
    .category-card {{
        background: white;
        border-radius: 8px;
        padding: 1.5rem;
        margin: 0.5rem 0;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        border-left: 5px solid {COLORS['primary']};
    }}
    
    .quality-issue-card {{
        border-left-color: {COLORS['danger']};
    }}
    
    /* Metric cards */
    .metric-card {{
        background: white;
        border-radius: 8px;
        padding: 1.5rem;
        text-align: center;
        box-shadow: 0 2px 10px rgba(0,0,0,0.05);
        border: 1px solid #E0E0E0;
        transition: transform 0.3s;
    }}
    
    .metric-card:hover {{
        transform: translateY(-5px);
        box-shadow: 0 5px 20px rgba(0,0,0,0.1);
    }}
    
    .metric-value {{
        font-size: 2.5rem;
        font-weight: 700;
        margin: 0.5rem 0;
    }}
    
    .metric-label {{
        color: {COLORS['dark']};
        font-size: 0.9rem;
        text-transform: uppercase;
        letter-spacing: 1px;
    }}
    
    /* FDA MDR alert */
    .fda-alert {{
        background: {COLORS['critical']};
        color: white;
        padding: 1rem;
        border-radius: 5px;
        font-weight: bold;
        text-align: center;
        margin: 1rem 0;
    }}
    
    /* Progress indicators */
    .quality-score {{
        background: linear-gradient(90deg, {COLORS['danger']} 0%, {COLORS['warning']} 50%, {COLORS['success']} 100%);
        height: 10px;
        border-radius: 5px;
        position: relative;
    }}
    
    /* Data tables */
    .dataframe {{
        font-size: 0.9rem;
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
        background: {COLORS['primary']};
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(0,0,0,0.2);
    }}
    
    /* Emergency button */
    .emergency-button {{
        background: {COLORS['injury']} !important;
        animation: pulse 2s infinite;
    }}
    
    /* Hide Streamlit branding */
    #MainMenu {{visibility: hidden;}}
    footer {{visibility: hidden;}}
    </style>
    """, unsafe_allow_html=True)

def initialize_session_state():
    """Initialize session state variables"""
    defaults = {
        # Data storage
        'raw_data': None,
        'processed_data': None,
        'categorized_data': None,
        'file_type': None,
        
        # Analysis results
        'category_analysis': {},
        'product_analysis': {},
        'quality_metrics': {},
        'injury_analysis': None,
        'ai_insights': None,
        
        # Injury tracking
        'injury_cases': [],
        'critical_injuries': 0,
        'fda_mdr_required': False,
        
        # Processing state
        'processing_complete': False,
        'file_uploaded': False,
        'analysis_ready': False,
        
        # AI components
        'ai_analyzer': None,
        'pdf_analyzer': None,
        'injury_detector': None,
        'file_detector': None,
        'column_mapper': None,
        
        # Filters
        'selected_asin': 'ALL',
        'date_range': 'All Time',
        'category_filter': 'All',
        'show_injuries_only': False,
        
        # Tracking
        'total_returns': 0,
        'quality_issues': 0,
        'total_injuries': 0,
        'processing_time': 0.0,
        'api_cost': 0.0
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

def check_api_keys():
    """Check for API keys"""
    keys_found = {}
    
    try:
        if hasattr(st, 'secrets'):
            # Check OpenAI
            for key in ['OPENAI_API_KEY', 'openai_api_key', 'openai']:
                if key in st.secrets:
                    keys_found['openai'] = str(st.secrets[key]).strip()
                    break
            
            # Check Claude
            for key in ['ANTHROPIC_API_KEY', 'anthropic_api_key', 'claude_api_key', 'claude']:
                if key in st.secrets:
                    keys_found['claude'] = str(st.secrets[key]).strip()
                    break
    except Exception as e:
        logger.warning(f"Error checking secrets: {e}")
    
    return keys_found

def initialize_analyzers():
    """Initialize all analyzer components"""
    if not st.session_state.ai_analyzer and AI_AVAILABLE:
        try:
            keys = check_api_keys()
            if keys:
                # Set environment variables
                if 'openai' in keys:
                    os.environ['OPENAI_API_KEY'] = keys['openai']
                if 'claude' in keys:
                    os.environ['ANTHROPIC_API_KEY'] = keys['claude']
                
                st.session_state.ai_analyzer = EnhancedAIAnalyzer(AIProvider.FASTEST)
                logger.info("AI analyzer initialized with injury detection")
        except Exception as e:
            logger.error(f"Failed to initialize AI: {e}")
    
    if not st.session_state.pdf_analyzer and PDF_AVAILABLE:
        st.session_state.pdf_analyzer = PDFAnalyzer()
    
    if not st.session_state.injury_detector and INJURY_AVAILABLE:
        st.session_state.injury_detector = InjuryDetector()
    
    if not st.session_state.file_detector and FILE_DETECTOR_AVAILABLE:
        st.session_state.file_detector = UniversalFileDetector()
    
    if not st.session_state.get('column_mapper') and MAPPER_AVAILABLE:
        st.session_state.column_mapper = SmartColumnMapper(st.session_state.ai_analyzer)

def display_header():
    """Display application header with injury alerts"""
    st.markdown("""
    <div class="main-header">
        <h1>🚨 Amazon Returns Quality Analyzer</h1>
        <p style="font-size: 1.2rem; margin-top: 0.5rem;">
            Medical Device Safety & Injury Detection System
        </p>
        <p style="font-size: 1rem; opacity: 0.9;">
            Automatic injury case identification • FDA MDR compliance • Quality improvement tracking
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Show critical injury alert if present
    if st.session_state.critical_injuries > 0:
        st.markdown(f"""
        <div class="injury-alert">
            <h2 style="margin: 0;">
                ⚠️ CRITICAL SAFETY ALERT: {st.session_state.critical_injuries} Critical Injury Case{'s' if st.session_state.critical_injuries > 1 else ''} Detected
            </h2>
            <p style="margin: 0.5rem 0 0 0; font-size: 1.1rem;">
                Immediate action required - Review injury cases below for potential FDA MDR reporting
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    # Show quality alerts
    elif st.session_state.quality_issues > 0:
        quality_rate = (st.session_state.quality_issues / st.session_state.total_returns * 100) if st.session_state.total_returns > 0 else 0
        if quality_rate > 30:
            st.markdown(f"""
            <div class="quality-alert">
                <h2 style="color: {COLORS['quality']}; margin: 0;">
                    ⚠️ {st.session_state.quality_issues} Quality-Related Returns ({quality_rate:.1f}%)
                </h2>
                <p style="margin: 0.5rem 0 0 0;">Review quality categories for improvement opportunities</p>
            </div>
            """, unsafe_allow_html=True)

def display_injury_summary():
    """Display detailed injury case summary"""
    if not st.session_state.ai_analyzer:
        return
    
    injury_summary = st.session_state.ai_analyzer.get_injury_summary()
    
    if injury_summary['total_injuries'] == 0:
        return
    
    st.markdown("### 🚨 Injury Case Summary")
    
    # Summary metrics
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card" style="border: 2px solid {COLORS['injury']};">
            <div class="metric-label">Total Injuries</div>
            <div class="metric-value" style="color: {COLORS['injury']};">{injury_summary['total_injuries']}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card" style="border: 2px solid {COLORS['critical']};">
            <div class="metric-label">Critical</div>
            <div class="metric-value" style="color: {COLORS['critical']};">{injury_summary['critical']}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-card" style="border: 2px solid {COLORS['danger']};">
            <div class="metric-label">High Severity</div>
            <div class="metric-value" style="color: {COLORS['danger']};">{injury_summary['high']}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class="metric-card" style="border: 2px solid {COLORS['warning']};">
            <div class="metric-label">Medium</div>
            <div class="metric-value" style="color: {COLORS['warning']};">{injury_summary['medium']}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col5:
        if injury_summary['fda_mdr_required']:
            st.markdown(f"""
            <div class="fda-alert">
                FDA MDR<br>REQUIRED
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Reportable</div>
                <div class="metric-value">{injury_summary['reportable_cases']}</div>
            </div>
            """, unsafe_allow_html=True)
    
    # Detailed injury cases
    st.markdown("#### 📋 Detailed Injury Cases")
    
    # Filter options
    col1, col2, col3 = st.columns([2, 2, 1])
    with col1:
        severity_filter = st.selectbox(
            "Filter by severity",
            ["All", "Critical", "High", "Medium"],
            key="injury_severity_filter"
        )
    with col2:
        asin_filter = st.selectbox(
            "Filter by ASIN",
            ["All"] + list(set(case.asin for case in injury_summary['cases'])),
            key="injury_asin_filter"
        )
    with col3:
        if st.button("Export Injury Report", type="primary"):
            export_injury_report()
    
    # Display cases
    filtered_cases = injury_summary['cases']
    
    if severity_filter != "All":
        filtered_cases = [c for c in filtered_cases if c.severity == severity_filter.lower()]
    
    if asin_filter != "All":
        filtered_cases = [c for c in filtered_cases if c.asin == asin_filter]
    
    for idx, case in enumerate(filtered_cases):
        severity_class = f"injury-severity-{case.severity}"
        
        st.markdown(f"""
        <div class="injury-case-card">
            <div style="display: flex; justify-content: space-between; align-items: start;">
                <div style="flex: 1;">
                    <h4 style="margin: 0;">Case #{idx + 1}: Order {case.order_id}</h4>
                    <div style="margin: 0.5rem 0;">
                        <span class="{severity_class}">{case.severity.upper()}</span>
                        {' <span style="background: #FF5722; color: white; padding: 0.2rem 0.5rem; border-radius: 15px; font-size: 0.8rem;">FDA MDR REQUIRED</span>' if case.reportable else ''}
                    </div>
                    <div style="color: #666; margin: 0.5rem 0;">
                        <strong>ASIN:</strong> {case.asin} | <strong>SKU:</strong> {case.sku} | <strong>Date:</strong> {case.return_date}
                    </div>
                    <div style="color: #666; margin: 0.5rem 0;">
                        <strong>Device Type:</strong> {case.device_type} | <strong>Potential Causes:</strong> {', '.join(case.potential_causes)}
                    </div>
                </div>
            </div>
            
            <div style="margin: 1rem 0;">
                <strong>Injury Keywords Detected:</strong>
                <div style="margin: 0.5rem 0;">
                    {' '.join([f'<span style="background: #FFE0E0; color: #D32F2F; padding: 0.2rem 0.5rem; margin: 0.2rem; border-radius: 15px; display: inline-block;">{kw}</span>' for kw in case.injury_keywords])}
                </div>
            </div>
            
            <div style="background: #F5F5F5; padding: 1rem; border-radius: 5px; margin: 1rem 0;">
                <strong>Customer Comment:</strong>
                <div style="margin-top: 0.5rem; font-style: italic;">"{case.full_comment}"</div>
            </div>
            
            <div style="background: #FFF3E0; padding: 1rem; border-radius: 5px; margin: 1rem 0;">
                <strong>Quality Team Recommendation:</strong>
                <div style="margin-top: 0.5rem;">{case.recommendation}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

def export_injury_report():
    """Export detailed injury report"""
    if st.session_state.ai_analyzer:
        report = st.session_state.ai_analyzer.export_injury_report()
        
        st.download_button(
            label="Download Injury Report",
            data=report,
            file_name=f"injury_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
            mime="text/plain"
        )

def upload_section():
    """File upload section for PDF and structured data files"""
    st.markdown("### 📁 Upload Amazon Return Data")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="category-card">
            <h4>📄 PDF from Seller Central</h4>
            <p>Export from Manage Returns → Print as PDF</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="category-card">
            <h4>📊 Structured Data Files</h4>
            <p>CSV, Excel, TXT files with return data</p>
            <small>• FBA Returns Report (.txt)<br>
            • Quality tracking spreadsheets<br>
            • Custom return exports</small>
        </div>
        """, unsafe_allow_html=True)
    
    # Injury detection notice
    st.info("""
    🚨 **Automatic Injury Detection**: The system will automatically identify and flag any potential injury cases for immediate review.
    All injury keywords like "hurt", "pain", "hospital", "bleeding" etc. will be detected and categorized by severity.
    """)
    
    # File format examples
    with st.expander("📋 Supported File Formats & Examples"):
        st.markdown("""
        **The tool uses AI to automatically detect columns and injury cases!**
        
        Common formats we handle:
        
        **1. Amazon FBA Returns (.txt)**
        - Headers: `return-date`, `order-id`, `sku`, `asin`, `reason`, `customer-comments`
        
        **2. Quality Tracking Spreadsheets**
        - Example: `Date Complaint was made`, `Product Identifier Tag`, `ASIN`, `Order #`, `Complaint`
        
        **3. Custom Exports**
        - Any file with columns for: dates, product IDs, order numbers, and complaint text
        
        **The AI will automatically:**
        - Map your columns to appropriate fields
        - Detect and flag injury cases
        - Categorize returns for quality analysis
        """)
    
    uploaded_file = st.file_uploader(
        "Choose file",
        type=['pdf', 'txt', 'csv', 'tsv', 'xlsx', 'xls'],
        help="Upload PDF from Seller Central or any structured data file with return information"
    )
    
    if uploaded_file:
        process_file(uploaded_file)

def process_file(uploaded_file):
    """Process uploaded file based on type"""
    try:
        initialize_analyzers()
        
        file_extension = uploaded_file.name.split('.')[-1].lower()
        file_content = uploaded_file.read()
        
        with st.spinner("🔍 Analyzing return data and detecting injury cases..."):
            start_time = time.time()
            
            if file_extension == 'pdf':
                process_pdf_file(file_content, uploaded_file.name)
            elif file_extension in ['txt', 'csv', 'tsv', 'xlsx', 'xls']:
                process_structured_file(file_content, uploaded_file.name, file_extension)
            else:
                st.error("Unsupported file type")
                return
            
            st.session_state.processing_time = time.time() - start_time
            st.session_state.processing_complete = True
            st.session_state.analysis_ready = True
            
            # Generate insights
            generate_quality_insights()
            
            # Check for injuries
            check_injury_cases()
            
            st.success(f"""
            ✅ Analysis Complete!
            - Total returns: {st.session_state.total_returns}
            - Quality issues: {st.session_state.quality_issues}
            - Injury cases: {st.session_state.total_injuries} {'⚠️ CRITICAL CASES FOUND!' if st.session_state.critical_injuries > 0 else ''}
            - Processing time: {st.session_state.processing_time:.1f}s
            """)
            
            st.rerun()
            
    except Exception as e:
        st.error(f"❌ Error processing file: {str(e)}")
        logger.error(f"File processing error: {e}", exc_info=True)

def process_pdf_file(content: bytes, filename: str):
    """Process PDF file from Amazon Seller Central"""
    st.session_state.file_type = 'pdf'
    
    # Extract data from PDF
    extracted_data = st.session_state.pdf_analyzer.extract_returns_from_pdf(content, filename)
    
    if not extracted_data or not extracted_data.get('returns'):
        st.error("No return data found in PDF.")
        return
    
    returns_data = extracted_data['returns']
    st.session_state.raw_data = returns_data
    
    # Process and categorize returns
    categorized_returns = []
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for idx, return_item in enumerate(returns_data):
        # Get return reason and buyer comment
        return_reason = return_item.get('return_reason', '')
        buyer_comment = return_item.get('buyer_comment', '') or return_item.get('customer_comment', '') or return_item.get('raw_content', '')
        
        # Create structured return item
        categorized_item = {
            'order_id': return_item.get('order_id', ''),
            'sku': return_item.get('sku', ''),
            'asin': return_item.get('asin', ''),
            'return_date': return_item.get('return_date', ''),
            'return_reason': return_reason,
            'buyer_comment': buyer_comment,
            'customer_comment': buyer_comment,
            'page': return_item.get('page', 0)
        }
        
        # Categorize using AI with injury detection
        if st.session_state.ai_analyzer:
            category, confidence, severity, language = st.session_state.ai_analyzer.categorize_return(
                complaint=buyer_comment,
                return_reason=return_reason,
                return_data=categorized_item
            )
            categorized_item['category'] = category
            categorized_item['confidence'] = confidence
            categorized_item['injury_severity'] = severity
            categorized_item['is_injury'] = severity != 'none'
        else:
            categorized_item['category'] = 'Other/Miscellaneous'
            categorized_item['confidence'] = 0.1
            categorized_item['injury_severity'] = 'none'
            categorized_item['is_injury'] = False
        
        categorized_returns.append(categorized_item)
        progress_bar.progress((idx + 1) / len(returns_data))
        status_text.text(f"Processing return {idx + 1} of {len(returns_data)}...")
    
    progress_bar.empty()
    status_text.empty()
    
    st.session_state.categorized_data = pd.DataFrame(categorized_returns)
    st.session_state.total_returns = len(categorized_returns)

def process_structured_file(content: bytes, filename: str, file_extension: str):
    """Process structured data files with smart column mapping"""
    st.session_state.file_type = 'structured'
    
    # Read file based on extension
    try:
        if file_extension in ['csv', 'txt', 'tsv']:
            # Detect encoding
            try:
                text = content.decode('utf-8')
            except:
                if CHARDET_AVAILABLE:
                    encoding = chardet.detect(content)['encoding']
                    text = content.decode(encoding)
                else:
                    # Try common encodings
                    for encoding in ['latin-1', 'cp1252', 'iso-8859-1']:
                        try:
                            text = content.decode(encoding)
                            break
                        except:
                            continue
                    else:
                        st.error("Could not decode file.")
                        return
            
            # Detect delimiter
            if file_extension == 'tsv' or '\t' in text.split('\n')[0]:
                delimiter = '\t'
            elif file_extension == 'csv' or ',' in text.split('\n')[0]:
                delimiter = ','
            else:
                first_line = text.split('\n')[0]
                delimiter = ',' if first_line.count(',') > first_line.count('\t') else '\t'
            
            df = pd.read_csv(io.StringIO(text), delimiter=delimiter)
            
        elif file_extension in ['xlsx', 'xls']:
            df = pd.read_excel(io.BytesIO(content))
        else:
            st.error(f"Cannot read {file_extension} files")
            return
        
        # Clean column names
        df.columns = [str(col).strip() for col in df.columns]
        
        st.info(f"📊 Loaded {len(df)} rows with {len(df.columns)} columns")
        
        # Check if it's an FBA return report
        fba_columns = ['return-date', 'order-id', 'sku', 'asin', 'reason', 'customer-comments']
        is_fba_report = all(col in df.columns for col in fba_columns)
        
        if is_fba_report:
            st.success("✅ Detected FBA Return Report format")
            process_fba_returns(df)
        else:
            process_with_column_mapper(df, filename)
        
    except Exception as e:
        st.error(f"Error processing file: {str(e)}")
        logger.error(f"Structured file processing error: {e}", exc_info=True)

def process_fba_returns(df: pd.DataFrame):
    """Process FBA return report with injury detection"""
    categorized_data = []
    progress_bar = st.progress(0)
    injury_count = 0
    
    for idx, row in df.iterrows():
        return_item = {
            'order_id': row.get('order-id', ''),
            'sku': row.get('sku', ''),
            'asin': row.get('asin', ''),
            'return_date': row.get('return-date', ''),
            'return_reason': row.get('reason', ''),
            'buyer_comment': row.get('customer-comments', ''),
            'customer_comment': row.get('customer-comments', ''),
            'fba_reason': row.get('reason', ''),
            'quantity': row.get('quantity', 1),
            'original_index': idx
        }
        
        # Categorize using AI with injury detection
        if st.session_state.ai_analyzer:
            category, confidence, severity, language = st.session_state.ai_analyzer.categorize_return(
                complaint=return_item['buyer_comment'],
                fba_reason=return_item['fba_reason'],
                return_reason=return_item['return_reason'],
                return_data=return_item
            )
            return_item['category'] = category
            return_item['confidence'] = confidence
            return_item['injury_severity'] = severity
            return_item['is_injury'] = severity != 'none'
            
            if return_item['is_injury']:
                injury_count += 1
        else:
            return_item['category'] = 'Other/Miscellaneous'
            return_item['confidence'] = 0.1
            return_item['injury_severity'] = 'none'
            return_item['is_injury'] = False
        
        categorized_data.append(return_item)
        progress_bar.progress((idx + 1) / len(df))
    
    progress_bar.empty()
    
    if injury_count > 0:
        st.warning(f"⚠️ Detected {injury_count} potential injury case{'s' if injury_count > 1 else ''}!")
    
    # Store results
    st.session_state.categorized_data = pd.DataFrame(categorized_data)
    st.session_state.raw_data = categorized_data
    st.session_state.total_returns = len(categorized_data)

def process_with_column_mapper(df: pd.DataFrame, filename: str):
    """Process file using smart column mapper with injury detection"""
    st.markdown("### 🔍 Column Detection & Mapping")
    
    # Use smart column mapper if available
    if MAPPER_AVAILABLE and st.session_state.column_mapper:
        with st.spinner("Using AI to detect column types..."):
            column_mapping = st.session_state.column_mapper.detect_columns(df)
    else:
        st.warning("Smart column mapper not available. Please map columns manually.")
        column_mapping = {}
    
    # Display detected mappings
    if column_mapping:
        st.success("✅ Columns detected automatically!")
    
    # Show mapping in expandable section
    with st.expander("View/Edit Column Mappings", expanded=True):
        # Create editable mapping interface
        edited_mapping = {}
        col1, col2 = st.columns(2)
        
        # Define expected column types
        column_types = {
            'date': 'Date/When complaint was made',
            'complaint': 'Customer complaint/comment text',
            'product_id': 'Product SKU/Identifier',
            'asin': 'Amazon ASIN',
            'order_id': 'Order number',
            'source': 'Complaint source',
            'agent': 'Agent/Investigator name',
            'ticket_id': 'Ticket/Case number',
            'udi': 'UDI (Device Identifier)'
        }
        
        # Show current mappings and allow editing
        for col_type, description in column_types.items():
            with col1 if list(column_types.keys()).index(col_type) < 5 else col2:
                current_mapping = column_mapping.get(col_type, '')
                
                # Create dropdown with all columns
                options = [''] + list(df.columns)
                default_index = options.index(current_mapping) if current_mapping in options else 0
                
                selected = st.selectbox(
                    f"{description}:",
                    options,
                    index=default_index,
                    key=f"map_{col_type}"
                )
                
                if selected:
                    edited_mapping[col_type] = selected
        
        # Use edited mapping if any changes made
        if edited_mapping:
            column_mapping = edited_mapping
    
    # Process returns with injury detection
    st.markdown("### 🤖 Categorizing Returns & Detecting Injuries...")
    
    categorized_data = []
    progress_bar = st.progress(0)
    injury_count = 0
    
    for idx, row in df.iterrows():
        return_item = {
            'order_id': row.get(column_mapping.get('order_id', ''), ''),
            'sku': row.get(column_mapping.get('product_id', ''), ''),
            'asin': row.get(column_mapping.get('asin', ''), ''),
            'return_date': row.get(column_mapping.get('date', ''), ''),
            'customer_comment': row.get(column_mapping.get('complaint', ''), ''),
            'buyer_comment': row.get(column_mapping.get('complaint', ''), ''),
            'source': row.get(column_mapping.get('source', ''), ''),
            'agent': row.get(column_mapping.get('agent', ''), ''),
            'original_index': idx
        }
        
        # Check for return reason if present
        return_reason = ''
        for col in df.columns:
            if 'reason' in col.lower() and col not in column_mapping.values():
                return_reason = row.get(col, '')
                break
        return_item['return_reason'] = return_reason
        
        # Categorize using AI with injury detection
        if st.session_state.ai_analyzer and return_item['customer_comment']:
            category, confidence, severity, language = st.session_state.ai_analyzer.categorize_return(
                complaint=return_item['customer_comment'],
                return_reason=return_reason,
                return_data=return_item
            )
            return_item['category'] = category
            return_item['confidence'] = confidence
            return_item['injury_severity'] = severity
            return_item['is_injury'] = severity != 'none'
            
            if return_item['is_injury']:
                injury_count += 1
        else:
            return_item['category'] = 'Other/Miscellaneous'
            return_item['confidence'] = 0.1
            return_item['injury_severity'] = 'none'
            return_item['is_injury'] = False
        
        categorized_data.append(return_item)
        progress_bar.progress((idx + 1) / len(df))
    
    progress_bar.empty()
    
    if injury_count > 0:
        st.warning(f"⚠️ Detected {injury_count} potential injury case{'s' if injury_count > 1 else ''}!")
    
    # Store results
    st.session_state.categorized_data = pd.DataFrame(categorized_data)
    st.session_state.raw_data = categorized_data
    st.session_state.total_returns = len(categorized_data)
    
    # Show summary
    st.markdown("### ✅ Processing Complete")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Columns Mapped", len(column_mapping))
    with col2:
        st.metric("Rows Processed", len(categorized_data))
    with col3:
        st.metric("Categories Found", len(set(item['category'] for item in categorized_data)))
    with col4:
        st.metric("Injury Cases", injury_count, delta=None if injury_count == 0 else "⚠️")

def check_injury_cases():
    """Check and summarize injury cases"""
    if st.session_state.ai_analyzer:
        injury_summary = st.session_state.ai_analyzer.get_injury_summary()
        st.session_state.total_injuries = injury_summary['total_injuries']
        st.session_state.critical_injuries = injury_summary['critical']
        st.session_state.fda_mdr_required = injury_summary['fda_mdr_required']
        st.session_state.injury_cases = injury_summary['cases']

def generate_quality_insights():
    """Generate quality-focused insights from categorized data"""
    df = st.session_state.categorized_data
    
    # Category analysis
    category_counts = df['category'].value_counts()
    st.session_state.category_analysis = category_counts.to_dict()
    
    # Quality issues count
    quality_issues = df[df['category'].isin(QUALITY_CATEGORIES)]
    st.session_state.quality_issues = len(quality_issues)
    
    # Product analysis
    if 'asin' in df.columns:
        product_analysis = defaultdict(lambda: defaultdict(int))
        
        for _, row in df.iterrows():
            asin = row.get('asin', 'Unknown')
            category = row['category']
            product_analysis[asin][category] += 1
            product_analysis[asin]['total'] += 1
            
            # Track injuries by product
            if row.get('is_injury', False):
                product_analysis[asin]['injuries'] += 1
                product_analysis[asin][f'injury_{row.get("injury_severity", "unknown")}'] += 1
        
        st.session_state.product_analysis = dict(product_analysis)
    
    # Quality metrics
    st.session_state.quality_metrics = {
        'quality_rate': (st.session_state.quality_issues / st.session_state.total_returns * 100) if st.session_state.total_returns > 0 else 0,
        'injury_rate': (st.session_state.total_injuries / st.session_state.total_returns * 100) if st.session_state.total_returns > 0 else 0,
        'top_quality_issue': quality_issues['category'].value_counts().index[0] if len(quality_issues) > 0 else 'None',
        'products_with_quality_issues': len(quality_issues['asin'].unique()) if 'asin' in quality_issues.columns else 0,
        'products_with_injuries': len([asin for asin, data in st.session_state.product_analysis.items() if data.get('injuries', 0) > 0])
    }

def display_dashboard():
    """Display main analysis dashboard with injury focus"""
    st.markdown("### 📊 Return Analysis Dashboard")
    
    # Key metrics
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Total Returns</div>
            <div class="metric-value">{st.session_state.total_returns:,}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        quality_rate = st.session_state.quality_metrics.get('quality_rate', 0)
        color = COLORS['danger'] if quality_rate > 30 else COLORS['warning'] if quality_rate > 15 else COLORS['success']
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Quality Issues</div>
            <div class="metric-value" style="color: {color};">{st.session_state.quality_issues:,}</div>
            <div style="color: {color}; font-size: 0.9rem;">{quality_rate:.1f}%</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        injury_rate = st.session_state.quality_metrics.get('injury_rate', 0)
        color = COLORS['critical'] if st.session_state.critical_injuries > 0 else COLORS['injury'] if st.session_state.total_injuries > 0 else COLORS['success']
        st.markdown(f"""
        <div class="metric-card" style="border: 2px solid {color};">
            <div class="metric-label">Injury Cases</div>
            <div class="metric-value" style="color: {color};">{st.session_state.total_injuries}</div>
            <div style="color: {color}; font-size: 0.9rem;">{injury_rate:.1f}%</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        top_category = max(st.session_state.category_analysis.items(), key=lambda x: x[1])[0] if st.session_state.category_analysis else 'None'
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Top Return Reason</div>
            <div class="metric-value" style="font-size: 1.2rem;">{top_category}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col5:
        unique_products = len(st.session_state.product_analysis) if st.session_state.product_analysis else 0
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Products Affected</div>
            <div class="metric-value">{unique_products}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col6:
        if st.session_state.fda_mdr_required:
            st.markdown(f"""
            <div class="fda-alert">
                FDA MDR<br>REQUIRED
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">AI Cost</div>
                <div class="metric-value">${st.session_state.api_cost:.4f}</div>
            </div>
            """, unsafe_allow_html=True)
    
    # Show injury summary if injuries detected
    if st.session_state.total_injuries > 0:
        st.markdown("---")
        display_injury_summary()

def display_category_analysis():
    """Display category breakdown and analysis"""
    st.markdown("### 📈 Return Category Analysis")
    
    if not st.session_state.category_analysis:
        st.info("No category data available")
        return
    
    # Create two columns
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Category distribution chart
        category_df = pd.DataFrame(
            list(st.session_state.category_analysis.items()),
            columns=['Category', 'Count']
        ).sort_values('Count', ascending=False)
        
        # Color mapping with injury highlight
        colors = []
        for cat in category_df['Category']:
            if cat == 'Medical/Health Concerns':
                colors.append(COLORS['injury'])
            elif cat in QUALITY_CATEGORIES:
                colors.append(COLORS['danger'])
            elif cat in ['Customer Error/Changed Mind', 'Price/Value']:
                colors.append(COLORS['success'])
            else:
                colors.append(COLORS['primary'])
        
        fig = px.bar(
            category_df,
            x='Count',
            y='Category',
            orientation='h',
            title='Returns by Category',
            color='Category',
            color_discrete_sequence=colors
        )
        fig.update_layout(showlegend=False, height=500)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Quality vs Non-Quality vs Injury breakdown
        quality_count = sum(count for cat, count in st.session_state.category_analysis.items() 
                          if cat in QUALITY_CATEGORIES and cat != 'Medical/Health Concerns')
        injury_count = st.session_state.category_analysis.get('Medical/Health Concerns', 0)
        non_quality_count = st.session_state.total_returns - quality_count - injury_count
        
        fig = go.Figure(data=[go.Pie(
            labels=['Injury/Health', 'Other Quality', 'Non-Quality'],
            values=[injury_count, quality_count, non_quality_count],
            hole=.3,
            marker_colors=[COLORS['injury'], COLORS['danger'], COLORS['primary']]
        )])
        fig.update_layout(title='Return Type Distribution', height=300)
        st.plotly_chart(fig, use_container_width=True)
        
        # Top insights
        st.markdown("#### 💡 Key Insights")
        
        insights = generate_category_insights()
        for insight in insights[:5]:
            st.markdown(f"• {insight}")

def generate_category_insights():
    """Generate insights from category analysis with injury focus"""
    insights = []
    
    if not st.session_state.category_analysis:
        return insights
    
    # Injury insights (highest priority)
    injury_rate = st.session_state.quality_metrics.get('injury_rate', 0)
    if st.session_state.critical_injuries > 0:
        insights.append(f"🚨 CRITICAL: {st.session_state.critical_injuries} critical injury case(s) require immediate action")
    elif st.session_state.total_injuries > 0:
        insights.append(f"⚠️ {st.session_state.total_injuries} injury cases detected ({injury_rate:.1f}% of returns)")
    
    # Quality issue insights
    quality_rate = st.session_state.quality_metrics.get('quality_rate', 0)
    if quality_rate > 30:
        insights.append(f"⚠️ Critical: {quality_rate:.1f}% of returns are quality-related")
    elif quality_rate > 15:
        insights.append(f"⚠️ High quality issue rate: {quality_rate:.1f}%")
    
    # Top category insights
    top_categories = sorted(st.session_state.category_analysis.items(), key=lambda x: x[1], reverse=True)[:3]
    for cat, count in top_categories:
        percentage = (count / st.session_state.total_returns * 100)
        if cat == 'Medical/Health Concerns':
            insights.append(f"🚨 {cat}: {count} returns ({percentage:.1f}%) - SAFETY PRIORITY")
        elif cat in QUALITY_CATEGORIES:
            insights.append(f"🔴 {cat}: {count} returns ({percentage:.1f}%)")
        else:
            insights.append(f"📊 {cat}: {count} returns ({percentage:.1f}%)")
    
    # Specific category insights
    if 'Size/Fit Issues' in st.session_state.category_analysis and st.session_state.category_analysis['Size/Fit Issues'] > 5:
        insights.append("👔 Size/fit issues suggest need for better sizing guide")
    
    if 'Wrong Product/Misunderstanding' in st.session_state.category_analysis and st.session_state.category_analysis['Wrong Product/Misunderstanding'] > 3:
        insights.append("📝 Product description may need clarification")
    
    return insights

def display_product_analysis():
    """Display product-specific analysis with injury tracking"""
    st.markdown("### 📦 Product Quality & Safety Analysis")
    
    if not st.session_state.product_analysis:
        st.info("No product data available")
        return
    
    # Products with injuries (highest priority)
    products_with_injuries = [
        (asin, data) for asin, data in st.session_state.product_analysis.items() 
        if data.get('injuries', 0) > 0
    ]
    
    if products_with_injuries:
        st.markdown("#### 🚨 Products with Injury Cases")
        
        injury_products_df = []
        for asin, data in products_with_injuries:
            injury_products_df.append({
                'ASIN': asin,
                'Total Returns': data.get('total', 0),
                'Total Injuries': data.get('injuries', 0),
                'Critical': data.get('injury_critical', 0),
                'High': data.get('injury_high', 0),
                'Medium': data.get('injury_medium', 0),
                'Injury Rate': (data.get('injuries', 0) / data.get('total', 1) * 100)
            })
        
        injury_df = pd.DataFrame(injury_products_df).sort_values('Total Injuries', ascending=False)
        
        for idx, product in injury_df.iterrows():
            severity_text = []
            if product['Critical'] > 0:
                severity_text.append(f"<span class='injury-severity-critical'>CRITICAL: {product['Critical']}</span>")
            if product['High'] > 0:
                severity_text.append(f"<span class='injury-severity-high'>HIGH: {product['High']}</span>")
            if product['Medium'] > 0:
                severity_text.append(f"<span class='injury-severity-medium'>MEDIUM: {product['Medium']}</span>")
            
            st.markdown(f"""
            <div class="injury-case-card">
                <div style="display: flex; justify-content: space-between;">
                    <div>
                        <strong>ASIN: {product['ASIN']}</strong>
                        <div style="margin-top: 0.5rem;">
                            {' '.join(severity_text)}
                        </div>
                    </div>
                    <div style="text-align: right;">
                        <div style="font-size: 1.5rem; font-weight: bold; color: {COLORS['injury']};">
                            {product['Total Injuries']} injuries
                        </div>
                        <div style="color: {COLORS['injury']}; font-size: 0.9rem;">
                            {product['Injury Rate']:.1f}% injury rate
                        </div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
    
    # Filter products with quality issues
    products_with_quality = []
    
    for asin, categories in st.session_state.product_analysis.items():
        quality_count = sum(count for cat, count in categories.items() 
                          if cat in QUALITY_CATEGORIES and cat != 'total' and not cat.startswith('injury'))
        total = categories.get('total', 0)
        
        if quality_count > 0:
            products_with_quality.append({
                'ASIN': asin,
                'Total Returns': total,
                'Quality Issues': quality_count,
                'Quality Rate': (quality_count / total * 100) if total > 0 else 0,
                'Top Issue': max([(cat, count) for cat, count in categories.items() 
                                if cat != 'total' and not cat.startswith('injury')], 
                               key=lambda x: x[1])[0],
                'Has Injuries': categories.get('injuries', 0) > 0
            })
    
    if products_with_quality:
        # Sort by quality issues
        products_df = pd.DataFrame(products_with_quality).sort_values('Quality Issues', ascending=False)
        
        # Display top problematic products
        st.markdown("#### 🔴 Products with Quality Issues")
        
        for idx, product in products_df.head(10).iterrows():
            color = COLORS['danger'] if product['Quality Rate'] > 50 else COLORS['warning'] if product['Quality Rate'] > 25 else COLORS['primary']
            injury_flag = " 🚨" if product['Has Injuries'] else ""
            
            st.markdown(f"""
            <div class="category-card quality-issue-card">
                <div style="display: flex; justify-content: space-between;">
                    <div>
                        <strong>ASIN: {product['ASIN']}{injury_flag}</strong>
                        <div style="color: #666; font-size: 0.9rem;">Top Issue: {product['Top Issue']}</div>
                    </div>
                    <div style="text-align: right;">
                        <div style="font-size: 1.5rem; font-weight: bold; color: {color};">
                            {product['Quality Issues']} / {product['Total Returns']}
                        </div>
                        <div style="color: {color}; font-size: 0.9rem;">
                            {product['Quality Rate']:.1f}% quality issues
                        </div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

def display_recommendations():
    """Display quality improvement recommendations with injury priority"""
    st.markdown("### 💡 Quality & Safety Improvement Recommendations")
    
    recommendations = generate_quality_recommendations()
    
    # Priority recommendations
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("#### 🎯 Priority Actions")
        
        for idx, rec in enumerate(recommendations['priority']):
            priority_color = {
                'IMMEDIATE': COLORS['critical'],
                'HIGH': COLORS['warning'],
                'MEDIUM': COLORS['primary']
            }.get(rec['priority'], COLORS['primary'])
            
            st.markdown(f"""
            <div class="category-card" style="border-left-color: {priority_color};">
                <div style="display: flex; justify-content: space-between;">
                    <div>
                        <strong>{idx + 1}. {rec['action']}</strong>
                        <div style="color: #666; font-size: 0.9rem; margin-top: 0.5rem;">
                            {rec['reason']}
                        </div>
                    </div>
                    <div>
                        <span style="background: {priority_color}; color: white; padding: 0.3rem 0.8rem; 
                                     border-radius: 15px; font-size: 0.8rem;">
                            {rec['priority']}
                        </span>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("#### 📊 Quality Score")
        
        # Calculate overall quality score
        quality_score = calculate_quality_score()
        
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Overall Quality Score</div>
            <div class="metric-value" style="font-size: 3rem; color: {get_score_color(quality_score)};">
                {quality_score}/100
            </div>
            <div class="quality-score" style="margin-top: 1rem;">
                <div style="width: {quality_score}%; background: {get_score_color(quality_score)}; 
                            height: 100%; border-radius: 5px;"></div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Score breakdown
        st.markdown("##### Score Factors:")
        st.markdown(f"• Quality issue rate: -{st.session_state.quality_metrics['quality_rate']:.0f} pts")
        if st.session_state.total_injuries > 0:
            st.markdown(f"• Injury cases: -{20 if st.session_state.critical_injuries > 0 else 10} pts")
        if st.session_state.fda_mdr_required:
            st.markdown(f"• FDA MDR required: -10 pts")

def generate_quality_recommendations():
    """Generate specific quality improvement recommendations with injury focus"""
    recommendations = {'priority': [], 'general': []}
    
    # Injury-based recommendations (highest priority)
    if st.session_state.critical_injuries > 0:
        recommendations['priority'].insert(0, {
            'priority': 'IMMEDIATE',
            'action': 'Address critical injury cases - FDA MDR reporting may be required',
            'reason': f'{st.session_state.critical_injuries} critical injury case(s) detected requiring immediate investigation'
        })
    elif st.session_state.total_injuries > 0:
        recommendations['priority'].append({
            'priority': 'HIGH',
            'action': 'Review all injury cases for safety improvements',
            'reason': f'{st.session_state.total_injuries} injury cases indicate potential safety issues'
        })
    
    # Based on quality rate
    quality_rate = st.session_state.quality_metrics.get('quality_rate', 0)
    if quality_rate > 30:
        recommendations['priority'].append({
            'priority': 'IMMEDIATE' if not st.session_state.critical_injuries else 'HIGH',
            'action': 'Conduct emergency quality audit',
            'reason': f'{quality_rate:.1f}% quality issue rate exceeds critical threshold'
        })
    elif quality_rate > 15:
        recommendations['priority'].append({
            'priority': 'HIGH',
            'action': 'Review manufacturing QC processes',
            'reason': f'{quality_rate:.1f}% quality issue rate is above target'
        })
    
    # Based on specific categories
    for category, count in sorted(st.session_state.category_analysis.items(), key=lambda x: x[1], reverse=True):
        if category == 'Product Defects/Quality' and count > 10:
            recommendations['priority'].append({
                'priority': 'HIGH',
                'action': 'Investigate defect patterns with manufacturer',
                'reason': f'{count} defect-related returns require root cause analysis'
            })
            break
        elif category == 'Size/Fit Issues' and count > 5:
            recommendations['priority'].append({
                'priority': 'MEDIUM',
                'action': 'Update product dimensions and sizing guide',
                'reason': f'{count} size-related returns indicate measurement issues'
            })
    
    # Product-specific recommendations
    if st.session_state.product_analysis:
        products_with_injuries = [
            asin for asin, data in st.session_state.product_analysis.items()
            if data.get('injuries', 0) > 0
        ]
        
        if len(products_with_injuries) > 0:
            recommendations['priority'].append({
                'priority': 'HIGH',
                'action': f'Prioritize safety review for {len(products_with_injuries)} products with injury cases',
                'reason': f'Products involved in injury cases: {", ".join(products_with_injuries[:3])}'
            })
    
    return recommendations

def calculate_quality_score():
    """Calculate overall quality score with injury penalties"""
    score = 100
    
    # Deduct for quality issue rate
    quality_rate = st.session_state.quality_metrics.get('quality_rate', 0)
    score -= min(quality_rate, 50)  # Max 50 point deduction
    
    # Deduct for injuries
    if st.session_state.critical_injuries > 0:
        score -= 30  # Severe penalty for critical injuries
    elif st.session_state.total_injuries > 0:
        score -= 15  # Moderate penalty for any injuries
    
    # Deduct for FDA MDR requirement
    if st.session_state.fda_mdr_required:
        score -= 10
    
    # Deduct for multiple products affected
    products_affected = st.session_state.quality_metrics.get('products_with_quality_issues', 0)
    if products_affected > 5:
        score -= 10
    elif products_affected > 3:
        score -= 5
    
    return max(0, int(score))

def get_score_color(score):
    """Get color based on quality score"""
    if score >= 90:
        return COLORS['success']
    elif score >= 70:
        return COLORS['warning']
    else:
        return COLORS['danger']

def export_analysis():
    """Export analysis results with injury report"""
    st.markdown("### 📥 Export Analysis Results")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if st.button("🚨 Export Injury Report", use_container_width=True, key="export_injury_main"):
            export_injury_report()
    
    with col2:
        if st.button("📊 Export Full Report (Excel)", use_container_width=True):
            export_excel_report()
    
    with col3:
        if st.button("📋 Export Categorized Data (CSV)", use_container_width=True):
            export_csv_data()
    
    with col4:
        if st.button("📄 Export Executive Summary", use_container_width=True):
            export_executive_summary()

def export_excel_report():
    """Export comprehensive Excel report with injury sheet"""
    try:
        output = io.BytesIO()
        
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            # Sheet 1: Categorized Returns
            st.session_state.categorized_data.to_excel(
                writer, sheet_name='Categorized_Returns', index=False
            )
            
            # Sheet 2: Injury Cases (if any)
            if st.session_state.total_injuries > 0:
                injury_cases = []
                for case in st.session_state.injury_cases:
                    injury_cases.append({
                        'Order ID': case.order_id,
                        'ASIN': case.asin,
                        'SKU': case.sku,
                        'Date': case.return_date,
                        'Severity': case.severity.upper(),
                        'Reportable': 'Yes' if case.reportable else 'No',
                        'Device Type': case.device_type,
                        'Injury Keywords': ', '.join(case.injury_keywords),
                        'Potential Causes': ', '.join(case.potential_causes),
                        'Comment': case.full_comment,
                        'Recommendation': case.recommendation
                    })
                
                injury_df = pd.DataFrame(injury_cases)
                injury_df.to_excel(writer, sheet_name='Injury_Cases', index=False)
            
            # Sheet 3: Category Summary
            category_summary = pd.DataFrame(
                list(st.session_state.category_analysis.items()),
                columns=['Category', 'Count']
            ).sort_values('Count', ascending=False)
            category_summary['Percentage'] = (category_summary['Count'] / st.session_state.total_returns * 100).round(1)
            category_summary.to_excel(writer, sheet_name='Category_Summary', index=False)
            
            # Sheet 4: Product Analysis
            if st.session_state.product_analysis:
                product_data = []
                for asin, categories in st.session_state.product_analysis.items():
                    row = {'ASIN': asin}
                    for cat, count in categories.items():
                        if cat != 'total' and not cat.startswith('injury'):
                            row[cat] = count
                    row['Total'] = categories.get('total', 0)
                    row['Injuries'] = categories.get('injuries', 0)
                    product_data.append(row)
                
                product_df = pd.DataFrame(product_data).fillna(0)
                product_df.to_excel(writer, sheet_name='Product_Analysis', index=False)
            
            # Sheet 5: Quality Metrics
            metrics_df = pd.DataFrame([st.session_state.quality_metrics])
            metrics_df.to_excel(writer, sheet_name='Quality_Metrics', index=False)
            
            # Add formatting
            workbook = writer.book
            header_format = workbook.add_format({
                'bold': True,
                'bg_color': '#2C3E50',
                'font_color': 'white',
                'border': 1
            })
            
            # Injury format for critical cases
            critical_format = workbook.add_format({
                'bg_color': '#DC143C',
                'font_color': 'white',
                'bold': True
            })
            
            # Format headers
            for sheet_name in writer.sheets:
                worksheet = writer.sheets[sheet_name]
                worksheet.set_row(0, 20, header_format)
                
                # Special formatting for injury sheet
                if sheet_name == 'Injury_Cases' and st.session_state.total_injuries > 0:
                    # Find rows with critical severity
                    for row_num, severity in enumerate(injury_df['Severity'], 1):
                        if severity == 'CRITICAL':
                            worksheet.set_row(row_num, None, critical_format)
        
        output.seek(0)
        
        st.download_button(
            label="Download Excel Report",
            data=output.getvalue(),
            file_name=f"amazon_returns_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
        
    except Exception as e:
        st.error(f"Export error: {str(e)}")

def export_csv_data():
    """Export categorized data as CSV"""
    csv = st.session_state.categorized_data.to_csv(index=False)
    
    st.download_button(
        label="Download CSV",
        data=csv,
        file_name=f"categorized_returns_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv"
    )

def export_executive_summary():
    """Export executive summary with injury focus"""
    summary = f"""
AMAZON RETURNS QUALITY & SAFETY ANALYSIS - EXECUTIVE SUMMARY
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

CRITICAL ALERTS
==============
FDA MDR Required: {'YES' if st.session_state.fda_mdr_required else 'NO'}
Critical Injuries: {st.session_state.critical_injuries}
Total Injuries: {st.session_state.total_injuries}

OVERVIEW
========
Total Returns Analyzed: {st.session_state.total_returns:,}
Quality-Related Returns: {st.session_state.quality_issues:,} ({st.session_state.quality_metrics['quality_rate']:.1f}%)
Injury Cases: {st.session_state.total_injuries} ({st.session_state.quality_metrics['injury_rate']:.1f}%)
Unique Products Affected: {len(st.session_state.product_analysis) if st.session_state.product_analysis else 0}
Products with Injuries: {st.session_state.quality_metrics.get('products_with_injuries', 0)}
Overall Quality Score: {calculate_quality_score()}/100

TOP RETURN CATEGORIES
====================
"""
    
    for cat, count in sorted(st.session_state.category_analysis.items(), key=lambda x: x[1], reverse=True)[:5]:
        percentage = (count / st.session_state.total_returns * 100)
        quality_flag = " [QUALITY ISSUE]" if cat in QUALITY_CATEGORIES else ""
        injury_flag = " [INJURY/SAFETY]" if cat == 'Medical/Health Concerns' else ""
        summary += f"{cat}{quality_flag}{injury_flag}: {count} ({percentage:.1f}%)\n"
    
    summary += f"""

KEY FINDINGS
============
"""
    
    insights = generate_category_insights()
    for insight in insights[:5]:
        summary += f"- {insight}\n"
    
    summary += f"""

PRIORITY RECOMMENDATIONS
=======================
"""
    
    recommendations = generate_quality_recommendations()
    for idx, rec in enumerate(recommendations['priority'][:5]):
        summary += f"{idx + 1}. [{rec['priority']}] {rec['action']}\n   Reason: {rec['reason']}\n\n"
    
    # Add injury summary if available
    if st.session_state.total_injuries > 0:
        summary += f"""

INJURY CASE SUMMARY
==================
Total Injury Cases: {st.session_state.total_injuries}
- Critical Severity: {st.session_state.critical_injuries}
- High Severity: {sum(1 for c in st.session_state.injury_cases if c.severity == 'high')}
- Medium Severity: {sum(1 for c in st.session_state.injury_cases if c.severity == 'medium')}

Products Involved in Injuries:
"""
        products_with_injuries = list(set(c.asin for c in st.session_state.injury_cases))
        for asin in products_with_injuries[:10]:
            injury_count = sum(1 for c in st.session_state.injury_cases if c.asin == asin)
            summary += f"- {asin}: {injury_count} case(s)\n"
        
        if st.session_state.fda_mdr_required:
            summary += """

FDA MDR REPORTING REQUIREMENTS
=============================
You have reportable injury cases that require FDA Medical Device Reporting.

IMMEDIATE ACTIONS REQUIRED:
1. Notify regulatory affairs team immediately
2. Preserve all complaint records and device samples
3. Conduct root cause analysis within 48 hours
4. Prepare MDR submission (5 days for critical, 30 days for other)
5. Evaluate need for product recall or safety alert
"""
    
    st.download_button(
        label="Download Summary",
        data=summary,
        file_name=f"executive_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
        mime="text/plain"
    )

def main():
    """Main application"""
    initialize_session_state()
    inject_professional_css()
    
    # Header
    display_header()
    
    # Sidebar
    with st.sidebar:
        st.markdown("### ⚙️ Analysis Settings")
        
        # File type info
        if st.session_state.file_type:
            st.info(f"📁 Current file type: {st.session_state.file_type.upper()}")
        
        # Injury filter toggle
        st.markdown("#### 🚨 Safety Filters")
        st.session_state.show_injuries_only = st.checkbox(
            "Show injury cases only",
            value=st.session_state.show_injuries_only
        )
        
        # Filters
        st.markdown("#### 🔍 Filters")
        
        # ASIN filter
        if st.session_state.product_analysis:
            asin_options = ['ALL'] + list(st.session_state.product_analysis.keys())
            st.session_state.selected_asin = st.selectbox(
                "Select ASIN",
                asin_options,
                index=0
            )
        
        # Category filter
        if st.session_state.category_analysis:
            category_options = ['All'] + list(st.session_state.category_analysis.keys())
            st.session_state.category_filter = st.selectbox(
                "Filter by Category",
                category_options,
                index=0
            )
        
        st.markdown("---")
        
        # Quality thresholds
        st.markdown("#### 📊 Quality Thresholds")
        quality_threshold = st.slider(
            "Quality issue alert threshold (%)",
            10, 50, 30,
            help="Alert when quality issues exceed this percentage"
        )
        
        st.markdown("---")
        
        # Session stats
        if st.session_state.processing_complete:
            st.markdown("#### 📈 Session Statistics")
            st.metric("Processing Time", f"{st.session_state.processing_time:.1f}s")
            st.metric("Total Injuries", st.session_state.total_injuries)
            
            if st.session_state.ai_analyzer:
                cost_summary = st.session_state.ai_analyzer.get_cost_summary()
                st.metric("API Cost", f"${cost_summary['total_cost']:.4f}")
                st.metric("API Calls", cost_summary['api_calls'])
        
        # Help
        with st.expander("📖 How to Use"):
            st.markdown("""
            **For PDF files:**
            1. Go to Seller Central → Manage Returns
            2. Filter by date/ASIN as needed
            3. Click Print → Save as PDF
            4. Upload PDF here
            
            **For FBA Returns:**
            1. Go to Reports → Fulfillment
            2. Download FBA Returns Report
            3. Upload .txt file here
            
            **Key Features:**
            - **Automatic injury detection**
            - FDA MDR compliance checks
            - Quality issue identification
            - Product-level insights
            - Export for action planning
            
            **Injury Detection:**
            System automatically flags:
            - Critical: Hospital, emergency, severe injury
            - High: Injury, bleeding, fracture
            - Medium: Pain, fall, bruise
            """)
    
    # Main content
    if not st.session_state.analysis_ready:
        upload_section()
    else:
        # Analysis tabs
        tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
            "📊 Dashboard",
            "🚨 Injuries" if st.session_state.total_injuries > 0 else "📈 Categories",
            "📈 Categories" if st.session_state.total_injuries > 0 else "📦 Products",
            "📦 Products" if st.session_state.total_injuries > 0 else "💡 Recommendations",
            "💡 Recommendations" if st.session_state.total_injuries > 0 else "📥 Export",
            "📥 Export" if st.session_state.total_injuries > 0 else "🔄 New Analysis",
            "🔄 New Analysis"
        ])
        
        with tab1:
            display_dashboard()
        
        if st.session_state.total_injuries > 0:
            with tab2:
                display_injury_summary()
            with tab3:
                display_category_analysis()
            with tab4:
                display_product_analysis()
            with tab5:
                display_recommendations()
            with tab6:
                export_analysis()
            with tab7:
                if st.button("🔄 Start New Analysis", type="primary", use_container_width=True):
                    # Reset state
                    for key in st.session_state:
                        if key not in ['ai_analyzer', 'pdf_analyzer', 'injury_detector', 'file_detector']:
                            st.session_state[key] = initialize_session_state.__defaults__[0].get(key)
                    st.rerun()
        else:
            with tab2:
                display_category_analysis()
            with tab3:
                display_product_analysis()
            with tab4:
                display_recommendations()
            with tab5:
                export_analysis()
            with tab6:
                if st.button("🔄 Start New Analysis", type="primary", use_container_width=True):
                    # Reset state
                    for key in st.session_state:
                        if key not in ['ai_analyzer', 'pdf_analyzer', 'injury_detector', 'file_detector']:
                            st.session_state[key] = initialize_session_state.__defaults__[0].get(key)
                    st.rerun()

if __name__ == "__main__":
    main()
