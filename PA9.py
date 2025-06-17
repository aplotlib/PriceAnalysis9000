"""
Amazon Returns Quality Analyzer - PDF & FBA Returns Processing
Version: 5.0 - Complete Quality Management System
For Quality Analysts to analyze returns, identify patterns, and drive quality improvements
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
import chardet

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import modules with error handling
try:
    from pdf_analyzer import PDFAnalyzer
    from injury_detector import InjuryDetector
    from universal_file_detector import UniversalFileDetector
    from enhanced_ai_analysis import EnhancedAIAnalyzer, AIProvider, MEDICAL_DEVICE_CATEGORIES, FBA_REASON_MAP
    from smart_column_mapper import SmartColumnMapper
    MODULES_AVAILABLE = True
    AI_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Some modules not found: {e}")
    MODULES_AVAILABLE = False
    AI_AVAILABLE = False
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

# App Configuration
st.set_page_config(
    page_title="Amazon Returns Quality Analyzer",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Professional color scheme
COLORS = {
    'primary': '#00D9FF',      # Cyan
    'secondary': '#FF006E',    # Pink
    'accent': '#FFB700',       # Gold
    'success': '#00F5A0',      # Green
    'warning': '#FF6B35',      # Orange
    'danger': '#FF0054',       # Red
    'critical': '#DC143C',     # Crimson
    'dark': '#2C3E50',         # Dark blue
    'light': '#ECF0F1',        # Light gray
    'quality': '#9B59B6'       # Purple for quality issues
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
    """Professional CSS styling for quality management"""
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
        'file_type': None,  # 'pdf' or 'structured'
        
        # Analysis results
        'category_analysis': {},
        'product_analysis': {},
        'quality_metrics': {},
        'injury_analysis': None,
        'ai_insights': None,
        
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
        
        # Tracking
        'total_returns': 0,
        'quality_issues': 0,
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
                logger.info("AI analyzer initialized")
        except Exception as e:
            logger.error(f"Failed to initialize AI: {e}")
    
    if not st.session_state.pdf_analyzer:
        st.session_state.pdf_analyzer = PDFAnalyzer()
    
    if not st.session_state.injury_detector:
        st.session_state.injury_detector = InjuryDetector()
    
    if not st.session_state.file_detector:
        st.session_state.file_detector = UniversalFileDetector()
    
    if not st.session_state.get('column_mapper'):
        st.session_state.column_mapper = SmartColumnMapper(st.session_state.ai_analyzer)

def display_header():
    """Display application header"""
    st.markdown("""
    <div class="main-header">
        <h1>📊 Amazon Returns Quality Analyzer</h1>
        <p style="font-size: 1.2rem; margin-top: 0.5rem;">
            AI-Powered Return Analysis for Quality Management
        </p>
        <p style="font-size: 1rem; opacity: 0.9;">
            Upload PDF or FBA Returns • Categorize Automatically • Identify Quality Issues • Export Insights
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Show quality alerts if issues found
    if st.session_state.quality_issues > 0:
        quality_rate = (st.session_state.quality_issues / st.session_state.total_returns * 100) if st.session_state.total_returns > 0 else 0
        st.markdown(f"""
        <div class="quality-alert">
            <h2 style="color: {COLORS['quality']}; margin: 0;">
                ⚠️ {st.session_state.quality_issues} Quality-Related Returns ({quality_rate:.1f}%)
            </h2>
            <p style="margin: 0.5rem 0 0 0;">Review quality categories for improvement opportunities</p>
        </div>
        """, unsafe_allow_html=True)

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
    
    # File format examples
    with st.expander("📋 Supported File Formats & Examples"):
        st.markdown("""
        **The tool uses AI to automatically detect columns in your file!**
        
        Common formats we handle:
        
        **1. Amazon FBA Returns (.txt)**
        - Headers: `return-date`, `order-id`, `sku`, `asin`, `reason`, `customer-comments`
        
        **2. Quality Tracking Spreadsheets**
        - Example: `Date Complaint was made`, `Product Identifier Tag`, `ASIN`, `Order #`, `Complaint`
        
        **3. Custom Exports**
        - Any file with columns for: dates, product IDs, order numbers, and complaint text
        
        **The AI will automatically map your columns** to the appropriate fields for analysis.
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
        
        with st.spinner("🔍 Analyzing return data..."):
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
            
            st.success(f"""
            ✅ Analysis Complete!
            - Total returns: {st.session_state.total_returns}
            - Quality issues: {st.session_state.quality_issues}
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
        st.error("No return data found in PDF. The PDF might be in an unexpected format.")
        
        # Try to show what was extracted for debugging
        if extracted_data and extracted_data.get('raw_text'):
            with st.expander("Show extracted text (first 1000 chars)"):
                st.text(extracted_data['raw_text'][:1000])
        return
    
    returns_data = extracted_data['returns']
    st.session_state.raw_data = returns_data
    
    # Convert to DataFrame for column mapping
    returns_df = pd.DataFrame(returns_data)
    
    # If we have a DataFrame, use smart column mapper
    if not returns_df.empty:
        st.markdown("### 🔍 Analyzing PDF Structure...")
        
        # Use column mapper to detect fields
        with st.spinner("Detecting data fields in PDF..."):
            column_mapping = st.session_state.column_mapper.detect_columns(returns_df)
        
        # Map to standardized format
        mapped_df = st.session_state.column_mapper.map_dataframe(returns_df, column_mapping)
        
        # Show what was found
        st.success(f"✅ Extracted {len(mapped_df)} returns from PDF")
        
        if column_mapping:
            with st.expander("Detected fields"):
                for field_type, col_name in column_mapping.items():
                    st.write(f"- **{field_type}**: {col_name}")
    else:
        # Fallback to list processing
        mapped_df = pd.DataFrame(returns_data)
    
    # Process and categorize returns
    categorized_returns = []
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for idx, row in mapped_df.iterrows():
        # Get complaint text
        complaint = row.get('customer_comment', '') or row.get('raw_content', '') or ''
        
        return_item = {
            'order_id': row.get('order_id', ''),
            'sku': row.get('sku', ''),
            'asin': row.get('asin', ''),
            'return_date': row.get('return_date', ''),
            'customer_comment': complaint,
            'page': row.get('page', 0)
        }
        
        if complaint and st.session_state.ai_analyzer:
            # Categorize using AI
            category, confidence, severity, language = st.session_state.ai_analyzer.categorize_return(complaint)
            return_item['category'] = category
            return_item['confidence'] = confidence
        else:
            return_item['category'] = 'Other/Miscellaneous'
            return_item['confidence'] = 0.1
        
        categorized_returns.append(return_item)
        progress_bar.progress((idx + 1) / len(mapped_df))
        status_text.text(f"Processing return {idx + 1} of {len(mapped_df)}...")
    
    progress_bar.empty()
    status_text.empty()
    
    st.session_state.categorized_data = pd.DataFrame(categorized_returns)
    st.session_state.total_returns = len(categorized_returns)
    
    # Check for injuries if injury detector is available
    if st.session_state.injury_detector and categorized_returns:
        st.session_state.injury_analysis = st.session_state.injury_detector.analyze_returns_for_injuries(categorized_returns)

def process_structured_file(content: bytes, filename: str, file_extension: str):
    """Process structured data files (CSV, TXT, Excel) with smart column mapping"""
    st.session_state.file_type = 'structured'
    
    # Read file based on extension
    try:
        if file_extension in ['csv', 'txt', 'tsv']:
            # Detect encoding
            try:
                text = content.decode('utf-8')
            except:
                import chardet
                encoding = chardet.detect(content)['encoding']
                text = content.decode(encoding)
            
            # Detect delimiter
            if file_extension == 'tsv' or '\t' in text.split('\n')[0]:
                delimiter = '\t'
            elif file_extension == 'csv' or ',' in text.split('\n')[0]:
                delimiter = ','
            else:
                # Try to detect
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
        
        # Show column mapping interface
        st.markdown("### 🔍 Column Detection & Mapping")
        
        # Use smart column mapper
        with st.spinner("Using AI to detect column types..."):
            column_mapping = st.session_state.column_mapper.detect_columns(df)
        
        # Display detected mappings
        st.success("✅ Columns detected automatically!")
        
        # Show mapping in expandable section
        with st.expander("View/Edit Column Mappings", expanded=True):
            st.markdown("**Detected Mappings:**")
            
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
        
        # Validate mapping
        validation = st.session_state.column_mapper.validate_mapping(df, column_mapping)
        
        if validation['missing_required']:
            st.warning(f"⚠️ Missing required columns: {', '.join(validation['missing_required'])}")
            st.info("The tool will still process available data, but results may be limited.")
        
        # Show data preview with mapped columns
        st.markdown("### 📋 Data Preview")
        
        # Map to standardized format
        mapped_df = st.session_state.column_mapper.map_dataframe(df, column_mapping)
        
        # Show preview of mapped data
        preview_cols = ['return_date', 'sku', 'asin', 'order_id', 'customer_comment']
        available_preview_cols = [col for col in preview_cols if col in mapped_df.columns and mapped_df[col].notna().any()]
        
        if available_preview_cols:
            preview_df = mapped_df[available_preview_cols].head(5)
            st.dataframe(preview_df, use_container_width=True)
        
        # Process returns
        st.markdown("### 🤖 Categorizing Returns...")
        
        categorized_data = []
        progress_bar = st.progress(0)
        
        for idx, row in mapped_df.iterrows():
            return_item = {
                'order_id': row.get('order_id', ''),
                'sku': row.get('sku', ''),
                'asin': row.get('asin', ''),
                'return_date': row.get('return_date', ''),
                'customer_comment': row.get('customer_comment', ''),
                'source': row.get('source', ''),
                'agent': row.get('agent', ''),
                'original_index': idx
            }
            
            # Check for FBA reason if present
            fba_reason = row.get('reason', '') or row.get('fba_reason', '')
            
            # Categorize using AI
            if st.session_state.ai_analyzer and return_item['customer_comment']:
                category, confidence, severity, language = st.session_state.ai_analyzer.categorize_return(
                    return_item['customer_comment'],
                    fba_reason
                )
                return_item['category'] = category
                return_item['confidence'] = confidence
            else:
                # Fallback categorization
                if fba_reason and fba_reason in FBA_REASON_MAP:
                    return_item['category'] = FBA_REASON_MAP[fba_reason]
                    return_item['confidence'] = 0.8
                else:
                    return_item['category'] = 'Other/Miscellaneous'
                    return_item['confidence'] = 0.1
            
            categorized_data.append(return_item)
            progress_bar.progress((idx + 1) / len(mapped_df))
        
        progress_bar.empty()
        
        # Store results
        st.session_state.categorized_data = pd.DataFrame(categorized_data)
        st.session_state.raw_data = categorized_data
        st.session_state.total_returns = len(categorized_data)
        
        # Run injury analysis if comments are available
        if any(item['customer_comment'] for item in categorized_data):
            if st.session_state.injury_detector:
                st.session_state.injury_analysis = st.session_state.injury_detector.analyze_returns_for_injuries(categorized_data)
        
        # Show column mapping summary
        st.markdown("### ✅ Processing Complete")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Columns Mapped", len(column_mapping))
        with col2:
            st.metric("Data Quality", f"{100 - len(validation['warnings']) * 10}%")
        with col3:
            st.metric("Rows Processed", len(categorized_data))
        
    except Exception as e:
        st.error(f"Error processing file: {str(e)}")
        logger.error(f"Structured file processing error: {e}", exc_info=True)

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
        
        st.session_state.product_analysis = dict(product_analysis)
    
    # Quality metrics
    st.session_state.quality_metrics = {
        'quality_rate': (st.session_state.quality_issues / st.session_state.total_returns * 100) if st.session_state.total_returns > 0 else 0,
        'top_quality_issue': quality_issues['category'].value_counts().index[0] if len(quality_issues) > 0 else 'None',
        'products_with_quality_issues': len(quality_issues['asin'].unique()) if 'asin' in quality_issues.columns else 0
    }

def display_dashboard():
    """Display main analysis dashboard"""
    st.markdown("### 📊 Return Analysis Dashboard")
    
    # Key metrics
    col1, col2, col3, col4, col5 = st.columns(5)
    
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
        top_category = max(st.session_state.category_analysis.items(), key=lambda x: x[1])[0] if st.session_state.category_analysis else 'None'
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Top Return Reason</div>
            <div class="metric-value" style="font-size: 1.2rem;">{top_category}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        unique_products = len(st.session_state.product_analysis) if st.session_state.product_analysis else 0
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Products Affected</div>
            <div class="metric-value">{unique_products}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col5:
        if st.session_state.injury_analysis:
            injuries = st.session_state.injury_analysis.get('total_injuries', 0)
            color = COLORS['critical'] if injuries > 0 else COLORS['success']
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Injury Cases</div>
                <div class="metric-value" style="color: {color};">{injuries}</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">AI Cost</div>
                <div class="metric-value">${st.session_state.api_cost:.4f}</div>
            </div>
            """, unsafe_allow_html=True)

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
        
        # Color mapping
        colors = []
        for cat in category_df['Category']:
            if cat in QUALITY_CATEGORIES:
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
        # Quality vs Non-Quality breakdown
        quality_count = sum(count for cat, count in st.session_state.category_analysis.items() if cat in QUALITY_CATEGORIES)
        non_quality_count = st.session_state.total_returns - quality_count
        
        fig = go.Figure(data=[go.Pie(
            labels=['Quality Issues', 'Other Returns'],
            values=[quality_count, non_quality_count],
            hole=.3,
            marker_colors=[COLORS['danger'], COLORS['primary']]
        )])
        fig.update_layout(title='Quality vs Non-Quality Returns', height=300)
        st.plotly_chart(fig, use_container_width=True)
        
        # Top insights
        st.markdown("#### 💡 Key Insights")
        
        insights = generate_category_insights()
        for insight in insights[:5]:
            st.markdown(f"• {insight}")

def generate_category_insights():
    """Generate insights from category analysis"""
    insights = []
    
    if not st.session_state.category_analysis:
        return insights
    
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
        if cat in QUALITY_CATEGORIES:
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
    """Display product-specific analysis"""
    st.markdown("### 📦 Product Quality Analysis")
    
    if not st.session_state.product_analysis:
        st.info("No product data available")
        return
    
    # Filter products with quality issues
    products_with_quality = []
    
    for asin, categories in st.session_state.product_analysis.items():
        quality_count = sum(count for cat, count in categories.items() if cat in QUALITY_CATEGORIES and cat != 'total')
        total = categories.get('total', 0)
        
        if quality_count > 0:
            products_with_quality.append({
                'ASIN': asin,
                'Total Returns': total,
                'Quality Issues': quality_count,
                'Quality Rate': (quality_count / total * 100) if total > 0 else 0,
                'Top Issue': max([(cat, count) for cat, count in categories.items() if cat != 'total'], key=lambda x: x[1])[0]
            })
    
    if products_with_quality:
        # Sort by quality issues
        products_df = pd.DataFrame(products_with_quality).sort_values('Quality Issues', ascending=False)
        
        # Display top problematic products
        st.markdown("#### 🚨 Products with Highest Quality Issues")
        
        for idx, product in products_df.head(10).iterrows():
            color = COLORS['danger'] if product['Quality Rate'] > 50 else COLORS['warning'] if product['Quality Rate'] > 25 else COLORS['primary']
            
            st.markdown(f"""
            <div class="category-card quality-issue-card">
                <div style="display: flex; justify-content: space-between;">
                    <div>
                        <strong>ASIN: {product['ASIN']}</strong>
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
        
        # Quality heatmap
        if st.checkbox("Show detailed category breakdown by product"):
            # Create matrix for heatmap
            products = list(st.session_state.product_analysis.keys())[:20]  # Top 20 products
            categories = list(set(cat for prod_data in st.session_state.product_analysis.values() for cat in prod_data.keys() if cat != 'total'))
            
            matrix = []
            for product in products:
                row = []
                for category in categories:
                    count = st.session_state.product_analysis[product].get(category, 0)
                    row.append(count)
                matrix.append(row)
            
            fig = px.imshow(
                matrix,
                labels=dict(x="Category", y="Product (ASIN)", color="Returns"),
                x=categories,
                y=products,
                color_continuous_scale='Reds',
                title="Return Categories by Product"
            )
            fig.update_layout(height=600)
            st.plotly_chart(fig, use_container_width=True)

def display_recommendations():
    """Display quality improvement recommendations"""
    st.markdown("### 💡 Quality Improvement Recommendations")
    
    recommendations = generate_quality_recommendations()
    
    # Priority recommendations
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("#### 🎯 Priority Actions")
        
        for idx, rec in enumerate(recommendations['priority']):
            priority_color = {
                'IMMEDIATE': COLORS['danger'],
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
        if st.session_state.injury_analysis and st.session_state.injury_analysis['total_injuries'] > 0:
            st.markdown(f"• Injury cases: -20 pts")
        
        products_affected = st.session_state.quality_metrics.get('products_with_quality_issues', 0)
        if products_affected > 5:
            st.markdown(f"• Multiple products affected: -10 pts")

def generate_quality_recommendations():
    """Generate specific quality improvement recommendations"""
    recommendations = {'priority': [], 'general': []}
    
    # Based on quality rate
    quality_rate = st.session_state.quality_metrics.get('quality_rate', 0)
    if quality_rate > 30:
        recommendations['priority'].append({
            'priority': 'IMMEDIATE',
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
                'priority': 'IMMEDIATE',
                'action': 'Investigate defect patterns with manufacturer',
                'reason': f'{count} defect-related returns require root cause analysis'
            })
            break
        elif category == 'Size/Fit Issues' and count > 5:
            recommendations['priority'].append({
                'priority': 'HIGH',
                'action': 'Update product dimensions and sizing guide',
                'reason': f'{count} size-related returns indicate measurement issues'
            })
        elif category == 'Performance/Effectiveness' and count > 5:
            recommendations['priority'].append({
                'priority': 'HIGH',
                'action': 'Review product specifications and testing',
                'reason': f'{count} performance issues reported by customers'
            })
    
    # Based on products
    if st.session_state.product_analysis:
        high_issue_products = [
            asin for asin, data in st.session_state.product_analysis.items()
            if sum(count for cat, count in data.items() if cat in QUALITY_CATEGORIES and cat != 'total') > 5
        ]
        
        if len(high_issue_products) > 3:
            recommendations['priority'].append({
                'priority': 'MEDIUM',
                'action': 'Review common components across affected products',
                'reason': f'{len(high_issue_products)} products show quality issues'
            })
    
    # Injury-based recommendations
    if st.session_state.injury_analysis and st.session_state.injury_analysis['total_injuries'] > 0:
        recommendations['priority'].insert(0, {
            'priority': 'IMMEDIATE',
            'action': 'Review injury cases for safety compliance',
            'reason': f"{st.session_state.injury_analysis['total_injuries']} potential injury cases detected"
        })
    
    return recommendations

def calculate_quality_score():
    """Calculate overall quality score (0-100)"""
    score = 100
    
    # Deduct for quality issue rate
    quality_rate = st.session_state.quality_metrics.get('quality_rate', 0)
    score -= min(quality_rate, 50)  # Max 50 point deduction
    
    # Deduct for injuries
    if st.session_state.injury_analysis and st.session_state.injury_analysis['total_injuries'] > 0:
        score -= 20
    
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
    """Export analysis results"""
    st.markdown("### 📥 Export Analysis Results")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📊 Export Full Report (Excel)", use_container_width=True):
            export_excel_report()
    
    with col2:
        if st.button("📋 Export Categorized Data (CSV)", use_container_width=True):
            export_csv_data()
    
    with col3:
        if st.button("📄 Export Executive Summary", use_container_width=True):
            export_executive_summary()

def export_excel_report():
    """Export comprehensive Excel report"""
    try:
        output = io.BytesIO()
        
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            # Sheet 1: Categorized Returns
            st.session_state.categorized_data.to_excel(
                writer, sheet_name='Categorized_Returns', index=False
            )
            
            # Sheet 2: Category Summary
            category_summary = pd.DataFrame(
                list(st.session_state.category_analysis.items()),
                columns=['Category', 'Count']
            ).sort_values('Count', ascending=False)
            category_summary['Percentage'] = (category_summary['Count'] / st.session_state.total_returns * 100).round(1)
            category_summary.to_excel(writer, sheet_name='Category_Summary', index=False)
            
            # Sheet 3: Product Analysis
            if st.session_state.product_analysis:
                product_data = []
                for asin, categories in st.session_state.product_analysis.items():
                    row = {'ASIN': asin}
                    for cat, count in categories.items():
                        if cat != 'total':
                            row[cat] = count
                    row['Total'] = categories.get('total', 0)
                    product_data.append(row)
                
                product_df = pd.DataFrame(product_data).fillna(0)
                product_df.to_excel(writer, sheet_name='Product_Analysis', index=False)
            
            # Sheet 4: Quality Metrics
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
            
            # Format headers
            for sheet_name in writer.sheets:
                worksheet = writer.sheets[sheet_name]
                worksheet.set_row(0, 20, header_format)
        
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
    """Export executive summary"""
    summary = f"""
AMAZON RETURNS QUALITY ANALYSIS - EXECUTIVE SUMMARY
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

OVERVIEW
========
Total Returns Analyzed: {st.session_state.total_returns:,}
Quality-Related Returns: {st.session_state.quality_issues:,} ({st.session_state.quality_metrics['quality_rate']:.1f}%)
Unique Products Affected: {len(st.session_state.product_analysis) if st.session_state.product_analysis else 0}
Overall Quality Score: {calculate_quality_score()}/100

TOP RETURN CATEGORIES
====================
"""
    
    for cat, count in sorted(st.session_state.category_analysis.items(), key=lambda x: x[1], reverse=True)[:5]:
        percentage = (count / st.session_state.total_returns * 100)
        quality_flag = " [QUALITY ISSUE]" if cat in QUALITY_CATEGORIES else ""
        summary += f"{cat}{quality_flag}: {count} ({percentage:.1f}%)\n"
    
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
    
    # Add injury information if available
    if st.session_state.injury_analysis and st.session_state.injury_analysis['total_injuries'] > 0:
        summary += f"""

SAFETY CONCERNS
===============
Potential Injury Cases: {st.session_state.injury_analysis['total_injuries']}
Critical Severity: {st.session_state.injury_analysis['severity_breakdown']['critical']}
High Severity: {st.session_state.injury_analysis['severity_breakdown']['high']}

IMMEDIATE ACTION REQUIRED: Review all injury cases for regulatory compliance.
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
            
            **Analysis Features:**
            - Automatic categorization
            - Quality issue identification
            - Product-level insights
            - Export for action planning
            """)
    
    # Main content
    if not st.session_state.analysis_ready:
        upload_section()
    else:
        # Analysis tabs
        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
            "📊 Dashboard",
            "📈 Categories",
            "📦 Products",
            "💡 Recommendations",
            "📥 Export",
            "🔄 New Analysis"
        ])
        
        with tab1:
            display_dashboard()
        
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
