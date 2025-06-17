"""
PA9.py - Amazon Return Analysis Tool
Professional quality management system for analyzing Amazon returns
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import time
import os
import sys
import re
import io
import base64
from typing import Dict, List, Tuple, Optional, Any
import logging
import traceback

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import modules with error handling
try:
    from enhanced_ai_analysis import EnhancedAIAnalyzer, AIProvider
    AI_AVAILABLE = True
except ImportError as e:
    logger.error(f"Failed to import AI module: {e}")
    AI_AVAILABLE = False

try:
    from universal_file_detector import UniversalFileDetector
    FILE_DETECTOR_AVAILABLE = True
except ImportError as e:
    logger.error(f"Failed to import file detector: {e}")
    FILE_DETECTOR_AVAILABLE = False

try:
    from pdf_analyzer import PDFAnalyzer
    PDF_AVAILABLE = True
except ImportError as e:
    logger.error(f"Failed to import PDF analyzer: {e}")
    PDF_AVAILABLE = False

try:
    from injury_detector import InjuryDetector
    INJURY_AVAILABLE = True
except ImportError as e:
    logger.error(f"Failed to import injury detector: {e}")
    INJURY_AVAILABLE = False

try:
    from smart_column_mapper import SmartColumnMapper
    MAPPER_AVAILABLE = True
except ImportError as e:
    logger.error(f"Failed to import column mapper: {e}")
    MAPPER_AVAILABLE = False

# Category definitions
QUALITY_CATEGORIES = ['QUALITY_DEFECTS', 'FUNCTIONALITY_ISSUES', 'COMPATIBILITY_ISSUES', 'INJURY_RISK']

# Color scheme
COLORS = {
    'primary': '#1E88E5',
    'secondary': '#FFC107',
    'success': '#4CAF50',
    'danger': '#F44336',
    'warning': '#FF9800',
    'info': '#00BCD4',
    'dark': '#212529',
    'light': '#F8F9FA',
    'critical': '#D32F2F'
}

# Define defaults dictionary globally for reuse
SESSION_DEFAULTS = {
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
    'api_cost': 0.0,
    'total_injuries': 0  # Added missing field
}

def display_header():
    """Display professional header with navigation"""
    # Add navigation menu
    col1, col2, col3 = st.columns([6, 3, 1])
    
    with col1:
        st.markdown("""
        <div style="background: linear-gradient(135deg, #1E88E5 0%, #00BCD4 100%); 
                    padding: 2rem; border-radius: 10px; margin-bottom: 2rem;">
            <h1 style="color: white; margin: 0; font-size: 2.5rem;">🔍 PriceAnalysis9000</h1>
            <p style="color: white; opacity: 0.9; margin: 0.5rem 0;">
                Professional Return Analysis & Quality Management System
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("### 🔗 Quick Links")
        # Navigation menu dropdown
        selected_app = st.selectbox(
            "Navigate to:",
            ["Current App", 
             "Amazon Review Analyzer", 
             "Return Categorization", 
             "AI Translator/Chatbot", 
             "Marketing ROI Tool"],
            key="nav_menu"
        )
        
        if selected_app != "Current App":
            app_urls = {
                "Amazon Review Analyzer": "https://azanalysis.streamlit.app/",
                "Return Categorization": "https://testenv.streamlit.app/",
                "AI Translator/Chatbot": "https://vivequalitychatbot.streamlit.app/",
                "Marketing ROI Tool": "https://kaizenroi.streamlit.app/"
            }
            st.markdown(f'<a href="{app_urls[selected_app]}" target="_blank" style="text-decoration: none;">'
                       f'<button style="background: #1E88E5; color: white; border: none; '
                       f'padding: 0.5rem 1rem; border-radius: 5px; cursor: pointer;">Go to {selected_app}</button></a>', 
                       unsafe_allow_html=True)
    
    with col3:
        if st.button("ℹ️", help="About this tool"):
            st.info("PriceAnalysis9000 v2.0 - AI-Powered Return Analysis")

def inject_professional_css():
    """Inject custom CSS for professional styling"""
    st.markdown(f"""
    <style>
    /* Global styles */
    .main {{
        padding: 0;
    }}
    
    /* Header styling */
    .main-header {{
        background: linear-gradient(135deg, {COLORS['primary']} 0%, {COLORS['info']} 100%);
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        box-shadow: 0 5px 20px rgba(0,0,0,0.1);
    }}
    
    /* Category badges */
    .category-badge {{
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 20px;
        font-size: 0.875rem;
        font-weight: 600;
        margin: 0.25rem;
    }}
    
    .quality-issue {{
        background: {COLORS['danger']};
        color: white;
    }}
    
    .size-issue {{
        background: {COLORS['warning']};
        color: white;
    }}
    
    .other-issue {{
        background: {COLORS['secondary']};
        color: {COLORS['dark']};
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
    
    /* Navigation styling */
    .nav-link {{
        display: inline-block;
        padding: 0.5rem 1rem;
        margin: 0.25rem;
        background: {COLORS['info']};
        color: white;
        text-decoration: none;
        border-radius: 5px;
        transition: all 0.3s;
    }}
    
    .nav-link:hover {{
        background: {COLORS['primary']};
        transform: translateY(-2px);
    }}
    
    /* Hide Streamlit branding */
    #MainMenu {{visibility: hidden;}}
    footer {{visibility: hidden;}}
    </style>
    """, unsafe_allow_html=True)

def initialize_session_state():
    """Initialize session state variables"""
    for key, value in SESSION_DEFAULTS.items():
        if key not in st.session_state:
            st.session_state[key] = value

def reset_analysis_state():
    """Reset analysis-related session state while preserving analyzers"""
    # List of keys to preserve (analyzers and detectors)
    preserve_keys = ['ai_analyzer', 'pdf_analyzer', 'injury_detector', 'file_detector', 'column_mapper']
    
    # Reset all other keys to their defaults
    for key, default_value in SESSION_DEFAULTS.items():
        if key not in preserve_keys:
            st.session_state[key] = default_value

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
    
    if not st.session_state.pdf_analyzer and PDF_AVAILABLE:
        st.session_state.pdf_analyzer = PDFAnalyzer()
    
    if not st.session_state.injury_detector and INJURY_AVAILABLE:
        st.session_state.injury_detector = InjuryDetector()
    
    if not st.session_state.file_detector and FILE_DETECTOR_AVAILABLE:
        st.session_state.file_detector = UniversalFileDetector()
    
    if not st.session_state.column_mapper and MAPPER_AVAILABLE:
        st.session_state.column_mapper = SmartColumnMapper()

def generate_basic_insights(category_analysis: Dict, product_analysis: Dict) -> str:
    """Generate basic insights without AI"""
    insights = []
    
    # Category insights
    if category_analysis:
        total_returns = sum(category_analysis.values())
        quality_returns = sum(count for cat, count in category_analysis.items() if cat in QUALITY_CATEGORIES)
        quality_rate = (quality_returns / total_returns * 100) if total_returns > 0 else 0
        
        insights.append(f"### 📊 Category Analysis")
        insights.append(f"- Total Returns: {total_returns:,}")
        insights.append(f"- Quality Issues: {quality_returns:,} ({quality_rate:.1f}%)")
        
        # Top categories
        top_categories = sorted(category_analysis.items(), key=lambda x: x[1], reverse=True)[:3]
        insights.append(f"\n**Top Return Categories:**")
        for cat, count in top_categories:
            pct = (count / total_returns * 100)
            insights.append(f"- {cat}: {count} ({pct:.1f}%)")
    
    # Product insights
    if product_analysis:
        insights.append(f"\n### 📦 Product Analysis")
        
        # High-risk products
        high_risk = [(asin, data) for asin, data in product_analysis.items() 
                     if (data['quality_issues'] / data['total_returns'] * 100) > 40]
        
        if high_risk:
            insights.append(f"\n**⚠️ High-Risk Products ({len(high_risk)}):**")
            for asin, data in sorted(high_risk, key=lambda x: x[1]['total_returns'], reverse=True)[:5]:
                quality_pct = (data['quality_issues'] / data['total_returns'] * 100)
                insights.append(f"- {asin}: {data['total_returns']} returns ({quality_pct:.1f}% quality)")
    
    # Recommendations
    insights.append(f"\n### 💡 Recommendations")
    if quality_rate > 30:
        insights.append("- 🚨 **Critical**: Quality issue rate exceeds 30%. Immediate investigation required.")
    if len(high_risk) > 0:
        insights.append(f"- ⚠️ **Action Required**: {len(high_risk)} products have >40% quality returns.")
    
    return '\n'.join(insights)

def process_uploaded_file(uploaded_file):
    """Process uploaded file and extract return data"""
    try:
        start_time = time.time()
        
        # Determine file type
        file_extension = uploaded_file.name.split('.')[-1].lower()
        
        if file_extension == 'pdf':
            # Process PDF
            if st.session_state.pdf_analyzer:
                with st.spinner("📄 Extracting data from PDF..."):
                    # Read PDF content
                    pdf_content = uploaded_file.read()
                    filename = uploaded_file.name
                    
                    # Extract returns from PDF
                    extracted_data = st.session_state.pdf_analyzer.extract_returns_from_pdf(pdf_content, filename)
                    
                    if 'error' in extracted_data:
                        st.error(f"PDF extraction error: {extracted_data['error']}")
                        return None
                    
                    # Convert returns list to DataFrame
                    returns_list = extracted_data.get('returns', [])
                    if not returns_list:
                        st.warning("No returns found in PDF. Please check the file format.")
                        return None
                    
                    # Create DataFrame with proper column mapping
                    data = pd.DataFrame(returns_list)
                    
                    # Ensure required columns exist
                    column_mapping = {
                        'order_id': 'Order ID',
                        'sku': 'SKU', 
                        'asin': 'ASIN',
                        'return_reason': 'Return Reason',
                        'customer_comment': 'Customer Comments',
                        'buyer_comment': 'Customer Comments',
                        'return_date': 'Return Date'
                    }
                    
                    # Rename columns to standard format
                    data = data.rename(columns=column_mapping)
                    
                    # Ensure Customer Comments column exists (use buyer_comment or customer_comment)
                    if 'Customer Comments' not in data.columns:
                        if 'buyer_comment' in data.columns:
                            data['Customer Comments'] = data['buyer_comment']
                        elif 'customer_comment' in data.columns:
                            data['Customer Comments'] = data['customer_comment']
                        else:
                            data['Customer Comments'] = ''
                    
                    st.session_state.file_type = 'pdf'
            else:
                st.error("PDF analyzer not available")
                return None
                
        elif file_extension in ['txt', 'csv']:
            # Process text/CSV file
            content = uploaded_file.read().decode('utf-8')
            
            # Use file detector if available
            if st.session_state.file_detector and FILE_DETECTOR_AVAILABLE:
                file_info = st.session_state.file_detector.detect_file_type(content, uploaded_file.name)
                
                if file_info['type'] == 'amazon_fba_returns':
                    st.session_state.file_type = 'fba_returns'
                    data = process_fba_returns(content)
                else:
                    # Try generic CSV processing
                    st.session_state.file_type = 'csv'
                    data = pd.read_csv(io.StringIO(content))
            else:
                # Fallback: Check if it's FBA returns format
                if 'return-date' in content and 'order-id' in content:
                    st.session_state.file_type = 'fba_returns'
                    data = process_fba_returns(content)
                else:
                    st.error("Unrecognized file format")
                    return None
                
        else:
            st.error(f"Unsupported file type: {file_extension}")
            return None
        
        # Store processing time
        st.session_state.processing_time = time.time() - start_time
        
        return data
        
    except Exception as e:
        logger.error(f"Error processing file: {e}")
        st.error(f"Error processing file: {str(e)}")
        return None

def process_fba_returns(content: str) -> pd.DataFrame:
    """Process FBA returns text file"""
    try:
        # Parse CSV content
        df = pd.read_csv(io.StringIO(content), sep='\t')
        
        # Map columns to standard format
        column_mapping = {
            'order-id': 'Order ID',
            'sku': 'SKU',
            'asin': 'ASIN',
            'product-name': 'Product Name',
            'reason': 'Return Reason',
            'customer-comments': 'Customer Comments',
            'return-date': 'Return Date',
            'quantity': 'Quantity'
        }
        
        # Rename columns
        df = df.rename(columns=column_mapping)
        
        # Ensure required columns exist
        required_columns = ['Order ID', 'SKU', 'ASIN', 'Return Reason']
        for col in required_columns:
            if col not in df.columns:
                df[col] = ''
        
        return df
        
    except Exception as e:
        logger.error(f"Error processing FBA returns: {e}")
        raise

def categorize_returns(df: pd.DataFrame) -> pd.DataFrame:
    """Categorize returns using AI or rule-based logic"""
    try:
        # Ensure required columns exist
        if 'Return Reason' not in df.columns:
            df['Return Reason'] = df.get('return_reason', '')
        if 'Customer Comments' not in df.columns:
            df['Customer Comments'] = df.get('customer_comment', df.get('buyer_comment', ''))
        if 'Order ID' not in df.columns:
            df['Order ID'] = df.get('order_id', '')
        if 'SKU' not in df.columns:
            df['SKU'] = df.get('sku', '') 
        if 'ASIN' not in df.columns:
            df['ASIN'] = df.get('asin', '')
            
        # Add category column
        df['Category'] = ''
        
        # Use AI if available
        if st.session_state.ai_analyzer:
            with st.spinner("🤖 AI categorizing returns..."):
                categories = []
                for idx, row in df.iterrows():
                    reason = str(row.get('Return Reason', ''))
                    comment = str(row.get('Customer Comments', ''))
                    
                    # Get AI categorization
                    result = st.session_state.ai_analyzer.categorize_return(
                        complaint=comment,  # Use comment as the main complaint text
                        return_reason=reason,
                        fba_reason=None,
                        return_data={'order_id': row.get('Order ID', ''), 'asin': row.get('ASIN', '')}
                    )
                    
                    # Extract category from result (it returns tuple or dict)
                    if isinstance(result, tuple):
                        category = result[0]  # First element is category
                    elif isinstance(result, dict):
                        category = result.get('category', 'OTHER')
                    else:
                        category = str(result)
                    
                    categories.append(category)
                    
                    # Update progress
                    if idx % 10 == 0:
                        progress = (idx + 1) / len(df)
                        st.progress(progress)
                
                df['Category'] = categories
        else:
            # Fallback to rule-based categorization
            df['Category'] = df.apply(lambda row: rule_based_categorization(
                str(row.get('Return Reason', '')),
                str(row.get('Customer Comments', ''))
            ), axis=1)
        
        # Check for injuries if detector available
        if st.session_state.injury_detector and INJURY_AVAILABLE:
            try:
                injuries = []
                for idx, row in df.iterrows():
                    text = str(row.get('Return Reason', '')) + ' ' + str(row.get('Customer Comments', ''))
                    
                    # Check if check_for_injury method exists
                    if hasattr(st.session_state.injury_detector, 'check_for_injury'):
                        has_injury = st.session_state.injury_detector.check_for_injury(text)
                    elif hasattr(st.session_state.injury_detector, 'detect_injury'):
                        result = st.session_state.injury_detector.detect_injury(text)
                        has_injury = result.get('has_injury', False) if isinstance(result, dict) else bool(result)
                    else:
                        # Simple keyword detection fallback
                        injury_keywords = ['injury', 'hurt', 'pain', 'hospital', 'emergency', 'bleeding']
                        has_injury = any(keyword in text.lower() for keyword in injury_keywords)
                    
                    injuries.append(has_injury)
                
                df['Has_Injury'] = injuries
                st.session_state.total_injuries = sum(injuries)
            except Exception as e:
                logger.warning(f"Injury detection failed: {e}")
                df['Has_Injury'] = False
                st.session_state.total_injuries = 0
        
        return df
        
    except Exception as e:
        logger.error(f"Error categorizing returns: {e}")
        raise

def rule_based_categorization(reason: str, comment: str) -> str:
    """Simple rule-based categorization as fallback"""
    text = (reason + ' ' + comment).lower()
    
    # Quality issues
    if any(word in text for word in ['defect', 'broken', 'damage', 'quality', 'faulty']):
        return 'QUALITY_DEFECTS'
    
    # Size issues
    elif any(word in text for word in ['small', 'large', 'size', 'fit']):
        return 'SIZE_FIT_ISSUES'
    
    # Wrong product
    elif any(word in text for word in ['wrong', 'incorrect', 'different']):
        return 'WRONG_PRODUCT'
    
    # Functionality
    elif any(word in text for word in ['work', 'function', 'operate']):
        return 'FUNCTIONALITY_ISSUES'
    
    # Buyer mistake
    elif any(word in text for word in ['mistake', 'accident', 'error']):
        return 'BUYER_MISTAKE'
    
    # Default
    else:
        return 'OTHER'

def analyze_categories(df: pd.DataFrame):
    """Analyze return categories"""
    # Category distribution
    st.session_state.category_analysis = df['Category'].value_counts().to_dict()
    
    # Quality metrics
    total_returns = len(df)
    quality_issues = len(df[df['Category'].isin(QUALITY_CATEGORIES)])
    
    st.session_state.total_returns = total_returns
    st.session_state.quality_issues = quality_issues
    st.session_state.quality_metrics = {
        'quality_rate': (quality_issues / total_returns * 100) if total_returns > 0 else 0,
        'top_quality_issue': df[df['Category'].isin(QUALITY_CATEGORIES)]['Category'].mode()[0] if quality_issues > 0 else 'None'
    }

def analyze_products(df: pd.DataFrame):
    """Analyze returns by product"""
    product_analysis = {}
    
    # Check if ASIN column exists
    if 'ASIN' not in df.columns:
        st.warning("ASIN column not found in data")
        st.session_state.product_analysis = {}
        return
    
    for asin in df['ASIN'].unique():
        if pd.notna(asin) and asin != '':
            asin_df = df[df['ASIN'] == asin]
            
            product_analysis[asin] = {
                'total_returns': len(asin_df),
                'quality_issues': len(asin_df[asin_df['Category'].isin(QUALITY_CATEGORIES)]),
                'categories': asin_df['Category'].value_counts().to_dict(),
                'sku': asin_df['SKU'].iloc[0] if 'SKU' in asin_df.columns and len(asin_df) > 0 else ''
            }
    
    st.session_state.product_analysis = product_analysis

def upload_section():
    """File upload section"""
    st.markdown("### 📤 Upload Return Data")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        uploaded_file = st.file_uploader(
            "Choose a file",
            type=['pdf', 'txt', 'csv'],
            help="Upload PDF from Seller Central or FBA Returns text file"
        )
    
    with col2:
        st.markdown("#### Supported Formats")
        st.markdown("""
        - 📄 **PDF**: Seller Central returns
        - 📝 **TXT**: FBA returns report
        - 📊 **CSV**: Custom format
        """)
    
    if uploaded_file:
        # Initialize analyzers
        initialize_analyzers()
        
        # Process file
        with st.spinner("Processing file..."):
            data = process_uploaded_file(uploaded_file)
            
            if data is not None and not data.empty:
                st.session_state.raw_data = data
                st.session_state.file_uploaded = True
                
                # Categorize returns
                categorized_data = categorize_returns(data)
                st.session_state.categorized_data = categorized_data
                
                # Analyze data
                analyze_categories(categorized_data)
                analyze_products(categorized_data)
                
                # Get AI insights if available
                if st.session_state.ai_analyzer:
                    with st.spinner("🤖 Generating AI insights..."):
                        try:
                            # Check if generate_insights method exists
                            if hasattr(st.session_state.ai_analyzer, 'generate_insights'):
                                st.session_state.ai_insights = st.session_state.ai_analyzer.generate_insights(
                                    st.session_state.category_analysis,
                                    st.session_state.product_analysis
                                )
                            else:
                                # Generate basic insights if method doesn't exist
                                st.session_state.ai_insights = generate_basic_insights(
                                    st.session_state.category_analysis,
                                    st.session_state.product_analysis
                                )
                        except Exception as e:
                            logger.warning(f"Could not generate AI insights: {e}")
                            st.session_state.ai_insights = None
                
                st.session_state.processing_complete = True
                st.session_state.analysis_ready = True
                
                st.success(f"✅ Successfully processed {len(data)} returns!")
                st.rerun()

def display_dashboard():
    """Display main dashboard"""
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
        color = COLORS['danger'] if quality_rate > 30 else COLORS['warning'] if quality_rate > 20 else COLORS['success']
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
        if st.session_state.total_injuries > 0:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Injury Cases</div>
                <div class="metric-value" style="color: {COLORS['critical']};">{st.session_state.total_injuries}</div>
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
        
        # Create bar chart
        st.bar_chart(category_df.set_index('Category'))
    
    with col2:
        # Category table
        st.markdown("#### Category Breakdown")
        for category, count in category_df.values:
            percentage = (count / st.session_state.total_returns * 100)
            badge_class = 'quality-issue' if category in QUALITY_CATEGORIES else 'other-issue'
            
            st.markdown(f"""
            <div style="margin: 0.5rem 0;">
                <span class="category-badge {badge_class}">{category}</span>
                <div style="margin-top: 0.25rem;">
                    <strong>{count}</strong> returns ({percentage:.1f}%)
                </div>
            </div>
            """, unsafe_allow_html=True)

def display_product_analysis():
    """Display product-level analysis"""
    st.markdown("### 📦 Product Analysis")
    
    if not st.session_state.product_analysis:
        st.info("No product data available")
        return
    
    # Sort products by total returns
    sorted_products = sorted(
        st.session_state.product_analysis.items(),
        key=lambda x: x[1]['total_returns'],
        reverse=True
    )
    
    # Display top 10 products
    st.markdown("#### Top 10 Products by Return Volume")
    
    for i, (asin, data) in enumerate(sorted_products[:10]):
        quality_rate = (data['quality_issues'] / data['total_returns'] * 100)
        color = COLORS['danger'] if quality_rate > 30 else COLORS['warning'] if quality_rate > 20 else COLORS['success']
        
        col1, col2, col3, col4 = st.columns([2, 1, 1, 1])
        
        with col1:
            st.markdown(f"**{asin}**")
            if data.get('sku'):
                st.caption(f"SKU: {data['sku']}")
        
        with col2:
            st.metric("Returns", data['total_returns'])
        
        with col3:
            st.metric("Quality Issues", data['quality_issues'])
        
        with col4:
            st.markdown(f"<div style='color: {color}; font-weight: bold;'>{quality_rate:.1f}%</div>", unsafe_allow_html=True)

def display_recommendations():
    """Display AI recommendations"""
    st.markdown("### 💡 AI Recommendations")
    
    if st.session_state.ai_insights:
        st.markdown(st.session_state.ai_insights)
    else:
        # Generate basic recommendations
        recommendations = []
        
        # Quality issue recommendations
        if st.session_state.quality_metrics.get('quality_rate', 0) > 30:
            recommendations.append("🚨 **Critical Quality Alert**: Over 30% of returns are quality-related. Immediate action required.")
        
        # Product-specific recommendations
        if st.session_state.product_analysis:
            high_risk_products = [
                asin for asin, data in st.session_state.product_analysis.items()
                if (data['quality_issues'] / data['total_returns'] * 100) > 40
            ]
            
            if high_risk_products:
                recommendations.append(f"⚠️ **High-Risk Products**: {len(high_risk_products)} products have >40% quality-related returns")
        
        # Display recommendations
        for rec in recommendations:
            st.markdown(rec)

def export_analysis():
    """Export analysis results"""
    st.markdown("### 📥 Export Options")
    
    if not st.session_state.categorized_data is None:
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # Export categorized data
            csv = st.session_state.categorized_data.to_csv(index=False)
            st.download_button(
                label="📊 Download Categorized Data",
                data=csv,
                file_name=f"categorized_returns_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
        
        with col2:
            # Export summary report
            summary = generate_summary_report()
            st.download_button(
                label="📄 Download Summary Report",
                data=summary,
                file_name=f"return_analysis_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                mime="text/plain"
            )
        
        with col3:
            # Export for Excel
            excel_buffer = create_excel_export()
            if excel_buffer:
                st.download_button(
                    label="📑 Download Excel Report",
                    data=excel_buffer,
                    file_name=f"return_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
            else:
                # Fallback to CSV with categories
                if st.session_state.categorized_data is not None:
                    csv = st.session_state.categorized_data.to_csv(index=False)
                    st.download_button(
                        label="📊 Download as CSV",
                        data=csv,
                        file_name=f"return_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )

def generate_summary_report() -> str:
    """Generate text summary report"""
    report = f"""
RETURN ANALYSIS SUMMARY REPORT
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

OVERVIEW
========
Total Returns: {st.session_state.total_returns:,}
Quality Issues: {st.session_state.quality_issues:,} ({st.session_state.quality_metrics.get('quality_rate', 0):.1f}%)
Products Affected: {len(st.session_state.product_analysis)}
Processing Time: {st.session_state.processing_time:.1f} seconds

CATEGORY BREAKDOWN
==================
"""
    
    for category, count in sorted(st.session_state.category_analysis.items(), key=lambda x: x[1], reverse=True):
        percentage = (count / st.session_state.total_returns * 100)
        report += f"{category}: {count} ({percentage:.1f}%)\n"
    
    report += f"""

TOP PRODUCTS BY RETURNS
=======================
"""
    
    sorted_products = sorted(
        st.session_state.product_analysis.items(),
        key=lambda x: x[1]['total_returns'],
        reverse=True
    )[:10]
    
    for asin, data in sorted_products:
        quality_rate = (data['quality_issues'] / data['total_returns'] * 100)
        report += f"\n{asin}:\n"
        report += f"  Total Returns: {data['total_returns']}\n"
        report += f"  Quality Issues: {data['quality_issues']} ({quality_rate:.1f}%)\n"
    
    if st.session_state.ai_insights:
        report += f"\n\nAI INSIGHTS\n===========\n{st.session_state.ai_insights}"
    
    return report

def create_excel_export():
    """Create Excel export with multiple sheets"""
    try:
        # Check if xlsxwriter is available
        try:
            import xlsxwriter
        except ImportError:
            st.warning("Excel export requires xlsxwriter. Please install: pip install xlsxwriter")
            return None
            
        from io import BytesIO
        
        output = BytesIO()
        workbook = xlsxwriter.Workbook(output)
        
        # Summary sheet
        summary_sheet = workbook.add_worksheet('Summary')
        summary_sheet.write(0, 0, 'Metric')
        summary_sheet.write(0, 1, 'Value')
        
        summary_data = [
            ['Total Returns', st.session_state.total_returns],
            ['Quality Issues', st.session_state.quality_issues],
            ['Quality Rate %', st.session_state.quality_metrics.get('quality_rate', 0)],
            ['Products Affected', len(st.session_state.product_analysis)],
            ['Processing Time (s)', st.session_state.processing_time]
        ]
        
        for i, (metric, value) in enumerate(summary_data, 1):
            summary_sheet.write(i, 0, metric)
            summary_sheet.write(i, 1, value)
        
        # Raw data sheet
        if st.session_state.categorized_data is not None:
            data_sheet = workbook.add_worksheet('Categorized Data')
            
            # Write headers
            for col, header in enumerate(st.session_state.categorized_data.columns):
                data_sheet.write(0, col, header)
            
            # Write data
            for row_idx, row in st.session_state.categorized_data.iterrows():
                for col_idx, value in enumerate(row):
                    data_sheet.write(row_idx + 1, col_idx, str(value))
        
        workbook.close()
        output.seek(0)
        
        return output.getvalue()
        
    except Exception as e:
        logger.error(f"Excel export error: {e}")
        st.warning(f"Excel export failed: {str(e)}. Using CSV export instead.")
        return None

def main():
    """Main application"""
    initialize_session_state()
    inject_professional_css()
    
    # Header with navigation
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
                try:
                    if hasattr(st.session_state.ai_analyzer, 'get_cost_summary'):
                        cost_summary = st.session_state.ai_analyzer.get_cost_summary()
                        st.metric("API Cost", f"${cost_summary.get('total_cost', 0):.4f}")
                        st.metric("API Calls", cost_summary.get('api_calls', 0))
                    else:
                        st.metric("API Cost", f"${st.session_state.api_cost:.4f}")
                except Exception as e:
                    logger.warning(f"Could not get cost summary: {e}")
                    st.metric("API Cost", f"${st.session_state.api_cost:.4f}")
        
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
        tabs = ["📊 Dashboard", "📈 Categories", "📦 Products", "💡 Recommendations", "📥 Export", "🔄 New Analysis"]
        
        # Add injury tab if injuries detected
        if st.session_state.total_injuries > 0:
            tabs.insert(1, "🚨 Injuries")
        
        tab_selection = st.tabs(tabs)
        
        # Dashboard
        with tab_selection[0]:
            display_dashboard()
        
        # Handle injury tab if present
        current_tab = 1
        if st.session_state.total_injuries > 0:
            with tab_selection[current_tab]:
                display_injury_analysis()
            current_tab += 1
        
        # Categories
        with tab_selection[current_tab]:
            display_category_analysis()
        current_tab += 1
        
        # Products
        with tab_selection[current_tab]:
            display_product_analysis()
        current_tab += 1
        
        # Recommendations
        with tab_selection[current_tab]:
            display_recommendations()
        current_tab += 1
        
        # Export
        with tab_selection[current_tab]:
            export_analysis()
        current_tab += 1
        
        # New Analysis
        with tab_selection[current_tab]:
            st.markdown("### 🔄 Start New Analysis")
            st.info("Click the button below to clear current data and start a new analysis.")
            
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                if st.button("🔄 Start New Analysis", type="primary", use_container_width=True):
                    # Reset state using the new function
                    reset_analysis_state()
                    st.rerun()

def display_injury_analysis():
    """Display injury-related returns analysis"""
    st.markdown("### 🚨 Injury-Related Returns Analysis")
    
    if st.session_state.categorized_data is None or 'Has_Injury' not in st.session_state.categorized_data.columns:
        st.info("No injury data available")
        return
    
    injury_df = st.session_state.categorized_data[st.session_state.categorized_data['Has_Injury'] == True]
    
    if injury_df.empty:
        st.success("✅ No injury-related returns detected")
        return
    
    # Injury metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card" style="border: 2px solid {COLORS['critical']};">
            <div class="metric-label" style="color: {COLORS['critical']};">Total Injury Cases</div>
            <div class="metric-value" style="color: {COLORS['critical']};">{len(injury_df)}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        injury_rate = (len(injury_df) / st.session_state.total_returns * 100)
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Injury Rate</div>
            <div class="metric-value" style="color: {COLORS['critical']};">{injury_rate:.2f}%</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        unique_products = injury_df['ASIN'].nunique()
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Products with Injuries</div>
            <div class="metric-value">{unique_products}</div>
        </div>
        """, unsafe_allow_html=True)
    
    # Injury details table
    st.markdown("#### Injury Case Details")
    
    # Prepare display dataframe
    display_df = injury_df[['Order ID', 'ASIN', 'SKU', 'Return Reason', 'Customer Comments']].copy()
    
    # Highlight injury keywords
    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True
    )
    
    # FDA MDR warning
    st.warning("""
    ⚠️ **FDA MDR Reporting Required**
    
    These returns contain potential injury information that may require FDA Medical Device Reporting (MDR).
    
    **Next Steps:**
    1. Review each case for severity
    2. Determine if MDR criteria are met
    3. File reports within 30 days if required
    4. Document investigation and corrective actions
    """)
    
    # Export injury report
    if st.button("📥 Export Injury Report for FDA MDR", type="primary"):
        injury_csv = injury_df.to_csv(index=False)
        st.download_button(
            label="Download Injury Report CSV",
            data=injury_csv,
            file_name=f"injury_returns_MDR_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )

if __name__ == "__main__":
    main()
