"""
PA9.py - Medical Device Return Analyzer
Fixed version with working AI support
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
from typing import Dict, List, Tuple, Optional, Any, Union
import logging
import traceback
import gc

# Configure page first
st.set_page_config(
    page_title="PriceAnalysis9000 - Medical Device Return Analyzer",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import modules with better error handling
modules_status = {}

# Try to import enhanced AI module
try:
    from enhanced_ai_analysis import EnhancedAIAnalyzer, AIProvider, FileProcessor
    AI_AVAILABLE = True
    modules_status['AI Analysis'] = "✅ Available"
    logger.info("AI module loaded successfully")
except ImportError as e:
    logger.error(f"Failed to import AI module: {e}")
    AI_AVAILABLE = False
    modules_status['AI Analysis'] = "❌ Not Available"
    
    # Define fallback FileProcessor
    class FileProcessor:
        @staticmethod
        def read_file(file, file_type):
            if 'pdf' in file_type.lower():
                st.error("PDF processing requires enhanced_ai_analysis module with pdfplumber")
                return pd.DataFrame()
            elif file.name.endswith(('.xlsx', '.xls')):
                return pd.read_excel(file)
            elif file.name.endswith('.txt') or file.name.endswith('.tsv'):
                # Check if tab-delimited
                file.seek(0)
                first_line = file.readline().decode('utf-8')
                file.seek(0)
                if '\t' in first_line:
                    return pd.read_csv(file, sep='\t')
                else:
                    return pd.read_csv(file)
            else:
                return pd.read_csv(file)
    
    class AIProvider:
        AUTO = "auto"
        PATTERN = "pattern"
        OPENAI = "openai"
        CLAUDE = "claude"

# Try to import other modules
try:
    from injury_detector import InjuryDetector
    INJURY_AVAILABLE = True
    modules_status['Injury Detection'] = "✅ Available"
except:
    INJURY_AVAILABLE = False
    modules_status['Injury Detection'] = "❌ Not Available"
    class InjuryDetector:
        def __init__(self, ai_analyzer=None):
            self.ai_analyzer = ai_analyzer
        
        def check_for_injury(self, text):
            injury_keywords = ['injury', 'hurt', 'pain', 'hospital', 'emergency', 'bleeding', 'fall', 'accident']
            return any(keyword in text.lower() for keyword in injury_keywords)

try:
    from pdf_analyzer import PDFAnalyzer
    PDF_AVAILABLE = True
    modules_status['PDF Analysis'] = "✅ Available"
except:
    PDF_AVAILABLE = False
    modules_status['PDF Analysis'] = "❌ Not Available"

try:
    from smart_column_mapper import SmartColumnMapper
    MAPPER_AVAILABLE = True
    modules_status['Smart Mapping'] = "✅ Available"
except:
    MAPPER_AVAILABLE = False
    modules_status['Smart Mapping'] = "❌ Not Available"

# Category definitions
QUALITY_CATEGORIES = [
    'QUALITY_DEFECTS',
    'FUNCTIONALITY_ISSUES',
    'COMPATIBILITY_ISSUES',
    'INJURY_RISK'
]

# All categories
ALL_CATEGORIES = [
    'QUALITY_DEFECTS',
    'FUNCTIONALITY_ISSUES', 
    'SIZE_FIT_ISSUES',
    'COMPATIBILITY_ISSUES',
    'WRONG_PRODUCT',
    'BUYER_MISTAKE',
    'NO_LONGER_NEEDED',
    'INJURY_RISK',
    'OTHER'
]

# Session state defaults
SESSION_DEFAULTS = {
    'uploaded_files': [],
    'combined_data': None,
    'processed_data': None,
    'analysis_complete': False,
    'ai_analyzer': None,
    'injury_detector': None,
    'pdf_analyzer': None,
    'column_mapper': None,
    'quality_metrics': {},
    'injury_report': {},
    'api_status': {},
    'processing_stats': {},
    'total_files': 0,
    'total_rows': 0,
    'api_provider': 'auto',
    'api_key_source': None,
    'ai_test_result': None
}

def initialize_session_state():
    """Initialize session state variables"""
    for key, value in SESSION_DEFAULTS.items():
        if key not in st.session_state:
            st.session_state[key] = value

def check_api_keys():
    """Check and validate API keys with better error handling"""
    api_status = {
        'openai': {'available': False, 'key_found': False, 'key_source': None},
        'anthropic': {'available': False, 'key_found': False, 'key_source': None}
    }
    
    # Check OpenAI
    openai_key = None
    key_source = None
    
    # Check different sources
    if hasattr(st, 'secrets'):
        if 'OPENAI_API_KEY' in st.secrets:
            openai_key = str(st.secrets['OPENAI_API_KEY']).strip()
            key_source = 'Streamlit secrets'
        elif 'openai_api_key' in st.secrets:
            openai_key = str(st.secrets['openai_api_key']).strip()
            key_source = 'Streamlit secrets'
    
    if not openai_key and os.getenv('OPENAI_API_KEY'):
        openai_key = os.getenv('OPENAI_API_KEY').strip()
        key_source = 'Environment variable'
    
    if openai_key and len(openai_key) > 10:  # Basic validation
        api_status['openai']['key_found'] = True
        api_status['openai']['key_source'] = key_source
        api_status['openai']['available'] = True
    
    # Check Anthropic
    anthropic_key = None
    key_source = None
    
    if hasattr(st, 'secrets'):
        if 'ANTHROPIC_API_KEY' in st.secrets:
            anthropic_key = str(st.secrets['ANTHROPIC_API_KEY']).strip()
            key_source = 'Streamlit secrets'
        elif 'anthropic_api_key' in st.secrets:
            anthropic_key = str(st.secrets['anthropic_api_key']).strip()
            key_source = 'Streamlit secrets'
    
    if not anthropic_key and os.getenv('ANTHROPIC_API_KEY'):
        anthropic_key = os.getenv('ANTHROPIC_API_KEY').strip()
        key_source = 'Environment variable'
    
    if anthropic_key and len(anthropic_key) > 10:  # Basic validation
        api_status['anthropic']['key_found'] = True
        api_status['anthropic']['key_source'] = key_source
        api_status['anthropic']['available'] = True
    
    st.session_state.api_status = api_status
    return api_status

def initialize_analyzers():
    """Initialize all analyzer components with better error handling"""
    # Initialize AI analyzer
    if AI_AVAILABLE and not st.session_state.ai_analyzer:
        try:
            provider = st.session_state.api_provider
            
            # Ensure we have the latest API key status
            if hasattr(st, 'secrets'):
                # Force reload of secrets if possible
                try:
                    import streamlit.runtime.secrets
                    if hasattr(streamlit.runtime.secrets, '_secrets'):
                        streamlit.runtime.secrets._secrets = None
                except:
                    pass
            
            # Create analyzer
            st.session_state.ai_analyzer = EnhancedAIAnalyzer(provider)
            
            # Check if it initialized properly
            if st.session_state.ai_analyzer.ai_available:
                logger.info(f"AI analyzer initialized successfully with {st.session_state.ai_analyzer.provider}")
                st.success(f"✅ AI initialized with {st.session_state.ai_analyzer.provider}")
            else:
                logger.info("AI analyzer in pattern matching mode")
                st.info("📊 Using pattern matching mode (no AI provider available)")
                
        except Exception as e:
            logger.error(f"Failed to initialize AI analyzer: {e}")
            st.error(f"AI initialization error: {str(e)}")
            # Create pattern-only analyzer
            try:
                st.session_state.ai_analyzer = EnhancedAIAnalyzer(AIProvider.PATTERN)
            except:
                pass
    
    # Initialize injury detector
    if INJURY_AVAILABLE and not st.session_state.injury_detector:
        try:
            st.session_state.injury_detector = InjuryDetector(st.session_state.ai_analyzer)
            logger.info("Injury detector initialized")
        except Exception as e:
            logger.error(f"Failed to initialize injury detector: {e}")
    elif not st.session_state.injury_detector:
        # Use fallback injury detector
        st.session_state.injury_detector = InjuryDetector()
    
    # Initialize PDF analyzer
    if PDF_AVAILABLE and not st.session_state.pdf_analyzer:
        try:
            st.session_state.pdf_analyzer = PDFAnalyzer()
            logger.info("PDF analyzer initialized")
        except Exception as e:
            logger.error(f"Failed to initialize PDF analyzer: {e}")
    
    # Initialize column mapper
    if MAPPER_AVAILABLE and not st.session_state.column_mapper:
        try:
            st.session_state.column_mapper = SmartColumnMapper(st.session_state.ai_analyzer)
            logger.info("Column mapper initialized")
        except Exception as e:
            logger.error(f"Failed to initialize column mapper: {e}")

def display_header():
    """Display application header"""
    col1, col2, col3 = st.columns([2, 2, 1])
    
    with col1:
        st.title("🏥 PriceAnalysis9000")
        st.markdown("**Medical Device Return Analyzer**")
        st.caption("Quality Management & Injury Detection System")
    
    with col2:
        # Display module status
        status_cols = st.columns(2)
        with status_cols[0]:
            for module, status in list(modules_status.items())[:2]:
                st.markdown(f"<small>{status} {module}</small>", unsafe_allow_html=True)
        with status_cols[1]:
            for module, status in list(modules_status.items())[2:]:
                st.markdown(f"<small>{status} {module}</small>", unsafe_allow_html=True)
    
    with col3:
        # API status indicator
        if st.session_state.ai_analyzer:
            if st.session_state.ai_analyzer.ai_available:
                provider = st.session_state.ai_analyzer.provider
                if provider == 'openai':
                    st.success("🟢 OpenAI Active")
                elif provider in ['claude', 'anthropic']:
                    st.success("🟢 Claude Active")
                else:
                    st.info("🔵 Pattern Mode")
            else:
                st.warning("🟡 Pattern Mode")
        else:
            st.info("⚪ Not Initialized")

def process_file(file) -> Optional[pd.DataFrame]:
    """Process a single file and return DataFrame"""
    try:
        file_type = file.type if hasattr(file, 'type') else 'unknown'
        file_name = file.name if hasattr(file, 'name') else 'unknown'
        
        # Use FileProcessor
        df = FileProcessor.read_file(file, file_type)
        
        # Add source file column
        if not df.empty:
            df['source_file'] = file_name
        
        return df
            
    except Exception as e:
        logger.error(f"Error processing file {file_name}: {e}")
        st.error(f"Error processing {file_name}: {str(e)}")
        return None

def detect_quality_issues(df: pd.DataFrame) -> Dict[str, Any]:
    """Detect quality issues and categorize returns"""
    quality_issues = {
        'total_returns': len(df),
        'quality_defects': 0,
        'injury_cases': 0,
        'categories': {cat: 0 for cat in ALL_CATEGORIES},
        'high_risk_products': [],
        'categorized_returns': []
    }
    
    # Find relevant columns - be more flexible
    reason_cols = ['reason', 'return_reason', 'Return Reason', 'issue', 'return-reason', 
                   'return reason', 'detailed-disposition', 'return_type']
    comment_cols = ['customer-comments', 'customer_comment', 'Customer Comments', 
                    'comments', 'buyer_comment', 'buyer-comment', 'buyer comment',
                    'customer comment', 'feedback', 'notes']
    asin_cols = ['asin', 'ASIN', 'product_asin', 'product-asin']
    order_cols = ['order-id', 'order_id', 'Order ID', 'order', 'order id', 'orderId']
    
    # Find columns by checking all variations
    reason_col = None
    comment_col = None
    asin_col = None
    order_col = None
    
    for col in df.columns:
        col_clean = col.lower().replace('-', '_').replace(' ', '_')
        
        if not reason_col:
            for rc in reason_cols:
                if rc.lower().replace('-', '_').replace(' ', '_') == col_clean:
                    reason_col = col
                    break
        
        if not comment_col:
            for cc in comment_cols:
                if cc.lower().replace('-', '_').replace(' ', '_') == col_clean:
                    comment_col = col
                    break
        
        if not asin_col:
            for ac in asin_cols:
                if ac.lower().replace('-', '_') == col_clean:
                    asin_col = col
                    break
        
        if not order_col:
            for oc in order_cols:
                if oc.lower().replace('-', '_').replace(' ', '_') == col_clean:
                    order_col = col
                    break
    
    # Log found columns
    logger.info(f"Found columns - Reason: {reason_col}, Comment: {comment_col}, ASIN: {asin_col}, Order: {order_col}")
    
    # Process returns
    if st.session_state.ai_analyzer and AI_AVAILABLE:
        # Use AI categorization
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for idx, row in df.iterrows():
            # Update progress
            if idx % 10 == 0:
                progress = (idx + 1) / len(df)
                progress_bar.progress(progress)
                status_text.text(f"Analyzing return {idx + 1} of {len(df)}...")
            
            # Get text data
            reason = str(row[reason_col]) if reason_col and pd.notna(row.get(reason_col, None)) else ''
            comment = str(row[comment_col]) if comment_col and pd.notna(row.get(comment_col, None)) else ''
            asin = str(row[asin_col]) if asin_col and pd.notna(row.get(asin_col, None)) else ''
            order_id = str(row[order_col]) if order_col and pd.notna(row.get(order_col, None)) else ''
            
            # Categorize
            result = st.session_state.ai_analyzer.categorize_return(
                complaint=comment,
                return_reason=reason,
                return_data={'asin': asin, 'order_id': order_id}
            )
            
            # Update counts
            category = result.get('category', 'OTHER')
            quality_issues['categories'][category] = quality_issues['categories'].get(category, 0) + 1
            
            if category in QUALITY_CATEGORIES:
                quality_issues['quality_defects'] += 1
            
            if result.get('has_injury', False):
                quality_issues['injury_cases'] += 1
            
            # Store categorized return
            quality_issues['categorized_returns'].append({
                'order_id': order_id,
                'asin': asin,
                'reason': reason,
                'comment': comment,
                'category': category,
                'has_injury': result.get('has_injury', False),
                'severity': result.get('severity', '')
            })
        
        progress_bar.empty()
        status_text.empty()
        
    else:
        # Fallback to pattern matching
        for idx, row in df.iterrows():
            # Combine all text from the row
            text_parts = []
            
            if reason_col and pd.notna(row.get(reason_col)):
                text_parts.append(str(row[reason_col]))
            
            if comment_col and pd.notna(row.get(comment_col)):
                text_parts.append(str(row[comment_col]))
            
            # If no specific columns found, use all text
            if not text_parts:
                text_parts = [str(val) for val in row.values if pd.notna(val)]
            
            text = ' '.join(text_parts).lower()
            
            # Check categories
            category = 'OTHER'
            
            # Check for injuries first (highest priority)
            if any(word in text for word in ['injury', 'hurt', 'pain', 'hospital', 'emergency', 
                                              'bleeding', 'fall', 'accident', 'burn']):
                category = 'INJURY_RISK'
                quality_issues['injury_cases'] += 1
            
            elif any(word in text for word in ['defect', 'broken', 'damaged', 'quality', 
                                                'malfunction', 'not working', 'faulty']):
                category = 'QUALITY_DEFECTS'
                quality_issues['quality_defects'] += 1
            
            elif any(word in text for word in ['too small', 'too large', 'size', 'fit', 
                                                "doesn't fit", 'wrong size']):
                category = 'SIZE_FIT_ISSUES'
            
            elif any(word in text for word in ['wrong', 'incorrect', 'different', 
                                                'not as described']):
                category = 'WRONG_PRODUCT'
            
            elif any(word in text for word in ['mistake', 'accident', 'error', 
                                                'accidentally', 'my fault']):
                category = 'BUYER_MISTAKE'
            
            elif any(word in text for word in ['compatible', 'compatibility', 
                                                'not compatible', "doesn't fit toilet"]):
                category = 'COMPATIBILITY_ISSUES'
                if any(word in text for word in ['quality', 'defect']):
                    quality_issues['quality_defects'] += 1
            
            elif any(word in text for word in ['uncomfortable', 'hard to use', 
                                                'difficult', 'not comfortable']):
                category = 'FUNCTIONALITY_ISSUES'
                quality_issues['quality_defects'] += 1
            
            elif any(word in text for word in ['no longer needed', 'changed mind', 
                                                "don't need"]):
                category = 'NO_LONGER_NEEDED'
            
            quality_issues['categories'][category] = quality_issues['categories'].get(category, 0) + 1
    
    # Calculate rates
    if quality_issues['total_returns'] > 0:
        quality_issues['quality_rate'] = (quality_issues['quality_defects'] / quality_issues['total_returns']) * 100
        quality_issues['injury_rate'] = (quality_issues['injury_cases'] / quality_issues['total_returns']) * 100
    else:
        quality_issues['quality_rate'] = 0
        quality_issues['injury_rate'] = 0
    
    return quality_issues

def generate_structured_export(df: pd.DataFrame, analysis: Dict[str, Any]) -> pd.DataFrame:
    """Generate structured export with analysis results"""
    export_df = df.copy()
    
    # Add analysis columns
    export_df['Analysis_Date'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    export_df['Quality_Issue'] = False
    export_df['Injury_Case'] = False
    export_df['Category'] = 'UNCATEGORIZED'
    export_df['Risk_Level'] = 'LOW'
    
    # If we have categorized returns, use them
    if 'categorized_returns' in analysis and analysis['categorized_returns']:
        # Create a mapping dictionary
        categorization_map = {}
        for cat_return in analysis['categorized_returns']:
            key = f"{cat_return.get('order_id', '')}_{cat_return.get('asin', '')}"
            categorization_map[key] = cat_return
        
        # Apply categorization
        for idx, row in export_df.iterrows():
            # Try to find matching categorization
            order_id = ''
            asin = ''
            
            # Look for order ID in various column names
            for col in ['order-id', 'order_id', 'Order ID']:
                if col in row:
                    order_id = str(row.get(col, ''))
                    break
            
            # Look for ASIN
            for col in ['asin', 'ASIN']:
                if col in row:
                    asin = str(row.get(col, ''))
                    break
            
            key = f"{order_id}_{asin}"
            
            if key in categorization_map:
                cat_data = categorization_map[key]
                export_df.at[idx, 'Category'] = cat_data['category']
                export_df.at[idx, 'Injury_Case'] = cat_data['has_injury']
                
                # Set quality issue flag
                if cat_data['category'] in QUALITY_CATEGORIES:
                    export_df.at[idx, 'Quality_Issue'] = True
                
                # Set risk level
                if cat_data['has_injury']:
                    severity = cat_data.get('severity', 'LOW')
                    export_df.at[idx, 'Risk_Level'] = severity
                elif cat_data['category'] in QUALITY_CATEGORIES:
                    export_df.at[idx, 'Risk_Level'] = 'MEDIUM'
    
    return export_df

def main():
    """Main application"""
    initialize_session_state()
    
    # Check API keys on startup
    check_api_keys()
    
    # Display header
    display_header()
    
    # Sidebar configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # API Provider section
        st.subheader("🤖 AI Provider")
        
        # Show key status
        col1, col2 = st.columns(2)
        with col1:
            if st.session_state.api_status.get('openai', {}).get('key_found'):
                st.success("✅ OpenAI Key")
                st.caption(f"Source: {st.session_state.api_status['openai'].get('key_source', 'Unknown')}")
            else:
                st.error("❌ OpenAI Key")
        
        with col2:
            if st.session_state.api_status.get('anthropic', {}).get('key_found'):
                st.success("✅ Claude Key")
                st.caption(f"Source: {st.session_state.api_status['anthropic'].get('key_source', 'Unknown')}")
            else:
                st.error("❌ Claude Key")
        
        # Provider selection
        providers = ['auto', 'pattern']
        
        if st.session_state.api_status.get('openai', {}).get('key_found'):
            providers.insert(1, 'openai')
        
        if st.session_state.api_status.get('anthropic', {}).get('key_found'):
            providers.insert(1, 'claude')
        
        st.session_state.api_provider = st.selectbox(
            "Select AI Provider",
            providers,
            help="Auto selects the best available provider"
        )
        
        # Initialize analyzers button
        if st.button("🚀 Initialize AI", type="primary", use_container_width=True):
            with st.spinner("Initializing AI components..."):
                initialize_analyzers()
                
                if st.session_state.ai_analyzer:
                    # Show which mode we're in
                    if st.session_state.ai_analyzer.ai_available:
                        st.info(f"Using {st.session_state.ai_analyzer.provider} AI")
                    else:
                        st.info("Using pattern matching mode")
                else:
                    st.error("❌ Failed to initialize AI")
        
        # Test AI button
        if st.button("🧪 Test AI Connection"):
            if not st.session_state.ai_analyzer:
                st.warning("Please initialize AI first")
            else:
                with st.spinner("Testing AI..."):
                    test_result = st.session_state.ai_analyzer.test_ai_connection()
                    st.session_state.ai_test_result = test_result
                    
                    if 'successful' in test_result.get('status', ''):
                        st.success(f"✅ {test_result['status']}")
                        st.info(f"Provider: {test_result.get('provider', 'Unknown')}")
                        st.info(f"Model: {test_result.get('model', 'Unknown')}")
                        if test_result.get('test_category'):
                            st.info(f"Test category: {test_result['test_category']}")
                    else:
                        st.warning(f"⚠️ {test_result['status']}")
        
        st.divider()
        
        # Processing options
        st.subheader("📊 Processing Options")
        
        enable_injury_detection = st.checkbox("Enable Injury Detection", value=True)
        enable_quality_analysis = st.checkbox("Enable Quality Analysis", value=True)
        enable_ai_categorization = st.checkbox("Enable AI Categorization", value=True)
        
        st.divider()
        
        # Help section
        with st.expander("📖 How to Use"):
            st.markdown("""
            1. **Initialize AI** - Click the button above
            2. **Upload Files** - PDF, Excel, CSV, or TXT
            3. **Analyze** - Get categorized returns
            4. **Export** - Download results
            
            **File Types:**
            - PDF: Amazon Seller Central exports
            - TXT: FBA return reports
            - Excel/CSV: Custom formats
            """)
        
        # API Key help
        with st.expander("🔑 API Key Setup"):
            st.markdown("""
            **Option 1: Streamlit Secrets**
            ```toml
            # .streamlit/secrets.toml
            OPENAI_API_KEY = "sk-..."
            ANTHROPIC_API_KEY = "sk-ant-..."
            ```
            
            **Option 2: Environment Variables**
            ```bash
            export OPENAI_API_KEY="sk-..."
            export ANTHROPIC_API_KEY="sk-ant-..."
            ```
            
            **Get API Keys:**
            - [OpenAI](https://platform.openai.com/api-keys)
            - [Anthropic](https://console.anthropic.com/)
            """)
    
    # Main content area
    st.header("📤 Upload Return Data")
    
    # Initialize AI if not done
    if not st.session_state.ai_analyzer and AI_AVAILABLE:
        st.info("👆 Please initialize AI in the sidebar first for best results")
    
    # File uploader
    uploaded_files = st.file_uploader(
        "Choose files (PDF, Excel, CSV, TXT)",
        type=['pdf', 'xlsx', 'xls', 'csv', 'txt', 'tsv'],
        accept_multiple_files=True,
        help="Upload Amazon return data files. Multiple files supported."
    )
    
    if uploaded_files:
        st.session_state.uploaded_files = uploaded_files
        st.session_state.total_files = len(uploaded_files)
        
        # Display file info
        total_size = sum(file.size for file in uploaded_files) / (1024 * 1024)  # MB
        st.info(f"📁 {len(uploaded_files)} file(s) uploaded | Total size: {total_size:.1f} MB")
        
        # Process button
        if st.button("🔍 Analyze Returns", type="primary", use_container_width=True):
            # Initialize analyzers if needed
            if not st.session_state.ai_analyzer:
                with st.spinner("Initializing AI..."):
                    initialize_analyzers()
            
            # Process files
            all_dataframes = []
            processing_errors = []
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            for idx, file in enumerate(uploaded_files):
                status_text.text(f"Processing {file.name}...")
                progress_bar.progress((idx + 1) / len(uploaded_files))
                
                df = process_file(file)
                if df is not None and not df.empty:
                    all_dataframes.append(df)
                else:
                    processing_errors.append(file.name)
            
            progress_bar.empty()
            status_text.empty()
            
            # Combine all dataframes
            if all_dataframes:
                combined_df = pd.concat(all_dataframes, ignore_index=True)
                st.session_state.combined_data = combined_df
                st.session_state.total_rows = len(combined_df)
                
                # Show data preview
                with st.expander("📊 Data Preview", expanded=False):
                    st.dataframe(combined_df.head(10))
                    st.caption(f"Showing first 10 of {len(combined_df)} rows")
                
                # Analyze data
                with st.spinner("🤖 Analyzing returns for quality issues and injuries..."):
                    analysis_results = detect_quality_issues(combined_df)
                    st.session_state.quality_metrics = analysis_results
                    
                    # Generate structured export
                    export_df = generate_structured_export(combined_df, analysis_results)
                    st.session_state.processed_data = export_df
                    st.session_state.analysis_complete = True
                
                # Show results
                st.success(f"✅ Analysis complete! Processed {len(combined_df):,} returns from {len(all_dataframes)} files")
                
                if processing_errors:
                    st.warning(f"⚠️ Failed to process: {', '.join(processing_errors)}")
                
                # Display metrics
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Total Returns", f"{analysis_results['total_returns']:,}")
                
                with col2:
                    st.metric(
                        "Quality Defects",
                        f"{analysis_results['quality_defects']:,}",
                        f"{analysis_results['quality_rate']:.1f}%",
                        delta_color="inverse"
                    )
                
                with col3:
                    injury_color = "🔴" if analysis_results['injury_cases'] > 0 else "🟢"
                    st.metric(
                        f"{injury_color} Injury Cases",
                        f"{analysis_results['injury_cases']:,}",
                        f"{analysis_results['injury_rate']:.1f}%",
                        delta_color="inverse"
                    )
                
                with col4:
                    high_risk = len(export_df[export_df['Risk_Level'].isin(['HIGH', 'CRITICAL'])])
                    st.metric("High Risk Items", f"{high_risk:,}")
                
                # Show category breakdown
                st.subheader("📊 Return Categories")
                
                # Create two columns for the breakdown
                cat_col1, cat_col2 = st.columns(2)
                
                with cat_col1:
                    # Top categories
                    st.markdown("**Top Categories:**")
                    sorted_cats = sorted(analysis_results['categories'].items(), 
                                       key=lambda x: x[1], reverse=True)
                    
                    for cat, count in sorted_cats[:5]:
                        if count > 0:
                            pct = (count / analysis_results['total_returns'] * 100)
                            st.markdown(f"- **{cat}**: {count} ({pct:.1f}%)")
                
                with cat_col2:
                    # Quality categories specifically
                    st.markdown("**Quality Issues:**")
                    for cat in QUALITY_CATEGORIES:
                        count = analysis_results['categories'].get(cat, 0)
                        if count > 0:
                            pct = (count / analysis_results['total_returns'] * 100)
                            st.markdown(f"- **{cat}**: {count} ({pct:.1f}%)")
                
                # Show injury alert if needed
                if analysis_results['injury_cases'] > 0:
                    st.error(f"""
                    🚨 **INJURY CASES DETECTED**
                    
                    Found {analysis_results['injury_cases']} potential injury cases requiring immediate review.
                    These may require FDA MDR reporting or other regulatory actions.
                    """)
                
                # Export options
                st.subheader("💾 Export Options")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    # CSV export
                    csv = export_df.to_csv(index=False)
                    st.download_button(
                        "📥 Download Analysis (CSV)",
                        data=csv,
                        file_name=f"return_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
                
                with col2:
                    # Excel export
                    try:
                        buffer = io.BytesIO()
                        with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
                            # Main data
                            export_df.to_excel(writer, sheet_name='Return Analysis', index=False)
                            
                            # Category summary
                            cat_df = pd.DataFrame(
                                [(cat, count, f"{(count/analysis_results['total_returns']*100):.1f}%") 
                                 for cat, count in sorted_cats if count > 0],
                                columns=['Category', 'Count', 'Percentage']
                            )
                            cat_df.to_excel(writer, sheet_name='Category Summary', index=False)
                        
                        buffer.seek(0)
                        st.download_button(
                            "📥 Download Analysis (Excel)",
                            data=buffer,
                            file_name=f"return_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                            use_container_width=True
                        )
                    except Exception as e:
                        st.error(f"Excel export failed: {str(e)}")
                
                with col3:
                    # Summary report
                    summary = f"""RETURN ANALYSIS REPORT
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

SUMMARY
=======
Total Returns: {analysis_results['total_returns']:,}
Quality Defects: {analysis_results['quality_defects']:,} ({analysis_results['quality_rate']:.1f}%)
Injury Cases: {analysis_results['injury_cases']:,} ({analysis_results['injury_rate']:.1f}%)

CATEGORIES
==========
"""
                    for cat, count in sorted_cats:
                        if count > 0:
                            pct = (count / analysis_results['total_returns'] * 100)
                            summary += f"{cat}: {count} ({pct:.1f}%)\n"
                    
                    st.download_button(
                        "📥 Download Summary (TXT)",
                        data=summary,
                        file_name=f"return_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                        mime="text/plain",
                        use_container_width=True
                    )
            
            else:
                st.error("❌ No files could be processed successfully")
    
    else:
        # Show instructions
        st.info("""
        👆 **Upload your return data files to begin analysis**
        
        **Supported file types:**
        - 📄 **PDF**: Amazon Seller Central return exports
        - 📝 **TXT**: FBA return reports (tab-delimited)
        - 📊 **Excel/CSV**: Custom return data formats
        
        **This tool will:**
        - 🔍 Automatically categorize returns
        - 🚨 Detect potential injury cases
        - 📊 Identify quality defects
        - 📈 Provide actionable insights
        - 💾 Export categorized data
        """)
        
        # Show sample FBA format
        with st.expander("📋 Sample FBA Return Format"):
            st.code("""
return-date	order-id	sku	asin	fnsku	product-name	quantity	fulfillment-center-id	detailed-disposition	reason	status	license-plate-number	customer-comments
2025-06-05T15:31:40+00:00	114-2962899-5762619	LVA1004-UPC	B00TZ73MUY	B00TZ73MUY	Vive Alternating Air Pressure Mattress Pad	1	LEX1	CUSTOMER_DAMAGED	NOT_COMPATIBLE	IMMEDIATE_DONATION	LPNNC5N165ZVR	
            """)

if __name__ == "__main__":
    main()
