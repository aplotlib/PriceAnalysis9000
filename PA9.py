"""
Amazon Returns Quality Analyzer - Enhanced Version
Version: 7.0 - AI-Powered Medical Device Safety Focus
Critical: Quality Manager's Tool for Return Analysis
"""

import streamlit as st
import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
import io
from typing import Dict, List, Any, Optional, Tuple, Union
import re
import json
from collections import Counter, defaultdict
import plotly.express as px
import plotly.graph_objects as go
import time
import os
import traceback

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configure page
st.set_page_config(
    page_title="Quality Analysis Tool - Amazon Returns",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Import modules with error handling
try:
    from enhanced_ai_analysis import (
        EnhancedAIAnalyzer, AIProvider, MEDICAL_DEVICE_CATEGORIES, 
        FBA_REASON_MAP, detect_injury_severity
    )
    AI_AVAILABLE = True
except ImportError:
    AI_AVAILABLE = False
    logger.error("enhanced_ai_analysis not available - AI features disabled")
    # Define categories directly
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

try:
    from smart_column_mapper import SmartColumnMapper
    MAPPER_AVAILABLE = True
except ImportError:
    MAPPER_AVAILABLE = False
    logger.warning("smart_column_mapper not available")

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

# Professional color scheme
COLORS = {
    'primary': '#1E88E5',
    'secondary': '#FFC107',
    'success': '#4CAF50',
    'danger': '#F44336',
    'warning': '#FF9800',
    'info': '#00BCD4',
    'critical': '#D32F2F',
    'dark': '#212529',
    'light': '#F8F9FA'
}

# Session state initialization
def init_session_state():
    """Initialize all session state variables"""
    defaults = {
        # Data storage
        'raw_data': [],
        'processed_data': None,
        'categorized_data': None,
        'sales_data': None,
        'uploaded_files': [],
        
        # Analysis results
        'category_analysis': {},
        'product_analysis': {},
        'quality_metrics': {},
        'injury_analysis': None,
        'ai_insights': None,
        'return_rate_analysis': {},
        
        # Processing state
        'processing_complete': False,
        'files_uploaded': False,
        'analysis_ready': False,
        'ai_available': AI_AVAILABLE,
        
        # AI components
        'ai_analyzer': None,
        'pdf_analyzer': None,
        'injury_detector': None,
        'file_detector': None,
        'column_mapper': None,
        
        # Tracking
        'total_returns': 0,
        'total_sales': 0,
        'quality_issues': 0,
        'processing_time': 0.0,
        'api_cost': 0.0,
        'total_injuries': 0,
        'unmapped_columns': [],
        'ai_failures': []
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

def display_header():
    """Display professional header"""
    st.markdown("""
    <div style="background: linear-gradient(135deg, #1E88E5 0%, #00BCD4 100%); 
                padding: 2rem; border-radius: 10px; margin-bottom: 2rem;">
        <h1 style="color: white; margin: 0; font-size: 2.5rem;">
            🏥 Quality Analysis Tool
        </h1>
        <p style="color: white; opacity: 0.9; margin: 0.5rem 0;">
            Medical Device Return Analysis & Quality Management System
        </p>
        <p style="color: white; opacity: 0.8; font-size: 0.9rem;">
            Focus: Amazon Returns Categorization | Injury Detection | Quality Insights
        </p>
    </div>
    """, unsafe_allow_html=True)

def display_ai_status():
    """Display AI availability status"""
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col1:
        if st.session_state.ai_available:
            st.success("✅ AI Analysis Available")
        else:
            st.error("❌ AI Analysis Not Available")
            st.info("System will show manual mapping options")
    
    with col2:
        st.info(f"📊 Files Uploaded: {len(st.session_state.uploaded_files)}")
    
    with col3:
        if st.session_state.total_returns > 0:
            st.metric("Total Returns", st.session_state.total_returns)

def file_uploader_section():
    """Enhanced file upload section with multi-file support"""
    st.markdown("### 📁 Upload Return Data Files")
    st.markdown("Supports: PDF, Excel, CSV, TSV, TXT files from Amazon Seller Central")
    
    uploaded_files = st.file_uploader(
        "Choose files",
        type=['pdf', 'xlsx', 'xls', 'csv', 'tsv', 'txt'],
        accept_multiple_files=True,
        help="Upload multiple files at once. Supports Amazon PDF returns, FBA reports, and spreadsheets."
    )
    
    # Sales data upload (optional)
    st.markdown("### 📈 Upload Sales Data (Optional)")
    col1, col2 = st.columns([2, 1])
    
    with col1:
        sales_file = st.file_uploader(
            "Upload sales data for return rate calculation",
            type=['xlsx', 'csv', 'tsv'],
            help="Upload sales data to calculate return rates by product"
        )
    
    with col2:
        st.markdown("#### Or Enter Manually")
        manual_sales = st.number_input(
            "Total Units Sold",
            min_value=0,
            value=0,
            help="Enter total sales for return rate calculation"
        )
        if manual_sales > 0:
            st.session_state.total_sales = manual_sales
    
    if uploaded_files:
        st.session_state.uploaded_files = uploaded_files
        st.session_state.files_uploaded = True
        
        # Process sales file if uploaded
        if sales_file:
            process_sales_file(sales_file)
        
        return True
    
    return False

def process_sales_file(sales_file):
    """Process sales data file"""
    try:
        if sales_file.name.endswith('.csv'):
            df = pd.read_csv(sales_file)
        elif sales_file.name.endswith('.tsv'):
            df = pd.read_csv(sales_file, sep='\\t')
        else:
            df = pd.read_excel(sales_file)
        
        st.session_state.sales_data = df
        st.success(f"✅ Sales data loaded: {len(df)} records")
        
        # Try to auto-detect sales quantity column
        quantity_cols = [col for col in df.columns if 'quantity' in col.lower() or 'sales' in col.lower()]
        if quantity_cols:
            total_sales = df[quantity_cols[0]].sum()
            st.session_state.total_sales = total_sales
            st.info(f"Total sales detected: {total_sales:,} units")
            
    except Exception as e:
        st.error(f"Error processing sales file: {str(e)}")

def process_files():
    """Process all uploaded files"""
    if not st.session_state.uploaded_files:
        return
    
    all_returns = []
    processing_errors = []
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for idx, file in enumerate(st.session_state.uploaded_files):
        status_text.text(f"Processing {file.name}...")
        
        try:
            # Read file content
            file_content = file.read()
            file.seek(0)  # Reset for potential re-reading
            
            # Detect file type and process accordingly
            if file.name.endswith('.pdf'):
                if PDF_AVAILABLE:
                    returns = process_pdf_file(file_content, file.name)
                else:
                    st.error(f"PDF processing not available for {file.name}")
                    st.info("Please install: pip install pdfplumber PyPDF2")
                    continue
            else:
                # Process structured files (CSV, Excel, etc.)
                returns = process_structured_file(file, file.name)
            
            if returns:
                all_returns.extend(returns)
                st.success(f"✅ Processed {file.name}: {len(returns)} returns found")
            else:
                st.warning(f"⚠️ No returns found in {file.name}")
                
        except Exception as e:
            error_msg = f"Error processing {file.name}: {str(e)}"
            processing_errors.append(error_msg)
            st.error(error_msg)
            logger.error(f"File processing error: {e}", exc_info=True)
        
        progress_bar.progress((idx + 1) / len(st.session_state.uploaded_files))
    
    progress_bar.empty()
    status_text.empty()
    
    # Store results
    if all_returns:
        st.session_state.raw_data = all_returns
        st.session_state.total_returns = len(all_returns)
        
        # Categorize returns
        categorize_returns(all_returns)
        
        st.success(f"✅ Total returns processed: {len(all_returns)}")
    else:
        st.error("No returns found in any uploaded files")
    
    if processing_errors:
        with st.expander("Processing Errors"):
            for error in processing_errors:
                st.error(error)

def process_pdf_file(content: bytes, filename: str) -> List[Dict]:
    """Process PDF file and extract returns"""
    if not PDF_AVAILABLE or not st.session_state.pdf_analyzer:
        st.error("PDF analyzer not initialized")
        return []
    
    try:
        result = st.session_state.pdf_analyzer.extract_returns_from_pdf(content, filename)
        
        if 'error' in result:
            st.error(f"PDF extraction error: {result['error']}")
            return []
        
        returns = result.get('returns', [])
        
        # Standardize the data format
        standardized_returns = []
        for ret in returns:
            standardized_returns.append({
                'order_id': ret.get('order_id', ''),
                'asin': ret.get('asin', ''),
                'sku': ret.get('sku', ''),
                'return_date': ret.get('return_date', ''),
                'return_reason': ret.get('return_reason', ''),
                'buyer_comment': ret.get('customer_comment', ret.get('buyer_comment', '')),
                'source_file': filename,
                'file_type': 'PDF'
            })
        
        return standardized_returns
        
    except Exception as e:
        st.error(f"Error processing PDF: {str(e)}")
        return []

def process_structured_file(file, filename: str) -> List[Dict]:
    """Process CSV, Excel, TSV files"""
    try:
        # Read file into dataframe
        if filename.endswith('.csv'):
            df = pd.read_csv(file)
        elif filename.endswith('.tsv') or filename.endswith('.txt'):
            # Try tab-separated first
            df = pd.read_csv(file, sep='\\t')
            # Check if it's actually comma-separated
            if len(df.columns) == 1 and ',' in str(df.columns[0]):
                file.seek(0)
                df = pd.read_csv(file)
        else:  # Excel
            df = pd.read_excel(file)
        
        # Clean column names
        df.columns = df.columns.str.strip()
        
        # Use AI column mapper if available
        if MAPPER_AVAILABLE and st.session_state.column_mapper:
            return process_with_ai_mapper(df, filename)
        else:
            return process_with_manual_mapping(df, filename)
            
    except Exception as e:
        st.error(f"Error reading {filename}: {str(e)}")
        return []

def process_with_ai_mapper(df: pd.DataFrame, filename: str) -> List[Dict]:
    """Process dataframe using AI column mapping"""
    st.info("🤖 Using AI to detect column mappings...")
    
    try:
        # Get AI mapping suggestions
        column_mapping = st.session_state.column_mapper.detect_columns(df)
        
        # Validate mapping
        validation = st.session_state.column_mapper.validate_mapping(df, column_mapping)
        
        if not validation['is_valid']:
            st.warning("AI couldn't map all required columns")
            st.info("Unmapped columns: " + ", ".join(validation['missing_required']))
            
            # Show what was detected
            if column_mapping:
                st.write("Detected mappings:")
                for col_type, col_name in column_mapping.items():
                    st.write(f"- {col_type}: {col_name}")
            
            # Allow manual override
            return process_with_manual_mapping(df, filename, column_mapping)
        
        # Apply mapping
        mapped_df = st.session_state.column_mapper.map_dataframe(df, column_mapping)
        
        # Convert to return format
        returns = dataframe_to_returns(mapped_df, filename)
        
        return returns
        
    except Exception as e:
        st.error(f"AI mapping failed: {str(e)}")
        st.session_state.ai_failures.append(f"Column mapping failed for {filename}")
        return process_with_manual_mapping(df, filename)

def process_with_manual_mapping(df: pd.DataFrame, filename: str, 
                               pre_mapping: Dict = None) -> List[Dict]:
    """Manual column mapping interface"""
    st.markdown("### 📊 Manual Column Mapping")
    st.write(f"File: {filename}")
    st.write(f"Rows: {len(df)}, Columns: {len(df.columns)}")
    
    # Show sample data
    with st.expander("View Sample Data"):
        st.dataframe(df.head(10))
    
    # Column mapping interface
    st.markdown("#### Map Columns to Return Fields")
    
    required_fields = {
        'order_id': 'Order ID',
        'asin': 'ASIN',
        'sku': 'SKU', 
        'return_date': 'Return Date',
        'return_reason': 'Return Reason',
        'buyer_comment': 'Customer Comment'
    }
    
    column_mapping = pre_mapping or {}
    
    col1, col2 = st.columns(2)
    
    for idx, (field, label) in enumerate(required_fields.items()):
        if idx % 2 == 0:
            with col1:
                default = column_mapping.get(field, 'Not mapped')
                options = ['Not mapped'] + list(df.columns)
                
                selected = st.selectbox(
                    f"{label}:",
                    options,
                    index=options.index(default) if default in options else 0,
                    key=f"map_{field}_{filename}"
                )
                
                if selected != 'Not mapped':
                    column_mapping[field] = selected
        else:
            with col2:
                default = column_mapping.get(field, 'Not mapped')
                options = ['Not mapped'] + list(df.columns)
                
                selected = st.selectbox(
                    f"{label}:",
                    options,
                    index=options.index(default) if default in options else 0,
                    key=f"map_{field}_{filename}"
                )
                
                if selected != 'Not mapped':
                    column_mapping[field] = selected
    
    # Check for unmapped columns
    unmapped = [col for col in df.columns if col not in column_mapping.values()]
    if unmapped:
        st.session_state.unmapped_columns.extend(unmapped)
        with st.expander(f"⚠️ Unmapped columns ({len(unmapped)})"):
            st.write(unmapped)
    
    # Process with mapping
    if st.button(f"Process {filename}", key=f"process_{filename}"):
        returns = []
        
        for idx, row in df.iterrows():
            return_item = {
                'source_file': filename,
                'file_type': filename.split('.')[-1].upper()
            }
            
            for field, column in column_mapping.items():
                if column in df.columns:
                    return_item[field] = row.get(column, '')
            
            # Ensure required fields exist
            for field in required_fields:
                if field not in return_item:
                    return_item[field] = ''
            
            returns.append(return_item)
        
        return returns
    
    return []

def dataframe_to_returns(df: pd.DataFrame, filename: str) -> List[Dict]:
    """Convert mapped dataframe to returns format"""
    returns = []
    
    for idx, row in df.iterrows():
        return_item = {
            'order_id': str(row.get('order_id', '')),
            'asin': str(row.get('asin', '')),
            'sku': str(row.get('sku', '')),
            'return_date': str(row.get('return_date', '')),
            'return_reason': str(row.get('return_reason', '')),
            'buyer_comment': str(row.get('buyer_comment', row.get('customer_comment', ''))),
            'source_file': filename,
            'file_type': filename.split('.')[-1].upper()
        }
        
        returns.append(return_item)
    
    return returns

def categorize_returns(returns: List[Dict]):
    """Categorize all returns using AI or show failure message"""
    st.markdown("### 🏷️ Categorizing Returns")
    
    categorized = []
    ai_failures = 0
    
    progress_bar = st.progress(0)
    
    for idx, return_item in enumerate(returns):
        if AI_AVAILABLE and st.session_state.ai_analyzer:
            try:
                # Use AI categorization
                category, confidence, severity, language = st.session_state.ai_analyzer.categorize_return(
                    complaint=return_item.get('buyer_comment', ''),
                    return_reason=return_item.get('return_reason', ''),
                    fba_reason=return_item.get('fba_reason', '')
                )
                
                return_item['category'] = category
                return_item['confidence'] = confidence
                return_item['severity'] = severity
                return_item['categorization_method'] = 'AI'
                
            except Exception as e:
                ai_failures += 1
                return_item['category'] = 'UNCATEGORIZED - AI FAILED'
                return_item['confidence'] = 0.0
                return_item['severity'] = 'unknown'
                return_item['categorization_method'] = 'FAILED'
                return_item['error'] = str(e)
                
        else:
            # No AI available
            return_item['category'] = 'UNCATEGORIZED - NO AI AVAILABLE'
            return_item['confidence'] = 0.0
            return_item['severity'] = 'unknown'
            return_item['categorization_method'] = 'NONE'
        
        categorized.append(return_item)
        progress_bar.progress((idx + 1) / len(returns))
    
    progress_bar.empty()
    
    # Store results
    st.session_state.categorized_data = pd.DataFrame(categorized)
    st.session_state.processing_complete = True
    
    # Show results summary
    if ai_failures > 0:
        st.error(f"❌ AI categorization failed for {ai_failures} returns")
        st.info("These returns are marked as 'UNCATEGORIZED - AI FAILED'")
    
    if not AI_AVAILABLE:
        st.error("❌ AI analysis is not available. All returns marked as 'UNCATEGORIZED - NO AI AVAILABLE'")
        st.info("To enable AI categorization, ensure enhanced_ai_analysis module is properly installed")

def display_analysis_results():
    """Display comprehensive analysis results"""
    if not st.session_state.processing_complete:
        return
    
    df = st.session_state.categorized_data
    
    # Calculate metrics
    total_returns = len(df)
    categorized_count = len(df[~df['category'].str.contains('UNCATEGORIZED')])
    
    # Display key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Returns", f"{total_returns:,}")
    
    with col2:
        st.metric("Categorized", f"{categorized_count:,}")
    
    with col3:
        if st.session_state.total_sales > 0:
            return_rate = (total_returns / st.session_state.total_sales) * 100
            st.metric("Return Rate", f"{return_rate:.2f}%")
        else:
            st.metric("Return Rate", "No sales data")
    
    with col4:
        quality_issues = len(df[df['category'].isin(['Product Defects/Quality', 
                                                     'Performance/Effectiveness',
                                                     'Design/Material Issues'])])
        st.metric("Quality Issues", f"{quality_issues:,}")
    
    # Category breakdown
    st.markdown("### 📊 Return Categories")
    
    category_counts = df['category'].value_counts()
    
    # Create visualization
    fig = px.pie(
        values=category_counts.values,
        names=category_counts.index,
        title="Return Category Distribution",
        color_discrete_sequence=px.colors.qualitative.Set3
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Detailed breakdown table
    st.markdown("### 📋 Detailed Category Breakdown")
    
    category_summary = df.groupby('category').agg({
        'order_id': 'count',
        'confidence': 'mean'
    }).round(2)
    
    category_summary.columns = ['Count', 'Avg Confidence']
    category_summary['Percentage'] = (category_summary['Count'] / total_returns * 100).round(2)
    
    st.dataframe(category_summary.sort_values('Count', ascending=False))
    
    # Product analysis if ASIN data available
    if 'asin' in df.columns and df['asin'].notna().any():
        st.markdown("### 🏷️ Product Analysis")
        
        product_returns = df.groupby('asin').agg({
            'order_id': 'count',
            'category': lambda x: x.mode()[0] if len(x.mode()) > 0 else 'Unknown'
        })
        
        product_returns.columns = ['Returns', 'Top Category']
        product_returns = product_returns.sort_values('Returns', ascending=False).head(20)
        
        fig2 = px.bar(
            product_returns.reset_index(),
            x='asin',
            y='Returns',
            color='Top Category',
            title="Top 20 Products by Return Count"
        )
        
        st.plotly_chart(fig2, use_container_width=True)
    
    # Show uncategorized items
    uncategorized = df[df['category'].str.contains('UNCATEGORIZED')]
    if len(uncategorized) > 0:
        with st.expander(f"⚠️ Uncategorized Returns ({len(uncategorized)})"):
            st.dataframe(uncategorized[['order_id', 'return_reason', 'buyer_comment', 'category']])

def export_results():
    """Export analysis results"""
    if not st.session_state.processing_complete:
        return
    
    st.markdown("### 💾 Export Results")
    
    df = st.session_state.categorized_data
    
    # Prepare Excel export with multiple sheets
    output = io.BytesIO()
    
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        # Main data sheet
        df.to_excel(writer, sheet_name='All Returns', index=False)
        
        # Category sheets - only for properly categorized items
        categorized_df = df[~df['category'].str.contains('UNCATEGORIZED')]
        
        for category in MEDICAL_DEVICE_CATEGORIES:
            category_df = categorized_df[categorized_df['category'] == category]
            if len(category_df) > 0:
                category_df.to_excel(writer, sheet_name=category[:30], index=False)
        
        # Summary sheet
        summary_data = {
            'Metric': ['Total Returns', 'Categorized', 'Uncategorized', 'Quality Issues', 
                      'Return Rate', 'Processing Time'],
            'Value': [
                len(df),
                len(categorized_df),
                len(df) - len(categorized_df),
                len(df[df['category'].isin(['Product Defects/Quality', 
                                           'Performance/Effectiveness'])]),
                f"{(len(df) / st.session_state.total_sales * 100):.2f}%" if st.session_state.total_sales > 0 else "N/A",
                f"{st.session_state.processing_time:.2f} seconds"
            ]
        }
        
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_excel(writer, sheet_name='Summary', index=False)
    
    output.seek(0)
    
    st.download_button(
        label="📥 Download Excel Report",
        data=output,
        file_name=f"return_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
    
    # CSV export option
    csv = df.to_csv(index=False)
    st.download_button(
        label="📥 Download CSV",
        data=csv,
        file_name=f"returns_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv"
    )

def main():
    """Main application flow"""
    init_session_state()
    
    # Initialize components if available
    if AI_AVAILABLE and not st.session_state.ai_analyzer:
        st.session_state.ai_analyzer = EnhancedAIAnalyzer(provider=AIProvider.FASTEST)
    
    if PDF_AVAILABLE and not st.session_state.pdf_analyzer:
        st.session_state.pdf_analyzer = PDFAnalyzer()
    
    if MAPPER_AVAILABLE and not st.session_state.column_mapper:
        st.session_state.column_mapper = SmartColumnMapper(st.session_state.ai_analyzer)
    
    # Display header
    display_header()
    
    # Display AI status
    display_ai_status()
    
    # Sidebar
    with st.sidebar:
        st.markdown("### ⚙️ Settings")
        
        if AI_AVAILABLE:
            ai_provider = st.selectbox(
                "AI Provider",
                ["Fastest", "OpenAI", "Claude", "Both"],
                help="Select AI provider for categorization"
            )
            
            if ai_provider != "Fastest" and st.session_state.ai_analyzer:
                provider_map = {
                    "OpenAI": AIProvider.OPENAI,
                    "Claude": AIProvider.CLAUDE,
                    "Both": AIProvider.BOTH
                }
                st.session_state.ai_analyzer.provider = provider_map.get(ai_provider, AIProvider.FASTEST)
        
        st.markdown("### 📊 Categories")
        st.markdown("Returns are categorized into:")
        for category in MEDICAL_DEVICE_CATEGORIES:
            st.markdown(f"• {category}")
    
    # Main content
    tab1, tab2, tab3 = st.tabs(["📤 Upload & Process", "📊 Analysis", "💾 Export"])
    
    with tab1:
        if file_uploader_section():
            if st.button("🚀 Process All Files", type="primary"):
                start_time = time.time()
                process_files()
                st.session_state.processing_time = time.time() - start_time
                st.success(f"✅ Processing complete in {st.session_state.processing_time:.2f} seconds")
    
    with tab2:
        if st.session_state.processing_complete:
            display_analysis_results()
        else:
            st.info("📤 Please upload and process files first")
    
    with tab3:
        if st.session_state.processing_complete:
            export_results()
        else:
            st.info("📤 Please upload and process files first")
    
    # Display any AI failures
    if st.session_state.ai_failures:
        with st.expander("⚠️ AI Processing Issues"):
            for failure in st.session_state.ai_failures:
                st.warning(failure)

if __name__ == "__main__":
    main()
