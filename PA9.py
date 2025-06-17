"""
Vive Health Quality Analysis Platform
Version: 5.0 - Simplified & Clean

Core Features:
- Medical device return categorization (15 Amazon categories)
- Column K export mode
- Universal file analysis
- Critical issue detection
"""

import streamlit as st
import pandas as pd
import numpy as np
import logging
from datetime import datetime
import io
from typing import Dict, List, Any, Optional, Tuple
import time
from collections import Counter, defaultdict
import re
import os
import gc
import asyncio
import json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page config
st.set_page_config(
    page_title="Quality Analysis Platform",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Import AI modules
try:
    from enhanced_ai_analysis import (
        EnhancedAIAnalyzer, AIProvider, FBA_REASON_MAP,
        MEDICAL_DEVICE_CATEGORIES as AI_CATEGORIES
    )
    AI_AVAILABLE = True
except ImportError as e:
    AI_AVAILABLE = False
    logger.error(f"AI module not available: {str(e)}")

# Import universal modules
try:
    from enhanced_ai_universal import UniversalAIAnalyzer
    from universal_file_detector import UniversalFileDetector, ProcessedFile
    UNIVERSAL_AVAILABLE = True
except ImportError as e:
    UNIVERSAL_AVAILABLE = False
    logger.error(f"Universal modules not available: {str(e)}")

# Check Excel support
try:
    import xlsxwriter
    EXCEL_AVAILABLE = True
except ImportError:
    EXCEL_AVAILABLE = False

# Amazon Medical Device Return Categories - EXACT from document
MEDICAL_DEVICE_CATEGORIES = [
    "Size/Fit Issues",
    "Comfort Issues",
    "Product Defects/Quality",
    "Performance/Effectiveness",
    "Stability/Positioning Issues",
    "Equipment Compatibility",
    "Design/Material Issues",
    "Wrong Product/Misunderstanding",
    "Missing Components",
    "Customer Error/Changed Mind",
    "Shipping/Fulfillment Issues",
    "Assembly/Usage Difficulty",
    "Medical/Health Concerns",
    "Price/Value",
    "Other/Miscellaneous"
]

# Clean, minimal styling
def inject_clean_css():
    """Minimal Apple-style CSS"""
    st.markdown("""
    <style>
    /* Clean typography */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&display=swap');
    
    html, body, .stApp {
        font-family: -apple-system, BlinkMacSystemFont, 'Inter', sans-serif;
        background-color: #FFFFFF;
    }
    
    /* Simple header */
    .main-header {
        padding: 2rem 0;
        border-bottom: 1px solid #E5E7EB;
        margin-bottom: 2rem;
    }
    
    .main-header h1 {
        font-size: 2rem;
        font-weight: 600;
        color: #111827;
        margin: 0;
    }
    
    .main-header p {
        color: #6B7280;
        margin: 0.5rem 0 0 0;
    }
    
    /* Clean cards */
    .info-card {
        background: #F9FAFB;
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
    }
    
    /* Status badges */
    .status-success {
        background: #D1FAE5;
        color: #065F46;
        padding: 0.25rem 0.75rem;
        border-radius: 6px;
        font-size: 0.875rem;
        display: inline-block;
    }
    
    .status-critical {
        background: #FEE2E2;
        color: #991B1B;
        padding: 0.25rem 0.75rem;
        border-radius: 6px;
        font-size: 0.875rem;
        display: inline-block;
    }
    
    /* Clean buttons */
    .stButton > button {
        background: #2563EB;
        color: white;
        border: none;
        padding: 0.5rem 1rem;
        border-radius: 8px;
        font-weight: 500;
        transition: background 0.2s;
    }
    
    .stButton > button:hover {
        background: #1D4ED8;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Simple progress */
    .stProgress > div > div {
        background: #2563EB;
    }
    </style>
    """, unsafe_allow_html=True)

def initialize_session_state():
    """Initialize session state"""
    defaults = {
        'mode': None,
        'data': None,
        'categorized': False,
        'column_mapping': {},
        'categories_summary': {},
        'critical_issues': [],
        'ai_analyzer': None,
        'export_ready': False,
        'processing_errors': []
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

def check_setup():
    """Check if APIs are configured"""
    try:
        if hasattr(st, 'secrets'):
            has_openai = any(key in st.secrets for key in ['OPENAI_API_KEY', 'openai_api_key'])
            has_claude = any(key in st.secrets for key in ['ANTHROPIC_API_KEY', 'claude_api_key'])
            return has_openai or has_claude
    except:
        pass
    
    # Check environment
    return bool(os.environ.get('OPENAI_API_KEY') or os.environ.get('ANTHROPIC_API_KEY'))

def display_header():
    """Simple header"""
    st.markdown("""
    <div class="main-header">
        <h1>Quality Analysis Platform</h1>
        <p>Medical device return categorization and analysis</p>
    </div>
    """, unsafe_allow_html=True)

def display_mode_selection():
    """Clean mode selection"""
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### 📊 Column K Export
        For structured files with complaints in Column I
        - Categorizes into Column K
        - Preserves file format
        - Excel/CSV export
        """)
        if st.button("Start Column K Mode", use_container_width=True):
            st.session_state.mode = 'column_k'
            st.rerun()
    
    with col2:
        st.markdown("""
        ### 🌐 Universal Analysis
        For any file type
        - PDFs from Amazon
        - FBA return reports
        - Review files
        """)
        if st.button("Start Universal Mode", use_container_width=True):
            st.session_state.mode = 'universal'
            st.rerun()

# ============================================================================
# COLUMN K MODE
# ============================================================================

def run_column_k_mode():
    """Column K mode - simplified"""
    st.markdown("## Column K Export Mode")
    
    # Back button
    if st.button("← Back"):
        st.session_state.mode = None
        st.rerun()
    
    st.markdown("---")
    
    # File upload
    uploaded_file = st.file_uploader(
        "Upload file (Excel, CSV, or TXT)",
        type=['xlsx', 'xls', 'csv', 'txt'],
        help="Complaints should be in Column I"
    )
    
    if uploaded_file:
        process_column_k_file(uploaded_file)
    
    # Show results if processed
    if st.session_state.categorized:
        display_column_k_results()

def process_column_k_file(uploaded_file):
    """Process Column K file"""
    try:
        # Read file
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file, dtype=str)
        elif uploaded_file.name.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(uploaded_file, dtype=str)
        elif uploaded_file.name.endswith('.txt'):
            df = pd.read_csv(uploaded_file, sep='\t', dtype=str)
        
        # Check structure
        if len(df.columns) < 11:
            st.error("File must have at least 11 columns (A-K)")
            return
        
        # Map columns
        cols = df.columns.tolist()
        column_mapping = {
            'complaint': cols[8] if len(cols) > 8 else None,  # Column I
            'sku': cols[1] if len(cols) > 1 else None,  # Column B
            'category': cols[10] if len(cols) > 10 else None  # Column K
        }
        
        # Validate
        if not column_mapping['complaint']:
            st.error("Column I (complaints) not found")
            return
        
        # Ensure Column K exists
        while len(df.columns) < 11:
            df[f'Col_{len(df.columns)}'] = ''
        
        if not column_mapping['category']:
            column_mapping['category'] = df.columns[10]
        
        # Clear Column K
        df[column_mapping['category']] = ''
        
        # Count valid complaints
        valid_mask = df[column_mapping['complaint']].notna() & \
                    (df[column_mapping['complaint']].str.strip() != '')
        valid_count = valid_mask.sum()
        
        if valid_count == 0:
            st.warning("No valid complaints found in Column I")
            return
        
        # Show info
        st.success(f"✓ File loaded: {len(df)} rows, {valid_count} complaints to categorize")
        
        # Store data
        st.session_state.data = df
        st.session_state.column_mapping = column_mapping
        
        # Process button
        if st.button("Categorize Returns", type="primary", use_container_width=True):
            categorize_returns()
            
    except Exception as e:
        st.error(f"Error reading file: {str(e)}")

def categorize_returns():
    """Categorize returns using AI"""
    if not AI_AVAILABLE:
        st.error("AI module not available")
        return
    
    # Initialize analyzer
    if not st.session_state.ai_analyzer:
        try:
            # Set API keys from secrets
            if hasattr(st, 'secrets'):
                for key in ['OPENAI_API_KEY', 'openai_api_key']:
                    if key in st.secrets:
                        os.environ['OPENAI_API_KEY'] = str(st.secrets[key])
                        break
                
                for key in ['ANTHROPIC_API_KEY', 'claude_api_key']:
                    if key in st.secrets:
                        os.environ['ANTHROPIC_API_KEY'] = str(st.secrets[key])
                        break
            
            st.session_state.ai_analyzer = EnhancedAIAnalyzer(AIProvider.FASTEST)
        except Exception as e:
            st.error(f"Failed to initialize AI: {str(e)}")
            return
    
    df = st.session_state.data
    column_mapping = st.session_state.column_mapping
    analyzer = st.session_state.ai_analyzer
    
    # Get valid rows
    complaint_col = column_mapping['complaint']
    category_col = column_mapping['category']
    
    valid_indices = df[df[complaint_col].notna() & 
                      (df[complaint_col].str.strip() != '')].index
    
    # Progress
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Clear previous issues
    st.session_state.critical_issues = []
    categories_count = defaultdict(int)
    
    # Process in batches
    batch_size = 50
    total = len(valid_indices)
    
    for i in range(0, total, batch_size):
        batch_indices = valid_indices[i:i+batch_size]
        batch_data = []
        
        # Prepare batch
        for idx in batch_indices:
            complaint = str(df.at[idx, complaint_col]).strip()
            batch_data.append({
                'index': idx,
                'complaint': complaint,
                'fba_reason': None
            })
        
        # Categorize
        try:
            results = analyzer.categorize_batch(batch_data, mode='standard')
            
            # Update dataframe
            for result in results:
                idx = result['index']
                category = result.get('category', 'Other/Miscellaneous')
                
                # Map to our categories if needed
                category = map_to_medical_category(category)
                
                df.at[idx, category_col] = category
                categories_count[category] += 1
                
                # Check for critical issues
                if category == 'Medical/Health Concerns' or \
                   result.get('severity') == 'critical':
                    st.session_state.critical_issues.append({
                        'row': idx,
                        'complaint': batch_data[results.index(result)]['complaint'],
                        'category': category
                    })
        
        except Exception as e:
            logger.error(f"Batch error: {e}")
            st.session_state.processing_errors.append(str(e))
        
        # Update progress
        progress = min((i + len(batch_indices)) / total, 1.0)
        progress_bar.progress(progress)
        status_text.text(f"Processing {i + len(batch_indices)}/{total}")
    
    # Complete
    progress_bar.empty()
    status_text.empty()
    
    st.session_state.categorized = True
    st.session_state.categories_summary = dict(categories_count)
    
    st.success(f"✓ Categorized {total} returns")
    
    # Show critical issues immediately
    if st.session_state.critical_issues:
        st.error(f"⚠️ {len(st.session_state.critical_issues)} critical issues found!")

def map_to_medical_category(ai_category):
    """Map AI category to standard medical device categories"""
    # Direct mapping
    for medical_cat in MEDICAL_DEVICE_CATEGORIES:
        if medical_cat.lower() in ai_category.lower():
            return medical_cat
    
    # Fuzzy mapping
    mappings = {
        'defect': 'Product Defects/Quality',
        'quality': 'Product Defects/Quality',
        'size': 'Size/Fit Issues',
        'fit': 'Size/Fit Issues',
        'comfort': 'Comfort Issues',
        'performance': 'Performance/Effectiveness',
        'stability': 'Stability/Positioning Issues',
        'compatible': 'Equipment Compatibility',
        'design': 'Design/Material Issues',
        'wrong': 'Wrong Product/Misunderstanding',
        'missing': 'Missing Components',
        'error': 'Customer Error/Changed Mind',
        'shipping': 'Shipping/Fulfillment Issues',
        'assembly': 'Assembly/Usage Difficulty',
        'medical': 'Medical/Health Concerns',
        'health': 'Medical/Health Concerns',
        'price': 'Price/Value'
    }
    
    ai_lower = ai_category.lower()
    for key, medical_cat in mappings.items():
        if key in ai_lower:
            return medical_cat
    
    return 'Other/Miscellaneous'

def display_column_k_results():
    """Display results for Column K mode"""
    st.markdown("---")
    st.markdown("### Results")
    
    # Summary metrics
    total_categorized = sum(st.session_state.categories_summary.values())
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Categorized", f"{total_categorized:,}")
    with col2:
        critical_count = len(st.session_state.critical_issues)
        if critical_count > 0:
            st.metric("Critical Issues", str(critical_count), delta_color="inverse")
        else:
            st.metric("Critical Issues", "0")
    with col3:
        quality_issues = sum(count for cat, count in st.session_state.categories_summary.items()
                           if 'Quality' in cat or 'Defect' in cat)
        st.metric("Quality Issues", f"{quality_issues:,}")
    
    # Category breakdown
    if st.session_state.categories_summary:
        st.markdown("### Category Distribution")
        
        # Sort by count
        sorted_cats = sorted(st.session_state.categories_summary.items(), 
                           key=lambda x: x[1], reverse=True)
        
        for cat, count in sorted_cats[:10]:
            pct = (count / total_categorized * 100) if total_categorized > 0 else 0
            
            # Determine if critical
            is_critical = cat in ['Medical/Health Concerns', 'Product Defects/Quality']
            
            st.markdown(f"""
            <div style="margin: 0.5rem 0;">
                <div style="display: flex; justify-content: space-between;">
                    <span>{cat}</span>
                    <span>{count} ({pct:.1f}%)</span>
                </div>
                <div style="background: #E5E7EB; height: 8px; border-radius: 4px; margin-top: 4px;">
                    <div style="background: {'#EF4444' if is_critical else '#3B82F6'}; 
                               width: {pct}%; height: 100%; border-radius: 4px;"></div>
                </div>
            </div>
            """, unsafe_allow_html=True)
    
    # Critical issues
    if st.session_state.critical_issues:
        with st.expander(f"⚠️ View {len(st.session_state.critical_issues)} Critical Issues"):
            for i, issue in enumerate(st.session_state.critical_issues[:20]):
                st.markdown(f"""
                **{i+1}. Row {issue['row']}**  
                Category: {issue['category']}  
                Complaint: {issue['complaint'][:200]}...
                """)
    
    # Export section
    st.markdown("---")
    st.markdown("### Export")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Excel export
        if st.button("Download Excel", use_container_width=True):
            try:
                output = export_to_excel()
                st.download_button(
                    label="💾 Download Excel File",
                    data=output,
                    file_name=f"categorized_{datetime.now().strftime('%Y%m%d_%H%M')}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
            except Exception as e:
                st.error(f"Export error: {str(e)}")
    
    with col2:
        # CSV export
        if st.button("Download CSV", use_container_width=True):
            try:
                output = export_to_csv()
                st.download_button(
                    label="💾 Download CSV File",
                    data=output,
                    file_name=f"categorized_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                    mime="text/csv"
                )
            except Exception as e:
                st.error(f"Export error: {str(e)}")

def export_to_excel():
    """Export to Excel with Column K highlighted"""
    output = io.BytesIO()
    
    if EXCEL_AVAILABLE:
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            st.session_state.data.to_excel(writer, index=False, sheet_name='Returns')
            
            # Format Column K
            workbook = writer.book
            worksheet = writer.sheets['Returns']
            
            green_format = workbook.add_format({
                'bg_color': '#D1FAE5',
                'font_color': '#065F46',
                'bold': True
            })
            
            # Column K is index 10
            worksheet.set_column(10, 10, 25, green_format)
    else:
        # Fallback to CSV
        st.session_state.data.to_csv(output, index=False)
    
    output.seek(0)
    return output.getvalue()

def export_to_csv():
    """Export to CSV"""
    output = io.StringIO()
    st.session_state.data.to_csv(output, index=False)
    return output.getvalue()

# ============================================================================
# UNIVERSAL MODE
# ============================================================================

def run_universal_mode():
    """Universal analysis mode - simplified"""
    st.markdown("## Universal Analysis Mode")
    
    # Back button
    if st.button("← Back"):
        st.session_state.mode = None
        st.rerun()
    
    st.markdown("---")
    
    if not UNIVERSAL_AVAILABLE:
        st.error("Universal analysis modules not installed")
        return
    
    # File upload
    uploaded_files = st.file_uploader(
        "Upload files",
        type=['pdf', 'csv', 'txt', 'xlsx', 'jpg', 'png'],
        accept_multiple_files=True
    )
    
    if uploaded_files:
        process_universal_files(uploaded_files)

def process_universal_files(files):
    """Process files for universal analysis"""
    results = []
    
    for file in files:
        with st.spinner(f"Processing {file.name}..."):
            try:
                content = file.read()
                
                # Process file
                processed = UniversalFileDetector.process_file(
                    content, 
                    file.name,
                    None
                )
                
                results.append(processed)
                
                # Check for critical issues
                if processed.critical_issues:
                    st.session_state.critical_issues.extend(processed.critical_issues)
                
                st.success(f"✓ {file.name}")
                
            except Exception as e:
                st.error(f"Error with {file.name}: {str(e)}")
    
    # Show summary
    if results:
        st.markdown("---")
        st.markdown("### Analysis Summary")
        
        total_returns = sum(1 for r in results if 'return' in r.content_category)
        total_reviews = sum(1 for r in results if 'review' in r.content_category)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Files Processed", str(len(results)))
        with col2:
            st.metric("Return Files", str(total_returns))
        with col3:
            st.metric("Critical Issues", str(len(st.session_state.critical_issues)))
        
        # Show critical issues
        if st.session_state.critical_issues:
            st.error(f"⚠️ {len(st.session_state.critical_issues)} critical issues found")
            
            with st.expander("View Critical Issues"):
                for i, issue in enumerate(st.session_state.critical_issues[:10]):
                    st.markdown(f"**Issue {i+1}**: {issue}")

# ============================================================================
# MAIN
# ============================================================================

def main():
    """Main application"""
    initialize_session_state()
    inject_clean_css()
    
    # Check setup
    if not check_setup():
        st.error("⚠️ API keys not configured")
        st.markdown("""
        Add to `.streamlit/secrets.toml`:
        ```
        openai_api_key = "sk-..."
        # or
        anthropic_api_key = "sk-ant-..."
        ```
        """)
        st.stop()
    
    # Display header
    display_header()
    
    # Run selected mode
    if st.session_state.mode is None:
        display_mode_selection()
    elif st.session_state.mode == 'column_k':
        run_column_k_mode()
    elif st.session_state.mode == 'universal':
        run_universal_mode()

if __name__ == "__main__":
    main()
