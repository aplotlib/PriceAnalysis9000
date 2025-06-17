"""
Amazon Quality Analysis Platform with Column K Export
Version: 3.0 - Unified with app.py functionality
Designed for Quality Analysts to analyze returns and export with categories in Column K
"""

import streamlit as st
import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
import io
from typing import Dict, List, Any, Optional, Tuple
import re
import asyncio
import json
from collections import Counter, defaultdict
import plotly.express as px
import plotly.graph_objects as go
import time
import gc

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import our modules
from enhanced_ai_universal import UniversalAIAnalyzer, FileAnalysis, MEDICAL_DEVICE_CATEGORIES
from universal_file_detector import UniversalFileDetector, ProcessedFile

# App Configuration
st.set_page_config(
    page_title="Quality Analysis Platform",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Professional color scheme
COLORS = {
    'primary': '#2E86AB',      # Professional blue
    'secondary': '#A23B72',    # Muted purple
    'success': '#58B368',      # Green
    'warning': '#F18F01',      # Orange
    'danger': '#C73E1D',       # Red
    'neutral': '#6C757D',      # Gray
    'light': '#F8F9FA',        # Light gray
    'dark': '#212529',         # Dark
    'cost': '#50C878',         # Cost green
}

# Quality categories for analysis
QUALITY_CATEGORIES = [
    'Product Defects/Quality',
    'Performance/Effectiveness',
    'Missing Components',
    'Design/Material Issues',
    'Stability/Positioning Issues',
    'Medical/Health Concerns'
]

def inject_professional_css():
    """Clean, professional CSS styling"""
    st.markdown(f"""
    <style>
    /* Professional styling */
    .main {{
        padding: 0;
    }}
    
    .stApp {{
        background-color: #FFFFFF;
    }}
    
    /* Header styling */
    .main-header {{
        background: linear-gradient(135deg, {COLORS['primary']} 0%, {COLORS['secondary']} 100%);
        color: white;
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
    }}
    
    .main-header h1 {{
        margin: 0;
        font-size: 2.5rem;
        font-weight: 700;
    }}
    
    .main-header p {{
        margin: 0.5rem 0 0 0;
        opacity: 0.9;
        font-size: 1.1rem;
    }}
    
    /* Card styling */
    .info-card {{
        background: white;
        border: 1px solid #E0E0E0;
        border-radius: 8px;
        padding: 1.5rem;
        margin-bottom: 1rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }}
    
    .info-card h3 {{
        color: {COLORS['primary']};
        margin-top: 0;
    }}
    
    /* Metric cards */
    .metric-card {{
        background: #F8F9FA;
        border-radius: 8px;
        padding: 1rem;
        text-align: center;
        border: 1px solid #E0E0E0;
    }}
    
    .metric-value {{
        font-size: 2rem;
        font-weight: 700;
        color: {COLORS['primary']};
    }}
    
    .metric-label {{
        color: {COLORS['neutral']};
        font-size: 0.9rem;
        margin-top: 0.5rem;
    }}
    
    /* Status badges */
    .status-badge {{
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 500;
    }}
    
    .status-critical {{
        background: #FEE;
        color: {COLORS['danger']};
        border: 1px solid {COLORS['danger']};
    }}
    
    .status-high {{
        background: #FFF4E6;
        color: {COLORS['warning']};
        border: 1px solid {COLORS['warning']};
    }}
    
    /* File upload area */
    .upload-area {{
        border: 2px dashed #CBD5E1;
        border-radius: 8px;
        padding: 2rem;
        text-align: center;
        background: #F8FAFC;
        margin: 1rem 0;
    }}
    
    /* Processing box */
    .processing-box {{
        background: linear-gradient(135deg, rgba(46, 134, 171, 0.1), rgba(162, 59, 114, 0.1));
        border: 1px solid {COLORS['primary']};
        border-radius: 8px;
        padding: 1.5rem;
        margin: 1rem 0;
    }}
    
    /* Hide Streamlit branding */
    #MainMenu {{visibility: hidden;}}
    footer {{visibility: hidden;}}
    
    /* Improve button styling */
    .stButton > button {{
        background: {COLORS['primary']};
        color: white;
        border: none;
        padding: 0.5rem 2rem;
        border-radius: 5px;
        font-weight: 500;
        transition: all 0.3s;
    }}
    
    .stButton > button:hover {{
        background: {COLORS['secondary']};
        transform: translateY(-1px);
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }}
    
    /* Success notification */
    .success-notification {{
        background: linear-gradient(135deg, {COLORS['success']} 0%, {COLORS['primary']} 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        margin-bottom: 1rem;
        animation: pulse 2s infinite;
        box-shadow: 0 0 30px rgba(88, 179, 104, 0.5);
    }}
    
    @keyframes pulse {{
        0% {{ transform: scale(1); box-shadow: 0 0 30px rgba(88, 179, 104, 0.5); }}
        50% {{ transform: scale(1.02); box-shadow: 0 0 40px rgba(88, 179, 104, 0.7); }}
        100% {{ transform: scale(1); box-shadow: 0 0 30px rgba(88, 179, 104, 0.5); }}
    }}
    </style>
    """, unsafe_allow_html=True)

def initialize_session_state():
    """Initialize session state variables"""
    defaults = {
        # Core data
        'uploaded_files': [],
        'processed_files': [],
        'current_analysis': None,
        'target_asin': '',
        'categorized_data': None,
        'column_k_data': None,
        
        # Processing state
        'processing_complete': False,
        'export_ready': False,
        'export_data': None,
        'export_filename': None,
        
        # UI state
        'current_tab': 'categorize',
        'show_ai_chat': False,
        'analysis_complete': False,
        
        # AI components
        'ai_analyzer': None,
        'chat_messages': [],
        
        # Settings
        'auto_analyze': True,
        'batch_size': 50,
        'chunk_size': 500,
        
        # Tracking
        'total_cost': 0.0,
        'api_calls_made': 0,
        'processing_time': 0.0,
        'total_rows_processed': 0,
        'processing_errors': [],
        
        # Cache
        'analysis_cache': {},
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

def display_header():
    """Display application header"""
    st.markdown("""
    <div class="main-header">
        <h1>🔍 Quality Analysis Platform</h1>
        <p>AI-Powered Return Categorization & Analysis for Medical Devices</p>
        <p style="font-size: 0.9em; opacity: 0.8;">✅ Column K Export | 📊 Multi-Source Analysis | 🤖 Dual AI Support</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Show completion notification if processing is done
    if st.session_state.processing_complete and st.session_state.export_ready:
        st.markdown("""
        <div class="success-notification">
            <h2 style="margin: 0;">🎉 Analysis Complete! Your results are ready for download 👇</h2>
        </div>
        """, unsafe_allow_html=True)
    
    # Quick stats
    if st.session_state.processed_files or st.session_state.categorized_data is not None:
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Files Loaded", len(st.session_state.processed_files))
        with col2:
            total_returns = st.session_state.total_rows_processed
            st.metric("Total Returns", f"{total_returns:,}")
        with col3:
            if st.session_state.target_asin:
                st.metric("Target ASIN", st.session_state.target_asin)
            else:
                st.metric("Mode", "All ASINs")
        with col4:
            ai_status = "🟢 Ready" if get_ai_status() else "🔴 Configure API"
            st.metric("AI Status", ai_status)

def get_ai_status():
    """Check AI availability"""
    try:
        if st.session_state.ai_analyzer is None:
            st.session_state.ai_analyzer = UniversalAIAnalyzer()
        
        providers = st.session_state.ai_analyzer.get_available_providers()
        return len(providers) > 0
    except Exception as e:
        logger.error(f"AI status check error: {e}")
        return False

def check_excel_support():
    """Check if Excel export is available"""
    try:
        import xlsxwriter
        return True
    except ImportError:
        return False

def display_categorization_tab():
    """Display the categorization tab (main functionality from app.py)"""
    st.markdown("### 📤 Upload Return Data Files")
    
    # ASIN input
    col1, col2 = st.columns([3, 1])
    with col1:
        asin_input = st.text_input(
            "Target ASIN (Optional - leave blank to process all)",
            value=st.session_state.target_asin,
            placeholder="B00XYZ1234",
            help="Enter ASIN to filter data, or leave blank to process all ASINs"
        )
        if asin_input:
            st.session_state.target_asin = asin_input.upper()
    
    with col2:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("🔄 Clear All", use_container_width=True):
            # Reset state
            for key in ['uploaded_files', 'processed_files', 'categorized_data', 
                       'column_k_data', 'processing_complete', 'export_ready']:
                st.session_state[key] = None if 'complete' in key or 'ready' in key else []
            st.rerun()
    
    # File upload instructions
    with st.expander("📖 File Format Instructions", expanded=True):
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            **Supported File Types:**
            - **FBA Return Reports** (.txt, .csv) - Direct export
            - **PDF Exports** - Print from Seller Central
            - **Excel Files** (.xlsx, .xls)
            - **CSV Files** - Any return data
            
            **Required Columns:**
            - Return reason/complaint text
            - ASIN or SKU (optional)
            - Order ID (optional)
            """)
        
        with col2:
            st.markdown("""
            **Export Format:**
            - Categories will be added to **Column K**
            - Original file structure preserved
            - Google Sheets compatible
            - Ready for pivot analysis
            
            **Processing Speed:**
            - ~100-200 returns/second
            - Automatic batch optimization
            """)
    
    # File upload area
    st.markdown("""
    <div class="upload-area">
        <h4>📁 Drop return data files here</h4>
        <p>Supports: PDF, Excel, CSV, TSV, TXT (FBA exports)</p>
    </div>
    """, unsafe_allow_html=True)
    
    uploaded_files = st.file_uploader(
        "Choose files",
        type=['pdf', 'csv', 'tsv', 'txt', 'xlsx', 'xls'],
        accept_multiple_files=True,
        label_visibility="collapsed"
    )
    
    if uploaded_files:
        process_uploaded_files_for_categorization(uploaded_files)
        
        # Show process button if files are loaded
        if st.session_state.processed_files and not st.session_state.processing_complete:
            st.markdown("---")
            
            # Count valid complaints
            total_valid = 0
            for file in st.session_state.processed_files:
                if file.data is not None and not file.data.empty:
                    # Look for complaint/reason columns
                    complaint_cols = [col for col in file.data.columns 
                                    if any(term in col.lower() for term in ['reason', 'comment', 'complaint'])]
                    if complaint_cols:
                        total_valid += len(file.data[file.data[complaint_cols[0]].notna()])
            
            if total_valid > 0:
                # Cost estimation
                est_cost = total_valid * 0.002  # Average cost per categorization
                
                st.markdown(f"""
                <div class="info-card" style="background: linear-gradient(135deg, rgba(80, 200, 120, 0.1), rgba(80, 200, 120, 0.2)); 
                            border-color: {COLORS['cost']}; text-align: center;">
                    <h4 style="color: {COLORS['cost']}; margin: 0;">💰 Estimated Processing Cost</h4>
                    <div style="font-size: 2em; font-weight: 700; color: {COLORS['cost']}; margin: 0.5rem 0;">
                        ${est_cost:.2f}
                    </div>
                    <div style="color: {COLORS['neutral']};">
                        for {total_valid:,} returns at ~$0.002 each
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                # Process button
                col1, col2, col3 = st.columns([1, 2, 1])
                with col2:
                    if st.button(f"🚀 Categorize {total_valid:,} Returns", 
                               type="primary", use_container_width=True):
                        process_returns_for_categorization()
    
    # Show results if processing is complete
    if st.session_state.processing_complete and st.session_state.categorized_data is not None:
        display_categorization_results()

def process_uploaded_files_for_categorization(files):
    """Process uploaded files for categorization"""
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for idx, file in enumerate(files):
        # Check if already processed
        if any(f.metadata.get('filename') == file.name for f in st.session_state.processed_files):
            continue
        
        status_text.text(f"Processing {file.name}...")
        progress_bar.progress((idx + 1) / len(files))
        
        try:
            # Read file content
            file_content = file.read()
            
            # Process with detector
            processed = UniversalFileDetector.process_file(
                file_content, 
                file.name,
                st.session_state.target_asin
            )
            
            # Store processed file
            st.session_state.processed_files.append(processed)
            
            # Show success
            if processed.warnings:
                st.warning(f"⚠️ {file.name}: {', '.join(processed.warnings)}")
            else:
                st.success(f"✅ {file.name} processed - {processed.metadata.get('row_count', 0)} rows")
                
        except Exception as e:
            st.error(f"❌ Error processing {file.name}: {str(e)}")
            logger.error(f"File processing error: {e}", exc_info=True)
    
    progress_bar.empty()
    status_text.empty()

def process_returns_for_categorization():
    """Process returns and add categories to Column K"""
    try:
        analyzer = st.session_state.ai_analyzer
        if not analyzer:
            st.error("AI analyzer not initialized")
            return
        
        # Combine all return data
        all_returns = []
        file_mapping = {}  # Track which file each row came from
        
        for file_idx, file in enumerate(st.session_state.processed_files):
            if file.data is not None and not file.data.empty:
                # Find complaint column
                complaint_cols = [col for col in file.data.columns 
                                if any(term in col.lower() for term in ['reason', 'comment', 'complaint', 'customer-comments'])]
                
                if complaint_cols:
                    complaint_col = complaint_cols[0]
                    
                    # Add file tracking
                    for idx, row in file.data.iterrows():
                        if pd.notna(row[complaint_col]) and str(row[complaint_col]).strip():
                            all_returns.append({
                                'file_idx': file_idx,
                                'row_idx': idx,
                                'complaint': str(row[complaint_col]),
                                'fba_reason': row.get('reason', '') if 'reason' in row else None,
                                'data': row.to_dict()
                            })
        
        if not all_returns:
            st.warning("No valid complaints found to categorize")
            return
        
        # Process in chunks with progress
        total = len(all_returns)
        st.info(f"🔍 Categorizing {total:,} returns...")
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        start_time = time.time()
        
        categorized_results = []
        chunk_size = st.session_state.batch_size
        
        for i in range(0, total, chunk_size):
            chunk = all_returns[i:i+chunk_size]
            chunk_num = (i // chunk_size) + 1
            total_chunks = (total + chunk_size - 1) // chunk_size
            
            status_text.text(f"Processing chunk {chunk_num}/{total_chunks}...")
            
            # Process chunk
            try:
                results = analyzer.categorize_batch(chunk, mode='standard')
                categorized_results.extend(results)
            except Exception as e:
                logger.error(f"Categorization error: {e}")
                # Add default categories for failed items
                for item in chunk:
                    item['category'] = 'Other/Miscellaneous'
                    categorized_results.append(item)
            
            # Update progress
            progress = min((i + chunk_size) / total, 1.0)
            progress_bar.progress(progress)
            
            # Small delay
            time.sleep(0.05)
        
        # Update original dataframes with categories
        for file in st.session_state.processed_files:
            if file.data is not None:
                # Ensure Column K exists
                if len(file.data.columns) < 11:
                    # Add columns to reach K (11th column, index 10)
                    while len(file.data.columns) < 11:
                        file.data[f'Column_{len(file.data.columns)}'] = ''
                
                # Set Column K as category column
                col_k = file.data.columns[10]
                file.data[col_k] = ''  # Initialize
        
        # Apply categories to Column K
        for result in categorized_results:
            file_idx = result['file_idx']
            row_idx = result['row_idx']
            category = result.get('category', 'Other/Miscellaneous')
            
            # Update Column K in the appropriate file
            file = st.session_state.processed_files[file_idx]
            col_k = file.data.columns[10]
            file.data.at[row_idx, col_k] = category
        
        # Combine all data for export
        all_dfs = []
        for file in st.session_state.processed_files:
            if file.data is not None:
                all_dfs.append(file.data)
        
        if all_dfs:
            st.session_state.categorized_data = pd.concat(all_dfs, ignore_index=True)
        
        # Calculate processing time and costs
        st.session_state.processing_time = time.time() - start_time
        st.session_state.total_rows_processed = len(categorized_results)
        
        # Get cost summary from analyzer
        cost_summary = analyzer.get_cost_summary()
        st.session_state.total_cost = cost_summary.get('total_cost', 0)
        st.session_state.api_calls_made = cost_summary.get('api_calls', 0)
        
        # Mark as complete
        st.session_state.processing_complete = True
        st.session_state.export_ready = True
        
        # Prepare export
        prepare_export_data()
        
        progress_bar.empty()
        status_text.empty()
        
        # Success message
        st.success(f"""
        ✅ Categorization complete! 
        - Processed: {st.session_state.total_rows_processed:,} returns
        - Time: {st.session_state.processing_time:.1f} seconds
        - Cost: ${st.session_state.total_cost:.4f}
        """)
        
        st.balloons()
        st.rerun()
        
    except Exception as e:
        st.error(f"Processing error: {str(e)}")
        logger.error(f"Categorization error: {e}", exc_info=True)

def prepare_export_data():
    """Prepare data for export with Column K"""
    try:
        if st.session_state.categorized_data is not None:
            df = st.session_state.categorized_data.copy()
            
            # Ensure we have at least 11 columns (up to K)
            while len(df.columns) < 11:
                df[f'Col_{len(df.columns)}'] = ''
            
            # Generate filename
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            
            # Check if Excel is available
            if check_excel_support():
                # Export as Excel
                output = io.BytesIO()
                
                with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                    df.to_excel(writer, index=False, sheet_name='Categorized_Returns')
                    
                    # Get workbook and worksheet
                    workbook = writer.book
                    worksheet = writer.sheets['Categorized_Returns']
                    
                    # Format column K (11th column, index 10)
                    category_format = workbook.add_format({
                        'bg_color': '#E6F5E6',
                        'font_color': '#006600',
                        'bold': True
                    })
                    
                    # Apply format to column K
                    worksheet.set_column(10, 10, 25, category_format)  # Column K with width 25
                    
                    # Add autofilter
                    worksheet.autofilter(0, 0, len(df), len(df.columns) - 1)
                
                output.seek(0)
                st.session_state.export_data = output.getvalue()
                st.session_state.export_filename = f'categorized_returns_{timestamp}.xlsx'
            else:
                # Export as CSV
                csv_buffer = io.StringIO()
                df.to_csv(csv_buffer, index=False)
                st.session_state.export_data = csv_buffer.getvalue().encode()
                st.session_state.export_filename = f'categorized_returns_{timestamp}.csv'
                
    except Exception as e:
        logger.error(f"Export preparation error: {e}")
        st.error(f"Error preparing export: {str(e)}")

def display_categorization_results():
    """Display categorization results and export options"""
    st.markdown("---")
    st.markdown("### 📊 Categorization Results")
    
    if st.session_state.categorized_data is not None:
        df = st.session_state.categorized_data
        
        # Find Column K (category column)
        if len(df.columns) > 10:
            category_col = df.columns[10]
            
            # Calculate statistics
            categorized_rows = df[df[category_col].notna() & (df[category_col] != '')][category_col]
            
            if not categorized_rows.empty:
                # Category distribution
                category_counts = categorized_rows.value_counts()
                
                # Metrics row
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Total Returns", f"{len(df):,}")
                
                with col2:
                    st.metric("Categorized", f"{len(categorized_rows):,}")
                
                with col3:
                    quality_count = sum(
                        count for cat, count in category_counts.items()
                        if any(q in cat for q in ['Quality', 'Defect', 'Performance'])
                    )
                    st.metric("Quality Issues", f"{quality_count:,}")
                
                with col4:
                    st.metric("Processing Cost", f"${st.session_state.total_cost:.4f}")
                
                # Category breakdown chart
                st.markdown("#### Category Distribution")
                
                # Create bar chart
                fig = px.bar(
                    x=category_counts.values,
                    y=category_counts.index,
                    orientation='h',
                    labels={'x': 'Count', 'y': 'Category'},
                    title="Returns by Category",
                    color=category_counts.values,
                    color_continuous_scale='Blues'
                )
                fig.update_layout(height=400, showlegend=False)
                st.plotly_chart(fig, use_container_width=True)
                
                # Top categories detail
                st.markdown("#### Top Return Categories")
                for i, (cat, count) in enumerate(category_counts.head(5).items()):
                    pct = count / len(categorized_rows) * 100
                    
                    # Determine priority color
                    if any(term in cat for term in ['Quality', 'Defect', 'Medical']):
                        color = COLORS['danger']
                        priority = "High Priority"
                    elif any(term in cat for term in ['Performance', 'Missing', 'Wrong']):
                        color = COLORS['warning']
                        priority = "Medium Priority"
                    else:
                        color = COLORS['primary']
                        priority = "Low Priority"
                    
                    st.markdown(f"""
                    <div class="info-card">
                        <div style="display: flex; justify-content: space-between; align-items: center;">
                            <div>
                                <strong>{i+1}. {cat}</strong>
                                <div style="color: {color}; font-size: 0.9em;">{priority}</div>
                            </div>
                            <div style="text-align: right;">
                                <div style="font-size: 1.2em; font-weight: 600;">{count:,} returns</div>
                                <div style="color: {COLORS['neutral']};">{pct:.1f}%</div>
                            </div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
    
    # Export section
    st.markdown("---")
    st.markdown("### 💾 Export Your Results")
    
    if st.session_state.export_ready and st.session_state.export_data:
        file_type = 'Excel' if st.session_state.export_filename.endswith('.xlsx') else 'CSV'
        mime_type = 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet' if file_type == 'Excel' else 'text/csv'
        
        # Success message
        st.markdown(f"""
        <div class="processing-box" style="text-align: center;">
            <h3 style="color: {COLORS['success']}; margin: 0;">✅ Export Ready!</h3>
            <p style="margin: 0.5rem 0;">Your categorized data with AI classifications in Column K</p>
            <p style="color: {COLORS['primary']}; font-weight: 600;">
                📄 {st.session_state.export_filename}
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        # Download button
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.download_button(
                label=f"⬇️ Download {file_type} File",
                data=st.session_state.export_data,
                file_name=st.session_state.export_filename,
                mime=mime_type,
                use_container_width=True,
                type="primary"
            )
            
            st.info("""
            ✅ **File Features:**
            - Categories in Column K
            - Original structure preserved
            - Google Sheets compatible
            - Ready for pivot analysis
            """)
        
        # Next steps
        with st.expander("📋 Next Steps & Analysis Tips"):
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("""
                **Quick Analysis in Excel/Sheets:**
                1. Create pivot table with Column K
                2. Filter by category priority
                3. Group by ASIN and category
                4. Identify problem products
                
                **Quality Actions:**
                - Focus on 'Product Defects/Quality' category
                - Review high-volume ASINs
                - Track improvement over time
                """)
            
            with col2:
                st.markdown("""
                **Import to Google Sheets:**
                1. Open Google Sheets
                2. File → Import → Upload
                3. Select your downloaded file
                4. Use Column K for filtering
                
                **Automated Analysis:**
                - Set up category alerts
                - Create quality dashboards
                - Monitor return trends
                """)

def display_analysis_tab():
    """Display the analysis tab for deeper insights"""
    st.markdown("### 🔍 Deep Analysis")
    
    if not st.session_state.categorized_data is not None:
        st.info("Upload and categorize return data in the 'Categorize Returns' tab first.")
        return
    
    df = st.session_state.categorized_data
    
    # Analysis options
    col1, col2 = st.columns(2)
    
    with col1:
        analysis_type = st.selectbox(
            "Analysis Type",
            ["Overview", "Quality Focus", "Product Analysis", "Trend Analysis", "Export Report"]
        )
    
    with col2:
        if 'asin' in df.columns:
            asins = ['All'] + sorted(df['asin'].dropna().unique().tolist())
            selected_asin = st.selectbox("Filter by ASIN", asins)
        else:
            selected_asin = 'All'
    
    # Filter data
    if selected_asin != 'All' and 'asin' in df.columns:
        filtered_df = df[df['asin'] == selected_asin]
    else:
        filtered_df = df
    
    # Display selected analysis
    if analysis_type == "Overview":
        display_overview_analysis(filtered_df)
    elif analysis_type == "Quality Focus":
        display_quality_analysis(filtered_df)
    elif analysis_type == "Product Analysis":
        display_product_analysis(filtered_df)
    elif analysis_type == "Trend Analysis":
        display_trend_analysis(filtered_df)
    elif analysis_type == "Export Report":
        generate_comprehensive_report(filtered_df)
    
    # AI Chat section
    st.markdown("---")
    st.markdown("### 💬 AI Assistant")
    
    if st.button("💬 Ask AI about the analysis", use_container_width=True):
        st.session_state.show_ai_chat = True
    
    if st.session_state.show_ai_chat:
        display_ai_chat()

def display_overview_analysis(df):
    """Display overview analysis"""
    st.markdown("#### 📊 Return Analysis Overview")
    
    # Find category column (Column K)
    if len(df.columns) > 10:
        category_col = df.columns[10]
        
        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        
        total_returns = len(df)
        categorized = df[df[category_col].notna() & (df[category_col] != '')]
        
        with col1:
            st.metric("Total Returns", f"{total_returns:,}")
        
        with col2:
            if 'return-date' in df.columns:
                df['return-date'] = pd.to_datetime(df['return-date'], errors='coerce')
                date_range = df['return-date'].max() - df['return-date'].min()
                st.metric("Date Range", f"{date_range.days} days")
        
        with col3:
            unique_asins = df['asin'].nunique() if 'asin' in df.columns else 'N/A'
            st.metric("Unique ASINs", unique_asins)
        
        with col4:
            unique_skus = df['sku'].nunique() if 'sku' in df.columns else 'N/A'
            st.metric("Unique SKUs", unique_skus)
        
        # Category pie chart
        if not categorized.empty:
            category_counts = categorized[category_col].value_counts()
            
            fig = px.pie(
                values=category_counts.values,
                names=category_counts.index,
                title="Return Categories Distribution",
                hole=0.4
            )
            fig.update_traces(textposition='inside', textinfo='percent+label')
            st.plotly_chart(fig, use_container_width=True)

def display_quality_analysis(df):
    """Display quality-focused analysis"""
    st.markdown("#### 🔍 Quality Issue Analysis")
    
    if len(df.columns) > 10:
        category_col = df.columns[10]
        
        # Filter for quality-related categories
        quality_categories = [
            'Product Defects/Quality',
            'Performance/Effectiveness',
            'Missing Components',
            'Design/Material Issues',
            'Medical/Health Concerns'
        ]
        
        quality_df = df[df[category_col].isin(quality_categories)]
        
        if not quality_df.empty:
            # Quality metrics
            col1, col2, col3 = st.columns(3)
            
            with col1:
                quality_rate = len(quality_df) / len(df) * 100
                st.metric("Quality Issue Rate", f"{quality_rate:.1f}%",
                         delta=f"{len(quality_df):,} returns")
            
            with col2:
                if 'asin' in quality_df.columns:
                    affected_asins = quality_df['asin'].nunique()
                    st.metric("Affected ASINs", affected_asins)
            
            with col3:
                most_common = quality_df[category_col].value_counts().index[0]
                st.metric("Top Quality Issue", most_common)
            
            # Quality issues by category
            quality_breakdown = quality_df[category_col].value_counts()
            
            fig = px.bar(
                x=quality_breakdown.values,
                y=quality_breakdown.index,
                orientation='h',
                title="Quality Issues Breakdown",
                labels={'x': 'Number of Returns', 'y': 'Issue Type'},
                color=quality_breakdown.values,
                color_continuous_scale='Reds'
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Top affected products
            if 'asin' in quality_df.columns:
                st.markdown("##### Top Products with Quality Issues")
                
                asin_quality = quality_df.groupby('asin').agg({
                    category_col: 'count',
                    'sku': 'first'
                }).sort_values(category_col, ascending=False).head(10)
                
                asin_quality.columns = ['Quality Returns', 'SKU']
                st.dataframe(asin_quality, use_container_width=True)
        else:
            st.info("No quality-related issues found in the data.")

def display_product_analysis(df):
    """Display product-level analysis"""
    st.markdown("#### 📦 Product Return Analysis")
    
    if 'asin' not in df.columns:
        st.warning("No ASIN data available for product analysis")
        return
    
    # Product metrics
    product_stats = df.groupby('asin').agg({
        'order-id': 'count',
        'sku': 'first'
    }).sort_values('order-id', ascending=False)
    
    product_stats.columns = ['Return Count', 'SKU']
    
    # Add return rate if possible
    product_stats['Return Rate'] = 'N/A'  # Would need sales data
    
    # Top 20 products
    st.markdown("##### Top 20 Products by Return Volume")
    st.dataframe(product_stats.head(20), use_container_width=True)
    
    # Product return distribution
    fig = px.histogram(
        x=product_stats['Return Count'],
        nbins=30,
        title="Return Volume Distribution",
        labels={'x': 'Number of Returns', 'y': 'Number of Products'}
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # High-risk products
    high_risk_threshold = product_stats['Return Count'].quantile(0.9)
    high_risk_products = product_stats[product_stats['Return Count'] > high_risk_threshold]
    
    st.markdown(f"##### High-Risk Products (>{high_risk_threshold:.0f} returns)")
    st.dataframe(high_risk_products, use_container_width=True)

def display_trend_analysis(df):
    """Display trend analysis"""
    st.markdown("#### 📈 Return Trend Analysis")
    
    if 'return-date' not in df.columns:
        st.warning("No date information available for trend analysis")
        return
    
    # Parse dates
    df['return-date'] = pd.to_datetime(df['return-date'], errors='coerce')
    df_with_dates = df.dropna(subset=['return-date'])
    
    if df_with_dates.empty:
        st.warning("No valid date data for trend analysis")
        return
    
    # Daily returns
    daily_returns = df_with_dates.groupby(df_with_dates['return-date'].dt.date).size()
    
    fig = px.line(
        x=daily_returns.index,
        y=daily_returns.values,
        title="Daily Return Volume",
        labels={'x': 'Date', 'y': 'Number of Returns'}
    )
    fig.update_traces(mode='lines+markers')
    st.plotly_chart(fig, use_container_width=True)
    
    # Monthly trend
    df_with_dates['month'] = df_with_dates['return-date'].dt.to_period('M')
    monthly_returns = df_with_dates.groupby('month').size()
    
    # Category trends over time
    if len(df.columns) > 10:
        category_col = df.columns[10]
        
        # Monthly category breakdown
        monthly_categories = df_with_dates.groupby(['month', category_col]).size().unstack(fill_value=0)
        
        # Plot stacked area chart for top categories
        top_categories = df_with_dates[category_col].value_counts().head(5).index
        
        fig_data = []
        for cat in top_categories:
            if cat in monthly_categories.columns:
                fig_data.append(go.Scatter(
                    x=monthly_categories.index.astype(str),
                    y=monthly_categories[cat],
                    mode='lines',
                    name=cat,
                    stackgroup='one'
                ))
        
        fig = go.Figure(data=fig_data)
        fig.update_layout(
            title="Category Trends Over Time",
            xaxis_title="Month",
            yaxis_title="Number of Returns",
            hovermode='x unified'
        )
        st.plotly_chart(fig, use_container_width=True)

def generate_comprehensive_report(df):
    """Generate comprehensive analysis report"""
    st.markdown("#### 📄 Comprehensive Analysis Report")
    
    report_buffer = io.StringIO()
    
    # Report header
    report_buffer.write("AMAZON QUALITY ANALYSIS REPORT\n")
    report_buffer.write("="*50 + "\n")
    report_buffer.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    report_buffer.write(f"Total Returns Analyzed: {len(df):,}\n")
    
    if st.session_state.target_asin:
        report_buffer.write(f"Target ASIN: {st.session_state.target_asin}\n")
    
    report_buffer.write("\n")
    
    # Executive Summary
    report_buffer.write("EXECUTIVE SUMMARY\n")
    report_buffer.write("-"*30 + "\n")
    
    if len(df.columns) > 10:
        category_col = df.columns[10]
        categorized = df[df[category_col].notna() & (df[category_col] != '')]
        
        if not categorized.empty:
            category_counts = categorized[category_col].value_counts()
            
            # Quality issues
            quality_categories = ['Product Defects/Quality', 'Performance/Effectiveness', 
                                'Missing Components', 'Design/Material Issues', 'Medical/Health Concerns']
            quality_count = sum(category_counts.get(cat, 0) for cat in quality_categories)
            quality_rate = quality_count / len(categorized) * 100
            
            report_buffer.write(f"Quality Issue Rate: {quality_rate:.1f}% ({quality_count:,} returns)\n")
            report_buffer.write(f"Top Issue: {category_counts.index[0]} ({category_counts.iloc[0]:,} returns)\n")
            
            # Category breakdown
            report_buffer.write("\nCATEGORY BREAKDOWN\n")
            report_buffer.write("-"*30 + "\n")
            
            for cat, count in category_counts.items():
                pct = count / len(categorized) * 100
                report_buffer.write(f"{cat}: {count:,} ({pct:.1f}%)\n")
    
    # Product analysis
    if 'asin' in df.columns:
        report_buffer.write("\nTOP PRODUCTS BY RETURN VOLUME\n")
        report_buffer.write("-"*30 + "\n")
        
        product_counts = df['asin'].value_counts().head(10)
        for asin, count in product_counts.items():
            report_buffer.write(f"{asin}: {count:,} returns\n")
    
    # Get report content
    report_content = report_buffer.getvalue()
    
    # Display report
    st.text_area("Report Preview", report_content, height=400)
    
    # Download button
    st.download_button(
        label="📥 Download Full Report",
        data=report_content,
        file_name=f"quality_analysis_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
        mime="text/plain"
    )

def display_ai_chat():
    """AI chat interface"""
    # Chat container
    chat_container = st.container()
    
    # Display messages
    with chat_container:
        for message in st.session_state.chat_messages:
            with st.chat_message(message["role"]):
                st.write(message["content"])
    
    # Input
    if prompt := st.chat_input("Ask about your return analysis..."):
        # Add user message
        st.session_state.chat_messages.append({"role": "user", "content": prompt})
        
        # Get AI response
        try:
            with st.spinner("Thinking..."):
                # Build context
                context = {
                    'has_analysis': st.session_state.categorized_data is not None,
                    'current_asin': st.session_state.target_asin,
                    'total_returns': st.session_state.total_rows_processed,
                    'categories_found': []
                }
                
                # Add category information to context
                if st.session_state.categorized_data is not None and len(st.session_state.categorized_data.columns) > 10:
                    category_col = st.session_state.categorized_data.columns[10]
                    categories = st.session_state.categorized_data[category_col].value_counts().to_dict()
                    context['categories_found'] = categories
                
                response = st.session_state.ai_analyzer.generate_chat_response(
                    prompt, context
                )
                
                st.session_state.chat_messages.append({"role": "assistant", "content": response})
        except Exception as e:
            logger.error(f"Chat error: {e}")
            st.session_state.chat_messages.append({
                "role": "assistant", 
                "content": f"I encountered an error: {str(e)}. Please try rephrasing your question."
            })
        
        st.rerun()

def main():
    """Main application"""
    initialize_session_state()
    inject_professional_css()
    
    # Header
    display_header()
    
    # Sidebar
    with st.sidebar:
        st.markdown("### ⚙️ Settings")
        
        # AI Provider status
        if get_ai_status():
            providers = st.session_state.ai_analyzer.get_available_providers()
            st.success(f"✅ AI Ready ({', '.join(providers)})")
            
            # Show cost summary
            if st.session_state.total_cost > 0:
                st.markdown("#### 💰 Session Costs")
                st.metric("Total Cost", f"${st.session_state.total_cost:.4f}")
                st.metric("API Calls", f"{st.session_state.api_calls_made:,}")
        else:
            st.error("❌ AI Not Configured")
            with st.expander("Setup Instructions"):
                st.markdown("""
                Add to `.streamlit/secrets.toml`:
                ```
                openai_api_key = "sk-..."
                claude_api_key = "sk-ant-..."
                ```
                """)
        
        st.markdown("---")
        
        # Processing options
        st.markdown("### 🚀 Processing Options")
        
        st.session_state.batch_size = st.slider(
            "API Batch Size",
            min_value=10,
            max_value=100,
            value=st.session_state.batch_size,
            help="Number of items per API call"
        )
        
        st.session_state.chunk_size = st.select_slider(
            "Processing Chunk Size",
            options=[100, 250, 500, 1000],
            value=st.session_state.chunk_size,
            help="Process large files in chunks"
        )
        
        # Help
        st.markdown("---")
        with st.expander("📖 Quick Guide"):
            st.markdown("""
            **Categorization Tab:**
            1. Upload return files (FBA, PDF, CSV)
            2. Set target ASIN (optional)
            3. Click Categorize
            4. Download with Column K
            
            **Analysis Tab:**
            1. Complete categorization first
            2. Choose analysis type
            3. Filter by ASIN if needed
            4. Ask AI for insights
            
            **Tips:**
            - Column K contains AI categories
            - Use Excel for pivot analysis
            - Focus on quality categories
            """)
    
    # Main content with tabs
    tab1, tab2 = st.tabs(["📋 Categorize Returns", "🔍 Deep Analysis"])
    
    with tab1:
        display_categorization_tab()
    
    with tab2:
        display_analysis_tab()

if __name__ == "__main__":
    main()
