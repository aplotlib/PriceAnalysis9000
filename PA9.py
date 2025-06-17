"""
Amazon Quality Analysis Platform - Unified Edition
Version: 3.0 - All-in-One Tool for Medical Device Quality Management
Combines return categorization, PDF analysis, and comprehensive quality insights

Key Features:
- Medical device return categorization (15 Amazon categories)
- Critical injury/safety detection with alerts
- PDF processing from Amazon Seller Central
- FBA return report analysis
- Review correlation
- AI-powered insights with OpenAI + Claude
- Export to categorized Excel reports
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

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import our enhanced modules
try:
    from enhanced_ai_universal import (
        UniversalAIAnalyzer, FileAnalysis, ReturnCategorization,
        AIProvider, MEDICAL_DEVICE_CATEGORIES, CRITICAL_KEYWORDS
    )
    from universal_file_detector import UniversalFileDetector, ProcessedFile
    AI_AVAILABLE = True
except ImportError as e:
    logger.error(f"Module import error: {e}")
    AI_AVAILABLE = False

# Page configuration
st.set_page_config(
    page_title="Quality Analysis Platform - Vive Health",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Professional color scheme for medical/quality focus
COLORS = {
    'primary': '#1E3A8A',      # Deep blue
    'secondary': '#3B82F6',    # Bright blue
    'success': '#10B981',      # Green
    'warning': '#F59E0B',      # Amber
    'danger': '#EF4444',       # Red
    'critical': '#991B1B',     # Dark red
    'neutral': '#6B7280',      # Gray
    'light': '#F3F4F6',        # Light gray
    'dark': '#1F2937',         # Dark gray
    'background': '#FFFFFF'    # White
}

def inject_custom_css():
    """Professional CSS styling for quality management"""
    st.markdown(f"""
    <style>
    /* Import fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    /* Global styles */
    .stApp {{
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        background-color: {COLORS['background']};
    }}
    
    /* Main header */
    .main-header {{
        background: linear-gradient(135deg, {COLORS['primary']} 0%, {COLORS['secondary']} 100%);
        color: white;
        padding: 2rem;
        border-radius: 12px;
        margin-bottom: 2rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }}
    
    .main-header h1 {{
        margin: 0;
        font-size: 2.5rem;
        font-weight: 700;
        letter-spacing: -0.02em;
    }}
    
    .main-header p {{
        margin: 0.5rem 0 0 0;
        opacity: 0.9;
        font-size: 1.125rem;
    }}
    
    /* Critical alert box */
    .critical-alert {{
        background-color: #FEE2E2;
        border: 2px solid {COLORS['danger']};
        border-radius: 8px;
        padding: 1rem;
        margin: 1rem 0;
        animation: pulse 2s infinite;
    }}
    
    @keyframes pulse {{
        0% {{ opacity: 1; }}
        50% {{ opacity: 0.8; }}
        100% {{ opacity: 1; }}
    }}
    
    /* Info cards */
    .info-card {{
        background: white;
        border: 1px solid #E5E7EB;
        border-radius: 8px;
        padding: 1.5rem;
        margin-bottom: 1rem;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
        transition: all 0.2s;
    }}
    
    .info-card:hover {{
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        transform: translateY(-1px);
    }}
    
    .info-card h3 {{
        color: {COLORS['primary']};
        margin-top: 0;
        font-weight: 600;
    }}
    
    /* Metric cards */
    .metric-card {{
        background: white;
        border: 1px solid #E5E7EB;
        border-radius: 8px;
        padding: 1.25rem;
        text-align: center;
        height: 100%;
    }}
    
    .metric-value {{
        font-size: 2.5rem;
        font-weight: 700;
        color: {COLORS['primary']};
        line-height: 1;
        margin-bottom: 0.5rem;
    }}
    
    .metric-label {{
        color: {COLORS['neutral']};
        font-size: 0.875rem;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }}
    
    /* Category badges */
    .category-badge {{
        display: inline-block;
        padding: 0.375rem 0.75rem;
        border-radius: 9999px;
        font-size: 0.875rem;
        font-weight: 500;
        margin: 0.25rem;
    }}
    
    .severity-critical {{
        background-color: #FEE2E2;
        color: {COLORS['critical']};
        border: 1px solid {COLORS['danger']};
    }}
    
    .severity-high {{
        background-color: #FEF3C7;
        color: #92400E;
        border: 1px solid {COLORS['warning']};
    }}
    
    .severity-medium {{
        background-color: #DBEAFE;
        color: #1E40AF;
        border: 1px solid {COLORS['secondary']};
    }}
    
    .severity-low {{
        background-color: {COLORS['light']};
        color: {COLORS['neutral']};
        border: 1px solid #E5E7EB;
    }}
    
    /* File upload area */
    .upload-area {{
        border: 2px dashed #CBD5E1;
        border-radius: 12px;
        padding: 3rem 2rem;
        text-align: center;
        background-color: #F9FAFB;
        transition: all 0.3s;
    }}
    
    .upload-area:hover {{
        border-color: {COLORS['secondary']};
        background-color: #EFF6FF;
    }}
    
    /* Progress indicators */
    .progress-step {{
        display: flex;
        align-items: center;
        padding: 1rem;
        border-radius: 8px;
        margin-bottom: 0.5rem;
        background: white;
        border: 1px solid #E5E7EB;
    }}
    
    .progress-step.active {{
        background: #EFF6FF;
        border-color: {COLORS['secondary']};
    }}
    
    .progress-step.completed {{
        background: #D1FAE5;
        border-color: {COLORS['success']};
    }}
    
    /* Buttons */
    .stButton > button {{
        background: {COLORS['primary']};
        color: white;
        border: none;
        padding: 0.625rem 1.25rem;
        border-radius: 6px;
        font-weight: 500;
        transition: all 0.2s;
    }}
    
    .stButton > button:hover {{
        background: {COLORS['secondary']};
        transform: translateY(-1px);
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }}
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {{
        gap: 0.5rem;
        background-color: #F3F4F6;
        padding: 0.25rem;
        border-radius: 8px;
    }}
    
    .stTabs [data-baseweb="tab"] {{
        height: 48px;
        padding: 0 1.5rem;
        background-color: transparent;
        border-radius: 6px;
        color: {COLORS['neutral']};
        font-weight: 500;
    }}
    
    .stTabs [data-baseweb="tab"][aria-selected="true"] {{
        background-color: white;
        color: {COLORS['primary']};
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
    }}
    
    /* Hide Streamlit branding */
    #MainMenu {{visibility: hidden;}}
    footer {{visibility: hidden;}}
    
    /* Data tables */
    .dataframe {{
        font-size: 0.875rem;
    }}
    
    /* Success animation */
    @keyframes slideIn {{
        from {{
            transform: translateY(-10px);
            opacity: 0;
        }}
        to {{
            transform: translateY(0);
            opacity: 1;
        }}
    }}
    
    .success-message {{
        animation: slideIn 0.3s ease-out;
    }}
    </style>
    """, unsafe_allow_html=True)

def initialize_session_state():
    """Initialize session state variables"""
    defaults = {
        # Core data
        'uploaded_files': [],
        'processed_files': [],
        'categorized_data': {},
        'current_analysis': None,
        'target_asin': '',
        
        # UI state
        'current_tab': 0,
        'processing_status': 'idle',
        'show_ai_chat': False,
        
        # Analysis results
        'critical_issues': [],
        'quality_metrics': {},
        'recommendations': [],
        
        # AI components
        'ai_analyzer': None,
        'ai_provider': AIProvider.FASTEST,
        'chat_messages': [],
        
        # Processing settings
        'batch_size': 50,
        'auto_categorize': True,
        'highlight_critical': True,
        
        # Cost tracking
        'total_cost': 0.0,
        'api_calls': 0,
        'processing_time': 0.0,
        
        # Export data
        'export_ready': False,
        'export_data': None
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

def display_header():
    """Display application header"""
    st.markdown("""
    <div class="main-header">
        <h1>🏥 Quality Analysis Platform</h1>
        <p>Medical Device Return Analysis & Quality Management System</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Critical issues alert
    if st.session_state.critical_issues:
        count = len(st.session_state.critical_issues)
        st.markdown(f"""
        <div class="critical-alert">
            <h3>⚠️ CRITICAL ALERT: {count} Safety/Injury Issues Detected</h3>
            <p>Immediate attention required - see Critical Issues tab for details</p>
        </div>
        """, unsafe_allow_html=True)

def get_ai_analyzer():
    """Get or create AI analyzer with medical device focus"""
    if st.session_state.ai_analyzer is None and AI_AVAILABLE:
        try:
            st.session_state.ai_analyzer = UniversalAIAnalyzer(st.session_state.ai_provider)
            logger.info(f"AI analyzer initialized with provider: {st.session_state.ai_provider.value}")
        except Exception as e:
            logger.error(f"Failed to initialize AI analyzer: {e}")
            st.error(f"AI initialization error: {str(e)}")
    
    return st.session_state.ai_analyzer

def check_api_status():
    """Check API configuration and availability"""
    if not AI_AVAILABLE:
        return {
            'status': 'error',
            'message': 'AI modules not available',
            'providers': []
        }
    
    analyzer = get_ai_analyzer()
    if analyzer:
        providers = analyzer.get_available_providers()
        if providers:
            return {
                'status': 'ready',
                'message': f"AI Ready ({', '.join(providers)})",
                'providers': providers
            }
        else:
            return {
                'status': 'not_configured',
                'message': 'No API keys configured',
                'providers': []
            }
    
    return {
        'status': 'error',
        'message': 'AI analyzer not initialized',
        'providers': []
    }

def display_sidebar():
    """Enhanced sidebar with quality focus"""
    with st.sidebar:
        st.markdown("### ⚙️ Configuration")
        
        # API Status
        api_status = check_api_status()
        if api_status['status'] == 'ready':
            st.success(f"✅ {api_status['message']}")
        else:
            st.error(f"❌ {api_status['message']}")
            with st.expander("Setup Instructions"):
                st.markdown("""
                Add to `.streamlit/secrets.toml`:
                ```toml
                openai_api_key = "sk-..."
                anthropic_api_key = "sk-ant-..."
                ```
                """)
        
        st.markdown("---")
        
        # AI Provider Selection
        st.markdown("### 🤖 AI Settings")
        provider_options = {
            'Fastest (Recommended)': AIProvider.FASTEST,
            'Most Accurate': AIProvider.ACCURATE,
            'OpenAI Only': AIProvider.OPENAI,
            'Claude Only': AIProvider.CLAUDE,
            'Both (Consensus)': AIProvider.BOTH
        }
        
        selected_provider = st.selectbox(
            "AI Provider",
            options=list(provider_options.keys()),
            index=0,
            help="Choose AI model for categorization"
        )
        st.session_state.ai_provider = provider_options[selected_provider]
        
        # Processing Settings
        st.markdown("---")
        st.markdown("### 🚀 Processing Settings")
        
        st.session_state.batch_size = st.slider(
            "Batch Size",
            min_value=10,
            max_value=100,
            value=50,
            step=10,
            help="Number of items to process at once"
        )
        
        st.session_state.auto_categorize = st.checkbox(
            "Auto-categorize on upload",
            value=True,
            help="Automatically categorize returns when files are uploaded"
        )
        
        st.session_state.highlight_critical = st.checkbox(
            "Highlight critical issues",
            value=True,
            help="Visually highlight safety and injury-related returns"
        )
        
        # Target ASIN
        st.markdown("---")
        st.markdown("### 🎯 Target Product")
        
        asin_input = st.text_input(
            "Target ASIN (Optional)",
            value=st.session_state.target_asin,
            placeholder="B00EXAMPLE",
            help="Filter analysis to specific ASIN"
        )
        
        if asin_input and len(asin_input) == 10:
            st.session_state.target_asin = asin_input.upper()
            st.success(f"Filtering for: {st.session_state.target_asin}")
        
        # Session Statistics
        st.markdown("---")
        st.markdown("### 📊 Session Statistics")
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Files Processed", len(st.session_state.processed_files))
            st.metric("API Calls", st.session_state.api_calls)
        
        with col2:
            st.metric("Total Cost", f"${st.session_state.total_cost:.4f}")
            if st.session_state.processing_time > 0:
                st.metric("Process Time", f"{st.session_state.processing_time:.1f}s")
        
        # Quick Actions
        st.markdown("---")
        if st.button("🗑️ Clear All Data", use_container_width=True):
            for key in ['uploaded_files', 'processed_files', 'categorized_data', 
                       'critical_issues', 'quality_metrics', 'recommendations']:
                st.session_state[key] = [] if key != 'categorized_data' else {}
            st.rerun()
        
        # Help
        st.markdown("---")
        with st.expander("📖 Quick Guide"):
            st.markdown("""
            **1. Upload Files**
            - PDF: Print from Manage Returns page
            - TXT: FBA return reports
            - Excel/CSV: Any format
            
            **2. Auto-Categorization**
            - Uses 15 medical device categories
            - Flags injuries & safety issues
            - Pattern + AI hybrid approach
            
            **3. Review Results**
            - Check Critical Issues tab first
            - Review categorization accuracy
            - Export for quality meetings
            
            **4. Take Action**
            - Follow recommendations
            - Track quality metrics
            - Monitor trends
            """)

def display_upload_tab():
    """File upload interface with drag-and-drop"""
    st.markdown("### 📤 Upload Return Data")
    
    # Instructions
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        <div class="info-card">
            <h3>Supported File Types</h3>
            <ul>
                <li><strong>PDF</strong>: Amazon Seller Central returns (print as PDF)</li>
                <li><strong>TXT</strong>: FBA return reports (tab-delimited)</li>
                <li><strong>Excel/CSV</strong>: Helium 10 reviews, custom exports</li>
                <li><strong>Images</strong>: Screenshots (JPG/PNG)</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="info-card">
            <h3>Quick Tips</h3>
            <ul>
                <li>🎯 Set Target ASIN for filtering</li>
                <li>⚡ Batch upload multiple files</li>
                <li>🔍 Auto-detects file types</li>
                <li>⚠️ Highlights critical issues</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    # Upload area
    st.markdown("""
    <div class="upload-area">
        <h2>📁 Drop files here or click to browse</h2>
        <p>Upload Amazon return reports, FBA exports, or review files</p>
    </div>
    """, unsafe_allow_html=True)
    
    uploaded_files = st.file_uploader(
        "Choose files",
        type=['pdf', 'csv', 'tsv', 'txt', 'xlsx', 'xls', 'jpg', 'jpeg', 'png'],
        accept_multiple_files=True,
        label_visibility="collapsed"
    )
    
    if uploaded_files:
        process_uploaded_files(uploaded_files)

def process_uploaded_files(files):
    """Process uploaded files with progress tracking"""
    progress_container = st.container()
    
    with progress_container:
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Process each file
        new_files = []
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
                
                # If needs AI analysis (PDF/Image)
                if processed.extraction_method in ['ai_required', 'vision_ai_required']:
                    analyzer = get_ai_analyzer()
                    if analyzer:
                        with st.spinner(f"AI analyzing {file.name}..."):
                            analysis = asyncio.run(analyzer.analyze_file(
                                file_content, file.name, processed.format
                            ))
                            
                            # Update processed file with AI results
                            if analysis.extracted_data:
                                if analysis.content_type == 'returns':
                                    processed.data = pd.DataFrame(analysis.extracted_data.get('returns', []))
                                    processed.content_category = 'returns'
                                    processed.critical_issues = analysis.critical_issues
                                    processed.metadata['ai_analysis'] = analysis.extracted_data
                
                st.session_state.processed_files.append(processed)
                new_files.append(processed)
                
                # Show result
                if processed.warnings:
                    st.warning(f"⚠️ {file.name}: {', '.join(processed.warnings)}")
                elif processed.critical_issues:
                    st.error(f"🚨 {file.name}: Found {len(processed.critical_issues)} critical issues!")
                else:
                    st.success(f"✅ {file.name} processed successfully")
                    
            except Exception as e:
                st.error(f"❌ Error processing {file.name}: {str(e)}")
                logger.error(f"File processing error: {e}", exc_info=True)
        
        progress_bar.empty()
        status_text.empty()
    
    # Auto-categorize if enabled
    if st.session_state.auto_categorize and new_files:
        categorize_returns(new_files)

def categorize_returns(files: List[ProcessedFile] = None):
    """Categorize returns using medical device categories"""
    if files is None:
        files = st.session_state.processed_files
    
    return_files = [f for f in files if f.content_category in ['returns', 'fba_returns'] and f.data is not None]
    
    if not return_files:
        st.warning("No return files to categorize")
        return
    
    analyzer = get_ai_analyzer()
    if not analyzer:
        st.error("AI analyzer not available")
        return
    
    # Process each file
    total_categorized = 0
    start_time = time.time()
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for file_idx, file in enumerate(return_files):
        df = file.data
        
        # Extract returns data
        returns_to_categorize = []
        
        # Handle different file formats
        if file.content_category == 'fba_returns':
            for idx, row in df.iterrows():
                returns_to_categorize.append({
                    'index': idx,
                    'reason': str(row.get('reason', '')),
                    'comment': str(row.get('customer_comments', '')),
                    'order_id': str(row.get('order_id', '')),
                    'sku': str(row.get('sku', '')),
                    'asin': str(row.get('asin', ''))
                })
        else:
            # Generic return format
            for idx, row in df.iterrows():
                returns_to_categorize.append({
                    'index': idx,
                    'reason': str(row.get('return_reason', row.get('reason', ''))),
                    'comment': str(row.get('customer_comment', row.get('comment', ''))),
                    'order_id': str(row.get('order_id', '')),
                    'sku': str(row.get('sku', '')),
                    'asin': str(row.get('asin', ''))
                })
        
        # Process in batches
        batch_size = st.session_state.batch_size
        all_results = []
        
        for i in range(0, len(returns_to_categorize), batch_size):
            batch = returns_to_categorize[i:i + batch_size]
            status_text.text(f"Categorizing {file.metadata.get('filename')}: {i}/{len(returns_to_categorize)}")
            
            # Categorize batch
            batch_results = asyncio.run(analyzer.process_batch_returns(batch, batch_size))
            all_results.extend(batch_results)
            
            # Update progress
            progress = (file_idx + (i + len(batch)) / len(returns_to_categorize)) / len(return_files)
            progress_bar.progress(progress)
        
        # Update dataframe with categories
        for result in all_results:
            idx = result['index']
            df.at[idx, 'category'] = result.get('category', 'Other/Miscellaneous')
            df.at[idx, 'severity'] = result.get('severity', 'low')
            df.at[idx, 'confidence'] = result.get('confidence', 0.0)
            
            # Check for critical flags
            if result.get('critical_flags'):
                st.session_state.critical_issues.append({
                    'file': file.metadata.get('filename'),
                    'order_id': result.get('order_id'),
                    'flags': result.get('critical_flags'),
                    'reason': result.get('reason'),
                    'comment': result.get('comment')
                })
        
        # Store categorized data
        file_key = file.metadata.get('filename', f'file_{file_idx}')
        st.session_state.categorized_data[file_key] = df
        total_categorized += len(all_results)
    
    # Clear progress
    progress_bar.empty()
    status_text.empty()
    
    # Update metrics
    st.session_state.processing_time = time.time() - start_time
    
    # Get cost summary
    if analyzer:
        cost_summary = analyzer.get_api_usage_summary()
        st.session_state.total_cost = cost_summary.get('total_cost', 0)
        st.session_state.api_calls = cost_summary.get('api_calls', 0)
    
    # Show summary
    st.success(f"""
    ✅ Categorization Complete!
    - Processed: {total_categorized} returns
    - Time: {st.session_state.processing_time:.1f} seconds
    - Cost: ${st.session_state.total_cost:.4f}
    - Critical Issues: {len(st.session_state.critical_issues)}
    """)

def display_analysis_tab():
    """Display comprehensive analysis results"""
    if not st.session_state.categorized_data:
        st.info("No categorized data available. Please upload and process files first.")
        return
    
    # Combine all categorized data
    all_returns = []
    for filename, df in st.session_state.categorized_data.items():
        df_copy = df.copy()
        df_copy['source_file'] = filename
        all_returns.append(df_copy)
    
    combined_df = pd.concat(all_returns, ignore_index=True)
    
    # Summary metrics
    st.markdown("### 📊 Quality Analysis Summary")
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{len(combined_df):,}</div>
            <div class="metric-label">Total Returns</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        critical_count = len(st.session_state.critical_issues)
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value" style="color: {COLORS['danger']};">{critical_count}</div>
            <div class="metric-label">Critical Issues</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        quality_defects = len(combined_df[combined_df['category'] == 'Product Defects/Quality'])
        defect_rate = (quality_defects / len(combined_df) * 100) if len(combined_df) > 0 else 0
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{defect_rate:.1f}%</div>
            <div class="metric-label">Quality Defect Rate</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        unique_skus = combined_df['sku'].nunique() if 'sku' in combined_df.columns else 0
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{unique_skus}</div>
            <div class="metric-label">Unique SKUs</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col5:
        avg_confidence = combined_df['confidence'].mean() if 'confidence' in combined_df.columns else 0
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{avg_confidence:.0%}</div>
            <div class="metric-label">Avg Confidence</div>
        </div>
        """, unsafe_allow_html=True)
    
    # Category breakdown
    st.markdown("---")
    st.markdown("### 📈 Return Categories Distribution")
    
    if 'category' in combined_df.columns:
        category_counts = combined_df['category'].value_counts()
        
        # Create visualizations
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Bar chart
            fig = px.bar(
                x=category_counts.values,
                y=category_counts.index,
                orientation='h',
                labels={'x': 'Count', 'y': 'Category'},
                title='Returns by Category',
                color=category_counts.values,
                color_continuous_scale='Reds'
            )
            fig.update_layout(height=500, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Top issues summary
            st.markdown("#### 🎯 Top Issues")
            
            for i, (cat, count) in enumerate(category_counts.head(5).items()):
                pct = (count / len(combined_df) * 100)
                severity = MEDICAL_DEVICE_CATEGORIES.get(cat, {}).get('priority', 'low')
                
                severity_class = f"severity-{severity}"
                st.markdown(f"""
                <div class="info-card">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <span><strong>{i+1}. {cat}</strong></span>
                        <span class="category-badge {severity_class}">{severity}</span>
                    </div>
                    <div style="margin-top: 0.5rem;">
                        <div style="background: #E5E7EB; height: 20px; border-radius: 10px;">
                            <div style="background: {COLORS['secondary']}; width: {pct}%; height: 100%; border-radius: 10px;"></div>
                        </div>
                        <small>{count} returns ({pct:.1f}%)</small>
                    </div>
                </div>
                """, unsafe_allow_html=True)
    
    # Product analysis
    if 'sku' in combined_df.columns:
        st.markdown("---")
        st.markdown("### 📦 Product-Level Analysis")
        
        # Top problematic products
        product_issues = combined_df.groupby('sku').agg({
            'category': 'count',
            'severity': lambda x: (x == 'critical').sum()
        }).rename(columns={'category': 'total_returns', 'severity': 'critical_issues'})
        
        product_issues = product_issues.sort_values('total_returns', ascending=False).head(10)
        
        # Display as table
        st.dataframe(
            product_issues.style.format({
                'total_returns': '{:,.0f}',
                'critical_issues': '{:,.0f}'
            }).background_gradient(cmap='Reds', subset=['total_returns', 'critical_issues']),
            use_container_width=True
        )
    
    # Time analysis if dates available
    date_columns = [col for col in combined_df.columns if 'date' in col.lower()]
    if date_columns:
        st.markdown("---")
        st.markdown("### 📅 Temporal Analysis")
        
        date_col = date_columns[0]
        combined_df['parsed_date'] = pd.to_datetime(combined_df[date_col], errors='coerce')
        
        # Group by month
        monthly_returns = combined_df.groupby(combined_df['parsed_date'].dt.to_period('M')).size()
        
        if not monthly_returns.empty:
            fig = px.line(
                x=monthly_returns.index.astype(str),
                y=monthly_returns.values,
                labels={'x': 'Month', 'y': 'Return Count'},
                title='Return Trend Over Time'
            )
            st.plotly_chart(fig, use_container_width=True)

def display_critical_issues_tab():
    """Display critical safety and injury issues"""
    st.markdown("### 🚨 Critical Issues Requiring Immediate Attention")
    
    if not st.session_state.critical_issues:
        st.success("✅ No critical safety or injury issues detected")
        return
    
    # Summary
    st.error(f"""
    ⚠️ **Found {len(st.session_state.critical_issues)} critical issues that may involve:**
    - Customer injuries or safety hazards
    - Medical emergencies or hospital visits
    - Severe product defects that could cause harm
    """)
    
    # Group by type
    issues_by_type = defaultdict(list)
    for issue in st.session_state.critical_issues:
        for flag in issue.get('flags', []):
            issues_by_type[flag].append(issue)
    
    # Display by type
    for issue_type, issues in issues_by_type.items():
        st.markdown(f"#### {issue_type.replace('_', ' ').title()} ({len(issues)} cases)")
        
        for issue in issues[:10]:  # Show first 10
            st.markdown(f"""
            <div class="critical-alert">
                <strong>Order ID:</strong> {issue.get('order_id', 'Unknown')}<br>
                <strong>File:</strong> {issue.get('file', 'Unknown')}<br>
                <strong>Reason:</strong> {issue.get('reason', 'N/A')}<br>
                <strong>Customer Comment:</strong> {issue.get('comment', 'N/A')}
            </div>
            """, unsafe_allow_html=True)
        
        if len(issues) > 10:
            st.info(f"... and {len(issues) - 10} more {issue_type} issues")

def display_recommendations_tab():
    """Display actionable recommendations"""
    st.markdown("### 💡 Quality Improvement Recommendations")
    
    if not st.session_state.categorized_data:
        st.info("Process files to generate recommendations")
        return
    
    # Generate recommendations based on analysis
    analyzer = get_ai_analyzer()
    if analyzer:
        # Get all processed files
        analysis_results = []
        for file in st.session_state.processed_files:
            if file.content_category in ['returns', 'fba_returns']:
                analysis_results.append(FileAnalysis(
                    file_type=file.file_type,
                    content_type=file.content_category,
                    extracted_data={'returns': file.data.to_dict('records') if file.data is not None else []},
                    confidence=file.confidence,
                    ai_provider='hybrid',
                    critical_issues=file.critical_issues
                ))
        
        # Generate quality report
        report = analyzer.generate_quality_report(analysis_results)
        recommendations = report.get('recommendations', [])
        
        if recommendations:
            for idx, rec in enumerate(recommendations):
                priority = rec.get('priority', 'MEDIUM')
                priority_color = {
                    'IMMEDIATE': COLORS['danger'],
                    'HIGH': COLORS['warning'],
                    'MEDIUM': COLORS['secondary'],
                    'LOW': COLORS['neutral']
                }.get(priority, COLORS['neutral'])
                
                st.markdown(f"""
                <div class="info-card">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <h3>{idx + 1}. {rec.get('recommendation', 'Action needed')}</h3>
                        <span class="category-badge" style="background: {priority_color}20; color: {priority_color}; border-color: {priority_color};">
                            {priority}
                        </span>
                    </div>
                    <p><strong>Category:</strong> {rec.get('category', 'General')}</p>
                    <p>{rec.get('details', '')}</p>
                    <h4>Action Items:</h4>
                    <ul>
                """, unsafe_allow_html=True)
                
                for action in rec.get('action_items', []):
                    st.markdown(f"<li>{action}</li>", unsafe_allow_html=True)
                
                st.markdown("</ul></div>", unsafe_allow_html=True)
        else:
            # Fallback recommendations
            st.markdown("""
            <div class="info-card">
                <h3>1. Review Critical Issues</h3>
                <p>Check the Critical Issues tab for any safety or injury-related returns</p>
            </div>
            
            <div class="info-card">
                <h3>2. Address Quality Defects</h3>
                <p>Focus on products with highest defect rates</p>
            </div>
            
            <div class="info-card">
                <h3>3. Improve Product Information</h3>
                <p>Update sizing guides and product descriptions for items with fit issues</p>
            </div>
            """, unsafe_allow_html=True)

def display_export_tab():
    """Export functionality with multiple formats"""
    st.markdown("### 💾 Export Analysis Results")
    
    if not st.session_state.categorized_data:
        st.info("No data to export. Process files first.")
        return
    
    # Export options
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="info-card">
            <h3>📊 Excel Export</h3>
            <p>Comprehensive report with multiple sheets:</p>
            <ul>
                <li>Summary statistics</li>
                <li>Categorized returns</li>
                <li>Critical issues</li>
                <li>Product analysis</li>
                <li>Recommendations</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("Generate Excel Report", use_container_width=True):
            excel_data = generate_excel_report()
            st.download_button(
                label="📥 Download Excel Report",
                data=excel_data,
                file_name=f"quality_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True
            )
    
    with col2:
        st.markdown("""
        <div class="info-card">
            <h3>📄 CSV Export</h3>
            <p>Simple categorized data export:</p>
            <ul>
                <li>All returns with categories</li>
                <li>Severity levels</li>
                <li>Confidence scores</li>
                <li>Critical flags</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("Generate CSV Export", use_container_width=True):
            csv_data = generate_csv_export()
            st.download_button(
                label="📥 Download CSV",
                data=csv_data,
                file_name=f"categorized_returns_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                use_container_width=True
            )

def generate_excel_report():
    """Generate comprehensive Excel report"""
    output = io.BytesIO()
    
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        workbook = writer.book
        
        # Define formats
        header_format = workbook.add_format({
            'bold': True,
            'bg_color': '#1E3A8A',
            'font_color': 'white',
            'align': 'center'
        })
        
        critical_format = workbook.add_format({
            'bg_color': '#FEE2E2',
            'font_color': '#991B1B'
        })
        
        # 1. Summary sheet
        summary_data = {
            'Metric': [
                'Total Returns',
                'Critical Issues',
                'Quality Defect Rate',
                'Files Processed',
                'Processing Cost',
                'Processing Time'
            ],
            'Value': [
                sum(len(df) for df in st.session_state.categorized_data.values()),
                len(st.session_state.critical_issues),
                f"{calculate_quality_defect_rate():.1f}%",
                len(st.session_state.processed_files),
                f"${st.session_state.total_cost:.4f}",
                f"{st.session_state.processing_time:.1f} seconds"
            ]
        }
        
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_excel(writer, sheet_name='Summary', index=False)
        
        # Format summary sheet
        worksheet = writer.sheets['Summary']
        for col_num, value in enumerate(summary_df.columns.values):
            worksheet.write(0, col_num, value, header_format)
        
        # 2. Categorized Returns sheet
        all_returns = []
        for filename, df in st.session_state.categorized_data.items():
            df_copy = df.copy()
            df_copy['source_file'] = filename
            all_returns.append(df_copy)
        
        if all_returns:
            combined_df = pd.concat(all_returns, ignore_index=True)
            
            # Reorder columns for better readability
            priority_cols = ['category', 'severity', 'order_id', 'asin', 'sku', 
                           'reason', 'customer_comments', 'source_file']
            other_cols = [col for col in combined_df.columns if col not in priority_cols]
            ordered_cols = [col for col in priority_cols if col in combined_df.columns] + other_cols
            
            combined_df = combined_df[ordered_cols]
            combined_df.to_excel(writer, sheet_name='Categorized Returns', index=False)
            
            # Apply conditional formatting
            worksheet = writer.sheets['Categorized Returns']
            for col_num, value in enumerate(combined_df.columns.values):
                worksheet.write(0, col_num, value, header_format)
            
            # Highlight critical rows
            if 'severity' in combined_df.columns:
                severity_col = combined_df.columns.get_loc('severity')
                for row_num, severity in enumerate(combined_df['severity'], 1):
                    if severity == 'critical':
                        worksheet.set_row(row_num, None, critical_format)
        
        # 3. Critical Issues sheet
        if st.session_state.critical_issues:
            critical_df = pd.DataFrame(st.session_state.critical_issues)
            critical_df.to_excel(writer, sheet_name='Critical Issues', index=False)
            
            worksheet = writer.sheets['Critical Issues']
            for col_num, value in enumerate(critical_df.columns.values):
                worksheet.write(0, col_num, value, header_format)
        
        # 4. Category Analysis sheet
        if all_returns:
            category_summary = combined_df['category'].value_counts().reset_index()
            category_summary.columns = ['Category', 'Count']
            category_summary['Percentage'] = (category_summary['Count'] / len(combined_df) * 100).round(1)
            
            # Add severity
            category_summary['Priority'] = category_summary['Category'].apply(
                lambda x: MEDICAL_DEVICE_CATEGORIES.get(x, {}).get('priority', 'low')
            )
            
            category_summary.to_excel(writer, sheet_name='Category Analysis', index=False)
            
            worksheet = writer.sheets['Category Analysis']
            for col_num, value in enumerate(category_summary.columns.values):
                worksheet.write(0, col_num, value, header_format)
    
    output.seek(0)
    return output.getvalue()

def generate_csv_export():
    """Generate simple CSV export"""
    all_returns = []
    for filename, df in st.session_state.categorized_data.items():
        df_copy = df.copy()
        df_copy['source_file'] = filename
        all_returns.append(df_copy)
    
    if all_returns:
        combined_df = pd.concat(all_returns, ignore_index=True)
        
        # Add critical flags
        combined_df['is_critical'] = combined_df.apply(
            lambda row: any(
                issue.get('order_id') == str(row.get('order_id', ''))
                for issue in st.session_state.critical_issues
            ), axis=1
        )
        
        return combined_df.to_csv(index=False)
    
    return ""

def calculate_quality_defect_rate():
    """Calculate overall quality defect rate"""
    total_returns = 0
    quality_defects = 0
    
    for df in st.session_state.categorized_data.values():
        total_returns += len(df)
        if 'category' in df.columns:
            quality_defects += len(df[df['category'] == 'Product Defects/Quality'])
    
    return (quality_defects / total_returns * 100) if total_returns > 0 else 0

def display_ai_chat_tab():
    """AI chat interface for quality insights"""
    st.markdown("### 💬 AI Quality Assistant")
    
    analyzer = get_ai_analyzer()
    if not analyzer:
        st.warning("AI not available. Please configure API keys.")
        return
    
    # Chat interface
    for message in st.session_state.chat_messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])
    
    # Input
    if prompt := st.chat_input("Ask about quality issues, trends, or get recommendations..."):
        # Add user message
        st.session_state.chat_messages.append({"role": "user", "content": prompt})
        
        # Get context
        context = {
            'has_analysis': bool(st.session_state.categorized_data),
            'critical_issues': len(st.session_state.critical_issues),
            'total_returns': sum(len(df) for df in st.session_state.categorized_data.values()),
            'quality_defect_rate': calculate_quality_defect_rate()
        }
        
        # Get AI response
        with st.spinner("Analyzing..."):
            response = asyncio.run(analyzer.generate_chat_response(prompt, context))
            st.session_state.chat_messages.append({"role": "assistant", "content": response})
        
        st.rerun()

def main():
    """Main application"""
    initialize_session_state()
    inject_custom_css()
    
    # Header
    display_header()
    
    # Sidebar
    display_sidebar()
    
    # Main content - Tabs
    tabs = st.tabs([
        "📤 Upload Files",
        "📊 Analysis",
        "🚨 Critical Issues",
        "💡 Recommendations", 
        "💾 Export",
        "💬 AI Assistant"
    ])
    
    with tabs[0]:
        display_upload_tab()
    
    with tabs[1]:
        display_analysis_tab()
    
    with tabs[2]:
        display_critical_issues_tab()
    
    with tabs[3]:
        display_recommendations_tab()
    
    with tabs[4]:
        display_export_tab()
    
    with tabs[5]:
        display_ai_chat_tab()

if __name__ == "__main__":
    main()
