"""
Amazon Quality Analysis Platform
Version: 2.0 - Universal File Support with AI Intelligence
Designed for Quality Analysts to quickly analyze returns and reviews
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
import importlib
import sys

# Import our modules
from enhanced_ai_universal import UniversalAIAnalyzer, FileAnalysis
from universal_file_detector import UniversalFileDetector, ProcessedFile

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
    'dark': '#212529'          # Dark
}

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
    
    .status-medium {{
        background: #E3F2FD;
        color: {COLORS['primary']};
        border: 1px solid {COLORS['primary']};
    }}
    
    .status-low {{
        background: #F3F4F6;
        color: {COLORS['neutral']};
        border: 1px solid {COLORS['neutral']};
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
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {{
        gap: 2rem;
    }}
    
    .stTabs [data-baseweb="tab"] {{
        height: 50px;
        padding: 0 2rem;
        font-weight: 500;
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
        
        # UI state
        'current_tab': 'upload',
        'show_ai_chat': False,
        'analysis_complete': False,
        
        # AI components
        'ai_analyzer': None,
        'chat_messages': [],
        
        # Settings
        'auto_analyze': True,
        'combine_sources': True,
        
        # Cache
        'analysis_cache': {},
        
        # Track initialization
        'ai_initialized': False
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

def display_header():
    """Display application header"""
    st.markdown("""
    <div class="main-header">
        <h1>🔍 Quality Analysis Platform</h1>
        <p>Intelligent Return & Review Analysis for Amazon Products</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Quick stats
    if st.session_state.processed_files:
        col1, col2, col3, col4 = st.columns(4)
        
        total_returns = sum(
            f.metadata.get('row_count', 0) 
            for f in st.session_state.processed_files 
            if f.content_category == 'returns'
        )
        
        with col1:
            st.metric("Files Loaded", len(st.session_state.processed_files))
        with col2:
            st.metric("Total Returns", f"{total_returns:,}")
        with col3:
            if st.session_state.target_asin:
                st.metric("Target ASIN", st.session_state.target_asin)
        with col4:
            ai_status = "🟢 Ready" if get_ai_status() else "🔴 Not configured"
            st.metric("AI Status", ai_status)

def get_ai_status():
    """Check AI availability with proper initialization"""
    try:
        if not st.session_state.ai_initialized or st.session_state.ai_analyzer is None:
            # Force module reload to ensure latest version
            import enhanced_ai_universal
            importlib.reload(enhanced_ai_universal)
            
            st.session_state.ai_analyzer = enhanced_ai_universal.UniversalAIAnalyzer()
            st.session_state.ai_initialized = True
            
        providers = st.session_state.ai_analyzer.get_available_providers()
        return len(providers) > 0
    except Exception as e:
        logger.error(f"AI initialization error: {e}")
        return False

def display_file_upload():
    """Simplified file upload interface"""
    st.markdown("### 📤 Upload Files")
    
    # ASIN input first
    col1, col2 = st.columns([3, 1])
    with col1:
        asin_input = st.text_input(
            "Target ASIN (Optional but recommended)",
            value=st.session_state.target_asin,
            placeholder="B00XYZ1234",
            help="Enter the ASIN to filter and focus analysis"
        )
        if asin_input and len(asin_input) == 10:
            st.session_state.target_asin = asin_input.upper()
    
    with col2:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("🔄 Clear All", use_container_width=True):
            st.session_state.uploaded_files = []
            st.session_state.processed_files = []
            st.session_state.current_analysis = None
            st.session_state.chat_messages = []
            st.rerun()
    
    # File upload area
    st.markdown("""
    <div class="upload-area">
        <h4>Drop files here or click to browse</h4>
        <p>Supports: PDF, Excel, CSV, TSV, TXT, Images (JPG/PNG)</p>
        <p style="color: #666; font-size: 0.9rem;">
        💡 Upload Amazon return reports, Helium 10 reviews, or any related files
        </p>
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
    """Process uploaded files"""
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
            
            # If needs AI analysis (PDF/Image), process with AI
            if processed.extraction_method in ['ai_required', 'vision_ai_required']:
                asyncio.run(process_with_ai(processed, file_content))
            
            st.session_state.processed_files.append(processed)
            
            # Show success
            if processed.warnings:
                st.warning(f"⚠️ {file.name}: {', '.join(processed.warnings)}")
            else:
                st.success(f"✅ {file.name} processed successfully")
                
        except Exception as e:
            st.error(f"❌ Error processing {file.name}: {str(e)}")
            logger.error(f"File processing error: {e}", exc_info=True)
    
    progress_bar.empty()
    status_text.empty()
    
    # Auto-analyze if enabled
    if st.session_state.auto_analyze and st.session_state.processed_files:
        run_analysis()

async def process_with_ai(processed_file: ProcessedFile, content: bytes):
    """Process files that need AI analysis"""
    if not get_ai_status():
        st.warning("AI not configured - some features unavailable")
        return
    
    try:
        analyzer = st.session_state.ai_analyzer
        
        # Analyze file with AI
        with st.spinner(f"AI analyzing {processed_file.metadata.get('filename')}..."):
            analysis = await analyzer.analyze_file(
                content,
                processed_file.metadata.get('filename', ''),
                processed_file.format
            )
            
            # Update processed file with AI results
            if analysis.extracted_data:
                processed_file.data = pd.DataFrame(analysis.extracted_data.get('returns', []))
                processed_file.content_category = analysis.content_type
                processed_file.metadata['ai_analysis'] = analysis.extracted_data
                
    except Exception as e:
        logger.error(f"AI processing error: {e}")
        st.error(f"AI analysis failed: {str(e)}")

def display_file_overview():
    """Display overview of processed files"""
    if not st.session_state.processed_files:
        st.info("No files uploaded yet. Upload files to begin analysis.")
        return
    
    st.markdown("### 📊 File Overview")
    
    # Create summary table
    file_summary = []
    for file in st.session_state.processed_files:
        summary = {
            'File': file.metadata.get('filename', 'Unknown'),
            'Type': file.content_category.replace('_', ' ').title(),
            'Format': file.format.upper(),
            'Records': file.metadata.get('row_count', 0) if file.data is not None else 'N/A',
            'Status': '✅' if file.confidence > 0.7 else '⚠️'
        }
        file_summary.append(summary)
    
    df_summary = pd.DataFrame(file_summary)
    st.dataframe(df_summary, use_container_width=True, hide_index=True)
    
    # Category breakdown
    categories = Counter(f.content_category for f in st.session_state.processed_files)
    
    if len(categories) > 1:
        col1, col2 = st.columns([2, 1])
        
        with col1:
            fig = px.pie(
                values=list(categories.values()),
                names=list(categories.keys()),
                title="Content Types"
            )
            fig.update_traces(textposition='inside', textinfo='percent+label')
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("#### Detection Confidence")
            for file in st.session_state.processed_files:
                confidence_color = COLORS['success'] if file.confidence > 0.8 else COLORS['warning']
                st.markdown(f"""
                <div style="margin: 0.5rem 0;">
                    <small>{file.metadata.get('filename', 'Unknown')[:30]}...</small><br>
                    <div style="background: #E0E0E0; height: 20px; border-radius: 10px;">
                        <div style="background: {confidence_color}; width: {file.confidence*100}%; 
                                    height: 100%; border-radius: 10px;"></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

def run_analysis():
    """Run comprehensive analysis on uploaded files"""
    with st.spinner("🔍 Analyzing data..."):
        try:
            # Separate files by type
            return_files = [f for f in st.session_state.processed_files 
                          if 'return' in f.content_category]
            review_files = [f for f in st.session_state.processed_files 
                          if 'review' in f.content_category]
            
            # Merge data if multiple files
            merged_returns = None
            merged_reviews = None
            
            if return_files:
                return_dfs = [f.data for f in return_files if f.data is not None]
                if return_dfs:
                    merged_returns = pd.concat(return_dfs, ignore_index=True)
            
            if review_files:
                review_dfs = [f.data for f in review_files if f.data is not None]
                if review_dfs:
                    merged_reviews = pd.concat(review_dfs, ignore_index=True)
            
            # Generate analysis
            if get_ai_status() and (merged_returns is not None or merged_reviews is not None):
                analyzer = st.session_state.ai_analyzer
                
                # Convert to format expected by AI
                returns_list = merged_returns.to_dict('records') if merged_returns is not None else []
                reviews_list = merged_reviews.to_dict('records') if merged_reviews is not None else []
                
                # Generate report
                report = analyzer.generate_return_analysis_report(
                    st.session_state.target_asin or 'ALL',
                    returns_list,
                    reviews_list
                )
                
                st.session_state.current_analysis = report
                st.session_state.analysis_complete = True
                st.success("✅ Analysis complete!")
            else:
                # Basic analysis without AI
                st.session_state.current_analysis = generate_basic_analysis(
                    merged_returns, merged_reviews
                )
                st.session_state.analysis_complete = True
                
        except Exception as e:
            st.error(f"Analysis error: {str(e)}")
            logger.error(f"Analysis error: {e}", exc_info=True)

def generate_basic_analysis(returns_df, reviews_df):
    """Generate basic analysis without AI"""
    analysis = {
        'executive_summary': {
            'total_returns': len(returns_df) if returns_df is not None else 0,
            'total_reviews': len(reviews_df) if reviews_df is not None else 0,
            'main_issues': []
        },
        'metrics': {}
    }
    
    if returns_df is not None and not returns_df.empty:
        # Basic return metrics
        if 'reason' in returns_df.columns:
            reason_counts = returns_df['reason'].value_counts()
            analysis['metrics']['top_return_reasons'] = reason_counts.head(5).to_dict()
            analysis['executive_summary']['main_issues'] = reason_counts.head(3).index.tolist()
    
    if reviews_df is not None and not reviews_df.empty:
        # Basic review metrics
        if 'Rating' in reviews_df.columns:
            analysis['metrics']['average_rating'] = reviews_df['Rating'].mean()
            analysis['metrics']['rating_distribution'] = reviews_df['Rating'].value_counts().to_dict()
    
    return analysis

def display_analysis_results():
    """Display analysis results"""
    if not st.session_state.current_analysis:
        st.info("No analysis available. Upload files and run analysis.")
        return
    
    analysis = st.session_state.current_analysis
    
    # Executive Summary
    st.markdown("### 📋 Executive Summary")
    
    summary = analysis.get('executive_summary', {})
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-value">{}</div>
            <div class="metric-label">Total Returns</div>
        </div>
        """.format(summary.get('total_returns', 0)), unsafe_allow_html=True)
    
    with col2:
        trend = summary.get('trend', 'Unknown')
        trend_color = COLORS['danger'] if trend == 'Increasing' else COLORS['success']
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value" style="color: {trend_color};">{trend}</div>
            <div class="metric-label">Return Trend</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        main_issues = summary.get('main_issues', [])
        if main_issues:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{len(main_issues)}</div>
                <div class="metric-label">Critical Issues</div>
            </div>
            """, unsafe_allow_html=True)
    
    with col4:
        if st.button("💬 Discuss with AI", use_container_width=True):
            st.session_state.show_ai_chat = True
    
    # Main Issues
    if summary.get('main_issues'):
        st.markdown("#### 🚨 Main Issues Identified")
        for issue in summary['main_issues']:
            st.markdown(f"- {issue}")
    
    # Category Breakdown
    if 'category_breakdown' in analysis:
        st.markdown("### 📊 Return Categories")
        
        categories = analysis['category_breakdown']
        
        # Create visualization
        cat_data = []
        for cat, info in categories.items():
            cat_data.append({
                'Category': cat.replace('_', ' ').title(),
                'Count': info['count'],
                'Percentage': info['percentage'],
                'Priority': info['priority']
            })
        
        if cat_data:
            df_cat = pd.DataFrame(cat_data)
            
            # Bar chart
            fig = px.bar(
                df_cat,
                x='Category',
                y='Count',
                color='Priority',
                color_discrete_map={
                    'critical': COLORS['danger'],
                    'high': COLORS['warning'],
                    'medium': COLORS['primary'],
                    'low': COLORS['neutral']
                },
                title="Returns by Category"
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Detailed table
            st.dataframe(df_cat, use_container_width=True, hide_index=True)
    
    # Business Impact
    if 'business_impact' in analysis:
        st.markdown("### 💼 Business Impact")
        
        impact = analysis['business_impact']
        severity = impact.get('severity', 'Unknown')
        
        severity_color = {
            'High': COLORS['danger'],
            'Medium': COLORS['warning'],
            'Low': COLORS['success']
        }.get(severity, COLORS['neutral'])
        
        st.markdown(f"""
        <div class="info-card">
            <h3 style="color: {severity_color};">Severity: {severity}</h3>
            <p><strong>Risk Assessment:</strong> {impact.get('risk_assessment', 'N/A')}</p>
            <p><strong>Critical Returns:</strong> {impact.get('critical_return_percentage', 0):.1f}%</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Action Items
    if 'action_items' in analysis:
        st.markdown("### 🎯 Recommended Actions")
        
        for idx, action in enumerate(analysis['action_items'], 1):
            priority = action.get('priority', 'MEDIUM')
            badge_class = {
                'IMMEDIATE': 'status-critical',
                'HIGH': 'status-high',
                'MEDIUM': 'status-medium',
                'LOW': 'status-low'
            }.get(priority, 'status-medium')
            
            st.markdown(f"""
            <div class="info-card">
                <span class="status-badge {badge_class}">{priority}</span>
                <h4>{idx}. {action.get('action', 'Action needed')}</h4>
                <p>{action.get('reason', '')}</p>
            </div>
            """, unsafe_allow_html=True)

def display_ai_chat():
    """AI chat interface with error handling"""
    st.markdown("### 💬 AI Assistant")
    
    # Check if AI is properly initialized
    if not get_ai_status():
        st.warning("AI is not configured. Please add API keys to enable chat.")
        return
    
    # Verify the method exists
    if not hasattr(st.session_state.ai_analyzer, 'generate_chat_response'):
        st.error("AI chat method not available. Please check the AI module.")
        
        # Try to reload the module
        try:
            import enhanced_ai_universal
            importlib.reload(enhanced_ai_universal)
            st.session_state.ai_analyzer = enhanced_ai_universal.UniversalAIAnalyzer()
            st.session_state.ai_initialized = True
        except Exception as e:
            logger.error(f"Failed to reload AI module: {e}")
            return
    
    # Chat container
    chat_container = st.container()
    
    # Display messages
    with chat_container:
        for message in st.session_state.chat_messages:
            with st.chat_message(message["role"]):
                st.write(message["content"])
    
    # Input
    if prompt := st.chat_input("Ask about your analysis..."):
        # Add user message
        st.session_state.chat_messages.append({"role": "user", "content": prompt})
        
        # Get AI response
        try:
            with st.spinner("Thinking..."):
                # Build context
                context = {
                    'has_analysis': st.session_state.analysis_complete,
                    'current_asin': st.session_state.target_asin,
                    'file_count': len(st.session_state.processed_files)
                }
                
                response = st.session_state.ai_analyzer.generate_chat_response(
                    prompt, context
                )
                
                st.session_state.chat_messages.append({"role": "assistant", "content": response})
        except AttributeError as e:
            logger.error(f"Chat method error: {e}")
            st.session_state.chat_messages.append({
                "role": "assistant", 
                "content": "I'm having trouble with the chat function. Please try reloading the app."
            })
        except Exception as e:
            logger.error(f"Chat error: {e}")
            st.session_state.chat_messages.append({
                "role": "assistant", 
                "content": f"An error occurred: {str(e)}"
            })
        
        st.rerun()

def display_export_options():
    """Export options"""
    st.markdown("### 💾 Export Results")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📄 Export Report (PDF)", use_container_width=True):
            # Generate PDF report
            st.info("PDF export coming soon")
    
    with col2:
        if st.button("📊 Export Data (Excel)", use_container_width=True):
            export_to_excel()
    
    with col3:
        if st.button("📋 Copy Summary", use_container_width=True):
            # Copy to clipboard
            st.info("Summary copied to clipboard")

def export_to_excel():
    """Export analysis to Excel"""
    try:
        output = io.BytesIO()
        
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            # Summary sheet
            if st.session_state.current_analysis:
                summary_data = {
                    'Metric': ['Total Returns', 'Return Trend', 'Main Issues'],
                    'Value': [
                        st.session_state.current_analysis.get('executive_summary', {}).get('total_returns', 0),
                        st.session_state.current_analysis.get('executive_summary', {}).get('trend', 'N/A'),
                        ', '.join(st.session_state.current_analysis.get('executive_summary', {}).get('main_issues', []))
                    ]
                }
                pd.DataFrame(summary_data).to_excel(writer, sheet_name='Summary', index=False)
            
            # Raw data sheets
            for idx, file in enumerate(st.session_state.processed_files):
                if file.data is not None:
                    sheet_name = f"{file.content_category}_{idx+1}"[:31]
                    file.data.to_excel(writer, sheet_name=sheet_name, index=False)
        
        output.seek(0)
        
        st.download_button(
            label="Download Excel Report",
            data=output,
            file_name=f"quality_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
        
    except Exception as e:
        st.error(f"Export error: {str(e)}")

def main():
    """Main application"""
    # Force reload modules if needed
    if 'modules_reloaded' not in st.session_state:
        try:
            import enhanced_ai_universal
            import universal_file_detector
            importlib.reload(enhanced_ai_universal)
            importlib.reload(universal_file_detector)
            st.session_state.modules_reloaded = True
        except Exception as e:
            logger.error(f"Module reload error: {e}")
    
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
        
        # Options
        st.checkbox("Auto-analyze on upload", 
                   value=st.session_state.auto_analyze,
                   key="auto_analyze_checkbox")
        st.session_state.auto_analyze = st.session_state.auto_analyze_checkbox
        
        st.checkbox("Combine similar files", 
                   value=st.session_state.combine_sources,
                   key="combine_checkbox")
        st.session_state.combine_sources = st.session_state.combine_checkbox
        
        # Help
        st.markdown("---")
        with st.expander("📖 Quick Guide"):
            st.markdown("""
            **1. Upload Files**
            - Print Amazon returns as PDF
            - Export FBA returns (.txt/.csv)
            - Add Helium 10 reviews
            
            **2. Set Target ASIN**
            - Optional but recommended
            - Filters data automatically
            
            **3. Analyze**
            - Auto-runs on upload
            - Or click Analyze button
            
            **4. Review & Export**
            - Check categorized returns
            - Export to Excel
            - Discuss with AI
            """)
    
    # Main content area with tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📤 Upload", "📊 Overview", "🔍 Analysis", "💬 AI Chat"])
    
    with tab1:
        display_file_upload()
        
        if st.session_state.processed_files:
            st.markdown("---")
            if st.button("🚀 Run Analysis", type="primary", use_container_width=True):
                run_analysis()
    
    with tab2:
        display_file_overview()
    
    with tab3:
        if st.session_state.analysis_complete:
            display_analysis_results()
            st.markdown("---")
            display_export_options()
        else:
            st.info("Upload files and run analysis to see results here.")
    
    with tab4:
        display_ai_chat()

if __name__ == "__main__":
    main()
