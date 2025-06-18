"""
PA9.py - Medical Device Return Analyzer
Enhanced version with full AI support and injury detection
Supports multiple files up to 1GB+ with comprehensive analysis
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

# Increase upload size limit
st._config.set_option('server.maxUploadSize', 1200)  # 1.2GB limit

# Import modules with error handling
modules_status = {}

try:
    from enhanced_ai_analysis import EnhancedAIAnalyzer, AIProvider, FileProcessor
    AI_AVAILABLE = True
    modules_status['AI Analysis'] = "✅ Available"
except ImportError as e:
    logger.error(f"Failed to import AI module: {e}")
    AI_AVAILABLE = False
    modules_status['AI Analysis'] = "❌ Not Available"
    
    # Define FileProcessor fallback
    class FileProcessor:
        @staticmethod
        def read_file(file, file_type):
            # Basic fallback implementation
            if 'csv' in file_type:
                return pd.read_csv(file)
            elif 'excel' in file_type or 'xlsx' in file_type:
                return pd.read_excel(file)
            else:
                return pd.read_csv(file, sep='\t')

try:
    from injury_detector import InjuryDetector
    INJURY_AVAILABLE = True
    modules_status['Injury Detection'] = "✅ Available"
except ImportError as e:
    logger.error(f"Failed to import injury detector: {e}")
    INJURY_AVAILABLE = False
    modules_status['Injury Detection'] = "❌ Not Available"

try:
    from pdf_analyzer import PDFAnalyzer
    PDF_AVAILABLE = True
    modules_status['PDF Analysis'] = "✅ Available"
except ImportError as e:
    logger.error(f"Failed to import PDF analyzer: {e}")
    PDF_AVAILABLE = False
    modules_status['PDF Analysis'] = "❌ Not Available"

try:
    from smart_column_mapper import SmartColumnMapper
    MAPPER_AVAILABLE = True
    modules_status['Smart Mapping'] = "✅ Available"
except ImportError as e:
    logger.error(f"Failed to import column mapper: {e}")
    MAPPER_AVAILABLE = False
    modules_status['Smart Mapping'] = "❌ Not Available"

# Category definitions for quality analysis
QUALITY_CATEGORIES = [
    'QUALITY_DEFECTS',
    'FUNCTIONALITY_ISSUES',
    'COMPATIBILITY_ISSUES',
    'INJURY_RISK',
    'SIZE_FIT_ISSUES',
    'WRONG_PRODUCT',
    'BUYER_MISTAKE',
    'OTHER'
]

# Injury keywords for detection
INJURY_KEYWORDS = {
    'critical': ['death', 'died', 'fatal', 'emergency', 'hospital', 'severe injury', 'surgery'],
    'high': ['injury', 'injured', 'hurt', 'wound', 'bleeding', 'broken', 'fracture', 'burn', 'fall', 'fell'],
    'medium': ['pain', 'discomfort', 'allergic', 'reaction', 'rash', 'swelling']
}

# Enhanced session state defaults
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
    'api_provider': 'auto'
}

def initialize_session_state():
    """Initialize session state variables"""
    for key, value in SESSION_DEFAULTS.items():
        if key not in st.session_state:
            st.session_state[key] = value

def check_api_keys():
    """Check and validate API keys"""
    api_status = {
        'openai': {'available': False, 'key_found': False, 'tested': False},
        'anthropic': {'available': False, 'key_found': False, 'tested': False}
    }
    
    # Check OpenAI
    openai_key = None
    if hasattr(st, 'secrets') and 'OPENAI_API_KEY' in st.secrets:
        openai_key = st.secrets['OPENAI_API_KEY']
    elif os.getenv('OPENAI_API_KEY'):
        openai_key = os.getenv('OPENAI_API_KEY')
    
    if openai_key:
        api_status['openai']['key_found'] = True
        try:
            import openai
            # Test the key
            client = openai.OpenAI(api_key=openai_key)
            # Simple test call
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": "test"}],
                max_tokens=5
            )
            api_status['openai']['available'] = True
            api_status['openai']['tested'] = True
        except Exception as e:
            logger.error(f"OpenAI test failed: {e}")
    
    # Check Anthropic
    anthropic_key = None
    if hasattr(st, 'secrets') and 'ANTHROPIC_API_KEY' in st.secrets:
        anthropic_key = st.secrets['ANTHROPIC_API_KEY']
    elif os.getenv('ANTHROPIC_API_KEY'):
        anthropic_key = os.getenv('ANTHROPIC_API_KEY')
    
    if anthropic_key:
        api_status['anthropic']['key_found'] = True
        try:
            import anthropic
            # Test the key
            client = anthropic.Anthropic(api_key=anthropic_key)
            response = client.messages.create(
                model="claude-3-haiku-20240307",
                messages=[{"role": "user", "content": "test"}],
                max_tokens=5
            )
            api_status['anthropic']['available'] = True
            api_status['anthropic']['tested'] = True
        except Exception as e:
            logger.error(f"Anthropic test failed: {e}")
    
    st.session_state.api_status = api_status
    return api_status

def initialize_analyzers():
    """Initialize all analyzer components"""
    # Initialize AI analyzer with selected provider
    if AI_AVAILABLE and not st.session_state.ai_analyzer:
        try:
            provider = st.session_state.api_provider
            if provider == 'auto':
                # Auto-select based on availability
                if st.session_state.api_status['openai']['available']:
                    provider = AIProvider.OPENAI
                elif st.session_state.api_status['anthropic']['available']:
                    provider = AIProvider.CLAUDE
                else:
                    provider = AIProvider.FASTEST  # Pattern matching fallback
            
            st.session_state.ai_analyzer = EnhancedAIAnalyzer(provider)
            logger.info(f"AI analyzer initialized with provider: {provider}")
        except Exception as e:
            logger.error(f"Failed to initialize AI analyzer: {e}")
            st.warning(f"AI initialization failed: {str(e)}")
    
    # Initialize injury detector
    if INJURY_AVAILABLE and not st.session_state.injury_detector:
        try:
            st.session_state.injury_detector = InjuryDetector(st.session_state.ai_analyzer)
            logger.info("Injury detector initialized")
        except Exception as e:
            logger.error(f"Failed to initialize injury detector: {e}")
    
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
                st.markdown(f"{status} {module}")
        with status_cols[1]:
            for module, status in list(modules_status.items())[2:]:
                st.markdown(f"{status} {module}")
    
    with col3:
        # API status indicator
        if st.session_state.api_status:
            if st.session_state.api_status['openai']['available']:
                st.success("🟢 OpenAI Active")
            elif st.session_state.api_status['anthropic']['available']:
                st.success("🟢 Claude Active")
            else:
                st.warning("🟡 Pattern Mode")

def process_file(file) -> Optional[pd.DataFrame]:
    """Process a single file and return DataFrame"""
    try:
        file_type = file.type if hasattr(file, 'type') else 'unknown'
        file_name = file.name if hasattr(file, 'name') else 'unknown'
        
        # Use FileProcessor to read the file
        if AI_AVAILABLE:
            df = FileProcessor.read_file(file, file_type)
        else:
            # Fallback file reading
            if file_name.lower().endswith('.pdf') or 'pdf' in file_type:
                st.error("PDF processing requires enhanced_ai_analysis module")
                return None
            elif file_name.lower().endswith(('.xlsx', '.xls')):
                df = pd.read_excel(file)
            elif file_name.lower().endswith('.tsv') or '\t' in file.read(1000).decode('utf-8', errors='ignore'):
                file.seek(0)
                df = pd.read_csv(file, sep='\t')
            else:
                df = pd.read_csv(file)
        
        return df
            
    except Exception as e:
        logger.error(f"Error processing file {file.name}: {e}")
        st.error(f"Error processing {file.name}: {str(e)}")
        return None

def detect_quality_issues(df: pd.DataFrame) -> Dict[str, Any]:
    """Detect quality issues in returns data"""
    quality_issues = {
        'total_returns': len(df),
        'quality_defects': 0,
        'injury_cases': 0,
        'categories': {},
        'high_risk_products': []
    }
    
    # Use AI analyzer if available
    if st.session_state.ai_analyzer and AI_AVAILABLE:
        try:
            # Process returns with AI
            for idx, row in df.iterrows():
                # Get return reason and comments
                reason = ''
                comment = ''
                
                # Try different column names
                reason_cols = ['reason', 'return_reason', 'Return Reason', 'issue']
                comment_cols = ['customer-comments', 'customer_comment', 'Customer Comments', 'comments']
                
                for col in reason_cols:
                    if col in row and pd.notna(row[col]):
                        reason = str(row[col])
                        break
                
                for col in comment_cols:
                    if col in row and pd.notna(row[col]):
                        comment = str(row[col])
                        break
                
                full_text = f"{reason} {comment}".strip()
                
                if full_text:
                    # AI categorization
                    result = st.session_state.ai_analyzer.categorize_return(reason, comment)
                    category = result.get('category', 'OTHER')
                    
                    # Update counts
                    quality_issues['categories'][category] = quality_issues['categories'].get(category, 0) + 1
                    
                    if category in ['QUALITY_DEFECTS', 'FUNCTIONALITY_ISSUES']:
                        quality_issues['quality_defects'] += 1
                    
                    # Check for injuries
                    if st.session_state.injury_detector:
                        injury_check = st.session_state.injury_detector.check_for_injury(full_text)
                        if injury_check:
                            quality_issues['injury_cases'] += 1
        
        except Exception as e:
            logger.error(f"AI analysis error: {e}")
            st.warning("AI analysis failed, using pattern matching")
    
    # Fallback to pattern matching
    else:
        for idx, row in df.iterrows():
            text = ' '.join([str(val) for val in row.values if pd.notna(val)]).lower()
            
            # Check for quality keywords
            quality_keywords = ['defect', 'broken', 'damaged', 'quality', 'malfunction']
            if any(keyword in text for keyword in quality_keywords):
                quality_issues['quality_defects'] += 1
            
            # Check for injury keywords
            injury_keywords = ['injury', 'hurt', 'pain', 'hospital', 'emergency']
            if any(keyword in text for keyword in injury_keywords):
                quality_issues['injury_cases'] += 1
    
    # Calculate percentages
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
    
    # Process each row
    for idx, row in export_df.iterrows():
        text = ' '.join([str(val) for val in row.values if pd.notna(val)]).lower()
        
        # Quality detection
        quality_keywords = ['defect', 'broken', 'damaged', 'quality', 'malfunction', 'faulty']
        if any(keyword in text for keyword in quality_keywords):
            export_df.at[idx, 'Quality_Issue'] = True
            export_df.at[idx, 'Risk_Level'] = 'MEDIUM'
        
        # Injury detection
        for severity, keywords in INJURY_KEYWORDS.items():
            if any(keyword in text for keyword in keywords):
                export_df.at[idx, 'Injury_Case'] = True
                export_df.at[idx, 'Risk_Level'] = severity.upper()
                break
        
        # Categorization
        if 'too small' in text or 'too large' in text or 'fit' in text:
            export_df.at[idx, 'Category'] = 'SIZE_FIT_ISSUES'
        elif any(word in text for word in ['defect', 'broken', 'quality']):
            export_df.at[idx, 'Category'] = 'QUALITY_DEFECTS'
        elif any(word in text for word in ['wrong', 'incorrect', 'different']):
            export_df.at[idx, 'Category'] = 'WRONG_PRODUCT'
        elif any(word in text for word in ['mistake', 'accident', 'error']):
            export_df.at[idx, 'Category'] = 'BUYER_MISTAKE'
    
    # Add summary statistics
    summary_row = {
        'Analysis_Date': 'SUMMARY',
        'Quality_Issue': f"Total: {export_df['Quality_Issue'].sum()}",
        'Injury_Case': f"Total: {export_df['Injury_Case'].sum()}",
        'Category': f"Total Returns: {len(export_df)}",
        'Risk_Level': f"High Risk: {len(export_df[export_df['Risk_Level'].isin(['HIGH', 'CRITICAL'])])}"
    }
    
    # Append summary as last row
    summary_df = pd.DataFrame([summary_row])
    export_df = pd.concat([export_df, summary_df], ignore_index=True)
    
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
        
        # API Provider selection
        st.subheader("🤖 AI Provider")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.session_state.api_status['openai']['available']:
                st.success("✅ OpenAI")
            else:
                st.error("❌ OpenAI")
        
        with col2:
            if st.session_state.api_status['anthropic']['available']:
                st.success("✅ Claude")
            else:
                st.error("❌ Claude")
        
        # Provider selection
        providers = ['auto']
        if st.session_state.api_status['openai']['available']:
            providers.append('openai')
        if st.session_state.api_status['anthropic']['available']:
            providers.append('claude')
        
        st.session_state.api_provider = st.selectbox(
            "Select AI Provider",
            providers,
            help="Auto selects the best available provider"
        )
        
        # Test AI button
        if st.button("🧪 Test AI Connection"):
            with st.spinner("Testing AI..."):
                initialize_analyzers()
                if st.session_state.ai_analyzer:
                    test_result = st.session_state.ai_analyzer.test_ai_connection()
                    if test_result['status'] == 'AI connection successful':
                        st.success("✅ AI is working!")
                    else:
                        st.error(f"❌ {test_result['status']}")
                else:
                    st.error("❌ AI analyzer not initialized")
        
        st.divider()
        
        # Processing options
        st.subheader("📊 Processing Options")
        
        enable_injury_detection = st.checkbox("Enable Injury Detection", value=True)
        enable_quality_analysis = st.checkbox("Enable Quality Analysis", value=True)
        enable_ai_categorization = st.checkbox("Enable AI Categorization", value=True)
        
        st.divider()
        
        # File info
        st.subheader("📁 Supported Files")
        st.markdown("""
        - **PDF**: Amazon Seller Central exports
        - **Excel**: .xlsx, .xls files
        - **CSV**: Comma-separated values
        - **TXT**: Tab-delimited FBA returns
        
        **Max file size**: 1.2GB per file
        **Multiple files**: ✅ Supported
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
    
    # File uploader
    uploaded_files = st.file_uploader(
        "Choose files (PDF, Excel, CSV, TXT)",
        type=['pdf', 'xlsx', 'xls', 'csv', 'txt', 'tsv'],
        accept_multiple_files=True,
        help="Upload up to 1.2GB per file. Multiple files supported."
    )
    
    if uploaded_files:
        st.session_state.uploaded_files = uploaded_files
        st.session_state.total_files = len(uploaded_files)
        
        # Display file info
        total_size = sum(file.size for file in uploaded_files) / (1024 * 1024)  # MB
        st.info(f"📁 {len(uploaded_files)} file(s) uploaded | Total size: {total_size:.1f} MB")
        
        # Process button
        if st.button("🔍 Analyze Returns", type="primary", use_container_width=True):
            # Initialize analyzers
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
                if df is not None:
                    df['source_file'] = file.name
                    all_dataframes.append(df)
                else:
                    processing_errors.append(file.name)
                
                # Clear memory for large files
                if file.size > 100 * 1024 * 1024:  # 100MB
                    gc.collect()
            
            progress_bar.empty()
            status_text.empty()
            
            # Combine all dataframes
            if all_dataframes:
                combined_df = pd.concat(all_dataframes, ignore_index=True)
                st.session_state.combined_data = combined_df
                st.session_state.total_rows = len(combined_df)
                
                # Analyze data
                with st.spinner("Analyzing for quality issues and injuries..."):
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
                        f"{analysis_results['quality_rate']:.1f}%"
                    )
                
                with col3:
                    injury_color = "🔴" if analysis_results['injury_cases'] > 0 else "🟢"
                    st.metric(
                        f"{injury_color} Injury Cases",
                        f"{analysis_results['injury_cases']:,}",
                        f"{analysis_results['injury_rate']:.1f}%"
                    )
                
                with col4:
                    high_risk = len(export_df[export_df['Risk_Level'].isin(['HIGH', 'CRITICAL'])])
                    st.metric("High Risk Items", f"{high_risk:,}")
                
                # Show injury alert if needed
                if analysis_results['injury_cases'] > 0:
                    st.error(f"""
                    🚨 **INJURY CASES DETECTED**
                    
                    Found {analysis_results['injury_cases']} potential injury cases requiring immediate review.
                    These may require FDA MDR reporting or other regulatory actions.
                    
                    Please review the detailed export for specific cases.
                    """)
                
                # Data preview
                st.subheader("📊 Data Preview")
                
                # Filter options
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    show_quality_only = st.checkbox("Quality Issues Only")
                
                with col2:
                    show_injuries_only = st.checkbox("Injury Cases Only")
                
                with col3:
                    show_high_risk_only = st.checkbox("High Risk Only")
                
                # Apply filters
                display_df = export_df.copy()
                
                if show_quality_only:
                    display_df = display_df[display_df['Quality_Issue'] == True]
                
                if show_injuries_only:
                    display_df = display_df[display_df['Injury_Case'] == True]
                
                if show_high_risk_only:
                    display_df = display_df[display_df['Risk_Level'].isin(['HIGH', 'CRITICAL'])]
                
                # Display filtered data
                st.dataframe(display_df, use_container_width=True, height=400)
                
                # Export options
                st.subheader("💾 Export Options")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    # CSV export
                    csv = export_df.to_csv(index=False)
                    st.download_button(
                        "📥 Download Full Analysis (CSV)",
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
                            
                            # Summary sheet
                            summary_data = {
                                'Metric': ['Total Returns', 'Quality Defects', 'Injury Cases', 'High Risk Items'],
                                'Count': [
                                    analysis_results['total_returns'],
                                    analysis_results['quality_defects'],
                                    analysis_results['injury_cases'],
                                    high_risk
                                ],
                                'Percentage': [
                                    '100%',
                                    f"{analysis_results['quality_rate']:.1f}%",
                                    f"{analysis_results['injury_rate']:.1f}%",
                                    f"{(high_risk/analysis_results['total_returns']*100):.1f}%"
                                ]
                            }
                            summary_df = pd.DataFrame(summary_data)
                            summary_df.to_excel(writer, sheet_name='Summary', index=False)
                            
                            # Category breakdown if available
                            if analysis_results.get('categories'):
                                cat_df = pd.DataFrame(
                                    list(analysis_results['categories'].items()),
                                    columns=['Category', 'Count']
                                )
                                cat_df.to_excel(writer, sheet_name='Categories', index=False)
                        
                        buffer.seek(0)
                        st.download_button(
                            "📥 Download Full Analysis (Excel)",
                            data=buffer,
                            file_name=f"return_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                            use_container_width=True
                        )
                    except Exception as e:
                        st.error(f"Excel export failed: {str(e)}")
                
                with col3:
                    # JSON export for further processing
                    json_data = {
                        'analysis_date': datetime.now().isoformat(),
                        'summary': analysis_results,
                        'processing_stats': {
                            'files_processed': len(all_dataframes),
                            'total_rows': st.session_state.total_rows,
                            'errors': processing_errors
                        }
                    }
                    json_str = json.dumps(json_data, indent=2)
                    st.download_button(
                        "📥 Download Analysis Report (JSON)",
                        data=json_str,
                        file_name=f"return_analysis_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                        mime="application/json",
                        use_container_width=True
                    )
                
            else:
                st.error("❌ No files could be processed successfully")
    
    else:
        # Show instructions when no files uploaded
        st.info("""
        👆 **Upload your return data files to begin analysis**
        
        This tool will:
        - 🔍 Detect quality defects and issues
        - 🚨 Identify potential injury cases
        - 📊 Categorize returns automatically
        - 📈 Provide structured data for analysis
        - 💾 Export results in multiple formats
        
        **Supported file types:**
        - PDF exports from Amazon Seller Central
        - Excel files with return data
        - CSV/TXT files from FBA reports
        - Multiple files up to 1.2GB each
        """)

if __name__ == "__main__":
    main()
