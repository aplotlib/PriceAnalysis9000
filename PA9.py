"""
FDA Medical Device Return Analyzer - Multi-File Support
Version: 9.0 - Complete Return Categorization + FDA Event Detection
Purpose: Categorize ALL returns AND identify FDA reportable events
"""

import streamlit as st
import pandas as pd
import numpy as np
import logging
from datetime import datetime
import io
from typing import Dict, List, Any, Optional, Union
import plotly.express as px
import plotly.graph_objects as go
import time
import gc

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configure page - MUST BE FIRST STREAMLIT COMMAND
st.set_page_config(
    page_title="FDA Medical Device Return Analyzer",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Import enhanced AI analysis module
try:
    from enhanced_ai_analysis import (
        EnhancedAIAnalyzer, AIProvider, MEDICAL_DEVICE_CATEGORIES,
        FDA_MDR_TRIGGERS, detect_fda_reportable_event, FileProcessor
    )
    AI_AVAILABLE = True
except ImportError as e:
    AI_AVAILABLE = False
    logger.error(f"Enhanced AI analysis not available: {e}")
    st.error("Critical module missing. Please check installation.")

# App styling
def apply_custom_css():
    """Apply custom CSS for professional appearance"""
    st.markdown("""
    <style>
    /* Main theme */
    .stApp {
        background-color: #f5f5f5;
    }
    
    /* Headers */
    h1, h2, h3 {
        color: #1e3a8a;
    }
    
    /* FDA Alert styling */
    .fda-alert {
        background-color: #dc2626;
        color: white;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
        font-weight: bold;
    }
    
    /* Category card */
    .category-card {
        background-color: #f3f4f6;
        border: 1px solid #e5e7eb;
        padding: 0.75rem;
        border-radius: 0.5rem;
        margin: 0.25rem 0;
    }
    
    /* Reportable event card */
    .reportable-event {
        background-color: #fee2e2;
        border: 2px solid #dc2626;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    
    /* Metrics */
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 0.5rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.12);
        text-align: center;
    }
    
    .metric-value {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1e3a8a;
    }
    
    .metric-label {
        color: #6b7280;
        font-size: 0.875rem;
        text-transform: uppercase;
    }
    
    /* Critical events */
    .critical {
        color: #dc2626;
        font-weight: bold;
    }
    
    /* Progress bars */
    .stProgress > div > div > div {
        background-color: #1e3a8a;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    </style>
    """, unsafe_allow_html=True)

def initialize_session_state():
    """Initialize session state variables"""
    defaults = {
        'uploaded_files': [],
        'combined_data': None,
        'processed_data': None,
        'categorization_complete': False,
        'fda_report': None,
        'file_upload_complete': False,
        'processing_complete': False,
        'ai_analyzer': None,
        'total_rows': 0,
        'reportable_events': 0,
        'critical_events': 0,
        'processing_stats': {}
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

def display_header():
    """Display app header"""
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.title("🏥 FDA Medical Device Return Analyzer")
        st.markdown("""
        **Complete Return Analysis**: Categorizes ALL returns + Identifies FDA reportable events
        
        **Categories**: Size/Fit, Quality Defects, Performance, Injuries, Falls, and more
        """)
    
    with col2:
        st.info("""
        **Multi-File Support**
        - Up to 7 files
        - 1GB+ capacity
        - All formats supported
        """)

def process_multiple_files(uploaded_files: List) -> pd.DataFrame:
    """Process multiple uploaded files and combine them"""
    all_dataframes = []
    file_stats = {}
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for idx, file in enumerate(uploaded_files):
        try:
            status_text.text(f"Processing file {idx + 1}/{len(uploaded_files)}: {file.name}")
            progress_bar.progress((idx + 1) / len(uploaded_files))
            
            # Get file type
            file_type = file.type if hasattr(file, 'type') else 'text/csv'
            
            # Process file
            df = FileProcessor.read_file(file, file_type)
            
            # Add source file column
            df['source_file'] = file.name
            
            # Standardize column names
            df.columns = df.columns.str.lower().str.strip().str.replace(' ', '-')
            
            all_dataframes.append(df)
            file_stats[file.name] = {'rows': len(df), 'status': 'Success'}
            
            # Clear memory for large files
            if len(df) > 10000:
                gc.collect()
                
        except Exception as e:
            file_stats[file.name] = {'rows': 0, 'status': f'Error: {str(e)}'}
            st.warning(f"Error processing {file.name}: {str(e)}")
    
    progress_bar.empty()
    status_text.empty()
    
    # Combine all dataframes
    if all_dataframes:
        combined_df = pd.concat(all_dataframes, ignore_index=True)
        
        # Display processing stats
        st.success(f"✅ Successfully processed {len(all_dataframes)} files")
        
        # Show file stats
        with st.expander("📊 File Processing Summary"):
            for filename, stats in file_stats.items():
                if stats['status'] == 'Success':
                    st.write(f"✓ {filename}: {stats['rows']:,} rows")
                else:
                    st.write(f"✗ {filename}: {stats['status']}")
        
        return combined_df
    else:
        st.error("No files could be processed successfully")
        return None

def display_categorization_analysis(df: pd.DataFrame):
    """Display comprehensive return categorization analysis"""
    st.header("📊 Return Categorization Analysis")
    
    # Overview metrics
    col1, col2, col3, col4 = st.columns(4)
    
    total_returns = len(df)
    quality_defects = len(df[df['category'] == 'Product Defects/Quality'])
    injury_events = len(df[df['category'] == 'Injury/Adverse Event'])
    unique_products = df['asin'].nunique() if 'asin' in df.columns else 0
    
    with col1:
        st.metric("Total Returns", f"{total_returns:,}")
    
    with col2:
        st.metric("Quality Defects", f"{quality_defects:,}", 
                 f"{quality_defects/total_returns*100:.1f}%")
    
    with col3:
        st.metric("Injury Events", f"{injury_events:,}",
                 f"{injury_events/total_returns*100:.1f}%")
    
    with col4:
        st.metric("Unique Products", f"{unique_products:,}")
    
    # Category distribution
    st.subheader("📈 Category Distribution")
    
    category_counts = df['category'].value_counts()
    
    # Create two columns for visualizations
    col1, col2 = st.columns(2)
    
    with col1:
        # Pie chart
        fig_pie = px.pie(
            values=category_counts.values,
            names=category_counts.index,
            title="Return Categories Distribution",
            color_discrete_sequence=px.colors.qualitative.Set3
        )
        fig_pie.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig_pie, use_container_width=True)
    
    with col2:
        # Bar chart
        fig_bar = px.bar(
            x=category_counts.values,
            y=category_counts.index,
            orientation='h',
            title="Returns by Category",
            labels={'x': 'Count', 'y': 'Category'},
            color=category_counts.values,
            color_continuous_scale='Blues'
        )
        st.plotly_chart(fig_bar, use_container_width=True)
    
    # Top products by category
    st.subheader("🏷️ Top Products by Return Category")
    
    selected_category = st.selectbox(
        "Select a category to view top products:",
        options=category_counts.index.tolist()
    )
    
    if 'product-name' in df.columns:
        category_products = df[df['category'] == selected_category]['product-name'].value_counts().head(10)
        
        if not category_products.empty:
            fig_products = px.bar(
                x=category_products.values,
                y=category_products.index,
                orientation='h',
                title=f"Top 10 Products - {selected_category}",
                labels={'x': 'Returns', 'y': 'Product'}
            )
            st.plotly_chart(fig_products, use_container_width=True)
    
    # Time trend if date available
    if 'return-date' in df.columns:
        try:
            df['return-date'] = pd.to_datetime(df['return-date'])
            daily_returns = df.groupby([df['return-date'].dt.date, 'category']).size().reset_index(name='count')
            
            fig_trend = px.line(
                daily_returns,
                x='return-date',
                y='count',
                color='category',
                title='Return Trends by Category Over Time'
            )
            st.plotly_chart(fig_trend, use_container_width=True)
        except:
            pass

def display_fda_analysis(df: pd.DataFrame, report: Dict[str, Any]):
    """Display FDA reportable events analysis"""
    st.header("🚨 FDA Reportable Events Analysis")
    
    if report['summary']['reportable_events'] > 0:
        st.markdown("""
        <div class="fda-alert">
        ⚠️ FDA REPORTABLE EVENTS DETECTED - IMMEDIATE REVIEW REQUIRED
        </div>
        """, unsafe_allow_html=True)
    
    # FDA metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Reportable Events", 
                 f"{report['summary']['reportable_events']:,}",
                 f"{report['summary']['reportable_events']/len(df)*100:.1f}% of returns")
    
    with col2:
        st.metric("Critical Severity", 
                 f"{report['summary']['critical_events']:,}",
                 "Immediate action")
    
    with col3:
        st.metric("High Severity",
                 f"{report['summary']['high_severity']:,}",
                 "30-day deadline")
    
    with col4:
        st.metric("MDR Required",
                 f"{report['summary']['mdr_required']:,}",
                 "File with FDA")
    
    # Event type breakdown
    if report['by_event_type']:
        st.subheader("📊 Reportable Events by Type")
        
        event_df = pd.DataFrame(
            list(report['by_event_type'].items()),
            columns=['Event Type', 'Count']
        ).sort_values('Count', ascending=False)
        
        # Create visual alert for fall-related events
        if 'falls' in report['by_event_type']:
            st.warning(f"⚠️ {report['by_event_type']['falls']} fall-related incidents detected!")
        
        fig_events = px.bar(
            event_df,
            x='Event Type',
            y='Count',
            title='FDA Reportable Events by Type',
            color='Count',
            color_continuous_scale=['#fee2e2', '#dc2626']
        )
        st.plotly_chart(fig_events, use_container_width=True)
    
    # Show detailed reportable events
    if report['summary']['reportable_events'] > 0:
        st.subheader("📋 Detailed Reportable Events")
        
        reportable_df = df[df['fda_reportable'] == True].copy()
        
        # Sort by severity
        severity_order = {'CRITICAL': 0, 'HIGH': 1, 'MODERATE': 2, 'LOW': 3}
        reportable_df['severity_rank'] = reportable_df['severity'].map(severity_order)
        reportable_df = reportable_df.sort_values('severity_rank')
        
        # Display key columns
        display_cols = ['order-id', 'asin', 'product-name', 'reason', 
                       'customer-comments', 'category', 'severity', 'event_types']
        
        available_cols = [col for col in display_cols if col in reportable_df.columns]
        
        st.dataframe(
            reportable_df[available_cols],
            use_container_width=True,
            height=400
        )

def main():
    """Main application flow"""
    # Initialize
    initialize_session_state()
    apply_custom_css()
    display_header()
    
    # Sidebar
    with st.sidebar:
        st.header("📁 File Upload")
        st.markdown("Upload up to 7 files (1GB+ supported)")
        
        uploaded_files = st.file_uploader(
            "Choose files",
            type=['csv', 'xlsx', 'xls', 'tsv', 'txt', 'pdf'],
            accept_multiple_files=True,
            help="Upload return data from Amazon Seller Central"
        )
        
        if uploaded_files:
            if len(uploaded_files) > 7:
                st.error("Maximum 7 files allowed")
                uploaded_files = uploaded_files[:7]
            
            st.success(f"{len(uploaded_files)} file(s) uploaded")
            for file in uploaded_files:
                st.write(f"• {file.name}")
            
            st.session_state.uploaded_files = uploaded_files
            st.session_state.file_upload_complete = True
        
        st.divider()
        
        # AI Provider selection
        if AI_AVAILABLE:
            st.subheader("🤖 AI Settings")
            provider = st.selectbox(
                "AI Provider",
                options=[AIProvider.FASTEST, AIProvider.QUALITY, AIProvider.OPENAI, AIProvider.CLAUDE],
                help="Select AI provider for categorization"
            )
        
        st.divider()
        
        # Processing options
        st.subheader("⚙️ Processing Options")
        
        chunk_size = st.select_slider(
            "Batch Size",
            options=[100, 500, 1000, 5000, 10000],
            value=1000,
            help="Larger batches process faster but use more memory"
        )
        
        st.divider()
        
        # Info
        st.info("""
        **Supported Formats:**
        - CSV, TSV, TXT (auto-delimiter)
        - Excel (XLSX, XLS)
        - PDF (with tables)
        
        **Detects:**
        - All return categories
        - FDA reportable events
        - Falls & injuries
        - Product defects
        """)
    
    # Main content area
    if st.session_state.file_upload_complete and uploaded_files:
        
        # Process button
        if st.button("🔍 Analyze Returns & Detect FDA Events", 
                     type="primary", 
                     use_container_width=True):
            
            start_time = time.time()
            
            with st.spinner("Processing files and categorizing returns..."):
                # Process all files
                combined_df = process_multiple_files(uploaded_files)
                
                if combined_df is not None:
                    st.session_state.combined_data = combined_df
                    st.session_state.total_rows = len(combined_df)
                    
                    # Initialize AI analyzer
                    if AI_AVAILABLE and not st.session_state.ai_analyzer:
                        st.session_state.ai_analyzer = EnhancedAIAnalyzer(provider)
                    
                    # Categorize returns with progress tracking
                    if st.session_state.ai_analyzer:
                        st.info(f"🤖 Categorizing {len(combined_df):,} returns using AI...")
                        
                        # Process in chunks for large datasets
                        if len(combined_df) > chunk_size:
                            processed_chunks = []
                            total_chunks = len(combined_df) // chunk_size + 1
                            
                            progress_bar = st.progress(0)
                            for i in range(0, len(combined_df), chunk_size):
                                chunk = combined_df.iloc[i:i+chunk_size]
                                processed_chunk = st.session_state.ai_analyzer.batch_categorize(chunk)
                                processed_chunks.append(processed_chunk)
                                
                                progress = (i + chunk_size) / len(combined_df)
                                progress_bar.progress(min(progress, 1.0))
                            
                            progress_bar.empty()
                            processed_df = pd.concat(processed_chunks, ignore_index=True)
                        else:
                            processed_df = st.session_state.ai_analyzer.batch_categorize(combined_df)
                        
                        st.session_state.processed_data = processed_df
                        st.session_state.categorization_complete = True
                        
                        # Generate FDA report
                        fda_report = st.session_state.ai_analyzer.generate_fda_report(processed_df)
                        st.session_state.fda_report = fda_report
                        st.session_state.processing_complete = True
                        
                        # Processing stats
                        processing_time = time.time() - start_time
                        st.session_state.processing_stats = {
                            'time': processing_time,
                            'rows_per_second': len(combined_df) / processing_time,
                            'api_calls': st.session_state.ai_analyzer.api_calls
                        }
                        
                        st.success(f"""
                        ✅ Processing Complete!
                        - Total returns: {len(processed_df):,}
                        - Processing time: {processing_time:.1f} seconds
                        - Speed: {len(combined_df) / processing_time:.0f} returns/second
                        """)
                    else:
                        st.error("AI analyzer not available")
    
    # Display results
    if st.session_state.processing_complete:
        # Create tabs for different views
        tab1, tab2, tab3, tab4 = st.tabs([
            "📊 Categorization Analysis",
            "🚨 FDA Reportable Events", 
            "📋 Full Data View",
            "💾 Export Results"
        ])
        
        with tab1:
            display_categorization_analysis(st.session_state.processed_data)
        
        with tab2:
            display_fda_analysis(st.session_state.processed_data, st.session_state.fda_report)
        
        with tab3:
            st.header("📋 Complete Processed Data")
            
            # Filters
            col1, col2, col3 = st.columns(3)
            
            with col1:
                category_filter = st.multiselect(
                    "Filter by Category",
                    options=st.session_state.processed_data['category'].unique().tolist(),
                    default=[]
                )
            
            with col2:
                if 'severity' in st.session_state.processed_data.columns:
                    severity_filter = st.multiselect(
                        "Filter by Severity",
                        options=st.session_state.processed_data['severity'].dropna().unique().tolist(),
                        default=[]
                    )
                else:
                    severity_filter = []
            
            with col3:
                show_fda_only = st.checkbox("Show FDA Reportable Only", value=False)
            
            # Apply filters
            filtered_df = st.session_state.processed_data.copy()
            
            if category_filter:
                filtered_df = filtered_df[filtered_df['category'].isin(category_filter)]
            
            if severity_filter:
                filtered_df = filtered_df[filtered_df['severity'].isin(severity_filter)]
            
            if show_fda_only:
                filtered_df = filtered_df[filtered_df['fda_reportable'] == True]
            
            st.dataframe(filtered_df, use_container_width=True, height=600)
        
        with tab4:
            st.header("💾 Export Options")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("📊 Full Categorized Data")
                csv_full = st.session_state.processed_data.to_csv(index=False)
                st.download_button(
                    "⬇️ Download Full Analysis (CSV)",
                    data=csv_full,
                    file_name=f"return_analysis_categorized_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                    use_container_width=True
                )
                
                # Excel export if available
                try:
                    buffer = io.BytesIO()
                    with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
                        st.session_state.processed_data.to_excel(writer, sheet_name='All Returns', index=False)
                        
                        # Add category summary
                        category_summary = st.session_state.processed_data['category'].value_counts().reset_index()
                        category_summary.columns = ['Category', 'Count']
                        category_summary.to_excel(writer, sheet_name='Category Summary', index=False)
                        
                        # Add FDA summary if events exist
                        if st.session_state.fda_report['summary']['reportable_events'] > 0:
                            fda_df = st.session_state.ai_analyzer.export_fda_summary(st.session_state.processed_data)
                            fda_df.to_excel(writer, sheet_name='FDA Reportable', index=False)
                    
                    buffer.seek(0)
                    st.download_button(
                        "⬇️ Download Full Analysis (Excel)",
                        data=buffer,
                        file_name=f"return_analysis_complete_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        use_container_width=True
                    )
                except:
                    st.info("Excel export requires openpyxl")
            
            with col2:
                st.subheader("🚨 FDA Report Only")
                
                if st.session_state.fda_report['summary']['reportable_events'] > 0:
                    fda_summary = st.session_state.ai_analyzer.export_fda_summary(st.session_state.processed_data)
                    
                    if not fda_summary.empty:
                        csv_fda = fda_summary.to_csv(index=False)
                        st.download_button(
                            "⬇️ Download FDA Summary (CSV)",
                            data=csv_fda,
                            file_name=f"fda_reportable_events_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                            mime="text/csv",
                            use_container_width=True
                        )
                        
                        # MDR template info
                        st.info("""
                        **Next Steps for FDA Reporting:**
                        1. Review all CRITICAL severity events immediately
                        2. Submit MDR within 30 days for serious injuries
                        3. Document corrective actions taken
                        4. Consider voluntary recall if pattern emerges
                        """)
                else:
                    st.success("✅ No FDA reportable events detected")
            
            # Processing statistics
            st.divider()
            st.subheader("📈 Processing Statistics")
            
            stats = st.session_state.processing_stats
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Processing Time", f"{stats['time']:.1f} seconds")
            
            with col2:
                st.metric("Processing Speed", f"{stats['rows_per_second']:.0f} rows/sec")
            
            with col3:
                if 'api_calls' in stats:
                    st.metric("AI API Calls", f"{stats['api_calls']:,}")
    
    else:
        # Show instructions
        st.markdown("""
        ### 🚀 Getting Started
        
        1. **Upload Files** - Use the sidebar to upload up to 7 return files (CSV, Excel, PDF, etc.)
        2. **Click Analyze** - Process all files to categorize returns and detect FDA events
        3. **Review Results** - See complete categorization + FDA reportable events
        4. **Export Data** - Download categorized data with FDA flags
        
        ### 📊 What This Tool Does
        
        **Return Categorization** (ALL returns are categorized):
        - Size/Fit Issues
        - Product Defects/Quality  
        - Performance/Effectiveness
        - Injury/Adverse Events
        - Stability/Safety Issues
        - Material/Component Failures
        - And 8+ more categories
        
        **FDA Event Detection**:
        - Deaths & Serious Injuries
        - Falls (fall, fell, fallen, slip, trip)
        - Product Malfunctions
        - Allergic Reactions
        - Infections
        
        ### 💡 Tips for Quality Managers
        
        - Process FBA return reports (.txt) AND PDF exports together for complete analysis
        - Filter by "Injury/Adverse Event" category to focus on safety issues
        - Export FDA summary for immediate MDR filing
        - Track category trends over time to identify quality improvement opportunities
        """)

if __name__ == "__main__":
    main()
