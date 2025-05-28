"""
Amazon Review Analyzer - Medical Device Quality Management Edition
Version 10.1 - Fixed Tutorial Display and Marketplace Analysis
"""

import streamlit as st
import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
import io
from typing import Dict, List, Any, Optional, Tuple
import re
from collections import Counter, defaultdict
from io import BytesIO
import requests
from bs4 import BeautifulSoup
import time
import json

# Import handling with fallbacks
try:
    import xlsxwriter
    EXCEL_AVAILABLE = True
except ImportError:
    EXCEL_AVAILABLE = False

try:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import letter
    from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, PageBreak
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch
    from reportlab.lib.enums import TA_CENTER
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    import enhanced_ai_analysis
    AI_AVAILABLE = True
except ImportError:
    AI_AVAILABLE = False
    logger.warning("AI module not available")

try:
    from amazon_file_detector import AmazonFileDetector
    DETECTOR_AVAILABLE = True
except ImportError:
    DETECTOR_AVAILABLE = False
    logger.warning("Amazon file detector module not available")

# Configuration
APP_CONFIG = {
    'title': 'Vive Health Review Intelligence',
    'version': '10.1',
    'company': 'Vive Health',
    'support_email': 'alexander.popoff@vivehealth.com'
}

COLORS = {
    'primary': '#00D9FF', 'secondary': '#FF006E', 'accent': '#FFB700',
    'success': '#00F5A0', 'warning': '#FF6B35', 'danger': '#FF0054',
    'dark': '#0A0A0F', 'light': '#1A1A2E', 'text': '#E0E0E0', 'muted': '#666680'
}

# Amazon scraping headers
HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
    'Accept-Language': 'en-US,en;q=0.9',
    'Accept-Encoding': 'gzip, deflate',
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
    'Connection': 'keep-alive',
}

def initialize_session_state():
    """Initialize session state with improved workflow tracking"""
    defaults = {
        # Data states
        'uploaded_data': None,
        'analysis_results': None,
        'marketplace_files': {
            'reimbursements': None,
            'fba_returns': None,
            'fbm_returns': None
        },
        'marketplace_data': None,
        
        # UI states
        'current_view': 'upload',
        'current_step': 1,  # Track workflow progress
        'processing': False,
        'show_ai_chat': False,
        
        # Analysis settings
        'selected_timeframe': 'all',
        'filter_rating': 'all',
        'analysis_depth': 'comprehensive',
        'analyze_all_reviews': True,
        'use_listing_details': False,
        
        # Listing data
        'listing_details': {
            'title': '', 'bullet_points': ['', '', '', '', ''], 'description': '',
            'backend_keywords': '', 'brand': '', 'category': '', 'asin': '', 'url': ''
        },
        'scraping_status': None,
        'auto_populated': False,
        
        # AI states
        'ai_analyzer': None,
        'chat_messages': [],
        
        # Workflow completion tracking
        'steps_completed': {
            'listing_details': False,
            'review_file': False,
            'marketplace_files': False,
            'analysis_run': False
        },
        
        # User guidance
        'show_tutorial': True,
        'workflow_errors': []
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

def get_workflow_status():
    """Get current workflow status and next steps"""
    steps = st.session_state.steps_completed
    
    # Determine current step
    if not steps['listing_details'] and not steps['review_file']:
        return 1, "Start by entering product details or uploading reviews"
    elif steps['listing_details'] and not steps['review_file']:
        return 2, "Now upload your Helium 10 review file"
    elif steps['review_file'] and not steps['listing_details']:
        return 2, "Optional: Add product details for enhanced analysis"
    elif steps['review_file'] and not steps['analysis_run']:
        return 3, "Ready to run analysis!"
    else:
        return 4, "Analysis complete - review results or start new analysis"

def inject_cyberpunk_css():
    """Inject enhanced cyberpunk CSS with workflow indicators"""
    st.markdown(f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&family=Rajdhani:wght@300;400;600;700&display=swap');
    
    :root {{
        --primary: {COLORS['primary']}; --secondary: {COLORS['secondary']};
        --accent: {COLORS['accent']}; --success: {COLORS['success']};
        --warning: {COLORS['warning']}; --danger: {COLORS['danger']};
        --dark: {COLORS['dark']}; --light: {COLORS['light']};
        --text: {COLORS['text']}; --muted: {COLORS['muted']};
    }}
    
    /* Base styles */
    html, body, .stApp {{
        background: linear-gradient(135deg, var(--dark) 0%, var(--light) 100%);
        color: var(--text); font-family: 'Rajdhani', sans-serif;
    }}
    
    h1, h2, h3 {{ font-family: 'Orbitron', sans-serif; text-transform: uppercase; letter-spacing: 0.1em; }}
    
    h1 {{
        background: linear-gradient(90deg, var(--primary) 0%, var(--secondary) 100%);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        text-shadow: 0 0 30px rgba(0, 217, 255, 0.4);
    }}
    
    /* Workflow progress bar */
    .workflow-progress {{
        background: rgba(10, 10, 15, 0.9);
        border: 1px solid var(--primary);
        border-radius: 10px;
        padding: 1rem;
        margin: 1rem 0;
    }}
    
    .progress-step {{
        display: inline-block;
        width: 24%;
        text-align: center;
        position: relative;
        color: var(--muted);
    }}
    
    .progress-step.active {{
        color: var(--primary);
    }}
    
    .progress-step.completed {{
        color: var(--success);
    }}
    
    .progress-step::after {{
        content: '';
        position: absolute;
        top: 20px;
        right: -50%;
        width: 100%;
        height: 2px;
        background: var(--muted);
    }}
    
    .progress-step.completed::after {{
        background: var(--success);
    }}
    
    .progress-step:last-child::after {{
        display: none;
    }}
    
    /* Enhanced boxes */
    .neon-box {{
        background: rgba(10, 10, 15, 0.9); border: 1px solid var(--primary);
        border-radius: 10px; padding: 1.5rem;
        box-shadow: 0 0 20px rgba(0, 217, 255, 0.4), inset 0 0 20px rgba(0, 217, 255, 0.1);
    }}
    
    .tutorial-box {{
        background: rgba(10, 10, 15, 0.95); 
        border: 2px solid var(--primary);
        border-radius: 15px; 
        padding: 2rem; 
        margin: 1rem 0;
        box-shadow: 0 0 30px rgba(0, 217, 255, 0.5);
    }}
    
    .tutorial-box h3 {{
        color: var(--primary);
        margin-top: 0;
    }}
    
    .tutorial-box ol {{
        margin-left: 1rem;
    }}
    
    .tutorial-box li {{
        margin: 1rem 0;
    }}
    
    .tutorial-box .highlight {{
        color: var(--accent);
        font-weight: bold;
    }}
    
    .workflow-box {{
        background: rgba(0, 217, 255, 0.1); 
        border: 2px solid var(--primary);
        border-radius: 15px; 
        padding: 2rem; 
        margin: 1rem 0;
        position: relative;
    }}
    
    .workflow-number {{
        position: absolute;
        top: -15px;
        left: 20px;
        background: var(--dark);
        color: var(--primary);
        font-family: 'Orbitron', sans-serif;
        font-size: 1.5rem;
        font-weight: bold;
        padding: 0 1rem;
        border: 2px solid var(--primary);
        border-radius: 20px;
    }}
    
    .help-tip {{
        background: rgba(255, 183, 0, 0.1); 
        border-left: 3px solid var(--accent);
        padding: 0.5rem 1rem; 
        margin: 0.5rem 0;
        font-size: 0.9rem;
    }}
    
    /* Status indicators */
    .status-badge {{
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 15px;
        font-size: 0.8rem;
        font-weight: 600;
        margin: 0 0.25rem;
    }}
    
    .status-badge.success {{
        background: rgba(0, 245, 160, 0.2);
        border: 1px solid var(--success);
        color: var(--success);
    }}
    
    .status-badge.pending {{
        background: rgba(255, 183, 0, 0.2);
        border: 1px solid var(--accent);
        color: var(--accent);
    }}
    
    .status-badge.optional {{
        background: rgba(102, 102, 128, 0.2);
        border: 1px solid var(--muted);
        color: var(--muted);
    }}
    
    /* Buttons */
    .stButton > button {{
        font-family: 'Rajdhani', sans-serif; font-weight: 600;
        background: linear-gradient(135deg, var(--primary) 0%, var(--secondary) 100%);
        color: var(--dark); border: none; padding: 0.75rem 2rem;
        border-radius: 5px; transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(0, 217, 255, 0.4);
    }}
    
    .stButton > button:hover {{
        transform: translateY(-2px); box-shadow: 0 6px 25px rgba(0, 217, 255, 0.6);
    }}
    
    .stButton > button:disabled {{
        background: var(--muted);
        cursor: not-allowed;
        opacity: 0.5;
    }}
    
    /* File uploader enhancement */
    .uploadedFile {{
        background: rgba(0, 245, 160, 0.1);
        border: 1px solid var(--success);
        border-radius: 5px;
        padding: 0.5rem;
    }}
    
    /* Hide Streamlit defaults */
    #MainMenu, footer, header {{ visibility: hidden; }}
    </style>
    """, unsafe_allow_html=True)

def display_workflow_progress():
    """Display visual workflow progress indicator"""
    steps = st.session_state.steps_completed
    
    st.markdown("""
    <div class="workflow-progress">
        <div class="progress-step """ + ("completed" if steps['listing_details'] else "active") + """">
            <div>📝</div>
            <div>Product Details</div>
        </div>
        <div class="progress-step """ + ("completed" if steps['review_file'] else ("active" if steps['listing_details'] else "")) + """">
            <div>📊</div>
            <div>Review Data</div>
        </div>
        <div class="progress-step """ + ("completed" if steps['marketplace_files'] else "") + """">
            <div>📂</div>
            <div>Marketplace</div>
            <span class="status-badge optional">Optional</span>
        </div>
        <div class="progress-step """ + ("completed" if steps['analysis_run'] else "") + """">
            <div>🚀</div>
            <div>Analysis</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

def display_header():
    """Display enhanced header with workflow status"""
    st.markdown("""
    <div class="cyber-header">
        <h1 style="font-size: 3em; margin: 0;">VIVE HEALTH REVIEW INTELLIGENCE</h1>
        <p style="color: var(--primary); text-transform: uppercase; letter-spacing: 3px;">
            Medical Device Quality Management Platform v10.1
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Quick status
    current_step, status_message = get_workflow_status()
    col1, col2, col3 = st.columns([2, 3, 1])
    
    with col1:
        if st.button("💬 AI Assistant", use_container_width=True, 
                    help="Get help at any stage of the process"):
            st.session_state.show_ai_chat = not st.session_state.show_ai_chat
            st.rerun()
    
    with col2:
        st.info(f"📍 {status_message}")
    
    with col3:
        if st.button("🔄 Start Over", use_container_width=True):
            # Reset everything except AI analyzer
            ai_analyzer = st.session_state.ai_analyzer
            for key in list(st.session_state.keys()):
                if key != 'ai_analyzer':
                    del st.session_state[key]
            initialize_session_state()
            st.session_state.ai_analyzer = ai_analyzer
            st.rerun()

def display_tutorial():
    """Display interactive tutorial for first-time users"""
    if st.session_state.show_tutorial:
        # Use columns for better layout
        col1, col2, col3 = st.columns([1, 6, 1])
        
        with col2:
            st.markdown('<div class="tutorial-box">', unsafe_allow_html=True)
            st.markdown('<h3 style="color: #00D9FF; margin-top: 0;">🎯 Quick Start Guide</h3>', unsafe_allow_html=True)
            st.markdown('Welcome to the Vive Health Review Intelligence Platform! Follow these steps:', unsafe_allow_html=True)
            
            st.markdown("")  # Space
            
            # Step 1
            st.markdown('**1. Enter Product Details** *(Optional but Recommended)*', unsafe_allow_html=True)
            st.markdown('   - Paste your Amazon product URL for auto-population, OR', unsafe_allow_html=True)
            st.markdown('   - Manually enter your ASIN and listing details', unsafe_allow_html=True)
            
            st.markdown("")  # Space
            
            # Step 2
            st.markdown('**2. Upload Review Data** *(Required)*', unsafe_allow_html=True)
            st.markdown('   - Export reviews from Helium 10 as CSV', unsafe_allow_html=True)
            st.markdown('   - Upload the file in the Review Data section', unsafe_allow_html=True)
            
            st.markdown("")  # Space
            
            # Step 3
            st.markdown('**3. Add Marketplace Files** *(Optional)*', unsafe_allow_html=True)
            st.markdown('   - Upload returns and reimbursement reports', unsafe_allow_html=True)
            st.markdown('   - Get deeper insights into quality issues', unsafe_allow_html=True)
            
            st.markdown("")  # Space
            
            # Step 4
            st.markdown('**4. Run Analysis**', unsafe_allow_html=True)
            st.markdown('   - Choose quick metrics or full AI analysis', unsafe_allow_html=True)
            st.markdown('   - Export results for action planning', unsafe_allow_html=True)
            
            st.markdown("")  # Space
            
            st.markdown('<p style="color: #FFB700; font-weight: bold;">💡 Pro Tip: Use the AI Assistant at any time for help or to discuss your results!</p>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
            
            st.markdown("")  # Space
            
            if st.button("Got it! Let's start", type="primary", use_container_width=True):
                st.session_state.show_tutorial = False
                st.rerun()

def validate_asin(asin: str) -> Tuple[bool, str]:
    """Validate ASIN format"""
    if not asin:
        return False, "ASIN is required"
    
    asin = asin.strip().upper()
    if not re.match(r'^[A-Z0-9]{10}$', asin):
        return False, "ASIN must be exactly 10 alphanumeric characters"
    
    return True, asin

def extract_asin_from_url(url: str) -> Optional[str]:
    """Extract ASIN from Amazon URL with validation"""
    try:
        patterns = [
            r'/dp/([A-Z0-9]{10})',
            r'/product/([A-Z0-9]{10})',
            r'asin=([A-Z0-9]{10})',
            r'/([A-Z0-9]{10})(?:/|\?|$)',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, url, re.IGNORECASE)
            if match:
                return match.group(1).upper()
        return None
    except Exception as e:
        logger.error(f"ASIN extraction error: {e}")
        return None

def clean_text(text: str) -> str:
    """Clean scraped text"""
    if not text:
        return ""
    
    text = re.sub(r'\s+', ' ', text.strip())
    text = re.sub(r'See more product details|Read more|Show more', '', text, flags=re.IGNORECASE)
    return text.strip()

def scrape_amazon_product(url: str) -> Dict[str, Any]:
    """Enhanced Amazon product scraping with better error handling"""
    try:
        # Validate URL
        if not url or 'amazon.' not in url.lower():
            return {'success': False, 'error': 'Please enter a valid Amazon URL'}
        
        # Extract ASIN
        asin = extract_asin_from_url(url)
        if not asin:
            return {'success': False, 'error': 'Could not find ASIN in URL. Make sure it\'s a product page URL.'}
        
        # Make request with retry logic
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = requests.get(url, headers=HEADERS, timeout=10)
                if response.status_code == 200:
                    break
                elif response.status_code == 503:
                    if attempt < max_retries - 1:
                        time.sleep(2 * (attempt + 1))
                        continue
                    else:
                        return {'success': False, 'error': 'Amazon is blocking requests. Try again in a few minutes.'}
                else:
                    return {'success': False, 'error': f'Could not access page (Error {response.status_code})'}
            except requests.RequestException as e:
                if attempt == max_retries - 1:
                    return {'success': False, 'error': f'Connection failed: Please check your internet connection'}
                time.sleep(1)
        
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Extract product information with enhanced selectors
        product_data = {
            'success': True,
            'asin': asin,
            'url': url,
            'title': '',
            'bullet_points': [],
            'description': '',
            'brand': '',
            'category': ''
        }
        
        # Title extraction
        title_selectors = [
            '#productTitle',
            '.product-title-word-break',
            'h1[data-automation-id="title"]',
            'h1.a-size-large'
        ]
        
        for selector in title_selectors:
            title_element = soup.select_one(selector)
            if title_element:
                product_data['title'] = clean_text(title_element.get_text())
                break
        
        # Bullet points extraction
        bullet_selectors = [
            '#feature-bullets ul li span',
            '#feature-bullets ul.a-unordered-list li span',
            '.a-unordered-list.a-vertical.a-spacing-mini li span',
        ]
        
        for selector in bullet_selectors:
            bullets = soup.select(selector)
            if bullets:
                bullet_texts = []
                for bullet in bullets:
                    text = clean_text(bullet.get_text())
                    if text and len(text) > 10 and not any(skip in text.lower() for skip in ['make sure', 'see more']):
                        bullet_texts.append(text)
                
                if bullet_texts:
                    product_data['bullet_points'] = bullet_texts[:5]
                    break
        
        # Brand extraction
        brand_selectors = [
            '#bylineInfo',
            'a#bylineInfo',
            '.po-brand .po-break-word',
            'a[href*="/stores/"]'
        ]
        
        for selector in brand_selectors:
            brand_element = soup.select_one(selector)
            if brand_element:
                brand_text = clean_text(brand_element.get_text())
                brand_text = re.sub(r'^(Visit the |Brand: |Store: )', '', brand_text, flags=re.IGNORECASE)
                if brand_text and len(brand_text) < 50:
                    product_data['brand'] = brand_text
                    break
        
        # Basic validation
        if not product_data['title'] and not product_data['bullet_points']:
            return {
                'success': False, 
                'error': 'Could not extract product information. The page structure may have changed or you may need to solve a CAPTCHA.'
            }
        
        return product_data
        
    except Exception as e:
        logger.error(f"Scraping error: {e}")
        return {'success': False, 'error': f'Unexpected error: {str(e)}'}

def display_step1_product_details():
    """Step 1: Product Details Entry with improved UX"""
    st.markdown("""
    <div class="workflow-box">
        <span class="workflow-number">1</span>
        <h3 style="margin-top: 0;">📝 Product Details 
            <span class="status-badge """ + ("success" if st.session_state.steps_completed['listing_details'] else "pending") + """">
                """ + ("✓ Complete" if st.session_state.steps_completed['listing_details'] else "Optional but Recommended") + """
            </span>
        </h3>
    </div>
    """, unsafe_allow_html=True)
    
    # Tab interface for clarity
    tab1, tab2 = st.tabs(["🔗 Auto-Populate from URL", "✏️ Manual Entry"])
    
    with tab1:
        st.markdown("""
        <div class="help-tip">
            💡 Paste your Amazon product URL to automatically extract listing details for more targeted analysis
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns([4, 1])
        with col1:
            amazon_url = st.text_input(
                "Amazon Product URL",
                placeholder="https://www.amazon.com/dp/B08XYZ1234",
                help="Full URL from your product page"
            )
        
        with col2:
            st.markdown("<br>", unsafe_allow_html=True)  # Spacing
            extract_button = st.button("🚀 Extract", type="primary", use_container_width=True)
        
        if extract_button and amazon_url:
            with st.spinner("🔍 Extracting product information..."):
                result = scrape_amazon_product(amazon_url)
                
                if result['success']:
                    # Update session state
                    st.session_state.listing_details.update({
                        'title': result.get('title', ''),
                        'brand': result.get('brand', ''),
                        'asin': result.get('asin', ''),
                        'url': result.get('url', '')
                    })
                    
                    # Update bullet points
                    bullets = result.get('bullet_points', [])
                    for i in range(5):
                        if i < len(bullets):
                            st.session_state.listing_details['bullet_points'][i] = bullets[i]
                    
                    st.session_state.auto_populated = True
                    st.session_state.use_listing_details = True
                    st.session_state.steps_completed['listing_details'] = True
                    
                    st.success("✅ Product details extracted successfully!")
                    st.rerun()
                else:
                    st.error(f"❌ {result.get('error', 'Unknown error')}")
    
    with tab2:
        st.markdown("""
        <div class="help-tip">
            💡 Enter at least the ASIN to enable marketplace data correlation
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            asin_input = st.text_input(
                "ASIN* (Required for marketplace data)",
                value=st.session_state.listing_details.get('asin', ''),
                max_chars=10,
                help="10-character Amazon product identifier"
            )
            
            if asin_input:
                valid, clean_asin = validate_asin(asin_input)
                if valid:
                    st.session_state.listing_details['asin'] = clean_asin
                    if not st.session_state.steps_completed['listing_details']:
                        st.session_state.steps_completed['listing_details'] = True
                        st.session_state.use_listing_details = True
                else:
                    st.error(clean_asin)  # Shows error message
        
        with col2:
            brand_input = st.text_input(
                "Brand Name",
                value=st.session_state.listing_details.get('brand', ''),
                help="Your product's brand"
            )
            if brand_input:
                st.session_state.listing_details['brand'] = brand_input
        
        # Title
        title_input = st.text_input(
            "Product Title",
            value=st.session_state.listing_details.get('title', ''),
            max_chars=200,
            help="Current Amazon listing title"
        )
        if title_input:
            st.session_state.listing_details['title'] = title_input
    
    # Show current details if populated
    if st.session_state.steps_completed['listing_details']:
        st.markdown("---")
        st.markdown("### ✅ Current Product Details")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.info(f"**ASIN:** {st.session_state.listing_details.get('asin', 'Not set')}")
        with col2:
            st.info(f"**Brand:** {st.session_state.listing_details.get('brand', 'Not set')}")
        with col3:
            if st.session_state.auto_populated:
                st.success("**Source:** Auto-populated")
            else:
                st.info("**Source:** Manual entry")

def display_step2_review_upload():
    """Step 2: Review File Upload with validation"""
    st.markdown("""
    <div class="workflow-box">
        <span class="workflow-number">2</span>
        <h3 style="margin-top: 0;">📊 Review Data Upload 
            <span class="status-badge """ + ("success" if st.session_state.steps_completed['review_file'] else "pending") + """">
                """ + ("✓ Complete" if st.session_state.steps_completed['review_file'] else "Required") + """
            </span>
        </h3>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="help-tip">
        💡 Export your reviews from Helium 10's Review Insights tool as a CSV file
    </div>
    """, unsafe_allow_html=True)
    
    # Analysis settings
    col1, col2 = st.columns([3, 1])
    with col1:
        st.checkbox(
            "🎯 Analyze ALL reviews with AI (recommended)",
            value=st.session_state.analyze_all_reviews,
            key="analyze_all_checkbox",
            help="When checked, AI analyzes all reviews regardless of filters. Filters only affect metric displays."
        )
        st.session_state.analyze_all_reviews = st.session_state.analyze_all_checkbox
    
    # File upload
    uploaded_file = st.file_uploader(
        "Upload Helium 10 Review Export",
        type=['csv', 'xlsx', 'xls'],
        help="CSV or Excel file with columns: Title, Body, Rating"
    )
    
    if uploaded_file:
        try:
            # Process file
            with st.spinner("🔄 Processing review data..."):
                if uploaded_file.name.endswith('.csv'):
                    try:
                        df = pd.read_csv(uploaded_file, encoding='utf-8')
                    except UnicodeDecodeError:
                        uploaded_file.seek(0)
                        df = pd.read_csv(uploaded_file, encoding='latin-1')
                else:
                    df = pd.read_excel(uploaded_file, engine='openpyxl')
                
                # Validate columns
                required_cols = ['Title', 'Body', 'Rating']
                missing = [col for col in required_cols if col not in df.columns]
                
                if missing:
                    st.error(f"❌ Missing required columns: {', '.join(missing)}")
                    st.info("Required columns: Title, Body, Rating")
                    
                    # Show sample of what was found
                    st.markdown("**Found columns:**")
                    st.write(", ".join(df.columns.tolist()))
                    return
                
                # Clean and validate data
                initial_count = len(df)
                df = df.dropna(subset=['Rating', 'Body'])
                df['Rating'] = pd.to_numeric(df['Rating'], errors='coerce')
                df = df[df['Rating'].between(1, 5)]
                
                if len(df) == 0:
                    st.error("❌ No valid reviews found after data cleaning")
                    return
                
                # Store data
                st.session_state.uploaded_data = {
                    'df': df,
                    'df_filtered': df,  # Will be filtered based on settings
                    'product_info': {
                        'asin': st.session_state.listing_details.get('asin', 'Unknown'),
                        'total_reviews': len(df),
                        'filtered_reviews': len(df)
                    },
                    'metrics': None  # Will be calculated when needed
                }
                
                st.session_state.steps_completed['review_file'] = True
                
                # Success message with stats
                st.success(f"✅ Loaded {len(df)} valid reviews from {initial_count} total rows")
                
                # Quick preview
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    avg_rating = df['Rating'].mean()
                    st.metric("Average Rating", f"{avg_rating:.2f}/5")
                with col2:
                    verified_count = sum(df.get('Verified', pd.Series()).eq('yes'))
                    st.metric("Verified Reviews", f"{verified_count}/{len(df)}")
                with col3:
                    recent_date = pd.to_datetime(df.get('Date', pd.Series()), errors='coerce').max()
                    if pd.notna(recent_date):
                        st.metric("Most Recent", recent_date.strftime('%b %Y'))
                with col4:
                    rating_dist = df['Rating'].value_counts().sort_index()
                    most_common = rating_dist.idxmax()
                    st.metric("Most Common Rating", f"{most_common}★")
                
        except Exception as e:
            st.error(f"❌ Error processing file: {str(e)}")
            logger.error(f"Upload error: {e}", exc_info=True)

def display_step3_marketplace_files():
    """Step 3: Optional Marketplace Files with clear guidance"""
    st.markdown("""
    <div class="workflow-box">
        <span class="workflow-number">3</span>
        <h3 style="margin-top: 0;">📂 Marketplace Data 
            <span class="status-badge optional">Optional - Enhances Analysis</span>
        </h3>
    </div>
    """, unsafe_allow_html=True)
    
    if not st.session_state.listing_details.get('asin'):
        st.warning("⚠️ Enter an ASIN in Step 1 to enable marketplace data correlation")
        return
    
    st.markdown(f"""
    <div class="help-tip">
        💡 Upload Amazon seller reports to analyze returns and reimbursements for ASIN: <strong>{st.session_state.listing_details['asin']}</strong>
    </div>
    """, unsafe_allow_html=True)
    
    if not DETECTOR_AVAILABLE:
        st.error("❌ Marketplace file detector module not available")
        return
    
    # File upload interface
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("#### 💰 Reimbursements")
        st.markdown('<small style="color: #666;">Transaction report</small>', unsafe_allow_html=True)
        reimb_file = st.file_uploader(
            "Upload",
            type=['csv'],
            key="reimb_upload",
            label_visibility="collapsed"
        )
        
        if reimb_file:
            process_marketplace_file(reimb_file, 'reimbursements')
    
    with col2:
        st.markdown("#### 📦 FBA Returns")
        st.markdown('<small style="color: #666;">Customer returns report</small>', unsafe_allow_html=True)
        fba_file = st.file_uploader(
            "Upload",
            type=['csv'],
            key="fba_upload",
            label_visibility="collapsed"
        )
        
        if fba_file:
            process_marketplace_file(fba_file, 'fba_returns')
    
    with col3:
        st.markdown("#### 🚚 FBM Returns")
        st.markdown('<small style="color: #666;">Manage returns report</small>', unsafe_allow_html=True)
        fbm_file = st.file_uploader(
            "Upload",
            type=['csv', 'tsv', 'txt'],
            key="fbm_upload",
            label_visibility="collapsed"
        )
        
        if fbm_file:
            process_marketplace_file(fbm_file, 'fbm_returns')
    
    # Display summary if files uploaded
    if any(st.session_state.marketplace_files.values()):
        st.markdown("---")
        st.markdown("### 📊 Marketplace Data Summary")
        
        # Process correlations to ensure data is up to date
        process_marketplace_correlations()
        
        # Get data from marketplace_data (processed correlations)
        if st.session_state.marketplace_data:
            total_returns = 0
            total_reimbursements = 0
            
            # Count returns from marketplace_data
            if 'return_patterns' in st.session_state.marketplace_data:
                for file_type, return_data in st.session_state.marketplace_data['return_patterns'].items():
                    if return_data and 'count' in return_data:
                        total_returns += return_data['count']
            
            # Get reimbursements from marketplace_data
            if 'financial_impact' in st.session_state.marketplace_data:
                if 'reimbursements' in st.session_state.marketplace_data['financial_impact']:
                    reimb_data = st.session_state.marketplace_data['financial_impact']['reimbursements']
                    total_reimbursements = reimb_data.get('total_amount', 0)
            
            col1, col2, col3 = st.columns(3)
            with col1:
                files_count = sum(1 for v in st.session_state.marketplace_files.values() if v)
                st.metric("Files Loaded", files_count)
            with col2:
                st.metric("Total Returns", total_returns)
                if total_returns == 0 and any(st.session_state.marketplace_files.values()):
                    st.caption("No returns found for this ASIN")
            with col3:
                st.metric("Reimbursements", f"${total_reimbursements:.2f}")
                if total_reimbursements == 0 and st.session_state.marketplace_files.get('reimbursements'):
                    st.caption("No reimbursements found for this ASIN")
        
        st.session_state.steps_completed['marketplace_files'] = True

def process_marketplace_file(uploaded_file, expected_type: str):
    """Process marketplace file with better error handling"""
    try:
        with st.spinner(f"Processing {expected_type.replace('_', ' ')}..."):
            # Read file
            file_content = uploaded_file.read()
            
            # Process with detector
            result = AmazonFileDetector.process_file(file_content, uploaded_file.name)
            
            if result['success']:
                detected_type = result['file_type']
                
                # Verify file type
                if detected_type != expected_type:
                    st.warning(f"⚠️ This appears to be a {detected_type.replace('_', ' ')} file")
                    if not st.checkbox(f"Process as {expected_type.replace('_', ' ')} anyway?", 
                                      key=f"override_{expected_type}_{uploaded_file.name}"):
                        return
                
                # Store processed data
                st.session_state.marketplace_files[expected_type] = {
                    'dataframe': result['dataframe'],
                    'summary': result['summary'],
                    'filename': uploaded_file.name
                }
                
                # Always process correlations after file upload
                process_marketplace_correlations()
                
                # Get ASIN-specific count
                target_asin = st.session_state.listing_details.get('asin')
                asin_count = 0
                
                if target_asin and st.session_state.marketplace_data:
                    if expected_type in ['fba_returns', 'fbm_returns']:
                        if 'return_patterns' in st.session_state.marketplace_data:
                            if expected_type in st.session_state.marketplace_data['return_patterns']:
                                asin_count = st.session_state.marketplace_data['return_patterns'][expected_type].get('count', 0)
                    elif expected_type == 'reimbursements':
                        if 'financial_impact' in st.session_state.marketplace_data:
                            if 'reimbursements' in st.session_state.marketplace_data['financial_impact']:
                                asin_count = st.session_state.marketplace_data['financial_impact']['reimbursements'].get('count', 0)
                
                # Show success with ASIN-specific info
                total_rows = result['summary']['row_count']
                if target_asin:
                    st.success(f"✅ Processed {total_rows} rows, found {asin_count} records for ASIN {target_asin}")
                else:
                    st.success(f"✅ Processed successfully: {total_rows} rows")
                
            else:
                st.error(f"❌ {result.get('error', 'Processing failed')}")
                
    except Exception as e:
        st.error(f"❌ Error: {str(e)}")
        logger.error(f"Marketplace file error: {e}", exc_info=True)

def process_marketplace_correlations():
    """Process correlations with current ASIN - FIXED to properly count returns"""
    target_asin = st.session_state.listing_details.get('asin')
    if not target_asin:
        logger.info("No ASIN provided for correlation")
        return
    
    # Gather all marketplace dataframes
    marketplace_dfs = {}
    for file_type, data in st.session_state.marketplace_files.items():
        if data and 'dataframe' in data:
            marketplace_dfs[file_type] = data['dataframe']
    
    if marketplace_dfs:
        logger.info(f"Processing correlations for ASIN {target_asin} with {len(marketplace_dfs)} files")
        
        # Get correlations using the detector
        correlations = AmazonFileDetector.correlate_with_asin(marketplace_dfs, target_asin)
        
        # Store the correlation results
        st.session_state.marketplace_data = correlations
        
        # Log what was found
        if correlations:
            logger.info(f"Correlation results: {json.dumps({k: len(v) if isinstance(v, dict) else v for k, v in correlations.items()})}")
            
            # Show insights if data found
            total_returns = 0
            if 'return_patterns' in correlations:
                for file_type, return_data in correlations['return_patterns'].items():
                    if return_data and 'count' in return_data:
                        total_returns += return_data['count']
                        logger.info(f"Found {return_data['count']} returns in {file_type}")
            
            if total_returns > 0 or correlations.get('financial_impact'):
                st.info(f"🔍 Found marketplace data for ASIN {target_asin}")
        else:
            logger.info(f"No correlations found for ASIN {target_asin}")

def display_step4_analysis():
    """Step 4: Run Analysis with clear options"""
    st.markdown("""
    <div class="workflow-box">
        <span class="workflow-number">4</span>
        <h3 style="margin-top: 0;">🚀 Run Analysis</h3>
    </div>
    """, unsafe_allow_html=True)
    
    if not st.session_state.steps_completed['review_file']:
        st.warning("⚠️ Please upload review data first (Step 2)")
        return
    
    # Analysis options
    col1, col2 = st.columns(2)
    
    with col1:
        # Quick metrics button
        if st.button("📊 Quick Metrics Only", use_container_width=True,
                    help="Fast statistical analysis without AI"):
            with st.spinner("Calculating metrics..."):
                df = apply_filters(st.session_state.uploaded_data['df'])
                metrics = calculate_advanced_metrics(df)
                
                if metrics:
                    st.session_state.uploaded_data['df_filtered'] = df
                    st.session_state.uploaded_data['metrics'] = metrics
                    st.session_state.current_view = 'metrics'
                    st.session_state.steps_completed['analysis_run'] = True
                    st.rerun()
                else:
                    st.error("Failed to calculate metrics")
    
    with col2:
        # AI analysis button
        ai_status = check_ai_status()
        
        if ai_status:
            button_label = "🤖 Full AI Analysis"
            if st.session_state.steps_completed['listing_details']:
                button_label += " ✨"
            if st.session_state.steps_completed['marketplace_files']:
                button_label += " 📂"
                
            if st.button(button_label, type="primary", use_container_width=True,
                        help="Comprehensive AI-powered optimization recommendations"):
                run_full_analysis()
        else:
            st.button("🤖 AI Unavailable", disabled=True, use_container_width=True)
            st.error("AI service not configured. Add OPENAI_API_KEY to enable.")
    
    # Show what will be analyzed
    st.markdown("---")
    st.markdown("### 📋 Analysis Scope")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        reviews_count = len(st.session_state.uploaded_data['df'])
        st.info(f"**Reviews:** {reviews_count} total")
        if st.session_state.analyze_all_reviews:
            st.success("AI will analyze ALL reviews")
        else:
            filtered_count = len(apply_filters(st.session_state.uploaded_data['df']))
            st.warning(f"AI will analyze {filtered_count} filtered reviews")
    
    with col2:
        if st.session_state.steps_completed['listing_details']:
            st.success(f"**ASIN:** {st.session_state.listing_details.get('asin', 'Set')}")
        else:
            st.info("**ASIN:** Not provided")
    
    with col3:
        if st.session_state.steps_completed['marketplace_files']:
            files_count = sum(1 for v in st.session_state.marketplace_files.values() if v)
            st.success(f"**Marketplace:** {files_count} files")
            
            # Show returns count if available
            if st.session_state.marketplace_data and 'return_patterns' in st.session_state.marketplace_data:
                total_returns = sum(
                    data.get('count', 0) 
                    for data in st.session_state.marketplace_data['return_patterns'].values()
                )
                if total_returns > 0:
                    st.caption(f"{total_returns} returns found")
        else:
            st.info("**Marketplace:** No files")

def run_full_analysis():
    """Run comprehensive analysis with progress tracking"""
    try:
        # First calculate metrics
        with st.spinner("📊 Calculating metrics..."):
            df = apply_filters(st.session_state.uploaded_data['df'])
            metrics = calculate_advanced_metrics(df)
            
            if not metrics:
                st.error("Failed to calculate metrics")
                return
            
            st.session_state.uploaded_data['df_filtered'] = df
            st.session_state.uploaded_data['metrics'] = metrics
        
        # Then run AI analysis
        with st.spinner("🤖 Running AI analysis... This may take a moment."):
            ai_results = run_comprehensive_ai_analysis(
                df,
                metrics,
                st.session_state.uploaded_data['product_info']
            )
            
            if ai_results:
                st.session_state.analysis_results = ai_results
                st.session_state.current_view = 'ai_results'
                st.session_state.steps_completed['analysis_run'] = True
                st.success("✅ Analysis complete!")
                time.sleep(1)  # Brief pause to show success
                st.rerun()
            else:
                st.error("AI analysis failed. Please check your configuration.")
                
    except Exception as e:
        st.error(f"Analysis error: {str(e)}")
        logger.error(f"Analysis error: {e}", exc_info=True)

def check_ai_status():
    """Check AI availability with initialization"""
    if not AI_AVAILABLE:
        return False
    
    try:
        if st.session_state.ai_analyzer is None:
            st.session_state.ai_analyzer = enhanced_ai_analysis.EnhancedAIAnalyzer()
        
        status = st.session_state.ai_analyzer.get_api_status()
        return status.get('available', False)
    except Exception as e:
        logger.error(f"AI status check error: {e}")
        return False

def display_ai_chat():
    """Enhanced AI chat with context awareness"""
    st.markdown("""
    <div class="neon-box">
        <h3 style="color: var(--primary);">🤖 AI Assistant</h3>
    </div>
    """, unsafe_allow_html=True)
    
    # Context-aware help
    current_step, _ = get_workflow_status()
    
    if not st.session_state.chat_messages:
        help_text = "I'm here to help! "
        
        if current_step == 1:
            help_text += "Need help finding your ASIN or understanding the workflow?"
        elif current_step == 2:
            help_text += "Having trouble with your review export or need help interpreting metrics?"
        elif current_step == 3:
            help_text += "Want to understand your analysis results or get implementation advice?"
        else:
            help_text += "I can explain any part of the analysis or suggest next steps."
        
        st.info(help_text)
    
    # Display chat history
    for message in st.session_state.chat_messages:
        role = "You" if message["role"] == "user" else "AI"
        with st.chat_message(message["role"]):
            st.write(f"**{role}:** {message['content']}")
    
    # Input
    user_input = st.chat_input("Ask me anything about your analysis or Amazon optimization...")
    
    if user_input:
        # Add user message
        st.session_state.chat_messages.append({"role": "user", "content": user_input})
        
        # Get AI response
        with st.spinner("Thinking..."):
            response = get_ai_chat_response(user_input)
            st.session_state.chat_messages.append({"role": "assistant", "content": response})
        
        st.rerun()
    
    # Clear chat button
    if st.session_state.chat_messages:
        if st.button("🔄 Clear Chat", use_container_width=True):
            st.session_state.chat_messages = []
            st.rerun()

def get_ai_chat_response(user_input: str) -> str:
    """Get contextual AI response"""
    if not check_ai_status():
        return "AI service is currently unavailable. Please check your API configuration."
    
    try:
        # Build context
        context_parts = []
        
        # Current workflow status
        current_step, status = get_workflow_status()
        context_parts.append(f"User is at step {current_step}: {status}")
        
        # Data status
        if st.session_state.listing_details.get('asin'):
            context_parts.append(f"Working with ASIN: {st.session_state.listing_details['asin']}")
        
        if st.session_state.uploaded_data:
            context_parts.append(f"Has {st.session_state.uploaded_data['product_info']['total_reviews']} reviews loaded")
        
        if st.session_state.analysis_results:
            context_parts.append("Has completed analysis")
        
        context = "\n".join(context_parts)
        
        system_prompt = f"""You are an expert Amazon listing optimization assistant for medical devices.
        Help users navigate the Review Intelligence Platform and understand their results.
        
        Current context:
        {context}
        
        Be helpful, specific, and actionable. Focus on medical device quality management when relevant."""
        
        result = st.session_state.ai_analyzer.api_client.call_api([
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_input}
        ], max_tokens=500, temperature=0.7)
        
        return result['result'] if result['success'] else f"Error: {result.get('error', 'Unknown')}"
        
    except Exception as e:
        logger.error(f"Chat error: {e}")
        return "I encountered an error. Please try rephrasing your question."

# Continue with remaining helper functions from original code...
# (parse_amazon_date, calculate_basic_stats, analyze_sentiment_patterns, etc.)
# These remain unchanged

def parse_amazon_date(date_string):
    """Parse Amazon review dates"""
    try:
        if pd.isna(date_string) or not date_string:
            return None
        date_part = str(date_string).split("on ")[-1] if "on " in str(date_string) else str(date_string)
        for fmt in ['%B %d, %Y', '%b %d, %Y', '%m/%d/%Y', '%Y-%m-%d']:
            try:
                return datetime.strptime(date_part.strip(), fmt).date()
            except:
                continue
        return pd.to_datetime(date_part, errors='coerce').date()
    except:
        return None

def apply_filters(df):
    """Apply user-selected filters"""
    df_filtered = df.copy()
    
    # Time filter
    if st.session_state.selected_timeframe != 'all' and 'Date' in df.columns:
        df_filtered['parsed_date'] = pd.to_datetime(df_filtered['Date'].apply(parse_amazon_date))
        days_map = {'30d': 30, '90d': 90, '180d': 180, '365d': 365}
        if st.session_state.selected_timeframe in days_map:
            cutoff = datetime.now() - timedelta(days=days_map[st.session_state.selected_timeframe])
            df_filtered = df_filtered[df_filtered['parsed_date'] >= cutoff]
    
    # Rating filter
    if st.session_state.filter_rating != 'all':
        if st.session_state.filter_rating in ['1', '2', '3', '4', '5']:
            df_filtered = df_filtered[df_filtered['Rating'] == int(st.session_state.filter_rating)]
        elif st.session_state.filter_rating == 'positive':
            df_filtered = df_filtered[df_filtered['Rating'] >= 4]
        elif st.session_state.filter_rating == 'negative':
            df_filtered = df_filtered[df_filtered['Rating'] <= 2]
    
    return df_filtered

def calculate_advanced_metrics(df):
    """Calculate all metrics - keeping original logic"""
    try:
        metrics = {
            'basic_stats': calculate_basic_stats(df),
            'sentiment_breakdown': analyze_sentiment_patterns(df),
            'keyword_analysis': extract_keywords(df),
            'temporal_trends': analyze_temporal_trends(df),
            'verified_vs_unverified': analyze_verification_impact(df),
            'review_quality_scores': calculate_review_quality(df),
            'issue_categories': categorize_issues(df),
            'competitor_mentions': find_competitor_mentions(df)
        }
        metrics['listing_health_score'] = calculate_listing_health_score(metrics)
        return metrics
    except Exception as e:
        logger.error(f"Error calculating metrics: {e}")
        return None

def calculate_basic_stats(df):
    """Calculate basic statistics"""
    try:
        ratings = df['Rating'].dropna()
        rating_counts = ratings.value_counts().sort_index().to_dict()
        
        stats = {
            'total_reviews': len(df),
            'average_rating': round(ratings.mean(), 2),
            'rating_distribution': rating_counts,
            'verified_count': sum(df['Verified'] == 'yes') if 'Verified' in df.columns else 0,
            '1_2_star_percentage': round((sum(ratings <= 2) / len(ratings)) * 100, 1) if len(ratings) > 0 else 0,
            '4_5_star_percentage': round((sum(ratings >= 4) / len(ratings)) * 100, 1) if len(ratings) > 0 else 0,
            'median_rating': ratings.median(),
            'rating_std': round(ratings.std(), 2)
        }
        
        if 'Date' in df.columns:
            df['parsed_date'] = df['Date'].apply(parse_amazon_date)
            valid_dates = df['parsed_date'].dropna()
            if len(valid_dates) > 0:
                stats['date_range'] = {
                    'earliest': valid_dates.min(),
                    'latest': valid_dates.max(),
                    'days_covered': (valid_dates.max() - valid_dates.min()).days
                }
        
        return stats
    except Exception as e:
        logger.error(f"Stats calculation error: {e}")
        return None

def analyze_sentiment_patterns(df):
    """Analyze sentiment in reviews"""
    sentiments = {
        'positive_keywords': ['love', 'great', 'excellent', 'perfect', 'amazing', 'best', 'wonderful', 'quality'],
        'negative_keywords': ['hate', 'terrible', 'awful', 'worst', 'horrible', 'poor', 'cheap', 'broken']
    }
    
    results = {'positive': 0, 'negative': 0, 'neutral': 0, 'mixed': 0}
    
    for _, row in df.iterrows():
        if pd.isna(row.get('Body')):
            continue
        text = str(row['Body']).lower()
        pos_count = sum(1 for word in sentiments['positive_keywords'] if word in text)
        neg_count = sum(1 for word in sentiments['negative_keywords'] if word in text)
        
        if pos_count > neg_count:
            results['positive'] += 1
        elif neg_count > pos_count:
            results['negative'] += 1
        elif pos_count == neg_count and pos_count > 0:
            results['mixed'] += 1
        else:
            results['neutral'] += 1
    
    return results

def extract_keywords(df, top_n=20):
    """Extract keywords from reviews"""
    positive_reviews = df[df['Rating'] >= 4]['Body'].dropna()
    negative_reviews = df[df['Rating'] <= 2]['Body'].dropna()
    
    def get_keywords(texts):
        all_words = []
        for text in texts:
            words = re.findall(r'\b[a-z]+\b', str(text).lower())
            all_words.extend(words)
        
        stopwords = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 
                    'with', 'is', 'was', 'it', 'this', 'that', 'have', 'has', 'i', 'my', 'your'}
        filtered_words = [w for w in all_words if w not in stopwords and len(w) > 3]
        return Counter(filtered_words).most_common(top_n)
    
    return {
        'positive_keywords': get_keywords(positive_reviews),
        'negative_keywords': get_keywords(negative_reviews)
    }

def analyze_temporal_trends(df):
    """Analyze rating trends over time"""
    if 'Date' not in df.columns:
        return {}
    
    df['parsed_date'] = df['Date'].apply(parse_amazon_date)
    df_with_dates = df.dropna(subset=['parsed_date'])
    
    if len(df_with_dates) == 0:
        return {}
    
    df_with_dates['month'] = pd.to_datetime(df_with_dates['parsed_date']).dt.to_period('M')
    monthly_avg = df_with_dates.groupby('month')['Rating'].agg(['mean', 'count'])
    monthly_avg.index = monthly_avg.index.astype(str)
    
    if len(monthly_avg) > 1:
        ratings = monthly_avg['mean'].values
        trend = 'improving' if ratings[-1] > ratings[0] else 'declining' if ratings[-1] < ratings[0] else 'stable'
    else:
        trend = 'insufficient_data'
    
    return {
        'trend': trend,
        'monthly_averages': monthly_avg.to_dict(),
        'recent_performance': monthly_avg.tail(3)['mean'].mean() if len(monthly_avg) >= 3 else None
    }

def categorize_issues(df):
    """Categorize issues from negative reviews"""
    categories = {
        'quality': ['quality', 'cheap', 'flimsy', 'broken', 'defect', 'poor'],
        'size_fit': ['size', 'fit', 'small', 'large', 'tight', 'loose'],
        'shipping': ['shipping', 'package', 'delivery', 'damaged', 'late'],
        'functionality': ['work', 'function', 'feature', 'button', 'operate'],
        'value': ['price', 'expensive', 'value', 'worth', 'money'],
        'durability': ['last', 'durable', 'broke', 'wear', 'tear'],
        'instructions': ['instructions', 'manual', 'setup', 'confusing'],
        'customer_service': ['service', 'support', 'response', 'help']
    }
    
    issue_counts = {cat: 0 for cat in categories}
    negative_reviews = df[df['Rating'] <= 3]['Body'].dropna()
    
    for review in negative_reviews:
        review_lower = str(review).lower()
        for category, keywords in categories.items():
            if any(keyword in review_lower for keyword in keywords):
                issue_counts[category] += 1
    
    return issue_counts

def calculate_review_quality(df):
    """Calculate review quality scores"""
    quality_scores = []
    
    for _, row in df.iterrows():
        if pd.isna(row.get('Body')):
            continue
        body = str(row['Body'])
        score = 0
        
        word_count = len(body.split())
        if word_count > 50: score += 3
        elif word_count > 20: score += 2
        elif word_count > 10: score += 1
        
        detail_keywords = ['size', 'color', 'material', 'quality', 'feature']
        score += sum(1 for keyword in detail_keywords if keyword in body.lower())
        
        if any(phrase in body.lower() for phrase in ['pros:', 'cons:', 'update:']):
            score += 2
        
        quality_scores.append(score)
    
    return {
        'avg_quality_score': np.mean(quality_scores) if quality_scores else 0,
        'high_quality_count': sum(1 for s in quality_scores if s >= 5),
        'low_quality_count': sum(1 for s in quality_scores if s <= 2)
    }

def analyze_verification_impact(df):
    """Compare verified vs unverified reviews"""
    if 'Verified' not in df.columns:
        return {}
    
    verified = df[df['Verified'] == 'yes']
    unverified = df[df['Verified'] != 'yes']
    
    return {
        'verified_avg_rating': verified['Rating'].mean() if len(verified) > 0 else None,
        'unverified_avg_rating': unverified['Rating'].mean() if len(unverified) > 0 else None,
        'verified_count': len(verified),
        'unverified_count': len(unverified)
    }

def find_competitor_mentions(df):
    """Find competitor mentions in reviews"""
    patterns = [r'better than\s+\w+', r'compared to\s+\w+', r'switch from\s+\w+']
    mentions = []
    
    for _, row in df.iterrows():
        if pd.isna(row.get('Body')):
            continue
        text = str(row['Body'])
        for pattern in patterns:
            mentions.extend(re.findall(pattern, text, re.IGNORECASE))
    
    return Counter(mentions).most_common(10)

def calculate_listing_health_score(metrics):
    """Calculate overall health score"""
    components = {
        'rating_score': (metrics['basic_stats']['average_rating'] / 5) * 25,
        'review_volume_score': min((metrics['basic_stats']['total_reviews'] / 100) * 15, 15),
        'sentiment_score': (metrics['sentiment_breakdown']['positive'] / sum(metrics['sentiment_breakdown'].values()) * 20) if sum(metrics['sentiment_breakdown'].values()) > 0 else 0,
        'trend_score': 15 if metrics['temporal_trends'].get('trend') == 'improving' else 10 if metrics['temporal_trends'].get('trend') == 'stable' else 5,
        'quality_score': min((metrics['review_quality_scores'].get('avg_quality_score', 0) / 8) * 15, 15),
        'issue_score': max(10 - (sum(metrics['issue_categories'].values()) / metrics['basic_stats']['total_reviews'] * 50), 0)
    }
    
    total_score = sum(components.values())
    
    return {
        'total_score': round(total_score, 1),
        'components': components,
        'grade': 'A' if total_score >= 85 else 'B' if total_score >= 70 else 'C' if total_score >= 55 else 'D' if total_score >= 40 else 'F',
        'status': 'Excellent' if total_score >= 85 else 'Good' if total_score >= 70 else 'Needs Improvement' if total_score >= 55 else 'Poor' if total_score >= 40 else 'Critical'
    }

def run_comprehensive_ai_analysis(df, metrics, product_info):
    """Run AI analysis with progress tracking"""
    if not check_ai_status():
        st.error("AI service is not available.")
        return None
    
    try:
        # Determine which dataframe to use
        if st.session_state.analyze_all_reviews and 'df' in st.session_state.uploaded_data:
            analysis_df = st.session_state.uploaded_data['df']
            analysis_note = f"Analyzing ALL {len(analysis_df)} reviews"
        else:
            analysis_df = df
            analysis_note = f"Analyzing {len(analysis_df)} filtered reviews"
        
        reviews = prepare_reviews_for_ai(analysis_df)
        if not reviews:
            st.error("No reviews to analyze")
            return None
        
        # Add marketplace data note
        if st.session_state.marketplace_data:
            analysis_note += " + marketplace data"
        
        logger.info(f"Running AI analysis: {analysis_note}")
        
        # Call enhanced analyzer
        result = st.session_state.ai_analyzer.analyze_reviews_for_listing_optimization(
            reviews=reviews,
            product_info=product_info,
            listing_details=st.session_state.listing_details if st.session_state.use_listing_details else None,
            metrics=metrics,
            marketplace_data=st.session_state.marketplace_data
        )
        
        if result:
            return {
                'success': True,
                'analysis': result,
                'timestamp': datetime.now(),
                'reviews_analyzed': len(reviews),
                'total_reviews': len(analysis_df),
                'analysis_scope': 'all_reviews' if st.session_state.analyze_all_reviews else 'filtered_reviews',
                'marketplace_data_included': bool(st.session_state.marketplace_data)
            }
        else:
            st.error("AI analysis failed")
            return None
            
    except Exception as e:
        logger.error(f"AI analysis error: {e}")
        st.error(f"Error: {str(e)}")
        return None

def prepare_reviews_for_ai(df):
    """Prepare reviews for AI analysis"""
    reviews = []
    
    for idx, row in df.iterrows():
        body = row.get('Body')
        if pd.isna(body) or not str(body).strip():
            continue
        
        rating = int(float(row.get('Rating', 3)))
        rating = max(1, min(5, rating))
        
        review = {
            'id': idx + 1,
            'rating': rating,
            'title': str(row.get('Title', '')).strip()[:200],
            'body': str(body).strip()[:1000],
            'verified': row.get('Verified', '') == 'yes',
            'date': str(row.get('Date', ''))
        }
        reviews.append(review)
    
    # Sort by rating (negative first)
    reviews.sort(key=lambda x: (x['rating'], x['date']))
    
    # Balance sampling
    if len(reviews) > 100:
        low = [r for r in reviews if r['rating'] <= 2][:25]
        mid = [r for r in reviews if r['rating'] == 3][:15]
        high = [r for r in reviews if r['rating'] >= 4][:35]
        reviews = low + mid + high
    
    return reviews

# View display functions
def display_metrics_view():
    """Display metrics dashboard"""
    if not st.session_state.uploaded_data:
        st.error("No data available")
        return
    
    metrics = st.session_state.uploaded_data['metrics']
    
    st.markdown('<h1>📊 METRICS DASHBOARD</h1>', unsafe_allow_html=True)
    
    # Health score card
    score = metrics['listing_health_score']['total_score']
    grade = metrics['listing_health_score']['grade']
    status = metrics['listing_health_score']['status']
    
    color_map = {'A': 'success', 'B': 'success', 'C': 'warning', 'D': 'warning', 'F': 'danger'}
    score_color = f"var(--{color_map.get(grade, 'warning')})"
    
    st.markdown(f"""
    <div class="neon-box" style="text-align: center;">
        <h2 style="color: {score_color}; font-size: 3rem; margin: 0;">{score:.0f}/100</h2>
        <h3 style="color: {score_color};">Grade: {grade} - {status}</h3>
    </div>
    """, unsafe_allow_html=True)
    
    # Quick metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Average Rating", f"{metrics['basic_stats']['average_rating']}/5",
                 f"{metrics['basic_stats']['average_rating'] - 4:.2f}" if metrics['basic_stats']['average_rating'] < 4 else "+Good")
    
    with col2:
        st.metric("Total Reviews", metrics['basic_stats']['total_reviews'])
    
    with col3:
        positive_pct = (metrics['sentiment_breakdown']['positive'] / sum(metrics['sentiment_breakdown'].values()) * 100) if sum(metrics['sentiment_breakdown'].values()) > 0 else 0
        st.metric("Positive Sentiment", f"{positive_pct:.0f}%")
    
    with col4:
        trend = metrics['temporal_trends'].get('trend', 'stable')
        trend_emoji = {'improving': '📈', 'declining': '📉', 'stable': '➡️', 'insufficient_data': '❓'}
        st.metric("Trend", trend.replace('_', ' ').title(), trend_emoji.get(trend, ''))
    
    # Detailed breakdowns
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Rating distribution
        st.markdown("### ⭐ Rating Distribution")
        rating_data = []
        for stars in range(5, 0, -1):
            count = metrics['basic_stats']['rating_distribution'].get(stars, 0)
            percentage = (count / metrics['basic_stats']['total_reviews'] * 100) if metrics['basic_stats']['total_reviews'] > 0 else 0
            rating_data.append({
                'Stars': f"{stars}★",
                'Count': count,
                'Percentage': percentage
            })
        
        df_ratings = pd.DataFrame(rating_data)
        st.dataframe(df_ratings, hide_index=True, use_container_width=True)
        
        # Issues breakdown
        st.markdown("### ⚠️ Top Issues")
        issues_sorted = sorted(metrics['issue_categories'].items(), key=lambda x: x[1], reverse=True)
        for category, count in issues_sorted[:5]:
            if count > 0:
                severity = 'danger' if count > 20 else 'warning' if count > 10 else 'success'
                st.markdown(f"""
                <div style="margin: 0.5rem 0;">
                    <span class="status-badge {severity}">{category.replace('_', ' ').title()}: {count}</span>
                </div>
                """, unsafe_allow_html=True)
    
    with col2:
        # Sentiment pie chart visualization
        st.markdown("### 💭 Sentiment Analysis")
        sentiment_data = []
        for sentiment_type, count in metrics['sentiment_breakdown'].items():
            if count > 0:
                sentiment_data.append({
                    'Type': sentiment_type.title(),
                    'Count': count,
                    'Percentage': f"{(count / sum(metrics['sentiment_breakdown'].values()) * 100):.1f}%"
                })
        
        df_sentiment = pd.DataFrame(sentiment_data)
        st.dataframe(df_sentiment, hide_index=True, use_container_width=True)
        
        # Keywords
        st.markdown("### 🔑 Top Keywords")
        
        if metrics['keyword_analysis']['positive_keywords']:
            st.markdown("**Positive:**")
            pos_words = ", ".join([f"`{word}` ({count})" for word, count in metrics['keyword_analysis']['positive_keywords'][:5]])
            st.markdown(pos_words)
        
        if metrics['keyword_analysis']['negative_keywords']:
            st.markdown("**Negative:**")
            neg_words = ", ".join([f"`{word}` ({count})" for word, count in metrics['keyword_analysis']['negative_keywords'][:5]])
            st.markdown(neg_words)
    
    # Action buttons
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    
    with col2:
        if check_ai_status():
            if st.button("🚀 Run AI Analysis", type="primary", use_container_width=True):
                run_full_analysis()
        else:
            st.button("🚀 AI Unavailable", disabled=True, use_container_width=True)

def display_ai_results():
    """Display AI analysis results"""
    if not st.session_state.analysis_results or not st.session_state.uploaded_data:
        st.error("No AI analysis results available")
        return
    
    results = st.session_state.analysis_results
    metrics = st.session_state.uploaded_data['metrics']
    
    # Header with enhancements used
    enhancements = []
    if st.session_state.use_listing_details:
        enhancements.append("📝 Listing Details")
    if results.get('marketplace_data_included'):
        enhancements.append("📂 Marketplace Data")
    
    st.markdown(f"""
    <h1>🤖 AI OPTIMIZATION ANALYSIS</h1>
    <div class="neon-box">
        <p>✅ Analysis complete! {' + '.join(enhancements) if enhancements else ''}</p>
        <p>📊 Analyzed {results['reviews_analyzed']} reviews ({results['analysis_scope'].replace('_', ' ')})</p>
        <p>🕐 {results['timestamp'].strftime('%B %d, %Y at %I:%M %p')}</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Quick actions
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("💬 Discuss with AI", use_container_width=True):
            st.session_state.show_ai_chat = True
            st.rerun()
    
    with col2:
        if st.button("📊 View Metrics", use_container_width=True):
            st.session_state.current_view = 'metrics'
            st.rerun()
    
    with col3:
        # Generate export
        export_data = generate_export_data(results, metrics)
        st.download_button(
            "📥 Export Report",
            data=export_data,
            file_name=f"amazon_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
            mime="text/plain",
            use_container_width=True
        )
    
    # Display AI insights in organized sections
    st.markdown("---")
    display_ai_insights(results['analysis'])

def display_ai_insights(analysis: str):
    """Display AI insights in structured format"""
    # Parse sections from the analysis
    sections = {
        'TITLE OPTIMIZATION': {'icon': '🎯', 'color': 'primary'},
        'BULLET POINT REWRITE': {'icon': '📝', 'color': 'accent'},
        'A9 ALGORITHM OPTIMIZATION': {'icon': '🔍', 'color': 'secondary'},
        'IMMEDIATE QUICK WINS': {'icon': '⚡', 'color': 'warning'},
        'QUALITY & SAFETY PRIORITIES': {'icon': '🏥', 'color': 'danger'},
        'RETURN REDUCTION STRATEGY': {'icon': '📦', 'color': 'success'}
    }
    
    # Split analysis into sections
    current_section = None
    section_content = {}
    
    for line in analysis.split('\n'):
        # Check if this is a section header
        found_section = False
        for section_name in sections:
            if section_name in line.upper():
                current_section = section_name
                section_content[current_section] = []
                found_section = True
                break
        
        if not found_section and current_section:
            section_content[current_section].append(line)
    
    # Display each section
    for section_name, config in sections.items():
        if section_name in section_content and section_content[section_name]:
            st.markdown(f"""
            <div class="neon-box">
                <h3 style="color: var(--{config['color']});">
                    {config['icon']} {section_name}
                </h3>
            </div>
            """, unsafe_allow_html=True)
            
            # Clean and display content
            content = '\n'.join(section_content[section_name]).strip()
            if content:
                # Format bullet points nicely
                if '•' in content:
                    lines = content.split('\n')
                    for line in lines:
                        if line.strip().startswith('•'):
                            st.markdown(line)
                        elif line.strip():
                            st.info(line)
                else:
                    st.write(content)
            
            st.markdown("")  # Spacing

def generate_export_data(results: Dict, metrics: Dict) -> str:
    """Generate comprehensive export report"""
    report = f"""
AMAZON LISTING OPTIMIZATION REPORT
Generated: {datetime.now().strftime('%B %d, %Y at %I:%M %p')}
Company: {APP_CONFIG['company']}

{'='*60}

EXECUTIVE SUMMARY
-----------------
Product ASIN: {st.session_state.listing_details.get('asin', 'Not provided')}
Brand: {st.session_state.listing_details.get('brand', 'Not provided')}
Health Score: {metrics['listing_health_score']['total_score']:.0f}/100 (Grade: {metrics['listing_health_score']['grade']})
Average Rating: {metrics['basic_stats']['average_rating']}/5
Total Reviews: {metrics['basic_stats']['total_reviews']}
Analysis Type: {'Enhanced' if st.session_state.use_listing_details else 'Standard'}

ANALYSIS CONFIGURATION
---------------------
Reviews Analyzed: {results['reviews_analyzed']} ({results['analysis_scope'].replace('_', ' ')})
Listing Details Used: {'Yes' if st.session_state.use_listing_details else 'No'}
Marketplace Data Used: {'Yes' if results.get('marketplace_data_included') else 'No'}

{'='*60}

AI OPTIMIZATION RECOMMENDATIONS
-------------------------------
{results['analysis']}

{'='*60}

KEY METRICS
-----------
Rating Distribution:
"""
    
    # Add rating distribution
    for stars in range(5, 0, -1):
        count = metrics['basic_stats']['rating_distribution'].get(stars, 0)
        percentage = (count / metrics['basic_stats']['total_reviews'] * 100) if metrics['basic_stats']['total_reviews'] > 0 else 0
        report += f"  {stars}★: {count} reviews ({percentage:.1f}%)\n"
    
    # Add sentiment
    report += "\nSentiment Analysis:\n"
    for sentiment, count in metrics['sentiment_breakdown'].items():
        percentage = (count / sum(metrics['sentiment_breakdown'].values()) * 100) if sum(metrics['sentiment_breakdown'].values()) > 0 else 0
        report += f"  {sentiment.title()}: {count} ({percentage:.1f}%)\n"
    
    # Add issues
    report += "\nTop Issues Identified:\n"
    issues_sorted = sorted(metrics['issue_categories'].items(), key=lambda x: x[1], reverse=True)
    for category, count in issues_sorted[:5]:
        if count > 0:
            report += f"  {category.replace('_', ' ').title()}: {count} mentions\n"
    
    # Add marketplace insights if available
    if st.session_state.marketplace_data:
        report += "\n" + "="*60 + "\n"
        report += "MARKETPLACE DATA INSIGHTS\n"
        report += "-" * 25 + "\n"
        
        marketplace = st.session_state.marketplace_data
        if 'return_patterns' in marketplace:
            total_returns = sum(
                data.get('count', 0) 
                for data in marketplace['return_patterns'].values()
            )
            report += f"Total Returns: {total_returns}\n"
            
            # Break down by type
            for file_type, data in marketplace['return_patterns'].items():
                if data and 'count' in data:
                    report += f"  {file_type.replace('_', ' ').title()}: {data['count']}\n"
        
        if 'financial_impact' in marketplace and 'reimbursements' in marketplace['financial_impact']:
            total_reimb = marketplace['financial_impact']['reimbursements'].get('total_amount', 0)
            reimb_count = marketplace['financial_impact']['reimbursements'].get('count', 0)
            report += f"Total Reimbursements: ${total_reimb:.2f} ({reimb_count} transactions)\n"
    
    report += "\n" + "="*60 + "\n"
    report += f"Report generated by {APP_CONFIG['title']} v{APP_CONFIG['version']}\n"
    report += f"Support: {APP_CONFIG['support_email']}\n"
    
    return report

# Main application flow
def main():
    st.set_page_config(
        page_title=APP_CONFIG['title'],
        page_icon="🚀",
        layout="wide",
        initial_sidebar_state="collapsed"
    )
    
    initialize_session_state()
    inject_cyberpunk_css()
    display_header()
    
    # Show AI chat if active
    if st.session_state.show_ai_chat:
        with st.container():
            display_ai_chat()
            st.markdown("<hr>", unsafe_allow_html=True)
    
    # Main content based on view
    if st.session_state.current_view == 'upload':
        # Show tutorial for new users
        if st.session_state.show_tutorial:
            display_tutorial()
        
        # Display workflow progress
        display_workflow_progress()
        
        # Step-by-step workflow
        display_step1_product_details()
        st.markdown("---")
        
        display_step2_review_upload()
        st.markdown("---")
        
        display_step3_marketplace_files()
        st.markdown("---")
        
        display_step4_analysis()
        
    elif st.session_state.current_view == 'metrics':
        display_metrics_view()
        
    elif st.session_state.current_view == 'ai_results':
        display_ai_results()
    
    # Footer
    st.markdown("---")
    st.markdown(f"""
    <div style="text-align: center; color: var(--muted); font-size: 0.9rem;">
        {APP_CONFIG['title']} v{APP_CONFIG['version']} | {APP_CONFIG['company']} | 
        <a href="mailto:{APP_CONFIG['support_email']}" style="color: var(--primary);">Support</a>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
