"""
return_analyzer_all_in_one.py - Complete Medical Device Return Analyzer
All functionality in one file for maximum simplicity
Just run: streamlit run return_analyzer_all_in_one.py
"""

import streamlit as st
import pandas as pd
import re
import os
from datetime import datetime
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page config
st.set_page_config(
    page_title="Medical Device Return Analyzer",
    page_icon="🏥",
    layout="wide"
)

# Categories and injury keywords
RETURN_CATEGORIES = {
    'QUALITY_DEFECTS': ['defect', 'broken', 'damaged', 'malfunction', 'not working', 'faulty', 'poor quality'],
    'SIZE_FIT_ISSUES': ['too small', 'too large', 'size', 'fit', "doesn't fit", 'wrong size'],
    'COMPATIBILITY_ISSUES': ['not compatible', 'incompatible', "doesn't fit toilet", "won't fit"],
    'FUNCTIONALITY_ISSUES': ['uncomfortable', 'hard to use', 'difficult', 'unstable', 'wobbles'],
    'WRONG_PRODUCT': ['wrong item', 'incorrect', 'not as described', 'different'],
    'BUYER_MISTAKE': ['mistake', 'accident', 'accidentally', 'error', 'my fault'],
    'NO_LONGER_NEEDED': ['no longer needed', 'changed mind', "don't need"],
    'INJURY_RISK': ['injury', 'injured', 'hurt', 'pain', 'hospital', 'emergency', 'bleeding', 'fall', 'burn']
}

INJURY_KEYWORDS = {
    'critical': ['death', 'died', 'hospital', 'emergency', 'severe injury', 'surgery'],
    'high': ['injury', 'injured', 'hurt', 'bleeding', 'fall', 'burn', 'fracture'],
    'medium': ['pain', 'bruise', 'discomfort', 'swelling', 'rash']
}

# Optional AI support
AI_AVAILABLE = False
ai_client = None
ai_type = None

# Try to set up AI if available
try:
    # Try OpenAI
    import openai
    api_key = os.getenv('OPENAI_API_KEY')
    if api_key:
        ai_client = openai.OpenAI(api_key=api_key)
        ai_type = 'openai'
        AI_AVAILABLE = True
        logger.info("OpenAI available")
except:
    pass

if not AI_AVAILABLE:
    try:
        # Try Anthropic
        import anthropic
        api_key = os.getenv('ANTHROPIC_API_KEY')
        if api_key:
            ai_client = anthropic.Anthropic(api_key=api_key)
            ai_type = 'anthropic'
            AI_AVAILABLE = True
            logger.info("Anthropic available")
    except:
        pass

def extract_returns_from_pdf(file) -> pd.DataFrame:
    """Extract returns from PDF"""
    try:
        import pdfplumber
    except ImportError:
        st.error("Please install pdfplumber: pip install pdfplumber")
        return pd.DataFrame()
    
    try:
        with pdfplumber.open(file) as pdf:
            # Get all text
            full_text = ""
            for page in pdf.pages:
                text = page.extract_text()
                if text:
                    full_text += text + "\n"
            
            # Find all order IDs and their context
            order_pattern = r'(\d{3}-\d{7}-\d{7})'
            returns = []
            
            # Split by order IDs
            segments = re.split(f'({order_pattern})', full_text)
            
            for i in range(1, len(segments), 2):
                if i + 1 < len(segments):
                    order_id = segments[i]
                    context = segments[i + 1]
                    
                    # Extract ASIN
                    asin_match = re.search(r'(B[A-Z0-9]{9})', context)
                    asin = asin_match.group(1) if asin_match else ''
                    
                    # Extract reason
                    reason = 'Not specified'
                    reason_match = re.search(r'(?:Return [Rr]eason|Reason):?\s*([^\n]+)', context)
                    if reason_match:
                        reason = reason_match.group(1).strip()
                    
                    # Extract comment
                    comment = ''
                    comment_match = re.search(r'(?:Customer [Cc]omment|Comment):?\s*([^\n]+(?:\n[^\n]+)*)', context)
                    if comment_match:
                        comment = ' '.join(comment_match.group(1).strip().split('\n')[:3])
                    
                    returns.append({
                        'order_id': order_id,
                        'asin': asin,
                        'return_reason': reason,
                        'customer_comment': comment,
                        'full_text': context[:500]
                    })
            
            return pd.DataFrame(returns) if returns else pd.DataFrame()
            
    except Exception as e:
        st.error(f"Error reading PDF: {e}")
        return pd.DataFrame()

def categorize_with_ai(reason: str, comment: str) -> str:
    """Try to categorize using AI"""
    if not AI_AVAILABLE:
        return None
    
    prompt = f"""Categorize this return into ONE category:
- QUALITY_DEFECTS
- SIZE_FIT_ISSUES  
- COMPATIBILITY_ISSUES
- FUNCTIONALITY_ISSUES
- WRONG_PRODUCT
- BUYER_MISTAKE
- NO_LONGER_NEEDED
- INJURY_RISK
- OTHER

Reason: {reason}
Comment: {comment}

Reply with ONLY the category name."""

    try:
        if ai_type == 'openai':
            response = ai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=20
            )
            return response.choices[0].message.content.strip()
        elif ai_type == 'anthropic':
            response = ai_client.messages.create(
                model="claude-3-haiku-20240307",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=20
            )
            return response.content[0].text.strip()
    except:
        return None

def categorize_return(reason: str, comment: str) -> str:
    """Categorize using AI or patterns"""
    # Try AI first
    if AI_AVAILABLE:
        ai_category = categorize_with_ai(reason, comment)
        if ai_category and ai_category in RETURN_CATEGORIES:
            return ai_category
    
    # Fall back to patterns
    text = f"{reason} {comment}".lower()
    for category, keywords in RETURN_CATEGORIES.items():
        if any(keyword in text for keyword in keywords):
            return category
    return 'OTHER'

def check_injury(text: str) -> tuple:
    """Check for injuries"""
    text_lower = text.lower()
    for severity, keywords in INJURY_KEYWORDS.items():
        for keyword in keywords:
            if keyword in text_lower:
                return True, severity, keyword
    return False, None, None

def main():
    st.title("🏥 Medical Device Return Analyzer")
    st.markdown("Extract, categorize, and identify injury cases from Amazon returns")
    
    # Sidebar
    with st.sidebar:
        st.markdown("### Status")
        if AI_AVAILABLE:
            st.success(f"✅ AI Active ({ai_type})")
        else:
            st.info("📊 Pattern Matching Mode")
        
        st.markdown("### Categories")
        st.markdown("""
        - **QUALITY_DEFECTS**
        - **SIZE_FIT_ISSUES**
        - **COMPATIBILITY_ISSUES**
        - **FUNCTIONALITY_ISSUES**
        - **WRONG_PRODUCT**
        - **BUYER_MISTAKE**
        - **NO_LONGER_NEEDED**
        - **INJURY_RISK** 🚨
        - **OTHER**
        """)
    
    # File upload
    file = st.file_uploader("Upload Amazon Return PDF", type=['pdf'])
    
    if file:
        # Extract returns
        with st.spinner("Extracting returns..."):
            df = extract_returns_from_pdf(file)
        
        if df.empty:
            st.error("No returns found in PDF")
            return
        
        st.success(f"Found {len(df)} returns")
        
        # Process returns
        with st.spinner("Analyzing..."):
            df['category'] = df.apply(lambda r: categorize_return(r['return_reason'], r['customer_comment']), axis=1)
            
            # Check injuries
            injury_results = df.apply(lambda r: check_injury(f"{r['return_reason']} {r['customer_comment']} {r['full_text']}"), axis=1)
            df['has_injury'] = injury_results.apply(lambda x: x[0])
            df['injury_severity'] = injury_results.apply(lambda x: x[1] or '')
            df['injury_keyword'] = injury_results.apply(lambda x: x[2] or '')
        
        # Metrics
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Returns", len(df))
        col2.metric("Quality Issues", len(df[df['category'].isin(['QUALITY_DEFECTS', 'FUNCTIONALITY_ISSUES'])]))
        
        injuries = len(df[df['has_injury']])
        if injuries > 0:
            col3.metric("🚨 Injuries", injuries)
            st.error("⚠️ INJURY CASES DETECTED - Review for FDA MDR reporting")
        else:
            col3.metric("✅ Injuries", 0)
        
        col4.metric("Categories", df['category'].nunique())
        
        # Tabs
        tab1, tab2, tab3 = st.tabs(["📊 Analysis", "🚨 Injuries", "💾 Export"])
        
        with tab1:
            # Category breakdown
            st.subheader("Category Breakdown")
            for cat, count in df['category'].value_counts().items():
                st.metric(cat, f"{count} ({count/len(df)*100:.1f}%)")
        
        with tab2:
            # Injury details
            injury_df = df[df['has_injury']]
            if not injury_df.empty:
                st.error(f"Found {len(injury_df)} injury cases")
                
                for _, row in injury_df.iterrows():
                    with st.expander(f"{row['order_id']} - {row['injury_severity'].upper()}"):
                        st.write(f"**ASIN:** {row['asin']}")
                        st.write(f"**Keyword:** {row['injury_keyword']}")
                        st.write(f"**Reason:** {row['return_reason']}")
                        st.write(f"**Comment:** {row['customer_comment']}")
            else:
                st.success("No injuries detected")
        
        with tab3:
            # Export
            csv = df.to_csv(index=False)
            st.download_button(
                "📥 Download Analysis",
                csv,
                f"returns_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                "text/csv"
            )
    
    else:
        st.info("👆 Upload a PDF from Amazon Seller Central to analyze returns")

if __name__ == "__main__":
    main()
