"""
Amazon Returns Quality Analyzer - FDA Reportable Event Detection
Version: 7.0 - Medical Device Injury Reporting Focus
Critical: Identifies FDA MDR (Medical Device Reporting) candidates
"""

# IMPORTANT: This is a module file - DO NOT use st.set_page_config() here
# st.set_page_config() should only be in the main app file

# Import Streamlit but don't call set_page_config
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
import time
import os
import traceback

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Optional imports for AI providers
try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    logger.warning("OpenAI not available")

try:
    import anthropic
    CLAUDE_AVAILABLE = True
except ImportError:
    CLAUDE_AVAILABLE = False
    logger.warning("Anthropic Claude not available")

# File parsing imports
try:
    import pdfplumber
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False
    logger.warning("pdfplumber not available")

try:
    import openpyxl
    EXCEL_AVAILABLE = True
except ImportError:
    EXCEL_AVAILABLE = False
    logger.warning("openpyxl not available")

try:
    import chardet
    CHARDET_AVAILABLE = True
except ImportError:
    CHARDET_AVAILABLE = False
    logger.warning("chardet not available")

# Medical device categories for FDA reporting
MEDICAL_DEVICE_CATEGORIES = [
    'Product Defects/Quality',
    'Injury/Adverse Event',
    'Performance/Effectiveness',
    'Size/Fit Issues',
    'Stability/Safety Issues',
    'Material/Component Failure',
    'Design Issues',
    'Comfort/Usability Issues',
    'Compatibility Issues',
    'Assembly/Installation Issues',
    'Wrong Product/Labeling',
    'Missing Components',
    'Customer Error',
    'Non-Medical Issue'
]

# FDA MDR triggers - events that may require reporting
FDA_MDR_TRIGGERS = {
    'death': ['death', 'died', 'fatal', 'fatality'],
    'serious_injury': [
        'injury', 'injured', 'hurt', 'wound', 'bleeding', 'blood',
        'fracture', 'broken', 'break', 'severe', 'emergency', 'ER',
        'hospital', 'hospitalized', 'surgery', 'operation',
        'permanent', 'disability', 'impairment'
    ],
    'falls': [
        'fall', 'fell', 'fallen', 'falling', 'dropped', 'collapsed',
        'slip', 'slipped', 'trip', 'tripped', 'tumble'
    ],
    'malfunction': [
        'malfunction', 'failed', 'failure', 'defect', 'broke',
        'exploded', 'fire', 'smoke', 'spark', 'electric shock',
        'sharp', 'exposed', 'hazard'
    ],
    'allergic_reaction': [
        'allergic', 'allergy', 'reaction', 'rash', 'hives',
        'swelling', 'anaphylaxis', 'breathing', 'throat'
    ],
    'infection': [
        'infection', 'infected', 'contaminated', 'bacteria',
        'sepsis', 'fever', 'pus', 'inflammation'
    ]
}

# FBA reason code mapping for FDA focus
FBA_REASON_MAP = {
    'DEFECTIVE': 'Product Defects/Quality',
    'QUALITY_UNACCEPTABLE': 'Product Defects/Quality',
    'DAMAGED_BY_CUSTOMER': 'Customer Error',
    'CUSTOMER_DAMAGED': 'Customer Error',
    'NOT_COMPATIBLE': 'Compatibility Issues',
    'FOUND_BETTER_PRICE': 'Non-Medical Issue',
    'NO_LONGER_WANTED': 'Non-Medical Issue',
    'UNWANTED_ITEM': 'Non-Medical Issue',
    'SWITCHEROO': 'Wrong Product/Labeling',
    'MISSED_ESTIMATED_DELIVERY': 'Non-Medical Issue',
    'MISSING_PARTS': 'Missing Components',
    'NOT_AS_DESCRIBED': 'Wrong Product/Labeling',
    'ORDERED_WRONG_ITEM': 'Customer Error',
    'UNAUTHORIZED_PURCHASE': 'Non-Medical Issue',
    'ITEM_DIFFERENT_WEBSITE': 'Wrong Product/Labeling',
    'DAMAGED_BY_CARRIER': 'Non-Medical Issue',
    'DAMAGED_BY_FC': 'Non-Medical Issue',
    'WRONG_ITEM': 'Wrong Product/Labeling'
}

# AI Provider enum
class AIProvider:
    OPENAI = "openai"
    CLAUDE = "claude"
    FASTEST = "fastest"
    QUALITY = "quality"

def detect_fda_reportable_event(text: str) -> Dict[str, Any]:
    """
    Detect potential FDA reportable events from return text
    Returns detailed analysis for MDR determination
    """
    if not text:
        return {
            'is_reportable': False,
            'severity': None,
            'event_types': [],
            'confidence': 0.0,
            'requires_immediate_review': False
        }
    
    text_lower = text.lower()
    detected_events = []
    severity = 'LOW'
    
    # Check for each MDR trigger type
    for event_type, keywords in FDA_MDR_TRIGGERS.items():
        if any(keyword in text_lower for keyword in keywords):
            detected_events.append(event_type)
    
    # Determine severity based on detected events
    if 'death' in detected_events:
        severity = 'CRITICAL'
    elif 'serious_injury' in detected_events or 'falls' in detected_events:
        severity = 'HIGH'
    elif any(event in detected_events for event in ['malfunction', 'allergic_reaction', 'infection']):
        severity = 'MODERATE'
    
    # Calculate confidence based on keyword matches
    total_keywords = sum(len(keywords) for keywords in FDA_MDR_TRIGGERS.values())
    matched_keywords = sum(
        sum(1 for keyword in keywords if keyword in text_lower)
        for event_type, keywords in FDA_MDR_TRIGGERS.items()
    )
    confidence = min(matched_keywords / 5, 1.0)  # Cap at 100%
    
    return {
        'is_reportable': len(detected_events) > 0,
        'severity': severity if detected_events else None,
        'event_types': detected_events,
        'confidence': confidence,
        'requires_immediate_review': severity in ['CRITICAL', 'HIGH']
    }

class EnhancedAIAnalyzer:
    """AI analyzer focused on FDA reportable event detection"""
    
    def __init__(self, provider: str = AIProvider.FASTEST):
        self.provider = provider
        self.categories = MEDICAL_DEVICE_CATEGORIES
        self.api_calls = 0
        self.total_cost = 0.0
        self.ai_client = None
        self.model = None
        self.ai_available = False
        self._initialize_ai()
        
    def _initialize_ai(self):
        """Initialize AI provider"""
        try:
            if self.provider in [AIProvider.OPENAI, AIProvider.FASTEST] and OPENAI_AVAILABLE:
                # Check for API key
                api_key = os.getenv('OPENAI_API_KEY')
                if not api_key and hasattr(st, 'secrets') and 'OPENAI_API_KEY' in st.secrets:
                    api_key = st.secrets['OPENAI_API_KEY']
                
                if api_key:
                    self.ai_client = openai.OpenAI(api_key=api_key)
                    self.model = "gpt-4o-mini" if self.provider == AIProvider.FASTEST else "gpt-4o"
                    self.ai_available = True
                    logger.info(f"OpenAI initialized successfully with model: {self.model}")
                else:
                    logger.warning("OpenAI API key not found - using pattern matching only")
                    self.ai_client = None
                    self.ai_available = False
            elif self.provider == AIProvider.CLAUDE and CLAUDE_AVAILABLE:
                # Check for Anthropic API key
                api_key = os.getenv('ANTHROPIC_API_KEY')
                if not api_key and hasattr(st, 'secrets') and 'ANTHROPIC_API_KEY' in st.secrets:
                    api_key = st.secrets['ANTHROPIC_API_KEY']
                
                if api_key:
                    self.ai_client = anthropic.Anthropic(api_key=api_key)
                    self.model = "claude-3-5-sonnet-20241022"
                    self.ai_available = True
                    logger.info(f"Claude initialized successfully with model: {self.model}")
                else:
                    logger.warning("Anthropic API key not found - using pattern matching only")
                    self.ai_client = None
                    self.ai_available = False
            else:
                logger.warning("No AI provider available - using pattern matching only")
                self.ai_client = None
                self.ai_available = False
        except Exception as e:
            logger.error(f"Failed to initialize AI: {e}")
            self.ai_client = None
            self.ai_available = False
    
    def get_status(self) -> Dict[str, Any]:
        """Get analyzer status"""
        return {
            'ai_available': self.ai_available,
            'provider': self.provider if self.ai_available else 'Pattern Matching',
            'model': self.model if self.ai_available else None,
            'api_calls': self.api_calls,
            'total_cost': self.total_cost
        }
    
    def test_ai_connection(self) -> Dict[str, Any]:
        """Test AI connection and categorization"""
        test_cases = [
            ("Product is too small for my needs", "Size/Fit Issues"),
            ("Item broke after 2 days", "Product Defects/Quality"),
            ("Caused injury when it fell on my foot", "Injury/Adverse Event"),
            ("Doesn't work with my existing equipment", "Compatibility Issues"),
            ("Ordered the wrong size by mistake", "Customer Error")
        ]
        
        if not self.ai_client:
            return {
                'status': 'No AI client available',
                'results': None
            }
        
        results = []
        for test_text, expected in test_cases[:2]:  # Test first 2 cases
            try:
                category = self._ai_categorize(test_text, "Test Product")
                results.append({
                    'text': test_text,
                    'expected': expected,
                    'actual': category,
                    'correct': category == expected
                })
            except Exception as e:
                results.append({
                    'text': test_text,
                    'expected': expected,
                    'actual': f"Error: {str(e)}",
                    'correct': False
                })
        
        return {
            'status': 'AI connection successful' if results else 'AI test failed',
            'results': results
        }
    
    def categorize_return(self, reason: str, comment: str = "", 
                         product_name: str = "", asin: str = "") -> Dict[str, Any]:
        """
        Categorize return with FDA reporting focus using AI
        """
        full_text = f"{reason} {comment}".strip()
        
        # First check for FDA reportable events
        fda_analysis = detect_fda_reportable_event(full_text)
        
        # Use AI for categorization if available
        if self.ai_client:
            category = self._ai_categorize(full_text, product_name)
            confidence = 0.95
        else:
            # Enhanced pattern matching fallback
            category, confidence = self._pattern_match_categorize(reason, full_text)
        
        # Override category if injury detected
        if fda_analysis['is_reportable'] and fda_analysis['severity'] in ['CRITICAL', 'HIGH']:
            category = 'Injury/Adverse Event'
            confidence = 1.0
        
        return {
            'category': category,
            'confidence': confidence,
            'fda_reportable': fda_analysis['is_reportable'],
            'severity': fda_analysis['severity'],
            'event_types': fda_analysis['event_types'],
            'requires_mdr': fda_analysis['severity'] in ['CRITICAL', 'HIGH'] if fda_analysis['severity'] else False,
            'requires_immediate_review': fda_analysis.get('requires_immediate_review', False),
            'product_name': product_name,
            'asin': asin
        }
    
    def _ai_categorize(self, text: str, product_name: str) -> str:
        """Use AI to categorize return reason"""
        if not self.ai_client:
            return 'Product Defects/Quality'
        
        prompt = f"""You are analyzing Amazon return reasons for medical devices. Categorize the following return into EXACTLY ONE of these categories:

1. Product Defects/Quality - broken, defective, damaged, poor quality, doesn't work
2. Size/Fit Issues - too small, too large, doesn't fit, wrong size
3. Performance/Effectiveness - doesn't work as expected, ineffective, poor performance
4. Injury/Adverse Event - injury, hurt, pain, hospital, medical attention
5. Stability/Safety Issues - unstable, tips over, falls, unsafe
6. Material/Component Failure - material defect, component broke, parts failed
7. Comfort/Usability Issues - uncomfortable, hard to use, difficult
8. Compatibility Issues - not compatible, doesn't fit with other products
9. Wrong Product/Labeling - wrong item, not as described, incorrect product
10. Missing Components - missing parts, incomplete
11. Customer Error - user mistake, ordered wrong, misunderstood
12. Non-Medical Issue - price, shipping, no longer needed, changed mind

Product: {product_name if product_name else 'Unknown'}
Return Reason: {text}

Analyze the return reason and respond with ONLY the category name exactly as shown above. Nothing else."""
        
        try:
            if isinstance(self.ai_client, openai.OpenAI):
                response = self.ai_client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.1,
                    max_tokens=50
                )
                category = response.choices[0].message.content.strip()
            else:  # Claude
                response = self.ai_client.messages.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.1,
                    max_tokens=50
                )
                category = response.content[0].text.strip()
            
            self.api_calls += 1
            
            # Log the response for debugging
            logger.info(f"AI categorized '{text[:50]}...' as: '{category}'")
            
            # Flexible matching - handle case variations and partial matches
            category_lower = category.lower().strip()
            
            # Direct match check (case-insensitive)
            for valid_category in MEDICAL_DEVICE_CATEGORIES:
                if category_lower == valid_category.lower():
                    return valid_category
            
            # Check if response contains a valid category
            for valid_category in MEDICAL_DEVICE_CATEGORIES:
                if valid_category.lower() in category_lower:
                    return valid_category
            
            # Common variations mapping
            category_mappings = {
                'quality': 'Product Defects/Quality',
                'defect': 'Product Defects/Quality',
                'broken': 'Product Defects/Quality',
                'size': 'Size/Fit Issues',
                'fit': 'Size/Fit Issues',
                'performance': 'Performance/Effectiveness',
                'injury': 'Injury/Adverse Event',
                'adverse': 'Injury/Adverse Event',
                'safety': 'Stability/Safety Issues',
                'stability': 'Stability/Safety Issues',
                'material': 'Material/Component Failure',
                'component': 'Material/Component Failure',
                'comfort': 'Comfort/Usability Issues',
                'usability': 'Comfort/Usability Issues',
                'compatibility': 'Compatibility Issues',
                'wrong': 'Wrong Product/Labeling',
                'labeling': 'Wrong Product/Labeling',
                'missing': 'Missing Components',
                'customer error': 'Customer Error',
                'non-medical': 'Non-Medical Issue'
            }
            
            for key, mapped_category in category_mappings.items():
                if key in category_lower:
                    return mapped_category
            
            # If no match found, log warning and return default
            logger.warning(f"AI returned unrecognized category: '{category}'. Defaulting to Product Defects/Quality")
            return 'Product Defects/Quality'
                
        except Exception as e:
            logger.error(f"AI categorization failed: {e}")
            return 'Product Defects/Quality'
    
    def _pattern_match_categorize(self, reason: str, full_text: str) -> tuple[str, float]:
        """Enhanced pattern matching for categorization when AI is not available"""
        text_lower = full_text.lower()
        
        # First check FBA reason mapping
        if reason in FBA_REASON_MAP:
            return FBA_REASON_MAP[reason], 0.8
        
        # Enhanced keyword matching with multiple patterns per category
        category_patterns = {
            'Size/Fit Issues': {
                'keywords': ['too small', 'too large', 'too big', 'size', 'fit', 'tight', 'loose', 
                           'doesn\'t fit', 'does not fit', 'wrong size', 'smaller', 'larger', 'bigger'],
                'confidence': 0.9
            },
            'Product Defects/Quality': {
                'keywords': ['broken', 'defective', 'damaged', 'defect', 'quality', 'poor quality',
                           'cheap', 'flimsy', 'broke', 'cracked', 'torn', 'ripped', 'faulty',
                           'doesn\'t work', 'does not work', 'stopped working', 'malfunction'],
                'confidence': 0.85
            },
            'Performance/Effectiveness': {
                'keywords': ['not effective', 'ineffective', 'doesn\'t help', 'does not help',
                           'not working as expected', 'poor performance', 'weak', 'useless'],
                'confidence': 0.8
            },
            'Injury/Adverse Event': {
                'keywords': ['injury', 'injured', 'hurt', 'pain', 'painful', 'hospital', 'doctor',
                           'emergency', 'bleeding', 'wound', 'burn', 'reaction', 'allergic'],
                'confidence': 0.95
            },
            'Stability/Safety Issues': {
                'keywords': ['unstable', 'wobble', 'tip', 'fall', 'unsafe', 'dangerous', 'hazard',
                           'not stable', 'tips over', 'fell over', 'collapsed'],
                'confidence': 0.85
            },
            'Material/Component Failure': {
                'keywords': ['material', 'fabric', 'plastic broke', 'metal bent', 'component',
                           'part broke', 'piece broke', 'strap broke', 'buckle broke'],
                'confidence': 0.8
            },
            'Comfort/Usability Issues': {
                'keywords': ['uncomfortable', 'not comfortable', 'hard to use', 'difficult',
                           'complicated', 'confusing', 'awkward', 'inconvenient'],
                'confidence': 0.75
            },
            'Compatibility Issues': {
                'keywords': ['not compatible', 'incompatible', 'doesn\'t fit with', 'does not fit with',
                           'won\'t work with', 'not suitable for'],
                'confidence': 0.85
            },
            'Wrong Product/Labeling': {
                'keywords': ['wrong', 'incorrect', 'not as described', 'different', 'not what ordered',
                           'wrong item', 'wrong product', 'mislabeled', 'not as advertised'],
                'confidence': 0.9
            },
            'Missing Components': {
                'keywords': ['missing', 'incomplete', 'parts missing', 'not included', 'forgot',
                           'didn\'t come with', 'no instructions'],
                'confidence': 0.85
            },
            'Customer Error': {
                'keywords': ['my mistake', 'ordered wrong', 'accident', 'my fault', 'didn\'t realize',
                           'misunderstood', 'wrong one', 'user error'],
                'confidence': 0.9
            },
            'Non-Medical Issue': {
                'keywords': ['price', 'cost', 'expensive', 'cheaper', 'no longer need',
                           'changed mind', 'don\'t want', 'found better', 'duplicate'],
                'confidence': 0.8
            }
        }
        
        # Score each category based on keyword matches
        best_category = 'Product Defects/Quality'
        best_score = 0
        best_confidence = 0.6
        
        for category, data in category_patterns.items():
            score = 0
            for keyword in data['keywords']:
                if keyword in text_lower:
                    score += 1
            
            if score > best_score:
                best_score = score
                best_category = category
                best_confidence = data['confidence']
        
        # If no keywords matched, try to use reason code patterns
        if best_score == 0:
            reason_upper = reason.upper()
            if 'DEFECT' in reason_upper or 'QUALITY' in reason_upper:
                return 'Product Defects/Quality', 0.7
            elif 'COMPATIBLE' in reason_upper:
                return 'Compatibility Issues', 0.7
            elif 'WRONG' in reason_upper:
                return 'Wrong Product/Labeling', 0.7
            elif 'DAMAGED' in reason_upper:
                return 'Product Defects/Quality', 0.7
        
        return best_category, best_confidence
    
    def batch_categorize(self, df: pd.DataFrame, 
                        reason_col: str = 'reason',
                        comment_col: str = 'customer-comments',
                        product_col: str = 'product-name',
                        asin_col: str = 'asin') -> pd.DataFrame:
        """
        Batch categorize returns with FDA reporting focus
        """
        results = []
        total = len(df)
        
        # Log available columns for debugging
        logger.info(f"Available columns: {list(df.columns)}")
        
        # Flexible column detection
        # Try to find columns even if they don't match exactly
        actual_reason_col = None
        actual_comment_col = None
        actual_product_col = None
        actual_asin_col = None
        
        for col in df.columns:
            col_lower = col.lower()
            if not actual_reason_col and ('reason' in col_lower or 'issue' in col_lower):
                actual_reason_col = col
            elif not actual_comment_col and ('comment' in col_lower or 'feedback' in col_lower or 'notes' in col_lower):
                actual_comment_col = col
            elif not actual_product_col and ('product' in col_lower or 'item' in col_lower or 'title' in col_lower):
                actual_product_col = col
            elif not actual_asin_col and 'asin' in col_lower:
                actual_asin_col = col
        
        # Use found columns or defaults
        reason_col = actual_reason_col or reason_col
        comment_col = actual_comment_col or comment_col
        product_col = actual_product_col or product_col
        asin_col = actual_asin_col or asin_col
        
        logger.info(f"Using columns - Reason: {reason_col}, Comment: {comment_col}, Product: {product_col}, ASIN: {asin_col}")
        
        # Create progress placeholder
        progress_placeholder = st.empty()
        status_placeholder = st.empty()
        
        # Track FDA reportable events
        reportable_count = 0
        critical_count = 0
        
        # Show sample of data being processed
        if total > 0 and reason_col in df.columns:
            sample_reasons = df[reason_col].dropna().head(3).tolist()
            st.info(f"Processing {total} returns. Sample reasons: {sample_reasons}")
        
        for idx, row in df.iterrows():
            # Update progress
            if idx % 10 == 0:
                progress = idx / total
                progress_placeholder.progress(progress, 
                    f"Analyzing returns for FDA reportable events... {idx}/{total} ({progress*100:.1f}%)")
                
                if reportable_count > 0:
                    status_placeholder.warning(
                        f"⚠️ Found {reportable_count} potential FDA reportable events "
                        f"({critical_count} critical)"
                    )
            
            # Extract data with fallbacks
            reason = str(row.get(reason_col, '')) if reason_col in df.columns else ''
            comment = str(row.get(comment_col, '')) if comment_col in df.columns else ''
            product = str(row.get(product_col, '')) if product_col in df.columns else ''
            asin = str(row.get(asin_col, '')) if asin_col in df.columns else ''
            
            # Skip empty rows
            if not reason and not comment:
                result = {
                    'category': 'Product Defects/Quality',
                    'confidence': 0.1,
                    'fda_reportable': False,
                    'severity': None,
                    'event_types': [],
                    'requires_mdr': False,
                    'requires_immediate_review': False,
                    'product_name': product,
                    'asin': asin
                }
            else:
                result = self.categorize_return(reason, comment, product, asin)
            
            results.append(result)
            
            # Track reportable events
            if result['fda_reportable']:
                reportable_count += 1
                if result['severity'] == 'CRITICAL':
                    critical_count += 1
        
        progress_placeholder.empty()
        status_placeholder.empty()
        
        # Add results to dataframe
        df['category'] = [r['category'] for r in results]
        df['confidence'] = [r['confidence'] for r in results]
        df['fda_reportable'] = [r['fda_reportable'] for r in results]
        df['severity'] = [r['severity'] for r in results]
        df['event_types'] = [r['event_types'] for r in results]
        df['requires_mdr'] = [r['requires_mdr'] for r in results]
        df['requires_immediate_review'] = [r['requires_immediate_review'] for r in results]
        
        # Show category distribution
        category_counts = df['category'].value_counts()
        st.info(f"Categorization complete! Top categories: {dict(category_counts.head(5))}")
        
        # Final status
        if reportable_count > 0:
            st.error(f"""
            🚨 **FDA REPORTABLE EVENTS DETECTED**
            - Total Reportable Events: {reportable_count}
            - Critical Severity: {critical_count}
            - Requires Immediate Review: {len(df[df['requires_immediate_review']])}
            
            **Action Required**: Review these cases for potential FDA MDR submission
            """)
        
        return df
    
    def generate_fda_report(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Generate FDA-focused analysis report"""
        reportable_df = df[df['fda_reportable'] == True]
        
        report = {
            'summary': {
                'total_returns': len(df),
                'reportable_events': len(reportable_df),
                'critical_events': len(reportable_df[reportable_df['severity'] == 'CRITICAL']),
                'high_severity': len(reportable_df[reportable_df['severity'] == 'HIGH']),
                'mdr_required': len(df[df['requires_mdr'] == True])
            },
            'by_event_type': {},
            'affected_products': {},
            'timeline': {},
            'recommendations': []
        }
        
        if len(reportable_df) > 0:
            # Analyze by event type
            all_events = []
            for events in reportable_df['event_types']:
                all_events.extend(events)
            event_counts = Counter(all_events)
            report['by_event_type'] = dict(event_counts)
            
            # Analyze affected products
            product_analysis = reportable_df.groupby(['product-name', 'asin']).agg({
                'severity': 'count',
                'requires_mdr': 'sum'
            }).sort_values('severity', ascending=False)
            
            report['affected_products'] = product_analysis.head(10).to_dict('index')
            
            # Generate recommendations
            if report['summary']['critical_events'] > 0:
                report['recommendations'].append(
                    "IMMEDIATE ACTION: Critical events detected. Initiate FDA MDR process within 30 days."
                )
            
            if 'falls' in report['by_event_type']:
                report['recommendations'].append(
                    "Multiple fall-related incidents reported. Review product stability and safety warnings."
                )
            
            if 'serious_injury' in report['by_event_type']:
                report['recommendations'].append(
                    "Serious injuries reported. Conduct root cause analysis and consider product recall."
                )
        
        return report
    
    def export_fda_summary(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create FDA-focused summary for export"""
        reportable_df = df[df['fda_reportable'] == True].copy()
        
        if len(reportable_df) > 0:
            # Prepare FDA summary
            reportable_df['event_summary'] = reportable_df['event_types'].apply(
                lambda x: ', '.join(x) if x else ''
            )
            
            summary_df = reportable_df[[
                'order-id', 'asin', 'product-name', 'reason', 
                'customer-comments', 'category', 'severity',
                'event_summary', 'requires_mdr'
            ]].sort_values('severity', ascending=False)
            
            return summary_df
        
        return pd.DataFrame()

class FileProcessor:
    """Universal file processor for multiple formats"""
    
    @staticmethod
    def _standardize_amazon_columns(df: pd.DataFrame) -> pd.DataFrame:
        """Standardize column names for Amazon return data"""
        # Common column name mappings
        column_mappings = {
            # Order ID variations
            'order id': 'order-id',
            'order_id': 'order-id',
            'order-id': 'order-id',
            'orderid': 'order-id',
            'order number': 'order-id',
            'order #': 'order-id',
            'order': 'order-id',
            'orderId': 'order-id',
            
            # ASIN variations
            'asin': 'asin',
            'product asin': 'asin',
            'item asin': 'asin',
            'product-asin': 'asin',
            
            # Return reason variations
            'return reason': 'reason',
            'reason': 'reason',
            'return_reason': 'reason',
            'return-reason': 'reason',
            'reason for return': 'reason',
            'issue': 'reason',
            'return type': 'reason',
            
            # Customer comments variations
            'customer comments': 'customer-comments',
            'customer comment': 'customer-comments',
            'customer-comments': 'customer-comments',
            'buyer comments': 'customer-comments',
            'buyer comment': 'customer-comments',
            'comments': 'customer-comments',
            'feedback': 'customer-comments',
            'notes': 'customer-comments',
            'customer feedback': 'customer-comments',
            'buyer_comments': 'customer-comments',
            'customer_comments': 'customer-comments',
            'buyer-comments': 'customer-comments',
            
            # Product name variations
            'product name': 'product-name',
            'product-name': 'product-name',
            'product': 'product-name',
            'item name': 'product-name',
            'product title': 'product-name',
            'title': 'product-name',
            'description': 'product-name',
            'product_name': 'product-name',
            'item': 'product-name',
            
            # SKU variations
            'sku': 'sku',
            'seller sku': 'sku',
            'merchant sku': 'sku',
            'seller-sku': 'sku',
            'merchant-sku': 'sku',
            
            # Date variations
            'return date': 'return-date',
            'return-date': 'return-date',
            'date': 'return-date',
            'return_date': 'return-date',
            'returned date': 'return-date',
            'date returned': 'return-date',
            'returned-date': 'return-date',
            
            # Quantity variations
            'quantity': 'quantity',
            'qty': 'quantity',
            'units': 'quantity',
            'amount': 'quantity',
        }
        
        # Create a copy of the dataframe
        result = df.copy()
        
        # First pass: collect all columns and their mappings
        new_column_names = {}
        seen_targets = set()
        
        for col in result.columns:
            col_lower = str(col).lower().strip()
            mapped = False
            
            # Check if it matches any known mapping
            for key, value in column_mappings.items():
                if col_lower == key:
                    # Check if we've already mapped to this target
                    if value in seen_targets:
                        # Add a suffix to make it unique
                        suffix = 1
                        new_name = f"{value}_{suffix}"
                        while new_name in seen_targets:
                            suffix += 1
                            new_name = f"{value}_{suffix}"
                        new_column_names[col] = new_name
                        seen_targets.add(new_name)
                    else:
                        new_column_names[col] = value
                        seen_targets.add(value)
                    mapped = True
                    break
            
            if not mapped:
                # Keep original but ensure uniqueness
                new_col = col_lower.replace(' ', '-').replace('_', '-')
                if new_col in seen_targets:
                    suffix = 1
                    unique_col = f"{new_col}_{suffix}"
                    while unique_col in seen_targets:
                        suffix += 1
                        unique_col = f"{new_col}_{suffix}"
                    new_column_names[col] = unique_col
                    seen_targets.add(unique_col)
                else:
                    new_column_names[col] = new_col
                    seen_targets.add(new_col)
        
        # Apply the new column names
        result.rename(columns=new_column_names, inplace=True)
        
        # Ensure required columns exist (even if empty)
        required_columns = ['order-id', 'asin', 'reason', 'customer-comments', 'product-name']
        for col in required_columns:
            if col not in result.columns:
                result[col] = ''
        
        # Log the column mapping for debugging
        logger.info(f"Column mapping applied: {new_column_names}")
        
        return result
    
    @staticmethod
    def read_file(file, file_type: str) -> pd.DataFrame:
        """Read various file formats and return DataFrame"""
        logger.info(f"Attempting to read file: {getattr(file, 'name', 'unknown')} as type: {file_type}")
        
        try:
            # Reset file pointer
            if hasattr(file, 'seek'):
                file.seek(0)
            
            # Handle PDF
            if 'pdf' in file_type.lower() or file_type == 'application/pdf' or (hasattr(file, 'name') and file.name.lower().endswith('.pdf')):
                # Check for PDF libraries
                pdf_library = None
                try:
                    import pdfplumber
                    pdf_library = 'pdfplumber'
                except ImportError:
                    try:
                        import PyPDF2
                        pdf_library = 'PyPDF2'
                    except ImportError:
                        pass
                
                if not pdf_library:
                    raise ValueError(
                        "PDF processing requires pdfplumber or PyPDF2. "
                        "Please install with: pip install pdfplumber\n"
                        "For now, please export your Amazon returns as CSV or Excel format."
                    )
                
                if pdf_library == 'pdfplumber':
                    import pdfplumber
                    
                    all_data = []
                    all_text = []
                    
                    with pdfplumber.open(file) as pdf:
                        for page_num, page in enumerate(pdf.pages):
                            try:
                                # Extract text first
                                text = page.extract_text()
                                if text:
                                    all_text.append(text)
                                
                                # Try to extract tables
                                tables = page.extract_tables()
                                if tables:
                                    for table_idx, table in enumerate(tables):
                                        if table and len(table) > 1:
                                            # Process table headers
                                            headers = []
                                            seen_headers = {}
                                            
                                            # Clean and make headers unique
                                            for col_idx, h in enumerate(table[0]):
                                                if h:
                                                    h_str = str(h).strip()
                                                else:
                                                    h_str = f'Column_{col_idx}'
                                                
                                                # Ensure uniqueness
                                                original_h = h_str
                                                counter = 1
                                                while h_str in seen_headers:
                                                    h_str = f"{original_h}_{counter}"
                                                    counter += 1
                                                seen_headers[h_str] = True
                                                headers.append(h_str)
                                            
                                            # Create DataFrame with unique headers
                                            try:
                                                # Filter out empty rows
                                                data_rows = [row for row in table[1:] if any(cell for cell in row if cell)]
                                                if data_rows:
                                                    df = pd.DataFrame(data_rows, columns=headers)
                                                    all_data.append(df)
                                            except Exception as e:
                                                logger.warning(f"Error creating DataFrame from table on page {page_num}: {e}")
                                                
                            except Exception as e:
                                logger.warning(f"Error processing page {page_num}: {e}")
                                continue
                    
                    # Try to process collected data
                    if all_data:
                        try:
                            # Combine all dataframes
                            if len(all_data) == 1:
                                result = all_data[0]
                            else:
                                # Stack dataframes vertically
                                result = pd.DataFrame()
                                for df in all_data:
                                    if result.empty:
                                        result = df
                                    else:
                                        # Try to align columns
                                        common_cols = set(result.columns) & set(df.columns)
                                        if common_cols:
                                            # Use only common columns
                                            result = pd.concat([result[list(common_cols)], df[list(common_cols)]], 
                                                             ignore_index=True, sort=False)
                                        else:
                                            # No common columns, append as new columns
                                            for col in df.columns:
                                                if col not in result.columns:
                                                    result[col] = None
                                            result = pd.concat([result, df], ignore_index=True, sort=False)
                        except Exception as e:
                            logger.warning(f"Error combining DataFrames: {e}")
                            # Use the largest table
                            result = max(all_data, key=len)
                    
                    # If no tables found, try to parse text
                    elif all_text:
                        logger.info("No tables found in PDF, attempting text parsing")
                        full_text = '\n'.join(all_text)
                        
                        # Look for Amazon return patterns
                        lines = full_text.split('\n')
                        data_rows = []
                        
                        # Try to identify data patterns
                        for line in lines:
                            # Skip empty lines
                            if not line.strip():
                                continue
                            
                            # Look for lines that might contain order IDs (XXX-XXXXXXX-XXXXXXX)
                            if re.search(r'\d{3}-\d{7}-\d{7}', line):
                                # Split by multiple spaces or tabs
                                parts = re.split(r'\s{2,}|\t', line.strip())
                                if len(parts) >= 3:  # Need at least order ID, ASIN, and reason
                                    data_rows.append(parts)
                        
                        if data_rows:
                            # Create DataFrame from parsed rows
                            max_cols = max(len(row) for row in data_rows)
                            
                            # Pad rows to have same number of columns
                            for row in data_rows:
                                while len(row) < max_cols:
                                    row.append('')
                            
                            # Create generic column names
                            columns = [f'Column_{i}' for i in range(max_cols)]
                            result = pd.DataFrame(data_rows, columns=columns)
                        else:
                            raise ValueError("No structured data found in PDF")
                    else:
                        raise ValueError("No data could be extracted from PDF")
                    
                    # Clean and standardize the result
                    if 'result' in locals() and not result.empty:
                        # Remove completely empty rows
                        result = result.dropna(how='all')
                        # Clean string columns
                        result = result.apply(lambda x: x.str.strip() if x.dtype == "object" else x)
                        # Standardize column names
                        result = FileProcessor._standardize_amazon_columns(result)
                        return result
                    else:
                        raise ValueError("Failed to extract data from PDF")
                
                else:  # PyPDF2
                    import PyPDF2
                    reader = PyPDF2.PdfReader(file)
                    text = ""
                    for page in reader.pages:
                        text += page.extract_text() + "\n"
                    
                    # Try to parse as structured data
                    lines = [line for line in text.split('\n') if line.strip()]
                    data = []
                    headers = None
                    
                    for line in lines:
                        # Try tab-separated first
                        if '\t' in line:
                            parts = line.split('\t')
                        else:
                            # Try multiple spaces as separator
                            parts = [p.strip() for p in line.split('  ') if p.strip()]
                        
                        if parts and len(parts) > 2:  # Need at least 3 columns for meaningful data
                            if not headers and any(col in ' '.join(parts).lower() for col in ['order', 'asin', 'reason', 'return']):
                                headers = parts
                            else:
                                data.append(parts)
                    
                    if data:
                        if headers:
                            # Ensure all rows have same number of columns as headers
                            max_cols = len(headers)
                            cleaned_data = []
                            for row in data:
                                if len(row) < max_cols:
                                    row.extend([''] * (max_cols - len(row)))
                                elif len(row) > max_cols:
                                    row = row[:max_cols]
                                cleaned_data.append(row)
                            
                            # Make headers unique
                            unique_headers = []
                            header_counts = {}
                            for h in headers:
                                if h in header_counts:
                                    header_counts[h] += 1
                                    unique_headers.append(f"{h}_{header_counts[h]}")
                                else:
                                    header_counts[h] = 0
                                    unique_headers.append(h)
                            
                            result = pd.DataFrame(cleaned_data, columns=unique_headers)
                        else:
                            result = pd.DataFrame(data)
                        
                        result = FileProcessor._standardize_amazon_columns(result)
                        return result
                    else:
                        raise ValueError("Could not extract structured data from PDF")
            
            # CSV handling
            elif file_type in ['csv', 'text/csv'] or file.name.lower().endswith('.csv'):
                # Try different encodings
                encodings = ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252', 'utf-16', 'windows-1252', 'utf-8-sig']
                last_error = None
                
                for encoding in encodings:
                    try:
                        file.seek(0)
                        # First try to read a sample to detect issues
                        sample = file.read(10000)
                        file.seek(0)
                        
                        # Try to decode the sample
                        try:
                            sample.decode(encoding)
                        except:
                            continue
                        
                        # If sample decodes successfully, read the full file
                        df = pd.read_csv(file, encoding=encoding, low_memory=False, on_bad_lines='skip')
                        if len(df) > 0:
                            df = FileProcessor._standardize_amazon_columns(df)
                            return df
                    except Exception as e:
                        last_error = e
                        continue
                
                # If all encodings fail, try with error replacement
                try:
                    file.seek(0)
                    df = pd.read_csv(file, encoding='utf-8', errors='replace', low_memory=False, on_bad_lines='skip')
                    df = FileProcessor._standardize_amazon_columns(df)
                    return df
                except:
                    raise ValueError(f"Could not decode CSV file. Last error: {last_error}")
            
            # Excel handling
            elif file_type in ['xlsx', 'xls', 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
                             'application/vnd.ms-excel']:
                try:
                    file.seek(0)
                    # Try with openpyxl first
                    try:
                        import openpyxl
                        return pd.read_excel(file, engine='openpyxl')
                    except ImportError:
                        # Try default engine
                        return pd.read_excel(file, engine=None)
                except ImportError:
                    raise ValueError(
                        "Excel file processing requires openpyxl or xlrd. "
                        "Please install with: pip install openpyxl\n"
                        "For now, please export your Amazon returns as CSV format."
                    )
            
            # TSV handling
            elif file_type in ['tsv', 'text/tab-separated-values'] or (file_type == 'text/plain' and file.name.lower().endswith('.tsv')):
                # Try different encodings for TSV files
                encodings = ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252', 'utf-16', 'windows-1252']
                last_error = None
                
                for encoding in encodings:
                    try:
                        file.seek(0)
                        df = pd.read_csv(file, sep='\t', encoding=encoding, low_memory=False, on_bad_lines='skip')
                        if len(df) > 0:
                            df = FileProcessor._standardize_amazon_columns(df)
                            return df
                    except Exception as e:
                        last_error = e
                        continue
                
                # If all encodings fail, try with error handling
                try:
                    file.seek(0)
                    df = pd.read_csv(file, sep='\t', encoding='utf-8', errors='replace', low_memory=False, on_bad_lines='skip')
                    df = FileProcessor._standardize_amazon_columns(df)
                    return df
                except:
                    raise ValueError(f"Could not decode TSV file. Last error: {last_error}")
            
            # TXT handling with delimiter detection
            elif file_type in ['txt', 'text/plain']:
                file.seek(0)
                
                # Check if it's actually a TSV file
                if file.name.lower().endswith('.tsv'):
                    # Handle as TSV with encoding detection
                    encodings = ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252', 'utf-16', 'windows-1252']
                    last_error = None
                    
                    for encoding in encodings:
                        try:
                            file.seek(0)
                            df = pd.read_csv(file, sep='\t', encoding=encoding, low_memory=False, on_bad_lines='skip')
                            if len(df) > 0:
                                df = FileProcessor._standardize_amazon_columns(df)
                                return df
                        except Exception as e:
                            last_error = e
                            continue
                    
                    # If all encodings fail, try with error handling
                    try:
                        file.seek(0)
                        df = pd.read_csv(file, sep='\t', encoding='utf-8', errors='replace', low_memory=False, on_bad_lines='skip')
                        df = FileProcessor._standardize_amazon_columns(df)
                        return df
                    except:
                        raise ValueError(f"Could not decode TSV file. Last error: {last_error}")
                
                else:
                    # Regular text file processing
                    # Read first few lines to detect delimiter
                    sample_lines = []
                    for _ in range(min(10, 5)):  # Read up to 10 lines
                        line = file.readline()
                        if not line:
                            break
                        if isinstance(line, bytes):
                            line = line.decode('utf-8', errors='ignore')
                        sample_lines.append(line)
                    file.seek(0)
                    
                    if not sample_lines:
                        raise ValueError("Empty text file")
                    
                    # Detect delimiter
                    delimiters = ['\t', '|', ',', ';']
                    delimiter_counts = {}
                    
                    for delim in delimiters:
                        counts = [line.count(delim) for line in sample_lines]
                        # Check if delimiter appears consistently
                        if counts and all(c > 0 for c in counts):
                            avg_count = sum(counts) / len(counts)
                            delimiter_counts[delim] = (avg_count, min(counts), max(counts))
                    
                    # Choose delimiter with most consistent count
                    if delimiter_counts:
                        # Sort by consistency (smallest difference between min and max)
                        best_delimiter = min(delimiter_counts.items(), 
                                           key=lambda x: x[1][2] - x[1][1])[0]
                    else:
                        # Default to tab for .txt files
                        best_delimiter = '\t'
                    
                    try:
                        df = pd.read_csv(file, sep=best_delimiter, low_memory=False, encoding='utf-8', errors='replace')
                        df = FileProcessor._standardize_amazon_columns(df)
                        return df
                    except Exception as e:
                        # Try with spaces as delimiter
                        file.seek(0)
                        df = pd.read_csv(file, delim_whitespace=True, low_memory=False, encoding='utf-8', errors='replace')
                        df = FileProcessor._standardize_amazon_columns(df)
                        return df
            
            else:
                # Try to read as CSV as last resort
                try:
                    file.seek(0)
                    # Try with different encodings
                    for encoding in ['utf-8', 'latin-1', 'cp1252', 'utf-8-sig']:
                        try:
                            file.seek(0)
                            df = pd.read_csv(file, encoding=encoding, low_memory=False, on_bad_lines='skip')
                            if not df.empty:
                                df = FileProcessor._standardize_amazon_columns(df)
                                return df
                        except:
                            continue
                    
                    # Final attempt with error replacement
                    file.seek(0)
                    df = pd.read_csv(file, encoding='utf-8', errors='replace', low_memory=False, on_bad_lines='skip')
                    df = FileProcessor._standardize_amazon_columns(df)
                    return df
                except Exception as e:
                    raise ValueError(f"Could not read file as any known format: {str(e)}")
                
        except Exception as e:
            logger.error(f"Error reading file: {e}")
            logger.error(f"File type: {file_type}, File name: {getattr(file, 'name', 'unknown')}")
            error_msg = str(e)
            
            # Provide helpful error messages
            if "PDF" in error_msg or "pdf" in file_type.lower():
                raise Exception(
                    "PDF file processing failed. Please ensure your PDF contains tabular data, "
                    "or export your Amazon returns as CSV/Excel format instead.\n"
                    "Error: " + error_msg
                )
            elif "Excel" in error_msg or "xlsx" in file_type.lower():
                raise Exception(
                    "Excel file processing failed. Try saving as CSV format, "
                    "or install openpyxl with: pip install openpyxl\n"
                    "Error: " + error_msg
                )
            elif "codec" in error_msg or "decode" in error_msg:
                raise Exception(
                    "File encoding issue detected. The file contains special characters that couldn't be read.\n"
                    "Solutions:\n"
                    "1. Save the file as UTF-8 encoded CSV\n"
                    "2. Remove special characters (like ™, ®, –) from the file\n"
                    "3. Open in Excel and save as CSV (UTF-8)\n"
                    "Error: " + error_msg
                )
            elif "Reindexing" in error_msg:
                raise Exception(
                    "File structure issue - duplicate column names detected.\n"
                    "Solutions:\n"
                    "1. Export from Amazon as CSV format (recommended)\n"
                    "2. Check that column headers are unique\n"
                    "3. Remove any merged cells or complex formatting\n"
                    "Error: " + error_msg
                )
            else:
                raise Exception(f"Failed to read file: {error_msg}")

# Export key components
__all__ = [
    'EnhancedAIAnalyzer',
    'AIProvider',
    'MEDICAL_DEVICE_CATEGORIES',
    'FBA_REASON_MAP',
    'FDA_MDR_TRIGGERS',
    'detect_fda_reportable_event',
    'FileProcessor'
]
