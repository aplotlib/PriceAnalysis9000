"""
Amazon Returns Quality Analyzer - FDA Reportable Event Detection
Version: 7.0 - Medical Device Injury Reporting Focus
Critical: Identifies FDA MDR (Medical Device Reporting) candidates
"""

# IMPORTANT: This is a module file - DO NOT use st.set_page_config() here
# st.set_page_config() should only be in the main app file

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

# Import Streamlit but don't call set_page_config
import streamlit as st

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
            'confidence': 0.0
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
        self._initialize_ai()
        
    def _initialize_ai(self):
        """Initialize AI provider"""
        try:
            if self.provider in [AIProvider.OPENAI, AIProvider.FASTEST] and OPENAI_AVAILABLE:
                self.ai_client = openai.OpenAI()
                self.model = "gpt-4o-mini" if self.provider == AIProvider.FASTEST else "gpt-4o"
            elif self.provider == AIProvider.CLAUDE and CLAUDE_AVAILABLE:
                self.ai_client = anthropic.Anthropic()
                self.model = "claude-3-5-sonnet-20241022"
            else:
                logger.warning("No AI provider available - using pattern matching only")
        except Exception as e:
            logger.error(f"Failed to initialize AI: {e}")
            self.ai_client = None
    
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
            # Fallback to FBA mapping only as last resort
            category = FBA_REASON_MAP.get(reason, 'Product Defects/Quality')
            confidence = 0.7
        
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
            'requires_mdr': fda_analysis['severity'] in ['CRITICAL', 'HIGH'],
            'requires_immediate_review': fda_analysis['requires_immediate_review'],
            'product_name': product_name,
            'asin': asin
        }
    
    def _ai_categorize(self, text: str, product_name: str) -> str:
        """Use AI to categorize return reason"""
        if not self.ai_client:
            return 'Product Defects/Quality'
        
        prompt = f"""
        You are an FDA medical device expert analyzing return reasons for potential adverse events.
        
        Product: {product_name}
        Return Text: {text}
        
        Categorize this return into ONE of these categories:
        {', '.join(MEDICAL_DEVICE_CATEGORIES)}
        
        Focus on identifying potential injuries, malfunctions, or safety issues that may require FDA reporting.
        
        Return only the category name, nothing else.
        """
        
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
            
            # Validate category
            if category in MEDICAL_DEVICE_CATEGORIES:
                return category
            else:
                return 'Product Defects/Quality'
                
        except Exception as e:
            logger.error(f"AI categorization failed: {e}")
            return 'Product Defects/Quality'
    
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
        
        # Create progress placeholder
        progress_placeholder = st.empty()
        status_placeholder = st.empty()
        
        # Track FDA reportable events
        reportable_count = 0
        critical_count = 0
        
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
            
            reason = str(row.get(reason_col, ''))
            comment = str(row.get(comment_col, ''))
            product = str(row.get(product_col, ''))
            asin = str(row.get(asin_col, ''))
            
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
    def read_file(file, file_type: str) -> pd.DataFrame:
        """Read various file formats and return DataFrame"""
        try:
            if file_type in ['csv', 'text/csv']:
                # Try different encodings
                encodings = ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252']
                for encoding in encodings:
                    try:
                        file.seek(0)
                        return pd.read_csv(file, encoding=encoding)
                    except:
                        continue
                raise ValueError("Could not decode CSV file")
                
            elif file_type in ['xlsx', 'xls', 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet']:
                return pd.read_excel(file, engine='openpyxl')
                
            elif file_type in ['tsv', 'text/tab-separated-values']:
                return pd.read_csv(file, sep='\t')
                
            elif file_type in ['txt', 'text/plain']:
                # Try to detect delimiter
                file.seek(0)
                first_line = file.readline().decode('utf-8', errors='ignore')
                file.seek(0)
                
                if '\t' in first_line:
                    return pd.read_csv(file, sep='\t')
                elif '|' in first_line:
                    return pd.read_csv(file, sep='|')
                else:
                    return pd.read_csv(file)
                    
            elif file_type == 'pdf' and PDF_AVAILABLE:
                # Extract tables from PDF
                import pdfplumber
                
                all_tables = []
                with pdfplumber.open(file) as pdf:
                    for page in pdf.pages:
                        tables = page.extract_tables()
                        for table in tables:
                            if table:
                                df = pd.DataFrame(table[1:], columns=table[0])
                                all_tables.append(df)
                
                if all_tables:
                    return pd.concat(all_tables, ignore_index=True)
                else:
                    raise ValueError("No tables found in PDF")
                    
            else:
                raise ValueError(f"Unsupported file type: {file_type}")
                
        except Exception as e:
            logger.error(f"Error reading file: {e}")
            raise

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
