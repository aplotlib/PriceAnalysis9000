"""
enhanced_ai_analysis.py - Enhanced AI Module for Medical Device Return Analysis
Supports both OpenAI and Anthropic with automatic fallback
Includes comprehensive injury detection and quality categorization
"""

import os
import logging
import json
import re
from typing import Dict, List, Any, Optional, Tuple, Union
from datetime import datetime
import pandas as pd
import time
from collections import Counter

logger = logging.getLogger(__name__)

# Try to import AI providers
OPENAI_AVAILABLE = False
ANTHROPIC_AVAILABLE = False

try:
    import openai
    OPENAI_AVAILABLE = True
    logger.info("OpenAI library available")
except ImportError:
    logger.warning("OpenAI library not available")

try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
    logger.info("Anthropic library available")
except ImportError:
    logger.warning("Anthropic library not available")

# Medical device return categories
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

# Enhanced injury detection patterns
INJURY_PATTERNS = {
    'death': {
        'keywords': ['death', 'died', 'fatal', 'fatality', 'deceased', 'passed away'],
        'severity': 'CRITICAL',
        'fda_reportable': True
    },
    'serious_injury': {
        'keywords': [
            'injury', 'injured', 'hurt', 'wound', 'wounded', 'bleeding', 'blood',
            'fracture', 'broken bone', 'break', 'severe', 'emergency', 'ER',
            'hospital', 'hospitalized', 'surgery', 'operation', 'amputation',
            'permanent damage', 'disability', 'impairment', 'paralysis',
            'unconscious', 'coma', 'concussion', 'trauma'
        ],
        'severity': 'HIGH',
        'fda_reportable': True
    },
    'falls': {
        'keywords': [
            'fall', 'fell', 'fallen', 'falling', 'dropped', 'collapsed',
            'slip', 'slipped', 'trip', 'tripped', 'tumble', 'toppled'
        ],
        'severity': 'HIGH',
        'fda_reportable': True
    },
    'burns': {
        'keywords': [
            'burn', 'burned', 'burnt', 'burning', 'scald', 'blister',
            'fire', 'flame', 'heat injury', 'thermal'
        ],
        'severity': 'HIGH',
        'fda_reportable': True
    },
    'allergic_reaction': {
        'keywords': [
            'allergic', 'allergy', 'reaction', 'rash', 'hives', 'itching',
            'swelling', 'anaphylaxis', 'breathing difficulty', 'throat closing'
        ],
        'severity': 'MODERATE',
        'fda_reportable': True
    },
    'infection': {
        'keywords': [
            'infection', 'infected', 'contaminated', 'bacteria', 'sepsis',
            'fever', 'pus', 'inflammation', 'abscess'
        ],
        'severity': 'MODERATE',
        'fda_reportable': True
    },
    'pain': {
        'keywords': [
            'pain', 'painful', 'ache', 'discomfort', 'sore', 'tender',
            'bruise', 'bruising', 'contusion'
        ],
        'severity': 'LOW',
        'fda_reportable': False
    }
}

# Quality issue patterns
QUALITY_PATTERNS = {
    'defective': ['defective', 'defect', 'faulty', 'malfunction', 'not working', 'broken', 'damaged'],
    'material': ['crack', 'tear', 'rip', 'hole', 'worn out', 'deteriorated', 'rust', 'corrosion'],
    'assembly': ['loose', 'wobbly', 'unstable', 'falls apart', 'came apart', 'disconnected'],
    'electrical': ['spark', 'smoke', 'shock', 'short circuit', 'electrical issue', 'power failure'],
    'mechanical': ['jammed', 'stuck', 'won\'t move', 'grinding', 'noise', 'vibration']
}

class AIProvider:
    """AI Provider options"""
    OPENAI = "openai"
    CLAUDE = "claude"
    ANTHROPIC = "anthropic"  # Alias for claude
    AUTO = "auto"
    FASTEST = "fastest"
    QUALITY = "quality"
    PATTERN = "pattern"  # Fallback pattern matching

class EnhancedAIAnalyzer:
    """Enhanced AI Analyzer with multi-provider support"""
    
    def __init__(self, provider: str = AIProvider.AUTO):
        self.provider = self._resolve_provider(provider)
        self.ai_client = None
        self.model = None
        self.api_calls = 0
        self.total_cost = 0.0
        self.ai_available = False
        self.pattern_mode = False
        self._initialize_ai()
    
    def _resolve_provider(self, provider: str) -> str:
        """Resolve provider selection"""
        if provider == AIProvider.AUTO:
            # Auto-detect best available provider
            if OPENAI_AVAILABLE and self._check_openai_key():
                return AIProvider.OPENAI
            elif ANTHROPIC_AVAILABLE and self._check_anthropic_key():
                return AIProvider.CLAUDE
            else:
                return AIProvider.PATTERN
        elif provider == AIProvider.FASTEST:
            # Prefer OpenAI for speed
            if OPENAI_AVAILABLE and self._check_openai_key():
                return AIProvider.OPENAI
            elif ANTHROPIC_AVAILABLE and self._check_anthropic_key():
                return AIProvider.CLAUDE
            else:
                return AIProvider.PATTERN
        elif provider == AIProvider.QUALITY:
            # Prefer Claude for quality
            if ANTHROPIC_AVAILABLE and self._check_anthropic_key():
                return AIProvider.CLAUDE
            elif OPENAI_AVAILABLE and self._check_openai_key():
                return AIProvider.OPENAI
            else:
                return AIProvider.PATTERN
        else:
            return provider
    
    def _check_openai_key(self) -> bool:
        """Check if OpenAI API key is available"""
        import streamlit as st
        if hasattr(st, 'secrets') and 'OPENAI_API_KEY' in st.secrets:
            return True
        return bool(os.getenv('OPENAI_API_KEY'))
    
    def _check_anthropic_key(self) -> bool:
        """Check if Anthropic API key is available"""
        import streamlit as st
        if hasattr(st, 'secrets') and 'ANTHROPIC_API_KEY' in st.secrets:
            return True
        return bool(os.getenv('ANTHROPIC_API_KEY'))
    
    def _initialize_ai(self):
        """Initialize AI provider with proper error handling"""
        import streamlit as st
        
        try:
            if self.provider == AIProvider.OPENAI and OPENAI_AVAILABLE:
                # Get OpenAI key
                api_key = None
                if hasattr(st, 'secrets') and 'OPENAI_API_KEY' in st.secrets:
                    api_key = st.secrets['OPENAI_API_KEY']
                elif os.getenv('OPENAI_API_KEY'):
                    api_key = os.getenv('OPENAI_API_KEY')
                
                if api_key:
                    self.ai_client = openai.OpenAI(api_key=api_key)
                    self.model = "gpt-4o-mini"  # Fast and cost-effective
                    self.ai_available = True
                    logger.info(f"OpenAI initialized with model: {self.model}")
                else:
                    logger.warning("OpenAI API key not found")
                    self._fallback_to_pattern()
            
            elif self.provider in [AIProvider.CLAUDE, AIProvider.ANTHROPIC] and ANTHROPIC_AVAILABLE:
                # Get Anthropic key
                api_key = None
                if hasattr(st, 'secrets') and 'ANTHROPIC_API_KEY' in st.secrets:
                    api_key = st.secrets['ANTHROPIC_API_KEY']
                elif os.getenv('ANTHROPIC_API_KEY'):
                    api_key = os.getenv('ANTHROPIC_API_KEY')
                
                if api_key:
                    self.ai_client = anthropic.Anthropic(api_key=api_key)
                    self.model = "claude-3-haiku-20240307"  # Fast Claude model
                    self.ai_available = True
                    logger.info(f"Anthropic initialized with model: {self.model}")
                else:
                    logger.warning("Anthropic API key not found")
                    self._fallback_to_pattern()
            
            else:
                logger.info("Using pattern matching mode")
                self._fallback_to_pattern()
                
        except Exception as e:
            logger.error(f"AI initialization failed: {e}")
            self._fallback_to_pattern()
    
    def _fallback_to_pattern(self):
        """Fallback to pattern matching mode"""
        self.provider = AIProvider.PATTERN
        self.ai_available = False
        self.pattern_mode = True
        logger.info("Falling back to pattern matching mode")
    
    def test_ai_connection(self) -> Dict[str, Any]:
        """Test AI connection with actual categorization"""
        test_cases = [
            ("Product broke after first use", "Product Defects/Quality"),
            ("Caused severe injury to my hand", "Injury/Adverse Event"),
            ("Too small for intended use", "Size/Fit Issues")
        ]
        
        results = []
        
        if self.ai_available and self.ai_client:
            try:
                # Test with a simple query first
                test_response = self._call_ai("Test connection", max_tokens=10)
                if test_response:
                    # Run categorization tests
                    for text, expected in test_cases[:2]:
                        category = self.categorize_return(text, "")
                        results.append({
                            'text': text,
                            'expected': expected,
                            'actual': category.get('category', 'Error'),
                            'correct': category.get('category') == expected
                        })
                    
                    return {
                        'status': 'AI connection successful',
                        'provider': self.provider,
                        'model': self.model,
                        'results': results
                    }
            except Exception as e:
                logger.error(f"AI test failed: {e}")
        
        return {
            'status': 'Pattern matching mode (no AI)',
            'provider': 'pattern',
            'model': None,
            'results': None
        }
    
    def _call_ai(self, prompt: str, max_tokens: int = 100) -> Optional[str]:
        """Call AI provider with error handling"""
        if not self.ai_client or not self.ai_available:
            return None
        
        try:
            if self.provider == AIProvider.OPENAI:
                response = self.ai_client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": "You are a medical device quality analyst. Categorize returns accurately."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.1,
                    max_tokens=max_tokens
                )
                self.api_calls += 1
                return response.choices[0].message.content.strip()
            
            elif self.provider in [AIProvider.CLAUDE, AIProvider.ANTHROPIC]:
                response = self.ai_client.messages.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.1,
                    max_tokens=max_tokens
                )
                self.api_calls += 1
                return response.content[0].text.strip()
                
        except Exception as e:
            logger.error(f"AI call failed: {e}")
            return None
    
    def categorize_return(self, reason: str, comment: str = "", 
                         product_name: str = "", asin: str = "") -> Dict[str, Any]:
        """Categorize return with injury detection"""
        full_text = f"{reason} {comment}".strip()
        
        # First check for injuries
        injury_analysis = self.detect_injuries(full_text)
        
        # Get category
        if self.ai_available:
            category = self._ai_categorize(full_text)
        else:
            category = self._pattern_categorize(full_text)
        
        # Override category if injury detected
        if injury_analysis['has_injury'] and injury_analysis['severity'] in ['CRITICAL', 'HIGH']:
            category = 'Injury/Adverse Event'
        
        # Check for quality issues
        quality_issues = self._detect_quality_issues(full_text)
        
        return {
            'category': category,
            'has_injury': injury_analysis['has_injury'],
            'injury_type': injury_analysis['injury_type'],
            'severity': injury_analysis['severity'],
            'fda_reportable': injury_analysis['fda_reportable'],
            'quality_issues': quality_issues,
            'confidence': 0.95 if self.ai_available else 0.7,
            'product_name': product_name,
            'asin': asin
        }
    
    def _ai_categorize(self, text: str) -> str:
        """Use AI to categorize return"""
        if not text:
            return 'Non-Medical Issue'
        
        prompt = f"""Categorize this medical device return into EXACTLY ONE category:

Categories:
- Product Defects/Quality
- Injury/Adverse Event
- Performance/Effectiveness
- Size/Fit Issues
- Stability/Safety Issues
- Material/Component Failure
- Design Issues
- Comfort/Usability Issues
- Compatibility Issues
- Assembly/Installation Issues
- Wrong Product/Labeling
- Missing Components
- Customer Error
- Non-Medical Issue

Return text: "{text}"

Respond with ONLY the category name, nothing else."""
        
        response = self._call_ai(prompt, max_tokens=50)
        
        if response:
            # Validate response
            for category in MEDICAL_DEVICE_CATEGORIES:
                if category.lower() in response.lower():
                    return category
        
        # Fallback to pattern matching
        return self._pattern_categorize(text)
    
    def _pattern_categorize(self, text: str) -> str:
        """Pattern-based categorization"""
        text_lower = text.lower()
        
        # Priority order categorization
        if any(word in text_lower for word in ['injury', 'injured', 'hurt', 'hospital', 'emergency']):
            return 'Injury/Adverse Event'
        
        elif any(word in text_lower for word in ['defect', 'broken', 'damaged', 'malfunction', 'not working']):
            return 'Product Defects/Quality'
        
        elif any(word in text_lower for word in ['too small', 'too large', 'too big', 'size', 'fit']):
            return 'Size/Fit Issues'
        
        elif any(word in text_lower for word in ['unstable', 'wobble', 'tip', 'fall over']):
            return 'Stability/Safety Issues'
        
        elif any(word in text_lower for word in ['material', 'fabric', 'plastic', 'metal', 'component']):
            return 'Material/Component Failure'
        
        elif any(word in text_lower for word in ['uncomfortable', 'comfort', 'hard to use']):
            return 'Comfort/Usability Issues'
        
        elif any(word in text_lower for word in ['compatible', 'fit with', 'work with']):
            return 'Compatibility Issues'
        
        elif any(word in text_lower for word in ['wrong', 'incorrect', 'not as described']):
            return 'Wrong Product/Labeling'
        
        elif any(word in text_lower for word in ['missing', 'incomplete', 'not included']):
            return 'Missing Components'
        
        elif any(word in text_lower for word in ['mistake', 'accident', 'my fault', 'ordered wrong']):
            return 'Customer Error'
        
        else:
            return 'Non-Medical Issue'
    
    def detect_injuries(self, text: str) -> Dict[str, Any]:
        """Comprehensive injury detection"""
        if not text:
            return {
                'has_injury': False,
                'injury_type': None,
                'severity': None,
                'fda_reportable': False,
                'keywords_found': []
            }
        
        text_lower = text.lower()
        injuries_found = []
        keywords_found = []
        max_severity = None
        fda_reportable = False
        
        # Check each injury pattern
        for injury_type, pattern_data in INJURY_PATTERNS.items():
            found_keywords = [kw for kw in pattern_data['keywords'] if kw in text_lower]
            
            if found_keywords:
                injuries_found.append(injury_type)
                keywords_found.extend(found_keywords)
                
                # Update severity
                if not max_severity or self._compare_severity(pattern_data['severity'], max_severity) > 0:
                    max_severity = pattern_data['severity']
                
                # Check FDA reportability
                if pattern_data['fda_reportable']:
                    fda_reportable = True
        
        return {
            'has_injury': len(injuries_found) > 0,
            'injury_type': injuries_found[0] if injuries_found else None,
            'all_types': injuries_found,
            'severity': max_severity,
            'fda_reportable': fda_reportable,
            'keywords_found': list(set(keywords_found))
        }
    
    def _compare_severity(self, sev1: str, sev2: str) -> int:
        """Compare severity levels"""
        severity_order = {'LOW': 0, 'MODERATE': 1, 'HIGH': 2, 'CRITICAL': 3}
        return severity_order.get(sev1, 0) - severity_order.get(sev2, 0)
    
    def _detect_quality_issues(self, text: str) -> List[str]:
        """Detect specific quality issues"""
        text_lower = text.lower()
        issues_found = []
        
        for issue_type, keywords in QUALITY_PATTERNS.items():
            if any(keyword in text_lower for keyword in keywords):
                issues_found.append(issue_type)
        
        return issues_found
    
    def check_for_injury(self, text: str) -> bool:
        """Simple injury check for compatibility"""
        result = self.detect_injuries(text)
        return result['has_injury']
    
    def batch_categorize(self, df: pd.DataFrame) -> pd.DataFrame:
        """Batch process returns with progress tracking"""
        import streamlit as st
        
        # Add result columns
        df['category'] = 'Uncategorized'
        df['has_injury'] = False
        df['injury_type'] = ''
        df['severity'] = ''
        df['fda_reportable'] = False
        df['quality_issues'] = ''
        
        total = len(df)
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Find relevant columns
        reason_col = None
        comment_col = None
        
        for col in df.columns:
            col_lower = col.lower()
            if not reason_col and 'reason' in col_lower:
                reason_col = col
            elif not comment_col and ('comment' in col_lower or 'feedback' in col_lower):
                comment_col = col
        
        # Process each row
        injury_count = 0
        quality_count = 0
        
        for idx, row in df.iterrows():
            # Update progress
            if idx % 10 == 0:
                progress = idx / total
                progress_bar.progress(progress)
                status_text.text(f"Processing returns... {idx}/{total} ({injury_count} injuries found)")
            
            # Get text
            reason = str(row[reason_col]) if reason_col and pd.notna(row.get(reason_col)) else ''
            comment = str(row[comment_col]) if comment_col and pd.notna(row.get(comment_col)) else ''
            
            # Categorize
            result = self.categorize_return(reason, comment)
            
            # Update row
            df.at[idx, 'category'] = result['category']
            df.at[idx, 'has_injury'] = result['has_injury']
            df.at[idx, 'injury_type'] = result.get('injury_type', '')
            df.at[idx, 'severity'] = result.get('severity', '')
            df.at[idx, 'fda_reportable'] = result.get('fda_reportable', False)
            df.at[idx, 'quality_issues'] = ', '.join(result.get('quality_issues', []))
            
            # Count issues
            if result['has_injury']:
                injury_count += 1
            if result['quality_issues']:
                quality_count += 1
        
        progress_bar.empty()
        status_text.empty()
        
        # Show summary
        if injury_count > 0:
            st.warning(f"⚠️ Found {injury_count} potential injury cases requiring review")
        
        st.info(f"✅ Categorized {total} returns: {quality_count} quality issues, {injury_count} injuries")
        
        return df
    
    def generate_fda_report(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Generate FDA-focused report"""
        reportable_df = df[df['fda_reportable'] == True]
        
        report = {
            'summary': {
                'total_returns': len(df),
                'reportable_events': len(reportable_df),
                'critical_events': len(df[df['severity'] == 'CRITICAL']),
                'high_severity': len(df[df['severity'] == 'HIGH']),
                'injury_cases': len(df[df['has_injury'] == True])
            },
            'by_category': df['category'].value_counts().to_dict(),
            'by_injury_type': df[df['has_injury']]['injury_type'].value_counts().to_dict() if len(df[df['has_injury']]) > 0 else {},
            'recommendations': []
        }
        
        # Generate recommendations
        if report['summary']['critical_events'] > 0:
            report['recommendations'].append(
                "IMMEDIATE ACTION: Critical events detected. Initiate FDA MDR process immediately."
            )
        
        if report['summary']['injury_cases'] > 5:
            report['recommendations'].append(
                "Multiple injury cases detected. Consider product safety review and potential recall."
            )
        
        return report
    
    def export_fda_summary(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create FDA summary export"""
        fda_df = df[df['fda_reportable'] == True].copy()
        
        if len(fda_df) > 0:
            # Select key columns
            export_cols = ['order-id', 'asin', 'product-name', 'reason', 'customer-comments',
                          'category', 'injury_type', 'severity', 'fda_reportable']
            
            available_cols = [col for col in export_cols if col in fda_df.columns]
            return fda_df[available_cols].sort_values('severity', ascending=False)
        
        return pd.DataFrame()

# Legacy function for compatibility
def detect_fda_reportable_event(text: str) -> Dict[str, Any]:
    """Legacy function for FDA event detection"""
    analyzer = EnhancedAIAnalyzer(AIProvider.PATTERN)
    injury_result = analyzer.detect_injuries(text)
    
    return {
        'is_reportable': injury_result['fda_reportable'],
        'severity': injury_result['severity'],
        'event_types': injury_result.get('all_types', []),
        'confidence': 0.8 if injury_result['has_injury'] else 0.0,
        'requires_immediate_review': injury_result['severity'] in ['CRITICAL', 'HIGH'] if injury_result['severity'] else False
    }

# Export all necessary components
__all__ = [
    'EnhancedAIAnalyzer',
    'AIProvider',
    'MEDICAL_DEVICE_CATEGORIES',
    'INJURY_PATTERNS',
    'detect_fda_reportable_event'
]
