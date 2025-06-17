"""
Enhanced AI Analysis Module - Universal File Understanding with Medical Device Focus
Version: 11.0 - Dual AI with Medical Device Categorization
"""

import logging
import os
import json
import re
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple, Union
from collections import Counter
import time
import base64
from io import BytesIO
from dataclasses import dataclass, field
import asyncio
from concurrent.futures import ThreadPoolExecutor, as_completed
from enum import Enum

# Standard imports
import pandas as pd
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Safe imports with fallbacks
def safe_import(module_name):
    try:
        return __import__(module_name), True
    except ImportError:
        logger.warning(f"Module {module_name} not available")
        return None, False

# Import required libraries
requests, has_requests = safe_import('requests')
anthropic_module, has_anthropic = safe_import('anthropic')
pdfplumber, has_pdfplumber = safe_import('pdfplumber')
PIL, has_pil = safe_import('PIL')
pytesseract, has_tesseract = safe_import('pytesseract')

# Model configurations
MODEL_CONFIG = {
    'openai': {
        'primary': 'gpt-3.5-turbo',  # Fast for categorization
        'vision': 'gpt-4-vision-preview',
        'fallback': 'gpt-3.5-turbo',
        'chat': 'gpt-3.5-turbo'
    },
    'claude': {
        'primary': 'claude-3-haiku-20240307',  # Fastest
        'vision': 'claude-3-haiku-20240307',
        'advanced': 'claude-3-sonnet-20240229',
        'chat': 'claude-3-haiku-20240307'
    }
}

# API settings for speed
API_TIMEOUT = 30
MAX_RETRIES = 2
BATCH_SIZE = 50  # Optimal batch size
MAX_WORKERS = 5

# Token limits
TOKEN_LIMITS = {
    'standard': 150,  # Reduced for speed
    'enhanced': 300,
    'chat': 500
}

# Pricing per 1K tokens
PRICING = {
    'gpt-3.5-turbo': {'input': 0.0005, 'output': 0.0015},
    'gpt-4': {'input': 0.03, 'output': 0.06},
    'claude-3-haiku-20240307': {'input': 0.00025, 'output': 0.00125},
    'claude-3-sonnet-20240229': {'input': 0.003, 'output': 0.015}
}

# Medical Device Return Categories - Column K categories
MEDICAL_DEVICE_CATEGORIES = [
    'Size/Fit Issues',
    'Comfort Issues', 
    'Product Defects/Quality',
    'Performance/Effectiveness',
    'Stability/Positioning Issues',
    'Equipment Compatibility',
    'Design/Material Issues',
    'Wrong Product/Misunderstanding',
    'Missing Components',
    'Customer Error/Changed Mind',
    'Shipping/Fulfillment Issues',
    'Assembly/Usage Difficulty',
    'Medical/Health Concerns',
    'Price/Value',
    'Other/Miscellaneous'
]

# FBA reason code mapping
FBA_REASON_MAP = {
    # Quality issues
    'DEFECTIVE': 'Product Defects/Quality',
    'DAMAGED_BY_FC': 'Product Defects/Quality',
    'DAMAGED_BY_CARRIER': 'Shipping/Fulfillment Issues',
    'QUALITY_NOT_ADEQUATE': 'Product Defects/Quality',
    'EXPIRED_ITEM': 'Product Defects/Quality',
    'DAMAGED_GLASS_VIAL': 'Product Defects/Quality',
    'DAMAGED': 'Product Defects/Quality',
    'BROKEN': 'Product Defects/Quality',
    'NOT_WORKING': 'Product Defects/Quality',
    'DOESNT_WORK': 'Product Defects/Quality',
    
    # Wrong item
    'NOT_AS_DESCRIBED': 'Wrong Product/Misunderstanding',
    'WRONG_ITEM': 'Wrong Product/Misunderstanding',
    'DIFFERENT_PRODUCT': 'Wrong Product/Misunderstanding',
    'SWITCHEROO': 'Wrong Product/Misunderstanding',
    'INACCURATE_WEBSITE_DESCRIPTION': 'Wrong Product/Misunderstanding',
    
    # Missing/Compatibility
    'MISSING_PARTS': 'Missing Components',
    'MISSING_ITEM': 'Missing Components',
    'NOT_COMPATIBLE': 'Equipment Compatibility',
    'NOT_COMPATIBLE_WITH_DEVICE': 'Equipment Compatibility',
    
    # Customer issues
    'UNWANTED_ITEM': 'Customer Error/Changed Mind',
    'UNAUTHORIZED_PURCHASE': 'Customer Error/Changed Mind',
    'CUSTOMER_DAMAGED': 'Customer Error/Changed Mind',
    'ORDERED_WRONG_ITEM': 'Customer Error/Changed Mind',
    'UNNEEDED_ITEM': 'Customer Error/Changed Mind',
    'BAD_GIFT': 'Customer Error/Changed Mind',
    
    # Size/Fit
    'DOES_NOT_FIT': 'Size/Fit Issues',
    'TOO_SMALL': 'Size/Fit Issues',
    'TOO_LARGE': 'Size/Fit Issues',
    
    # Other
    'BETTER_PRICE_AVAILABLE': 'Price/Value',
    'NOT_DELIVERED': 'Shipping/Fulfillment Issues',
    'ARRIVED_LATE': 'Shipping/Fulfillment Issues',
    'UNCOMFORTABLE': 'Comfort Issues',
    'DIFFICULT_TO_USE': 'Assembly/Usage Difficulty',
    'UNSATISFACTORY_PRODUCT': 'Performance/Effectiveness'
}

# Quick categorization patterns
QUICK_PATTERNS = {
    'Size/Fit Issues': [
        r'too (small|large|big|tight|loose)', r'doesn[\']?t fit', r'wrong size',
        r'size issue', r'(small|large)r than expected', r'fit issue', r'too (tall|short|wide)'
    ],
    'Product Defects/Quality': [
        r'defect', r'broken', r'damaged', r'poor quality', r'doesn[\']?t work',
        r'stopped working', r'malfunction', r'fell apart', r'ripped', r'torn',
        r'dead on arrival', r'doa', r'bad velcro', r'won[\']?t inflate'
    ],
    'Wrong Product/Misunderstanding': [
        r'wrong (item|product)', r'not as described', r'different than',
        r'thought it was', r'expected', r'not what I ordered', r'wrong model'
    ],
    'Customer Error/Changed Mind': [
        r'changed mind', r'don[\']?t need', r'ordered by mistake',
        r'accidentally', r'no longer need', r'bought wrong', r'my (fault|mistake)'
    ],
    'Comfort Issues': [
        r'uncomfort', r'hurts', r'painful', r'too (hard|soft|firm)',
        r'causes pain', r'irritat', r'not comfortable', r'too stiff'
    ],
    'Equipment Compatibility': [
        r'doesn[\']?t fit (my|the|walker|wheelchair|toilet)', r'not compatible', 
        r'incompatible', r'doesn[\']?t work with', r'won[\']?t fit'
    ],
    'Missing Components': [
        r'missing (parts|pieces|components)', r'incomplete', r'not included',
        r'no instructions', r'parts missing'
    ],
    'Performance/Effectiveness': [
        r'ineffective', r'doesn[\']?t (help|work well)', r'not enough support',
        r'not enough compression', r'unstable', r'not cold enough'
    ]
}

# Compile patterns for speed
COMPILED_PATTERNS = {
    category: [re.compile(pattern, re.IGNORECASE) for pattern in patterns]
    for category, patterns in QUICK_PATTERNS.items()
}

# Return reason categories for general analysis
RETURN_CATEGORIES = {
    'QUALITY_DEFECTS': {
        'keywords': ['defective', 'broken', 'damaged', 'doesn\'t work', 'poor quality'],
        'priority': 'critical',
        'maps_to': 'Product Defects/Quality'
    },
    'SIZE_FIT_ISSUES': {
        'keywords': ['too small', 'too large', 'doesn\'t fit', 'wrong size'],
        'priority': 'high',
        'maps_to': 'Size/Fit Issues'
    },
    'FUNCTIONALITY_ISSUES': {
        'keywords': ['not comfortable', 'hard to use', 'unstable', 'difficult'],
        'priority': 'high',
        'maps_to': 'Performance/Effectiveness'
    },
    'WRONG_PRODUCT': {
        'keywords': ['wrong item', 'not as described', 'different'],
        'priority': 'medium',
        'maps_to': 'Wrong Product/Misunderstanding'
    },
    'BUYER_MISTAKE': {
        'keywords': ['bought by mistake', 'accidentally ordered', 'my fault'],
        'priority': 'low',
        'maps_to': 'Customer Error/Changed Mind'
    }
}

class AIProvider(Enum):
    OPENAI = "openai"
    CLAUDE = "claude"
    BOTH = "both"
    FASTEST = "fastest"

@dataclass
class FileAnalysis:
    """Results from file analysis"""
    file_type: str
    content_type: str  # 'returns', 'reviews', 'other'
    extracted_data: Dict[str, Any]
    confidence: float
    ai_provider: str
    needs_clarification: bool = False
    clarification_questions: List[str] = field(default_factory=list)

@dataclass
class CostEstimate:
    """Cost tracking"""
    provider: str
    model: str
    input_tokens: int
    output_tokens: int
    total_cost: float

class CostTracker:
    """Track API costs"""
    def __init__(self):
        self.total_cost = 0.0
        self.api_calls = 0
        self.quick_categorizations = 0
        self.ai_categorizations = 0
        
    def add_cost(self, cost: float):
        """Add cost to total"""
        self.total_cost += cost
        self.api_calls += 1
        
    def get_summary(self):
        """Get cost summary"""
        return {
            'total_cost': round(self.total_cost, 4),
            'api_calls': self.api_calls,
            'quick_categorizations': self.quick_categorizations,
            'ai_categorizations': self.ai_categorizations
        }

def quick_categorize(complaint: str, fba_reason: str = None) -> Optional[str]:
    """Quick pattern-based categorization for speed"""
    if not complaint:
        return None
    
    # Check FBA reason first
    if fba_reason and fba_reason in FBA_REASON_MAP:
        return FBA_REASON_MAP[fba_reason]
    
    complaint_lower = complaint.lower()
    
    # Check compiled patterns
    for category, patterns in COMPILED_PATTERNS.items():
        for pattern in patterns:
            if pattern.search(complaint_lower):
                return category
    
    return None

def estimate_tokens(text: str) -> int:
    """Estimate token count"""
    return max(len(text) // 4, len(text.split()) * 4 // 3)

class UniversalAIAnalyzer:
    """Universal AI analyzer with medical device focus"""
    
    def __init__(self, provider: AIProvider = AIProvider.FASTEST):
        self.provider = provider
        self.openai_key = self._get_api_key('openai')
        self.claude_key = self._get_api_key('claude')
        self.executor = ThreadPoolExecutor(max_workers=MAX_WORKERS)
        self.cost_tracker = CostTracker()  # Initialize cost tracker
        
        # Initialize Claude client if available
        self.claude_client = None
        if self.claude_key and has_anthropic:
            try:
                from anthropic import Anthropic
                self.claude_client = Anthropic(api_key=self.claude_key)
                logger.info("Claude API initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize Claude: {e}")
        
        # Session for connection pooling
        self.session = None
        if has_requests:
            self.session = requests.Session()
            
        self.openai_configured = bool(self.openai_key and has_requests)
        self.claude_configured = bool(self.claude_client)
        
        logger.info(f"AI Analyzer initialized - OpenAI: {self.openai_configured}, Claude: {self.claude_configured}")
        
    def _get_api_key(self, provider: str) -> Optional[str]:
        """Get API key from Streamlit secrets or environment"""
        try:
            import streamlit as st
            if hasattr(st, 'secrets'):
                # Try various key names
                if provider == 'openai':
                    for key in ['openai_api_key', 'OPENAI_API_KEY', 'openai']:
                        if key in st.secrets:
                            return str(st.secrets[key]).strip()
                elif provider == 'claude':
                    for key in ['claude_api_key', 'anthropic_api_key', 'ANTHROPIC_API_KEY', 'claude']:
                        if key in st.secrets:
                            return str(st.secrets[key]).strip()
        except:
            pass
        
        # Try environment variables
        env_map = {
            'openai': ['OPENAI_API_KEY'],
            'claude': ['ANTHROPIC_API_KEY', 'CLAUDE_API_KEY']
        }
        
        for env_name in env_map.get(provider, []):
            if env_name in os.environ:
                return os.environ[env_name].strip()
        
        return None
    
    def get_available_providers(self) -> List[str]:
        """Get list of available AI providers"""
        providers = []
        if self.openai_configured:
            providers.append('openai')
        if self.claude_configured:
            providers.append('claude')
        return providers
    
    def get_api_status(self) -> Dict[str, Any]:
        """Get API status"""
        return {
            'available': self.openai_configured or self.claude_configured,
            'openai_configured': self.openai_configured,
            'claude_configured': self.claude_configured,
            'provider': self.provider.value,
            'cost_summary': self.cost_tracker.get_summary()
        }
    
    def categorize_return(self, complaint: str, fba_reason: str = None, mode: str = 'standard') -> Tuple[str, float, str, str]:
        """Categorize return with medical device categories"""
        if not complaint or not complaint.strip():
            return 'Other/Miscellaneous', 0.1, 'none', 'en'
        
        # Try quick categorization first
        quick_category = quick_categorize(complaint, fba_reason)
        if quick_category:
            self.cost_tracker.quick_categorizations += 1
            return quick_category, 0.9, 'standard', 'en'
        
        # AI categorization
        self.cost_tracker.ai_categorizations += 1
        
        # Build prompts
        system_prompt = """You are a medical device quality expert. Categorize this return into exactly one category from the provided list. Respond with ONLY the category name, nothing else."""
        
        categories_list = '\n'.join(f'- {cat}' for cat in MEDICAL_DEVICE_CATEGORIES)
        
        user_prompt = f"""Complaint: "{complaint}"

Categories:
{categories_list}

Category:"""
        
        # Call AI based on provider setting
        if self.provider == AIProvider.FASTEST and self.claude_configured:
            response = self._call_claude_sync(user_prompt, system_prompt, mode)
        elif self.openai_configured:
            response = self._call_openai_sync(user_prompt, system_prompt, mode)
        else:
            response = None
        
        if response:
            category = self._clean_category_response(response)
            return category, 0.85, 'standard', 'en'
        
        return 'Other/Miscellaneous', 0.3, 'none', 'en'
    
    def categorize_batch(self, complaints: List[Dict[str, Any]], mode: str = 'standard') -> List[Dict[str, Any]]:
        """Categorize multiple complaints in parallel"""
        results = []
        futures = []
        
        # Submit all tasks
        for item in complaints:
            future = self.executor.submit(
                self.categorize_return,
                item.get('complaint', ''),
                item.get('fba_reason'),
                mode
            )
            futures.append((future, item))
        
        # Collect results
        for future, item in futures:
            try:
                category, confidence, severity, language = future.result(timeout=API_TIMEOUT)
                result = item.copy()
                result.update({
                    'category': category,
                    'confidence': confidence,
                    'severity': severity,
                    'language': language
                })
                results.append(result)
            except Exception as e:
                logger.error(f"Batch categorization error: {e}")
                result = item.copy()
                result.update({
                    'category': 'Other/Miscellaneous',
                    'confidence': 0.1,
                    'severity': 'none',
                    'language': 'en'
                })
                results.append(result)
        
        return results
    
    def _call_openai_sync(self, prompt: str, system_prompt: str, mode: str = 'standard') -> Optional[str]:
        """Synchronous OpenAI call"""
        if not self.openai_configured:
            return None
            
        model = MODEL_CONFIG['openai']['primary']
        max_tokens = TOKEN_LIMITS[mode]
        
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.openai_key}"
        }
        
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.1,
            "max_tokens": max_tokens
        }
        
        try:
            response = (self.session or requests).post(
                "https://api.openai.com/v1/chat/completions",
                headers=headers,
                json=payload,
                timeout=API_TIMEOUT
            )
            
            if response.status_code == 200:
                result = response.json()
                content = result["choices"][0]["message"]["content"].strip()
                
                # Track cost
                usage = result.get("usage", {})
                input_tokens = usage.get("prompt_tokens", 0)
                output_tokens = usage.get("completion_tokens", 0)
                cost = (input_tokens * 0.0005 + output_tokens * 0.0015) / 1000
                self.cost_tracker.add_cost(cost)
                
                return content
                
        except Exception as e:
            logger.error(f"OpenAI error: {e}")
            
        return None
    
    def _call_claude_sync(self, prompt: str, system_prompt: str, mode: str = 'standard') -> Optional[str]:
        """Synchronous Claude call"""
        if not self.claude_configured:
            return None
            
        try:
            message = self.claude_client.messages.create(
                model=MODEL_CONFIG['claude']['primary'],
                max_tokens=TOKEN_LIMITS[mode],
                temperature=0.1,
                system=system_prompt,
                messages=[{"role": "user", "content": prompt}]
            )
            
            content = message.content[0].text.strip()
            
            # Track cost (approximate)
            input_tokens = estimate_tokens(system_prompt + prompt)
            output_tokens = estimate_tokens(content)
            cost = (input_tokens * 0.00025 + output_tokens * 0.00125) / 1000
            self.cost_tracker.add_cost(cost)
            
            return content
            
        except Exception as e:
            logger.error(f"Claude error: {e}")
            
        return None
    
    async def _call_claude(self, text: str, prompt: str) -> Dict[str, Any]:
        """Async Claude call for file analysis"""
        if not self.claude_configured:
            return {'success': False, 'error': 'Claude not configured'}
        
        try:
            message = self.claude_client.messages.create(
                model=MODEL_CONFIG['claude']['primary'],
                max_tokens=1000,
                temperature=0.3,
                system="You are an expert at analyzing e-commerce data, especially Amazon returns and reviews.",
                messages=[{"role": "user", "content": f"{prompt}\n\nContent:\n{text[:4000]}"}]
            )
            
            return {
                'success': True,
                'response': message.content[0].text,
                'provider': 'claude'
            }
            
        except Exception as e:
            logger.error(f"Claude API error: {e}")
            return {'success': False, 'error': str(e)}
    
    async def _call_openai(self, text: str, prompt: str) -> Dict[str, Any]:
        """Async OpenAI call for file analysis"""
        if not self.openai_configured:
            return {'success': False, 'error': 'OpenAI not configured'}
        
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.openai_key}"
        }
        
        payload = {
            "model": MODEL_CONFIG['openai']['primary'],
            "messages": [
                {
                    "role": "system",
                    "content": "You are an expert at analyzing e-commerce data, especially Amazon returns and reviews."
                },
                {
                    "role": "user",
                    "content": f"{prompt}\n\nContent:\n{text[:4000]}"
                }
            ],
            "temperature": 0.3,
            "max_tokens": 1000
        }
        
        try:
            response = requests.post(
                "https://api.openai.com/v1/chat/completions",
                headers=headers,
                json=payload,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                return {
                    'success': True,
                    'response': result['choices'][0]['message']['content'],
                    'provider': 'openai'
                }
            else:
                return {'success': False, 'error': f'API error {response.status_code}'}
                
        except Exception as e:
            logger.error(f"OpenAI API error: {e}")
            return {'success': False, 'error': str(e)}
    
    def _clean_category_response(self, response: str) -> str:
        """Clean AI response to extract category"""
        response = response.strip().strip('"').strip("'").strip()
        
        # Remove common prefixes
        prefixes = ['Category:', 'The category is:', 'Answer:']
        for prefix in prefixes:
            if response.startswith(prefix):
                response = response[len(prefix):].strip()
        
        # Try exact match first
        for valid_cat in MEDICAL_DEVICE_CATEGORIES:
            if response == valid_cat or response.lower() == valid_cat.lower():
                return valid_cat
        
        # Try partial match
        response_lower = response.lower()
        for valid_cat in MEDICAL_DEVICE_CATEGORIES:
            if valid_cat.lower() in response_lower:
                return valid_cat
        
        # Try keyword match
        for valid_cat in MEDICAL_DEVICE_CATEGORIES:
            cat_words = set(valid_cat.lower().split('/'))
            response_words = set(response_lower.split())
            if cat_words & response_words:
                return valid_cat
        
        return 'Other/Miscellaneous'
    
    async def analyze_file(self, file_content: bytes, filename: str, 
                          file_type: str = None) -> FileAnalysis:
        """Analyze any file type and extract relevant data"""
        
        # For PDF/Image files that need AI
        if file_type in ['pdf', 'image']:
            return FileAnalysis(
                file_type=file_type,
                content_type='pending_analysis',
                extracted_data={'message': 'PDF/Image analysis requires AI processing'},
                confidence=0.0,
                ai_provider='none',
                needs_clarification=True
            )
        
        # Basic text analysis
        try:
            text = file_content.decode('utf-8', errors='ignore')
            return FileAnalysis(
                file_type='text',
                content_type='data',
                extracted_data={'text': text[:1000]},
                confidence=0.5,
                ai_provider='none'
            )
        except:
            return FileAnalysis(
                file_type='unknown',
                content_type='error',
                extracted_data={'error': 'Unable to process file'},
                confidence=0.0,
                ai_provider='none'
            )
    
    def generate_return_analysis_report(self, asin: str, returns_data: List[Dict],
                                      reviews_data: List[Dict] = None) -> Dict[str, Any]:
        """Generate comprehensive return analysis report"""
        
        # Filter returns for specific ASIN
        if asin != 'ALL':
            asin_returns = [r for r in returns_data if r.get('asin') == asin]
        else:
            asin_returns = returns_data
        
        if not asin_returns:
            return {'error': f'No returns found for ASIN {asin}'}
        
        # Categorize returns if not already done
        categorized = {}
        for return_item in asin_returns:
            category = return_item.get('category', 'Other/Miscellaneous')
            if category not in categorized:
                categorized[category] = []
            categorized[category].append(return_item)
        
        # Generate report
        total_returns = len(asin_returns)
        
        report = {
            'executive_summary': {
                'asin': asin,
                'total_returns': total_returns,
                'main_issues': self._identify_main_issues(categorized),
                'trend': 'Stable'  # Would need date analysis
            },
            'category_breakdown': {
                category: {
                    'count': len(returns),
                    'percentage': len(returns) / total_returns * 100 if total_returns > 0 else 0,
                    'priority': self._get_priority(category)
                }
                for category, returns in categorized.items()
            },
            'quality_metrics': self._calculate_quality_metrics(categorized, total_returns),
            'action_items': self._generate_action_items(categorized)
        }
        
        return report
    
    def _identify_main_issues(self, categorized: Dict) -> List[str]:
        """Identify main issues from categorized returns"""
        issues = []
        
        # Sort by count
        sorted_cats = sorted(categorized.items(), key=lambda x: len(x[1]), reverse=True)
        
        # Get top 3 high-priority issues
        for category, returns in sorted_cats[:5]:
            if self._get_priority(category) in ['critical', 'high']:
                issues.append(f"{category} ({len(returns)} returns)")
                if len(issues) >= 3:
                    break
        
        return issues
    
    def _get_priority(self, category: str) -> str:
        """Get priority level for category"""
        critical = ['Product Defects/Quality', 'Medical/Health Concerns']
        high = ['Performance/Effectiveness', 'Missing Components', 'Wrong Product/Misunderstanding']
        
        if category in critical:
            return 'critical'
        elif category in high:
            return 'high'
        else:
            return 'medium'
    
    def _calculate_quality_metrics(self, categorized: Dict, total: int) -> Dict[str, Any]:
        """Calculate quality-specific metrics"""
        quality_categories = [
            'Product Defects/Quality',
            'Performance/Effectiveness',
            'Missing Components',
            'Design/Material Issues',
            'Medical/Health Concerns'
        ]
        
        quality_returns = sum(
            len(returns) for cat, returns in categorized.items()
            if cat in quality_categories
        )
        
        return {
            'quality_issue_rate': quality_returns / total * 100 if total > 0 else 0,
            'quality_return_count': quality_returns,
            'top_quality_issue': max(
                [(cat, len(returns)) for cat, returns in categorized.items() 
                 if cat in quality_categories],
                key=lambda x: x[1],
                default=('None', 0)
            )[0]
        }
    
    def _generate_action_items(self, categorized: Dict) -> List[Dict[str, str]]:
        """Generate action items based on categories"""
        actions = []
        
        # Quality defects
        if 'Product Defects/Quality' in categorized and len(categorized['Product Defects/Quality']) > 5:
            actions.append({
                'priority': 'IMMEDIATE',
                'action': 'Conduct quality audit with manufacturer',
                'reason': f"{len(categorized['Product Defects/Quality'])} quality-related returns"
            })
        
        # Size/fit issues
        if 'Size/Fit Issues' in categorized and len(categorized['Size/Fit Issues']) > 3:
            actions.append({
                'priority': 'HIGH',
                'action': 'Update product dimensions and sizing guide',
                'reason': 'Multiple size-related returns'
            })
        
        # Wrong product
        if 'Wrong Product/Misunderstanding' in categorized:
            actions.append({
                'priority': 'MEDIUM',
                'action': 'Review and clarify product listing',
                'reason': 'Customer confusion about product'
            })
        
        return actions[:5]  # Top 5 actions
    
    def get_cost_summary(self) -> Dict[str, Any]:
        """Get cost summary"""
        return self.cost_tracker.get_summary()
    
    def get_cost_summary(self) -> Dict[str, Any]:
        """Get cost summary"""
        return self.cost_tracker.get_summary()
    
    def generate_chat_response(self, user_message: str, context: Dict[str, Any] = None) -> str:
        """Generate contextual chat responses"""
        if context is None:
            context = {}
        
        providers = self.get_available_providers()
        
        if not providers:
            return "AI chat is not available. Please configure your OpenAI or Claude API key."
        
        try:
            # Build context-aware prompt
            system_prompt = """You are a helpful Amazon quality analysis assistant specializing in medical device returns and quality improvement.
            Provide clear, actionable advice based on the user's question and the analysis context.
            Focus on practical implementation steps for quality improvement."""
            
            # Add context
            if context.get('has_analysis'):
                system_prompt += "\nThe user has completed an analysis of their Amazon returns."
            
            if context.get('current_asin'):
                system_prompt += f"\nCurrently analyzing ASIN: {context['current_asin']}"
            
            if context.get('categories_found'):
                top_categories = sorted(context['categories_found'].items(), 
                                      key=lambda x: x[1], reverse=True)[:3]
                system_prompt += f"\nTop return categories: {', '.join([f'{cat[0]} ({cat[1]})' for cat in top_categories])}"
            
            # Get response
            if 'claude' in providers and self.claude_configured:
                response = self._call_claude_sync(
                    user_message,
                    system_prompt,
                    'chat'
                )
                if response:
                    return response
            
            if 'openai' in providers and self.openai_configured:
                response = self._call_openai_sync(
                    user_message,
                    system_prompt,
                    'chat'
                )
                if response:
                    return response
            
            return "I'm having trouble processing your request. Please try again."
            
        except Exception as e:
            logger.error(f"Chat error: {e}")
            return f"Error: {str(e)}"
    
    def __del__(self):
        """Cleanup resources"""
        if hasattr(self, 'executor'):
            self.executor.shutdown(wait=False)
        if hasattr(self, 'session') and self.session:
            self.session.close()

# Export all components
__all__ = [
    'UniversalAIAnalyzer',
    'FileAnalysis',
    'MEDICAL_DEVICE_CATEGORIES',
    'FBA_REASON_MAP',
    'AIProvider',
    'quick_categorize',
    'CostTracker'
]
