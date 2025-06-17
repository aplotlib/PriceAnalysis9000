"""
Enhanced AI Analysis Module - Universal File Understanding with Medical Device Focus
Version: 11.0 - Dual AI with Medical Device Return Categories and Injury Detection

Key Features:
- Medical device return categorization per Amazon standards
- Critical injury/safety detection
- Dual AI support (OpenAI + Claude) with speed optimization
- Universal file understanding (PDF, FBA returns, reviews)
- Pattern matching + AI for accuracy and speed
"""

import logging
import os
import json
import re
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple, Union
from collections import Counter, defaultdict
import time
import base64
from io import BytesIO
from dataclasses import dataclass
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

# API Configuration
API_TIMEOUT = 30
MAX_RETRIES = 2
BATCH_SIZE = 50  # Optimized for speed
MAX_WORKERS = 5

# Model configurations - optimized for speed and accuracy
MODEL_CONFIG = {
    'openai': {
        'fast': 'gpt-3.5-turbo',
        'accurate': 'gpt-4o-mini',
        'vision': 'gpt-4o-mini',
        'chat': 'gpt-3.5-turbo'
    },
    'claude': {
        'fast': 'claude-3-haiku-20240307',
        'accurate': 'claude-3-sonnet-20240229',
        'vision': 'claude-3-haiku-20240307',
        'chat': 'claude-3-haiku-20240307'
    }
}

# Medical Device Return Categories - MUST match Amazon standards exactly
MEDICAL_DEVICE_CATEGORIES = {
    'Size/Fit Issues': {
        'keywords': ['too large', 'too small', 'doesn\'t fit', 'wrong size', 'too tight', 
                    'too loose', 'too tall', 'too short', 'too wide', 'sizing issues', 
                    'bad fit', 'didn\'t fit', 'doesn\'t fit well'],
        'priority': 'medium',
        'patterns': [r'too (large|big|small|tight|loose|tall|short|wide)', r'doesn[\']?t fit', 
                    r'wrong size', r'siz(e|ing) issue']
    },
    'Comfort Issues': {
        'keywords': ['uncomfortable', 'hurts', 'hurts customer', 'too firm', 'too hard', 
                    'too stiff', 'too soft', 'not enough padding', 'not enough cushion'],
        'priority': 'high',
        'patterns': [r'uncomfort', r'hurt', r'too (firm|hard|stiff|soft)', r'pain']
    },
    'Product Defects/Quality': {
        'keywords': ['defective', 'does not work properly', 'poor quality', 'ripped', 'torn',
                    'bad velcro', 'velcro doesn\'t stick', 'defective handles', 'defective suction cups',
                    'won\'t inflate', 'inflation issues', 'not working', 'broken', 'damaged'],
        'priority': 'critical',
        'patterns': [r'defect', r'broken', r'ripped', r'torn', r'doesn[\']?t work', 
                    r'poor quality', r'damaged', r'malfunction']
    },
    'Performance/Effectiveness': {
        'keywords': ['ineffective', 'not as expected', 'does not meet expectations', 
                    'not enough support', 'poor support', 'not enough compression', 
                    'not cold enough', 'not hot enough', 'inaccurate'],
        'priority': 'high',
        'patterns': [r'ineffective', r'not (as )?expected', r'poor support', r'doesn[\']?t help']
    },
    'Stability/Positioning Issues': {
        'keywords': ['doesn\'t stay in place', 'doesn\'t stay fastened', 'slides around',
                    'slides off', 'slides up', 'slippery', 'unstable', 'flattens'],
        'priority': 'high',
        'patterns': [r'slides? (around|off|up)', r'doesn[\']?t stay', r'unstable', r'slippery']
    },
    'Equipment Compatibility': {
        'keywords': ['doesn\'t fit walker', 'doesn\'t fit bariatric walker', 'doesn\'t fit knee walker',
                    'doesn\'t fit wheelchair', 'doesn\'t fit toilet', 'doesn\'t fit shower',
                    'doesn\'t fit bed', 'doesn\'t fit machine', 'doesn\'t fit handle',
                    'doesn\'t fit finger', 'not compatible', 'does not work with compression stockings'],
        'priority': 'medium',
        'patterns': [r'doesn[\']?t fit (walker|wheelchair|toilet|shower|bed|machine)', 
                    r'not compatible', r'incompatible']
    },
    'Design/Material Issues': {
        'keywords': ['too bulky', 'too thick', 'too heavy', 'too thin', 'flimsy', 
                    'small pulley', 'grip too small', 'fingers too long', 'fingers too short'],
        'priority': 'medium',
        'patterns': [r'too (bulky|thick|heavy|thin)', r'flimsy', r'poor design']
    },
    'Wrong Product/Misunderstanding': {
        'keywords': ['wrong item', 'wrong color', 'not as advertised', 'different from website description',
                    'thought it was something else', 'thought it was scooter', 'thought it was crutches',
                    'thought pump was included', 'brace for wrong hand', 'immobilizer for wrong hand',
                    'style not as expected'],
        'priority': 'low',
        'patterns': [r'wrong (item|product|color)', r'not as advertised', r'thought it was']
    },
    'Missing Components': {
        'keywords': ['missing pieces', 'missing parts', 'missing accessories', 'no instructions',
                    'thought pump was included'],
        'priority': 'medium',
        'patterns': [r'missing (pieces?|parts?|accessories)', r'no instructions']
    },
    'Customer Error/Changed Mind': {
        'keywords': ['ordered wrong item', 'bought by mistake', 'changed mind', 'no longer needed',
                    'unauthorized purchase', 'no issue', 'customer error'],
        'priority': 'low',
        'patterns': [r'ordered wrong', r'bought by mistake', r'changed mind', r'no longer need']
    },
    'Shipping/Fulfillment Issues': {
        'keywords': ['arrived too late', 'received used item', 'received damaged item', 'item never arrived'],
        'priority': 'medium',
        'patterns': [r'arrived (too )?late', r'received (used|damaged)', r'never arrived']
    },
    'Assembly/Usage Difficulty': {
        'keywords': ['difficult to use', 'difficult to adjust', 'difficult to assemble',
                    'difficult to open valve', 'installation issues'],
        'priority': 'medium',
        'patterns': [r'difficult to (use|adjust|assemble)', r'hard to', r'complicated']
    },
    'Medical/Health Concerns': {
        'keywords': ['doctor did not approve', 'allergic reaction', 'bad smell', 'bad odor',
                    'injury', 'injured', 'hospital', 'emergency', 'pain', 'swelling'],
        'priority': 'critical',
        'patterns': [r'doctor', r'allerg', r'injury', r'injured', r'hospital', r'emergency']
    },
    'Price/Value': {
        'keywords': ['better price available', 'found better price', 'too expensive'],
        'priority': 'low',
        'patterns': [r'better price', r'too expensive', r'overpriced']
    },
    'Other/Miscellaneous': {
        'keywords': ['other', 'incompatible or not useful'],
        'priority': 'low',
        'patterns': []
    }
}

# Compile patterns for speed
COMPILED_PATTERNS = {}
for category, info in MEDICAL_DEVICE_CATEGORIES.items():
    COMPILED_PATTERNS[category] = [re.compile(pattern, re.IGNORECASE) 
                                  for pattern in info.get('patterns', [])]

# Critical keywords that require immediate attention
CRITICAL_KEYWORDS = {
    'injury': ['injury', 'injured', 'hurt myself', 'caused injury', 'got hurt', 'wound'],
    'medical_emergency': ['hospital', 'emergency', 'emergency room', 'ER', 'ambulance', 
                         'doctor visit', 'medical attention'],
    'safety': ['dangerous', 'unsafe', 'hazard', 'risk', 'accident', 'fall', 'fell'],
    'severe_pain': ['severe pain', 'extreme pain', 'unbearable', 'excruciating'],
    'allergic': ['allergic', 'reaction', 'rash', 'hives', 'swelling', 'breathing']
}

class AIProvider(Enum):
    OPENAI = "openai"
    CLAUDE = "claude"
    BOTH = "both"
    FASTEST = "fastest"
    ACCURATE = "accurate"

@dataclass
class FileAnalysis:
    """Results from file analysis"""
    file_type: str
    content_type: str  # 'returns', 'reviews', 'other'
    extracted_data: Dict[str, Any]
    confidence: float
    ai_provider: str
    critical_issues: List[Dict[str, Any]] = None
    needs_clarification: bool = False
    clarification_questions: List[str] = None

@dataclass
class ReturnCategorization:
    """Return categorization result"""
    category: str
    confidence: float
    severity: str  # 'critical', 'high', 'medium', 'low'
    critical_flags: List[str]  # injury, medical_emergency, safety
    ai_provider: str
    processing_time: float

class UniversalAIAnalyzer:
    """Universal AI analyzer with medical device focus"""
    
    def __init__(self, provider: AIProvider = AIProvider.FASTEST):
        self.provider = provider
        self.openai_key = self._get_api_key('openai')
        self.claude_key = self._get_api_key('claude')
        self.executor = ThreadPoolExecutor(max_workers=MAX_WORKERS)
        
        # Initialize Claude client if available
        self.claude_client = None
        if self.claude_key and has_anthropic:
            try:
                from anthropic import Anthropic
                self.claude_client = Anthropic(api_key=self.claude_key)
                logger.info("Claude API initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize Claude: {e}")
        
        # Track API usage and costs
        self.api_calls = {'openai': 0, 'claude': 0}
        self.total_cost = 0.0
        self.categorization_cache = {}  # Cache for repeated returns
        
        # Session for connection pooling
        self.session = None
        if has_requests:
            self.session = requests.Session()
    
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
        if self.openai_key:
            providers.append('openai')
        if self.claude_client:
            providers.append('claude')
        return providers
    
    def detect_critical_issues(self, text: str) -> List[str]:
        """Detect critical safety/injury issues in text"""
        critical_flags = []
        text_lower = text.lower()
        
        for flag_type, keywords in CRITICAL_KEYWORDS.items():
            if any(keyword in text_lower for keyword in keywords):
                critical_flags.append(flag_type)
        
        return critical_flags
    
    def quick_categorize(self, complaint: str, comment: str = "") -> Optional[Tuple[str, float]]:
        """Quick pattern-based categorization for speed"""
        if not complaint and not comment:
            return None
        
        full_text = f"{complaint} {comment}".lower()
        
        # Check each category's patterns
        best_match = None
        best_score = 0
        
        for category, patterns in COMPILED_PATTERNS.items():
            score = 0
            # Check patterns
            for pattern in patterns:
                if pattern.search(full_text):
                    score += 2
            
            # Check keywords
            keywords = MEDICAL_DEVICE_CATEGORIES[category].get('keywords', [])
            for keyword in keywords:
                if keyword.lower() in full_text:
                    score += 1
            
            if score > best_score:
                best_score = score
                best_match = category
        
        if best_match and best_score >= 2:
            confidence = min(0.9, 0.7 + (best_score * 0.05))
            return best_match, confidence
        
        return None
    
    async def categorize_return(self, reason: str, comment: str = "", 
                              order_id: str = "", sku: str = "") -> ReturnCategorization:
        """Categorize a return with medical device categories"""
        start_time = time.time()
        
        # Check cache first
        cache_key = f"{reason}|{comment}"
        if cache_key in self.categorization_cache:
            cached = self.categorization_cache[cache_key]
            return ReturnCategorization(
                category=cached['category'],
                confidence=cached['confidence'],
                severity=cached['severity'],
                critical_flags=self.detect_critical_issues(f"{reason} {comment}"),
                ai_provider='cache',
                processing_time=0.001
            )
        
        # Detect critical issues
        critical_flags = self.detect_critical_issues(f"{reason} {comment}")
        
        # Try quick categorization first
        quick_result = self.quick_categorize(reason, comment)
        if quick_result:
            category, confidence = quick_result
            severity = self._determine_severity(category, critical_flags)
            
            # Cache result
            self.categorization_cache[cache_key] = {
                'category': category,
                'confidence': confidence,
                'severity': severity
            }
            
            return ReturnCategorization(
                category=category,
                confidence=confidence,
                severity=severity,
                critical_flags=critical_flags,
                ai_provider='pattern',
                processing_time=time.time() - start_time
            )
        
        # Use AI for complex cases
        result = await self._ai_categorize(reason, comment)
        
        # Cache result
        self.categorization_cache[cache_key] = {
            'category': result.category,
            'confidence': result.confidence,
            'severity': result.severity
        }
        
        result.critical_flags = critical_flags
        result.processing_time = time.time() - start_time
        
        return result
    
    async def _ai_categorize(self, reason: str, comment: str) -> ReturnCategorization:
        """Use AI to categorize return"""
        # Build prompt
        categories_list = '\n'.join([f"- {cat}" for cat in MEDICAL_DEVICE_CATEGORIES.keys()])
        
        system_prompt = """You are a medical device quality expert. Categorize this return into exactly one category.
Focus on identifying quality defects, safety issues, and medical concerns.
Respond with ONLY the category name, nothing else."""
        
        user_prompt = f"""Return Reason: {reason}
Customer Comment: {comment}

Categories:
{categories_list}

Category:"""
        
        # Choose AI based on provider setting
        if self.provider == AIProvider.FASTEST:
            # Use Claude Haiku for speed
            if self.claude_client:
                result = await self._call_claude(user_prompt, system_prompt, 'fast')
                if result:
                    category = self._clean_category_response(result)
                    critical_flags = self.detect_critical_issues(f"{reason} {comment}")
                    severity = self._determine_severity(category, critical_flags)
                    return ReturnCategorization(
                        category=category,
                        confidence=0.85,
                        severity=severity,
                        critical_flags=[],
                        ai_provider='claude_haiku',
                        processing_time=0
                    )
            
            # Fallback to OpenAI
            if self.openai_key:
                result = await self._call_openai(user_prompt, system_prompt, 'fast')
                if result:
                    category = self._clean_category_response(result)
                    critical_flags = self.detect_critical_issues(f"{reason} {comment}")
                    severity = self._determine_severity(category, critical_flags)
                    return ReturnCategorization(
                        category=category,
                        confidence=0.85,
                        severity=severity,
                        critical_flags=[],
                        ai_provider='openai_fast',
                        processing_time=0
                    )
        
        elif self.provider == AIProvider.ACCURATE:
            # Use better models for accuracy
            tasks = []
            if self.openai_key:
                tasks.append(self._call_openai(user_prompt, system_prompt, 'accurate'))
            if self.claude_client:
                tasks.append(self._call_claude(user_prompt, system_prompt, 'accurate'))
            
            if tasks:
                results = await asyncio.gather(*tasks, return_exceptions=True)
                valid_results = [r for r in results if r and not isinstance(r, Exception)]
                
                if valid_results:
                    # Use consensus if multiple results
                    categories = [self._clean_category_response(r) for r in valid_results]
                    category = Counter(categories).most_common(1)[0][0]
                    confidence = 0.95 if len(set(categories)) == 1 else 0.85
                    
                    critical_flags = self.detect_critical_issues(f"{reason} {comment}")
                    severity = self._determine_severity(category, critical_flags)
                    
                    return ReturnCategorization(
                        category=category,
                        confidence=confidence,
                        severity=severity,
                        critical_flags=[],
                        ai_provider='consensus',
                        processing_time=0
                    )
        
        # Default fallback
        return ReturnCategorization(
            category='Other/Miscellaneous',
            confidence=0.3,
            severity='low',
            critical_flags=self.detect_critical_issues(f"{reason} {comment}"),
            ai_provider='fallback',
            processing_time=0
        )
    
    async def _call_openai(self, prompt: str, system_prompt: str, mode: str = 'fast') -> Optional[str]:
        """Call OpenAI API"""
        if not self.openai_key:
            return None
        
        model = MODEL_CONFIG['openai'][mode]
        
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
            "max_tokens": 100
        }
        
        try:
            response = (self.session or requests).post(
                "https://api.openai.com/v1/chat/completions",
                headers=headers,
                json=payload,
                timeout=API_TIMEOUT
            )
            
            if response.status_code == 200:
                self.api_calls['openai'] += 1
                return response.json()["choices"][0]["message"]["content"].strip()
            else:
                logger.error(f"OpenAI API error {response.status_code}")
                return None
                
        except Exception as e:
            logger.error(f"OpenAI call error: {e}")
            return None
    
    async def _call_claude(self, prompt: str, system_prompt: str, mode: str = 'fast') -> Optional[str]:
        """Call Claude API"""
        if not self.claude_client:
            return None
        
        try:
            message = self.claude_client.messages.create(
                model=MODEL_CONFIG['claude'][mode],
                max_tokens=100,
                temperature=0.1,
                system=system_prompt,
                messages=[{"role": "user", "content": prompt}]
            )
            
            self.api_calls['claude'] += 1
            return message.content[0].text.strip()
            
        except Exception as e:
            logger.error(f"Claude API error: {e}")
            return None
    
    def _clean_category_response(self, response: str) -> str:
        """Clean AI response to extract category"""
        response = response.strip().strip('"').strip("'").strip()
        
        # Remove common prefixes
        prefixes = ['Category:', 'The category is:', 'Answer:']
        for prefix in prefixes:
            if response.startswith(prefix):
                response = response[len(prefix):].strip()
        
        # Try exact match first
        for category in MEDICAL_DEVICE_CATEGORIES.keys():
            if response == category or response.lower() == category.lower():
                return category
        
        # Try partial match
        response_lower = response.lower()
        for category in MEDICAL_DEVICE_CATEGORIES.keys():
            if category.lower() in response_lower or response_lower in category.lower():
                return category
        
        return 'Other/Miscellaneous'
    
    def _determine_severity(self, category: str, critical_flags: List[str]) -> str:
        """Determine severity level"""
        # Critical flags override
        if critical_flags:
            return 'critical'
        
        # Category-based severity
        priority = MEDICAL_DEVICE_CATEGORIES.get(category, {}).get('priority', 'low')
        
        severity_map = {
            'critical': 'critical',
            'high': 'high',
            'medium': 'medium',
            'low': 'low'
        }
        
        return severity_map.get(priority, 'low')
    
    async def analyze_file(self, file_content: bytes, filename: str, 
                          file_type: str = None) -> FileAnalysis:
        """Analyze any file type and extract relevant data"""
        
        # Determine file type if not provided
        if not file_type:
            file_type = self._detect_file_type(filename)
        
        # Route to appropriate handler
        if file_type == 'pdf':
            return await self._analyze_pdf(file_content, filename)
        elif file_type in ['jpg', 'jpeg', 'png']:
            return await self._analyze_image(file_content, filename)
        elif file_type in ['csv', 'tsv', 'txt']:
            return await self._analyze_text_file(file_content, filename, file_type)
        elif file_type in ['xlsx', 'xls']:
            return await self._analyze_excel(file_content, filename)
        else:
            return await self._analyze_unknown(file_content, filename)
    
    def _detect_file_type(self, filename: str) -> str:
        """Detect file type from filename"""
        ext = filename.lower().split('.')[-1]
        return ext
    
    async def _analyze_pdf(self, content: bytes, filename: str) -> FileAnalysis:
        """Analyze PDF files - especially Amazon return reports"""
        if not has_pdfplumber:
            return FileAnalysis(
                file_type='pdf',
                content_type='error',
                extracted_data={'error': 'PDF processing not available'},
                confidence=0.0,
                ai_provider='none'
            )
        
        try:
            extracted_data = {
                'returns': [],
                'summary': {},
                'raw_text': '',
                'critical_issues': []
            }
            
            with pdfplumber.open(BytesIO(content)) as pdf:
                for page_num, page in enumerate(pdf.pages):
                    text = page.extract_text() or ""
                    extracted_data['raw_text'] += f"\n--- Page {page_num + 1} ---\n{text}"
                    
                    # Extract Amazon return data
                    returns = self._extract_amazon_returns_from_text(text)
                    extracted_data['returns'].extend(returns)
                    
                    # Check for critical issues
                    critical_flags = self.detect_critical_issues(text)
                    if critical_flags:
                        extracted_data['critical_issues'].append({
                            'page': page_num + 1,
                            'flags': critical_flags,
                            'context': text[:200]
                        })
            
            # Categorize all returns
            if extracted_data['returns']:
                categorized_returns = []
                for return_data in extracted_data['returns']:
                    categorization = await self.categorize_return(
                        return_data.get('return_reason', ''),
                        return_data.get('customer_comment', ''),
                        return_data.get('order_id', ''),
                        return_data.get('sku', '')
                    )
                    
                    return_data['category'] = categorization.category
                    return_data['severity'] = categorization.severity
                    return_data['critical_flags'] = categorization.critical_flags
                    return_data['confidence'] = categorization.confidence
                    
                    categorized_returns.append(return_data)
                
                extracted_data['returns'] = categorized_returns
                extracted_data['summary'] = self._generate_return_summary(categorized_returns)
            
            return FileAnalysis(
                file_type='pdf',
                content_type='returns',
                extracted_data=extracted_data,
                confidence=0.9 if extracted_data['returns'] else 0.5,
                ai_provider='hybrid',
                critical_issues=extracted_data.get('critical_issues', [])
            )
            
        except Exception as e:
            logger.error(f"PDF analysis error: {e}")
            return FileAnalysis(
                file_type='pdf',
                content_type='error',
                extracted_data={'error': str(e)},
                confidence=0.0,
                ai_provider='none'
            )
    
    def _extract_amazon_returns_from_text(self, text: str) -> List[Dict]:
        """Extract return information from Amazon PDF text"""
        returns = []
        
        # Pattern for order IDs
        order_pattern = r'\b\d{3}-\d{7}-\d{7}\b'
        
        # Split text into potential return blocks
        lines = text.split('\n')
        current_return = {}
        
        for i, line in enumerate(lines):
            # Check for order ID
            order_match = re.search(order_pattern, line)
            if order_match:
                if current_return:
                    returns.append(current_return)
                current_return = {'order_id': order_match.group()}
            
            # Extract ASIN (10 alphanumeric)
            asin_match = re.search(r'\b[A-Z0-9]{10}\b', line)
            if asin_match and current_return:
                current_return['asin'] = asin_match.group()
            
            # Look for return reason indicators
            if any(indicator in line.lower() for indicator in ['reason:', 'return reason', 'defect']):
                # Get the rest of the line and potentially next line
                reason_text = line.split(':', 1)[-1].strip()
                if i + 1 < len(lines) and not re.search(order_pattern, lines[i + 1]):
                    reason_text += ' ' + lines[i + 1].strip()
                if current_return:
                    current_return['return_reason'] = reason_text
            
            # Look for customer comments
            if any(indicator in line.lower() for indicator in ['comment:', 'customer comment', 'notes:']):
                comment_text = line.split(':', 1)[-1].strip()
                if i + 1 < len(lines) and not re.search(order_pattern, lines[i + 1]):
                    comment_text += ' ' + lines[i + 1].strip()
                if current_return:
                    current_return['customer_comment'] = comment_text
        
        if current_return:
            returns.append(current_return)
        
        return returns
    
    def _generate_return_summary(self, returns: List[Dict]) -> Dict[str, Any]:
        """Generate summary statistics from categorized returns"""
        if not returns:
            return {}
        
        summary = {
            'total_returns': len(returns),
            'by_category': Counter(r.get('category', 'Unknown') for r in returns),
            'by_severity': Counter(r.get('severity', 'unknown') for r in returns),
            'critical_returns': [r for r in returns if r.get('critical_flags')],
            'injury_related': [r for r in returns if 'injury' in r.get('critical_flags', [])],
            'quality_defects': [r for r in returns if r.get('category') == 'Product Defects/Quality'],
            'top_categories': []
        }
        
        # Calculate percentages
        total = summary['total_returns']
        summary['category_percentages'] = {
            cat: (count / total * 100) for cat, count in summary['by_category'].items()
        }
        
        # Top categories
        summary['top_categories'] = summary['by_category'].most_common(5)
        
        # Quality metrics
        quality_categories = ['Product Defects/Quality', 'Performance/Effectiveness', 
                            'Design/Material Issues', 'Assembly/Usage Difficulty']
        quality_returns = sum(summary['by_category'].get(cat, 0) for cat in quality_categories)
        summary['quality_issue_rate'] = (quality_returns / total * 100) if total > 0 else 0
        
        return summary
    
    async def analyze_fba_returns(self, df: pd.DataFrame) -> FileAnalysis:
        """Analyze FBA return report with medical device categories"""
        try:
            categorized_returns = []
            
            # Process each return
            for _, row in df.iterrows():
                return_data = {
                    'order_id': row.get('order-id', ''),
                    'asin': row.get('asin', ''),
                    'sku': row.get('sku', ''),
                    'return_date': row.get('return-date', ''),
                    'reason': row.get('reason', ''),
                    'customer_comments': row.get('customer-comments', ''),
                    'quantity': row.get('quantity', 1),
                    'product_name': row.get('product-name', '')
                }
                
                # Categorize the return
                categorization = await self.categorize_return(
                    return_data['reason'],
                    return_data['customer_comments'],
                    return_data['order_id'],
                    return_data['sku']
                )
                
                return_data['category'] = categorization.category
                return_data['severity'] = categorization.severity
                return_data['critical_flags'] = categorization.critical_flags
                return_data['confidence'] = categorization.confidence
                
                categorized_returns.append(return_data)
            
            # Generate summary
            summary = self._generate_return_summary(categorized_returns)
            
            # Identify critical issues
            critical_issues = [r for r in categorized_returns if r.get('critical_flags')]
            
            return FileAnalysis(
                file_type='fba_returns',
                content_type='returns',
                extracted_data={
                    'returns': categorized_returns,
                    'summary': summary,
                    'total_returns': len(categorized_returns),
                    'critical_count': len(critical_issues)
                },
                confidence=0.95,
                ai_provider='hybrid',
                critical_issues=critical_issues
            )
            
        except Exception as e:
            logger.error(f"FBA return analysis error: {e}")
            return FileAnalysis(
                file_type='fba_returns',
                content_type='error',
                extracted_data={'error': str(e)},
                confidence=0.0,
                ai_provider='none'
            )
    
    def generate_quality_report(self, analysis_results: List[FileAnalysis]) -> Dict[str, Any]:
        """Generate comprehensive quality report from multiple analyses"""
        report = {
            'executive_summary': {},
            'critical_issues': [],
            'category_analysis': defaultdict(list),
            'product_analysis': defaultdict(lambda: defaultdict(int)),
            'recommendations': [],
            'metrics': {}
        }
        
        # Aggregate all returns
        all_returns = []
        for analysis in analysis_results:
            if analysis.content_type == 'returns':
                returns = analysis.extracted_data.get('returns', [])
                all_returns.extend(returns)
                
                # Collect critical issues
                if analysis.critical_issues:
                    report['critical_issues'].extend(analysis.critical_issues)
        
        if not all_returns:
            return report
        
        # Executive summary
        total_returns = len(all_returns)
        critical_returns = [r for r in all_returns if r.get('critical_flags')]
        quality_defects = [r for r in all_returns if r.get('category') == 'Product Defects/Quality']
        
        report['executive_summary'] = {
            'total_returns': total_returns,
            'critical_returns': len(critical_returns),
            'quality_defects': len(quality_defects),
            'critical_rate': (len(critical_returns) / total_returns * 100) if total_returns > 0 else 0,
            'quality_defect_rate': (len(quality_defects) / total_returns * 100) if total_returns > 0 else 0
        }
        
        # Category analysis
        for return_data in all_returns:
            category = return_data.get('category', 'Unknown')
            report['category_analysis'][category].append(return_data)
        
        # Product analysis (by SKU/ASIN)
        for return_data in all_returns:
            sku = return_data.get('sku', 'Unknown')
            category = return_data.get('category', 'Unknown')
            report['product_analysis'][sku][category] += 1
            report['product_analysis'][sku]['total'] += 1
            
            if return_data.get('critical_flags'):
                report['product_analysis'][sku]['critical'] += 1
        
        # Generate recommendations
        report['recommendations'] = self._generate_recommendations(report)
        
        return report
    
    def _generate_recommendations(self, report: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate actionable recommendations based on analysis"""
        recommendations = []
        
        # Critical issues recommendation
        if report['critical_issues']:
            recommendations.append({
                'priority': 'IMMEDIATE',
                'category': 'Safety',
                'recommendation': 'Investigate all injury/safety reports immediately',
                'details': f"Found {len(report['critical_issues'])} critical safety issues requiring immediate attention",
                'action_items': [
                    'Contact affected customers',
                    'Document incidents for FDA reporting',
                    'Review product design for safety improvements'
                ]
            })
        
        # Quality defect recommendations
        quality_rate = report['executive_summary'].get('quality_defect_rate', 0)
        if quality_rate > 10:
            recommendations.append({
                'priority': 'HIGH',
                'category': 'Quality',
                'recommendation': 'Implement quality improvement program',
                'details': f"Quality defect rate of {quality_rate:.1f}% exceeds acceptable threshold",
                'action_items': [
                    'Audit manufacturing processes',
                    'Increase quality control inspections',
                    'Review supplier quality agreements'
                ]
            })
        
        # Category-specific recommendations
        for category, returns in report['category_analysis'].items():
            if len(returns) > 10:
                if category == 'Size/Fit Issues':
                    recommendations.append({
                        'priority': 'MEDIUM',
                        'category': 'Product Information',
                        'recommendation': 'Improve sizing information and guides',
                        'details': f"{len(returns)} returns due to sizing issues",
                        'action_items': [
                            'Create detailed sizing charts',
                            'Add measurement instructions to listings',
                            'Include size comparison images'
                        ]
                    })
                elif category == 'Assembly/Usage Difficulty':
                    recommendations.append({
                        'priority': 'MEDIUM',
                        'category': 'Documentation',
                        'recommendation': 'Enhance product instructions',
                        'details': f"{len(returns)} returns due to usage difficulties",
                        'action_items': [
                            'Create video tutorials',
                            'Redesign instruction manuals',
                            'Add QR codes for online help'
                        ]
                    })
        
        return recommendations
    
    async def process_batch_returns(self, returns: List[Dict], batch_size: int = 50) -> List[Dict]:
        """Process returns in batches for efficiency"""
        results = []
        
        for i in range(0, len(returns), batch_size):
            batch = returns[i:i + batch_size]
            
            # Process batch in parallel
            tasks = []
            for return_data in batch:
                task = self.categorize_return(
                    return_data.get('reason', ''),
                    return_data.get('customer_comments', ''),
                    return_data.get('order_id', ''),
                    return_data.get('sku', '')
                )
                tasks.append(task)
            
            # Wait for all categorizations
            categorizations = await asyncio.gather(*tasks)
            
            # Combine results
            for return_data, categorization in zip(batch, categorizations):
                return_data['category'] = categorization.category
                return_data['severity'] = categorization.severity
                return_data['critical_flags'] = categorization.critical_flags
                return_data['confidence'] = categorization.confidence
                results.append(return_data)
        
        return results
    
    def get_api_usage_summary(self) -> Dict[str, Any]:
        """Get API usage and cost summary"""
        return {
            'api_calls': self.api_calls,
            'total_cost': self.total_cost,
            'cache_size': len(self.categorization_cache),
            'available_providers': self.get_available_providers()
        }
    
    async def generate_chat_response(self, user_message: str, context: Dict[str, Any] = None) -> str:
        """Generate contextual chat responses for quality analysis"""
        if context is None:
            context = {}
        
        providers = self.get_available_providers()
        if not providers:
            return "AI chat is not available. Please configure your OpenAI or Claude API key."
        
        # Build quality-focused system prompt
        system_prompt = """You are an expert medical device quality analyst and regulatory compliance specialist.
        Help users understand return patterns, identify quality issues, and provide actionable recommendations.
        Focus on safety, FDA compliance, and quality improvement strategies."""
        
        # Add context
        if context.get('current_analysis'):
            analysis = context['current_analysis']
            system_prompt += f"\n\nCurrent analysis shows {analysis.get('total_returns', 0)} returns with {analysis.get('critical_returns', 0)} critical issues."
        
        # Use appropriate AI
        if 'claude' in providers:
            result = await self._call_claude(user_message, system_prompt, 'chat')
            if result:
                return result
        
        if 'openai' in providers:
            result = await self._call_openai(user_message, system_prompt, 'chat')
            if result:
                return result
        
        return "Unable to process your request. Please try again."

# Export all components
__all__ = [
    'UniversalAIAnalyzer',
    'FileAnalysis',
    'ReturnCategorization',
    'AIProvider',
    'MEDICAL_DEVICE_CATEGORIES',
    'CRITICAL_KEYWORDS'
]
