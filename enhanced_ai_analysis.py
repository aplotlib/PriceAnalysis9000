"""
Enhanced AI Analysis Module - Dual AI with Advanced Injury Detection
Version 15.0 - Medical Device Safety Focus

Key Features:
- Advanced injury detection and severity classification
- FDA MDR compliance checks
- Automatic flagging of reportable events
- Detailed injury case extraction
"""

import logging
import os
import json
import re
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple, Union
from enum import Enum
import time
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Safe imports
def safe_import(module_name):
    try:
        return __import__(module_name), True
    except ImportError:
        logger.warning(f"Module {module_name} not available")
        return None, False

# Check for dependencies
requests, has_requests = safe_import('requests')

# API Configuration
API_TIMEOUT = 30
MAX_RETRIES = 2
BATCH_SIZE = 20
MAX_WORKERS = 5

# Token configurations by mode
TOKEN_LIMITS = {
    'standard': 100,
    'enhanced': 200,     
    'extreme': 400,      
    'chat': 500,
    'injury_analysis': 300  # Special mode for injury detection
}

# Model configurations
MODELS = {
    'openai': {
        'standard': 'gpt-3.5-turbo',
        'enhanced': 'gpt-3.5-turbo',
        'extreme': 'gpt-4',
        'chat': 'gpt-3.5-turbo',
        'injury_analysis': 'gpt-4'  # Use GPT-4 for injury analysis
    },
    'claude': {
        'standard': 'claude-3-haiku-20240307',
        'enhanced': 'claude-3-haiku-20240307',
        'extreme': 'claude-3-sonnet-20240229',
        'chat': 'claude-3-haiku-20240307',
        'injury_analysis': 'claude-3-sonnet-20240229'  # Use Sonnet for injury analysis
    }
}

# Updated pricing per 1K tokens
PRICING = {
    # OpenAI
    'gpt-3.5-turbo': {'input': 0.0005, 'output': 0.0015},
    'gpt-4': {'input': 0.03, 'output': 0.06},
    'gpt-4-turbo': {'input': 0.01, 'output': 0.03},
    # Claude (Anthropic)
    'claude-3-haiku-20240307': {'input': 0.00025, 'output': 0.00125},
    'claude-3-sonnet-20240229': {'input': 0.003, 'output': 0.015},
    'claude-3-opus-20240229': {'input': 0.015, 'output': 0.075}
}

# Medical Device Return Categories with injury risk indicators
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
    'Medical/Health Concerns',  # HIGH INJURY RISK
    'Price/Value',
    'Other/Miscellaneous'
]

# Injury risk categories
INJURY_RISK_CATEGORIES = {
    'Medical/Health Concerns': 'CRITICAL',
    'Product Defects/Quality': 'HIGH',
    'Stability/Positioning Issues': 'HIGH',
    'Performance/Effectiveness': 'MEDIUM',
    'Design/Material Issues': 'MEDIUM',
    'Assembly/Usage Difficulty': 'MEDIUM',
    'Comfort Issues': 'LOW'
}

# Enhanced injury keywords for medical devices
INJURY_KEYWORDS = {
    'critical': [
        'hospital', 'emergency', 'emergency room', 'er visit', 'urgent care',
        'ambulance', 'died', 'death', 'fatal', 'life threatening',
        'severe injury', 'serious injury', 'surgery', 'operation',
        'permanent damage', 'disability', 'paralyzed', 'unconscious',
        'bleeding profusely', 'severe bleeding', 'hemorrhage',
        'anaphylactic', 'seizure', 'cardiac', 'heart attack',
        'stroke', 'respiratory failure', 'suffocation', 'choking'
    ],
    'high': [
        'injured', 'hurt badly', 'hurt seriously', 'broken bone', 'fracture',
        'bleeding', 'blood', 'wound', 'laceration', 'cut deep', 'stitches',
        'concussion', 'head injury', 'knocked out', 'passed out', 'fainted',
        'burn', 'burned', 'severe pain', 'excruciating', 'unbearable pain',
        'infection', 'infected', 'swollen badly', 'allergic reaction',
        'can\'t walk', 'can\'t move', 'immobilized', 'nerve damage',
        'hospitalized', 'medical attention', 'doctor visit', 'emergency visit',
        'fell down', 'collapsed', 'dropped me', 'gave way'
    ],
    'medium': [
        'hurt', 'pain', 'painful', 'ache', 'sore', 'bruise', 'bruised',
        'swelling', 'swollen', 'inflammation', 'rash', 'irritation',
        'cut', 'scrape', 'scratch', 'minor bleeding', 'discomfort',
        'sprain', 'strain', 'pulled muscle', 'dizzy', 'nausea',
        'fell', 'fall', 'dropped', 'slipped', 'tripped', 'stumbled',
        'pinched', 'squeezed', 'pressure', 'numbness', 'tingling',
        'doctor recommended against', 'unsafe', 'dangerous'
    ]
}

# FBA reason code mapping - Enhanced with injury indicators
FBA_REASON_MAP = {
    # Original mappings
    'NOT_COMPATIBLE': 'Equipment Compatibility',
    'DAMAGED_BY_FC': 'Product Defects/Quality',
    'DAMAGED_BY_CARRIER': 'Shipping/Fulfillment Issues',
    'DEFECTIVE': 'Product Defects/Quality',
    'NOT_AS_DESCRIBED': 'Wrong Product/Misunderstanding',
    'WRONG_ITEM': 'Wrong Product/Misunderstanding',
    'MISSING_PARTS': 'Missing Components',
    'QUALITY_NOT_ADEQUATE': 'Product Defects/Quality',
    'UNWANTED_ITEM': 'Customer Error/Changed Mind',
    'UNAUTHORIZED_PURCHASE': 'Customer Error/Changed Mind',
    'CUSTOMER_DAMAGED': 'Customer Error/Changed Mind',
    'SWITCHEROO': 'Wrong Product/Misunderstanding',
    'EXPIRED_ITEM': 'Product Defects/Quality',
    'DAMAGED_GLASS_VIAL': 'Product Defects/Quality',
    'DIFFERENT_PRODUCT': 'Wrong Product/Misunderstanding',
    'MISSING_ITEM': 'Missing Components',
    'NOT_DELIVERED': 'Shipping/Fulfillment Issues',
    'ORDERED_WRONG_ITEM': 'Customer Error/Changed Mind',
    'UNNEEDED_ITEM': 'Customer Error/Changed Mind',
    'BAD_GIFT': 'Customer Error/Changed Mind',
    'INACCURATE_WEBSITE_DESCRIPTION': 'Wrong Product/Misunderstanding',
    'BETTER_PRICE_AVAILABLE': 'Price/Value',
    'DOES_NOT_FIT': 'Size/Fit Issues',
    'NOT_COMPATIBLE_WITH_DEVICE': 'Equipment Compatibility',
    'UNSATISFACTORY_PRODUCT': 'Performance/Effectiveness',
    'ARRIVED_LATE': 'Shipping/Fulfillment Issues',
    # Additional mappings
    'TOO_SMALL': 'Size/Fit Issues',
    'TOO_LARGE': 'Size/Fit Issues',
    'UNCOMFORTABLE': 'Comfort Issues',
    'DIFFICULT_TO_USE': 'Assembly/Usage Difficulty',
    'DAMAGED': 'Product Defects/Quality',
    'BROKEN': 'Product Defects/Quality',
    'POOR_QUALITY': 'Product Defects/Quality',
    'NOT_WORKING': 'Product Defects/Quality',
    'DOESNT_WORK': 'Product Defects/Quality',
    # Injury-related mappings
    'CAUSED_INJURY': 'Medical/Health Concerns',
    'SAFETY_ISSUE': 'Medical/Health Concerns',
    'HEALTH_CONCERN': 'Medical/Health Concerns'
}

# Quick categorization patterns - Enhanced with injury detection
QUICK_PATTERNS = {
    'Medical/Health Concerns': [
        r'injur', r'hurt', r'pain', r'hospital', r'emergency', r'doctor',
        r'medical', r'health', r'safety', r'dangerous', r'unsafe',
        r'allergic', r'reaction', r'bleeding', r'blood', r'wound',
        r'burn', r'infection', r'fever', r'sick', r'ill'
    ],
    'Size/Fit Issues': [
        r'too (small|large|big|tight|loose)', r'doesn[\']?t fit', r'wrong size',
        r'size issue', r'(small|large)r than expected', r'fit issue', r'too (tall|short|wide)',
        r'sizing issues', r'bad fit', r'didn[\']?t fit', r'doesn[\']?t fit well'
    ],
    'Product Defects/Quality': [
        r'defect', r'broken', r'damaged', r'poor quality', r'didn[\']?t work',
        r'stopped working', r'malfunction', r'fell apart', r'ripped', r'torn',
        r'does not work properly', r'bad velcro', r'velcro doesn[\']?t stick',
        r'defective handles', r'defective suction cups', r'won[\']?t inflate',
        r'inflation issues', r'not working', r'broke while using', r'collapsed'
    ],
    # ... (rest of patterns remain the same)
}

# Compile patterns for speed
COMPILED_PATTERNS = {
    category: [re.compile(pattern, re.IGNORECASE) for pattern in patterns]
    for category, patterns in QUICK_PATTERNS.items()
}

# Injury detection patterns
INJURY_PATTERNS = {
    severity: [re.compile(r'\b' + keyword + r'\b', re.IGNORECASE) for keyword in keywords]
    for severity, keywords in INJURY_KEYWORDS.items()
}

class AIProvider(Enum):
    OPENAI = "openai"
    CLAUDE = "claude"
    BOTH = "both"
    FASTEST = "fastest"

@dataclass
class InjuryCase:
    """Detailed injury case information"""
    order_id: str
    asin: str
    sku: str
    return_date: str
    severity: str  # critical, high, medium
    injury_keywords: List[str]
    full_comment: str
    category: str
    confidence: float
    reportable: bool  # FDA MDR requirement
    device_type: str
    potential_causes: List[str]
    recommendation: str

@dataclass
class CostEstimate:
    """Cost estimation data class"""
    provider: str
    model: str
    input_tokens: int
    output_tokens: int
    input_cost: float
    output_cost: float
    total_cost: float
    
    def to_dict(self):
        return {
            'provider': self.provider,
            'model': self.model,
            'input_tokens': self.input_tokens,
            'output_tokens': self.output_tokens,
            'input_cost': self.input_cost,
            'output_cost': self.output_cost,
            'total_cost': self.total_cost
        }

def estimate_tokens(text: str) -> int:
    """Estimate token count (rough approximation)"""
    return max(len(text) // 4, len(text.split()) * 4 // 3)

def calculate_cost(model: str, input_tokens: int, output_tokens: int) -> CostEstimate:
    """Calculate cost for API usage"""
    if model not in PRICING:
        logger.warning(f"Model {model} not in pricing table")
        return CostEstimate("unknown", model, input_tokens, output_tokens, 0, 0, 0)
    
    pricing = PRICING[model]
    input_cost = (input_tokens / 1000) * pricing['input']
    output_cost = (output_tokens / 1000) * pricing['output']
    total_cost = input_cost + output_cost
    
    provider = 'openai' if 'gpt' in model else 'claude'
    
    return CostEstimate(
        provider=provider,
        model=model,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        input_cost=input_cost,
        output_cost=output_cost,
        total_cost=total_cost
    )

def detect_injury_severity(complaint: str) -> Tuple[str, List[str]]:
    """Detect injury severity and keywords"""
    if not complaint:
        return 'none', []
    
    complaint_lower = complaint.lower()
    found_keywords = []
    
    # Check each severity level
    for severity in ['critical', 'high', 'medium']:
        for pattern in INJURY_PATTERNS[severity]:
            matches = pattern.findall(complaint_lower)
            if matches:
                found_keywords.extend(matches)
        
        if found_keywords:
            return severity, list(set(found_keywords))
    
    return 'none', []

def quick_categorize(complaint: str, fba_reason: str = None, return_reason: str = None) -> Optional[str]:
    """Quick pattern-based categorization with injury priority"""
    if not complaint:
        complaint = ""
    
    # Check for injury indicators first
    injury_severity, injury_keywords = detect_injury_severity(complaint)
    if injury_severity in ['critical', 'high']:
        return 'Medical/Health Concerns'
    
    # Check FBA reason
    if fba_reason and fba_reason in FBA_REASON_MAP:
        return FBA_REASON_MAP[fba_reason]
    
    # Check Amazon return reason
    if return_reason:
        return_reason_lower = return_reason.lower().strip()
        # Add injury-specific checks
        if any(word in return_reason_lower for word in ['injury', 'hurt', 'medical', 'safety']):
            return 'Medical/Health Concerns'
    
    # Combined text for analysis
    combined_text = f"{return_reason or ''} {complaint}".lower().strip()
    
    # Check compiled patterns (Medical/Health Concerns first)
    pattern_order = ['Medical/Health Concerns'] + [cat for cat in COMPILED_PATTERNS.keys() if cat != 'Medical/Health Concerns']
    
    for category in pattern_order:
        if category in COMPILED_PATTERNS:
            for pattern in COMPILED_PATTERNS[category]:
                if pattern.search(combined_text):
                    return category
    
    return None

class CostTracker:
    """Track API costs across sessions"""
    
    def __init__(self):
        self.session_costs = []
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_cost = 0.0
        self.api_calls = 0
        self.start_time = datetime.now()
        self.quick_categorizations = 0
        self.ai_categorizations = 0
        self.injury_cases_detected = 0
    
    def add_cost(self, cost_estimate: CostEstimate):
        """Add cost to tracking"""
        self.session_costs.append(cost_estimate)
        self.total_input_tokens += cost_estimate.input_tokens
        self.total_output_tokens += cost_estimate.output_tokens
        self.total_cost += cost_estimate.total_cost
        self.api_calls += 1
    
    def add_quick_categorization(self):
        """Track quick categorization"""
        self.quick_categorizations += 1
    
    def add_ai_categorization(self):
        """Track AI categorization"""
        self.ai_categorizations += 1
    
    def add_injury_case(self):
        """Track injury case detection"""
        self.injury_cases_detected += 1
    
    def get_summary(self) -> Dict[str, Any]:
        """Get cost summary"""
        duration = (datetime.now() - self.start_time).total_seconds() / 60
        
        return {
            'total_cost': round(self.total_cost, 4),
            'api_calls': self.api_calls,
            'total_tokens': self.total_input_tokens + self.total_output_tokens,
            'quick_categorizations': self.quick_categorizations,
            'ai_categorizations': self.ai_categorizations,
            'injury_cases_detected': self.injury_cases_detected,
            'speed_improvement': f"{self.quick_categorizations / max(1, self.quick_categorizations + self.ai_categorizations) * 100:.1f}%",
            'average_cost_per_call': round(self.total_cost / max(1, self.api_calls), 4),
            'duration_minutes': round(duration, 1),
            'breakdown_by_provider': self._get_provider_breakdown()
        }
    
    def _get_provider_breakdown(self) -> Dict[str, Dict]:
        """Get cost breakdown by provider"""
        breakdown = {'openai': {'calls': 0, 'cost': 0}, 'claude': {'calls': 0, 'cost': 0}}
        
        for cost in self.session_costs:
            provider = cost.provider
            if provider in breakdown:
                breakdown[provider]['calls'] += 1
                breakdown[provider]['cost'] += cost.total_cost
        
        return breakdown

class EnhancedAIAnalyzer:
    """Main AI analyzer with injury detection focus"""
    
    def __init__(self, provider: AIProvider = AIProvider.FASTEST):
        self.provider = provider
        self.openai_key = self._get_api_key('openai')
        self.claude_key = self._get_api_key('claude')
        
        # Initialize tracking
        self.cost_tracker = CostTracker()
        self.injury_cases = []  # Store all detected injury cases
        
        # Initialize API availability
        self.openai_configured = bool(self.openai_key and has_requests)
        self.claude_configured = bool(self.claude_key and has_requests)
        
        # Thread pool for parallel processing
        self.executor = ThreadPoolExecutor(max_workers=MAX_WORKERS)
        
        # Session for connection pooling
        self.session = None
        if has_requests:
            self.session = requests.Session()
        
        logger.info(f"AI Analyzer initialized - OpenAI: {self.openai_configured}, Claude: {self.claude_configured}, Mode: {provider.value}")
    
    def _get_api_key(self, provider: str) -> Optional[str]:
        """Get API key from multiple sources"""
        # Try Streamlit secrets first
        try:
            import streamlit as st
            if hasattr(st, 'secrets'):
                if provider == 'openai':
                    for key_name in ["OPENAI_API_KEY", "openai_api_key", "openai"]:
                        if key_name in st.secrets:
                            key_value = str(st.secrets[key_name]).strip()
                            if key_value and key_value.startswith('sk-'):
                                logger.info(f"Found {provider} key in Streamlit secrets")
                                return key_value
                elif provider == 'claude':
                    for key_name in ["ANTHROPIC_API_KEY", "anthropic_api_key", "claude_api_key", "claude"]:
                        if key_name in st.secrets:
                            key_value = str(st.secrets[key_name]).strip()
                            if key_value:
                                logger.info(f"Found {provider} key in Streamlit secrets")
                                return key_value
        except Exception as e:
            logger.debug(f"Streamlit secrets not available: {e}")
        
        # Try environment variables
        env_vars = {
            'openai': ["OPENAI_API_KEY", "OPENAI_API"],
            'claude': ["ANTHROPIC_API_KEY", "CLAUDE_API_KEY"]
        }
        
        for env_name in env_vars.get(provider, []):
            api_key = os.environ.get(env_name, '').strip()
            if api_key:
                logger.info(f"Found {provider} key in environment")
                return api_key
        
        logger.warning(f"No {provider} API key found")
        return None
    
    def get_api_status(self) -> Dict[str, Any]:
        """Get API status with injury tracking"""
        status = {
            'available': self.openai_configured or self.claude_configured,
            'openai_configured': self.openai_configured,
            'claude_configured': self.claude_configured,
            'dual_ai_available': self.openai_configured and self.claude_configured,
            'provider': self.provider.value,
            'cost_summary': self.cost_tracker.get_summary(),
            'injury_cases_found': len(self.injury_cases),
            'message': ''
        }
        
        if status['dual_ai_available']:
            status['message'] = 'Both OpenAI and Claude APIs are configured'
        elif self.openai_configured:
            status['message'] = 'OpenAI API is configured (Claude not available)'
        elif self.claude_configured:
            status['message'] = 'Claude API is configured (OpenAI not available)'
        else:
            status['message'] = 'No APIs configured'
        
        return status
    
    def _call_openai(self, prompt: str, system_prompt: str, mode: str = 'standard') -> Tuple[Optional[str], Optional[CostEstimate]]:
        """Call OpenAI API with cost tracking"""
        if not self.openai_configured:
            return None, None
        
        model = MODELS['openai'][mode]
        max_tokens = TOKEN_LIMITS[mode]
        
        # Estimate input tokens
        input_tokens = estimate_tokens(system_prompt + prompt)
        
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
        
        for attempt in range(MAX_RETRIES):
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
                    
                    # Get actual token usage
                    usage = result.get("usage", {})
                    actual_input = usage.get("prompt_tokens", input_tokens)
                    actual_output = usage.get("completion_tokens", len(content.split()))
                    
                    # Calculate cost
                    cost = calculate_cost(model, actual_input, actual_output)
                    self.cost_tracker.add_cost(cost)
                    
                    return content, cost
                
                elif response.status_code == 429:
                    wait_time = min(2 ** attempt, 10)
                    logger.warning(f"OpenAI rate limited, waiting {wait_time}s")
                    time.sleep(wait_time)
                    continue
                
                else:
                    logger.error(f"OpenAI API error {response.status_code}")
                    return None, None
                    
            except Exception as e:
                logger.error(f"OpenAI call error: {e}")
                if attempt == MAX_RETRIES - 1:
                    return None, None
                time.sleep(1)
        
        return None, None
    
    def _call_claude(self, prompt: str, system_prompt: str, mode: str = 'standard') -> Tuple[Optional[str], Optional[CostEstimate]]:
        """Call Claude API with cost tracking"""
        if not self.claude_configured:
            return None, None
        
        model = MODELS['claude'][mode]
        max_tokens = TOKEN_LIMITS[mode]
        
        # Estimate input tokens
        input_tokens = estimate_tokens(system_prompt + prompt)
        
        headers = {
            "Content-Type": "application/json",
            "x-api-key": self.claude_key,
            "anthropic-version": "2023-06-01"
        }
        
        payload = {
            "model": model,
            "max_tokens": max_tokens,
            "temperature": 0.1,
            "system": system_prompt,
            "messages": [{"role": "user", "content": prompt}]
        }
        
        for attempt in range(MAX_RETRIES):
            try:
                response = (self.session or requests).post(
                    "https://api.anthropic.com/v1/messages",
                    headers=headers,
                    json=payload,
                    timeout=API_TIMEOUT
                )
                
                if response.status_code == 200:
                    result = response.json()
                    content = result["content"][0]["text"].strip()
                    
                    # Get actual token usage
                    usage = result.get("usage", {})
                    actual_input = usage.get("input_tokens", input_tokens)
                    actual_output = usage.get("output_tokens", len(content.split()))
                    
                    # Calculate cost
                    cost = calculate_cost(model, actual_input, actual_output)
                    self.cost_tracker.add_cost(cost)
                    
                    return content, cost
                
                elif response.status_code == 429:
                    wait_time = min(2 ** attempt, 10)
                    logger.warning(f"Claude rate limited, waiting {wait_time}s")
                    time.sleep(wait_time)
                    continue
                
                else:
                    logger.error(f"Claude API error {response.status_code}: {response.text}")
                    return None, None
                    
            except Exception as e:
                logger.error(f"Claude call error: {e}")
                if attempt == MAX_RETRIES - 1:
                    return None, None
                time.sleep(1)
        
        return None, None
    
    def detect_injury_case(self, return_data: Dict[str, Any]) -> Optional[InjuryCase]:
        """Detect if return involves injury and create detailed case"""
        complaint = return_data.get('customer_comment', '') or return_data.get('buyer_comment', '')
        if not complaint:
            return None
        
        # Detect injury severity
        severity, keywords = detect_injury_severity(complaint)
        if severity == 'none':
            return None
        
        # Determine if reportable (FDA MDR)
        reportable = severity in ['critical', 'high'] or any(
            keyword in ['death', 'permanent', 'surgery', 'hospitalization'] 
            for keyword in keywords
        )
        
        # Analyze device type and potential causes
        device_type = self._identify_device_type(return_data)
        potential_causes = self._analyze_potential_causes(complaint, device_type)
        
        # Generate recommendation
        recommendation = self._generate_safety_recommendation(severity, device_type, potential_causes)
        
        # Create injury case
        injury_case = InjuryCase(
            order_id=return_data.get('order_id', 'Unknown'),
            asin=return_data.get('asin', 'Unknown'),
            sku=return_data.get('sku', 'Unknown'),
            return_date=return_data.get('return_date', 'Unknown'),
            severity=severity,
            injury_keywords=keywords,
            full_comment=complaint,
            category='Medical/Health Concerns',
            confidence=0.95 if severity == 'critical' else 0.85,
            reportable=reportable,
            device_type=device_type,
            potential_causes=potential_causes,
            recommendation=recommendation
        )
        
        # Track injury case
        self.injury_cases.append(injury_case)
        self.cost_tracker.add_injury_case()
        
        return injury_case
    
    def _identify_device_type(self, return_data: Dict[str, Any]) -> str:
        """Identify medical device type from SKU/product info"""
        sku = str(return_data.get('sku', '')).upper()
        product_name = str(return_data.get('product_name', '')).lower()
        
        # Device type patterns
        device_patterns = {
            'mobility': ['MOB', 'WALK', 'CANE', 'CRUTCH', 'WHEEL', 'SCOOT'],
            'support': ['SUP', 'BRACE', 'SLING', 'COMPRESS', 'IMMOBIL'],
            'bathroom': ['BATH', 'TOIL', 'SHOWER', 'COMMODE', 'RAIL'],
            'respiratory': ['CPAP', 'OXYGEN', 'NEBUL', 'BREATH'],
            'monitoring': ['MONITOR', 'SENSOR', 'ALARM', 'GLUCOSE', 'PRESSURE']
        }
        
        for device_type, patterns in device_patterns.items():
            if any(pattern in sku for pattern in patterns):
                return device_type
            if any(pattern.lower() in product_name for pattern in patterns):
                return device_type
        
        return 'medical_device'
    
    def _analyze_potential_causes(self, complaint: str, device_type: str) -> List[str]:
        """Analyze potential causes of injury"""
        causes = []
        complaint_lower = complaint.lower()
        
        # Common medical device failure modes
        failure_patterns = {
            'structural_failure': ['broke', 'collapsed', 'gave way', 'snapped', 'cracked'],
            'instability': ['tipped', 'unstable', 'wobbled', 'fell over', 'slipped'],
            'sharp_edges': ['cut', 'sharp', 'rough edge', 'scraped', 'laceration'],
            'material_issue': ['allergic', 'rash', 'irritation', 'reaction', 'burn'],
            'design_flaw': ['pinched', 'trapped', 'caught', 'squeezed'],
            'instruction_issue': ['confusing', 'unclear', 'no instructions', 'didn\'t know']
        }
        
        for cause, patterns in failure_patterns.items():
            if any(pattern in complaint_lower for pattern in patterns):
                causes.append(cause)
        
        if not causes:
            causes.append('unknown_cause')
        
        return causes
    
    def _generate_safety_recommendation(self, severity: str, device_type: str, causes: List[str]) -> str:
        """Generate safety recommendation based on injury"""
        if severity == 'critical':
            return "IMMEDIATE ACTION: Investigate immediately. Consider product recall. Report to FDA within 5 days if death/serious injury."
        elif severity == 'high':
            return "URGENT: Conduct root cause analysis within 48 hours. Evaluate need for safety alert. Monitor for similar cases."
        else:
            return "Monitor for trends. Review design/instructions if pattern emerges. Document for quality system."
    
    def categorize_return(self, complaint: str, fba_reason: str = None, return_reason: str = None, 
                         mode: str = 'standard', return_data: Dict[str, Any] = None) -> Tuple[str, float, str, str]:
        """Categorize return with injury detection priority"""
        
        # First check for injury case
        if return_data:
            injury_case = self.detect_injury_case(return_data)
            if injury_case:
                return 'Medical/Health Concerns', injury_case.confidence, injury_case.severity, 'en'
        
        # Try quick categorization
        quick_category = quick_categorize(complaint, fba_reason, return_reason)
        if quick_category:
            self.cost_tracker.add_quick_categorization()
            severity = 'none'
            if quick_category == 'Medical/Health Concerns':
                severity, _ = detect_injury_severity(complaint)
            return quick_category, 0.9, severity, 'en'
        
        # AI categorization
        self.cost_tracker.add_ai_categorization()
        
        # Build prompts with injury focus
        system_prompt = """You are a medical device safety expert categorizing returns. 
CRITICAL: Identify ANY potential injury or safety concerns immediately.

Your task:
1. First check for ANY injury, safety, or health concerns
2. If injury/safety issue found, ALWAYS categorize as "Medical/Health Concerns"
3. Otherwise, categorize into the most appropriate category

IMPORTANT: Be vigilant for injuries - even minor ones. Words like hurt, pain, bleeding, 
fell, hospital, doctor, unsafe, dangerous should trigger "Medical/Health Concerns"."""
        
        categories_list = '\n'.join(f'- {cat}' for cat in MEDICAL_DEVICE_CATEGORIES)
        
        user_prompt = f"""Return Reason: "{return_reason or 'Not specified'}"
Customer Comment: "{complaint}"

Available Categories:
{categories_list}

Category:"""
        
        # Use appropriate AI mode
        if self.provider == AIProvider.FASTEST:
            mode = 'injury_analysis' if any(word in complaint.lower() for word in ['hurt', 'pain', 'injur', 'hospital']) else 'standard'
        
        # Make AI call
        response = None
        if self.provider == AIProvider.OPENAI and self.openai_configured:
            response, _ = self._call_openai(user_prompt, system_prompt, mode)
        elif self.provider == AIProvider.CLAUDE and self.claude_configured:
            response, _ = self._call_claude(user_prompt, system_prompt, mode)
        elif self.provider == AIProvider.FASTEST:
            if self.claude_configured:
                response, _ = self._call_claude(user_prompt, system_prompt, mode)
            elif self.openai_configured:
                response, _ = self._call_openai(user_prompt, system_prompt, mode)
        
        if response:
            category = self._clean_category_response(response)
            severity = 'none'
            if category == 'Medical/Health Concerns':
                severity, _ = detect_injury_severity(complaint)
            return category, 0.85, severity, 'en'
        
        # Fallback
        return 'Other/Miscellaneous', 0.3, 'none', 'en'
    
    def categorize_batch(self, complaints: List[Dict[str, Any]], mode: str = 'standard') -> List[Dict[str, Any]]:
        """Categorize multiple complaints with injury detection"""
        results = []
        futures = []
        
        # Submit all tasks
        for item in complaints:
            future = self.executor.submit(
                self.categorize_return,
                item.get('complaint', ''),
                item.get('fba_reason'),
                item.get('return_reason'),
                mode,
                item  # Pass full data for injury detection
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
                    'language': language,
                    'is_injury': severity != 'none'
                })
                results.append(result)
            except Exception as e:
                logger.error(f"Batch categorization error: {e}")
                result = item.copy()
                result.update({
                    'category': 'Other/Miscellaneous',
                    'confidence': 0.1,
                    'severity': 'none',
                    'language': 'en',
                    'is_injury': False
                })
                results.append(result)
        
        return results
    
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
    
    def get_injury_summary(self) -> Dict[str, Any]:
        """Get summary of all injury cases detected"""
        if not self.injury_cases:
            return {
                'total_injuries': 0,
                'critical': 0,
                'high': 0,
                'medium': 0,
                'reportable_cases': 0,
                'cases': []
            }
        
        severity_counts = {'critical': 0, 'high': 0, 'medium': 0}
        reportable_count = 0
        
        for case in self.injury_cases:
            severity_counts[case.severity] += 1
            if case.reportable:
                reportable_count += 1
        
        return {
            'total_injuries': len(self.injury_cases),
            'critical': severity_counts['critical'],
            'high': severity_counts['high'],
            'medium': severity_counts['medium'],
            'reportable_cases': reportable_count,
            'cases': self.injury_cases,
            'fda_mdr_required': reportable_count > 0,
            'immediate_action_required': severity_counts['critical'] > 0
        }
    
    def export_injury_report(self) -> str:
        """Export detailed injury report"""
        summary = self.get_injury_summary()
        
        report = f"""MEDICAL DEVICE INJURY REPORT
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

EXECUTIVE SUMMARY
================
Total Injury Cases: {summary['total_injuries']}
Critical Severity: {summary['critical']}
High Severity: {summary['high']}
Medium Severity: {summary['medium']}
FDA MDR Required: {'YES' if summary['fda_mdr_required'] else 'NO'}
Immediate Action Required: {'YES' if summary['immediate_action_required'] else 'NO'}

DETAILED INJURY CASES
====================
"""
        
        for idx, case in enumerate(summary['cases'], 1):
            report += f"""
Case #{idx}
--------
Order ID: {case.order_id}
ASIN: {case.asin}
SKU: {case.sku}
Date: {case.return_date}
Severity: {case.severity.upper()}
Device Type: {case.device_type}
Reportable: {'YES' if case.reportable else 'NO'}

Injury Keywords: {', '.join(case.injury_keywords)}
Potential Causes: {', '.join(case.potential_causes)}

Customer Comment:
{case.full_comment}

Recommendation:
{case.recommendation}

---
"""
        
        if summary['fda_mdr_required']:
            report += """
FDA MDR REPORTING REQUIREMENTS
=============================
You have reportable injury cases that may require FDA Medical Device Reporting (MDR).

Timeline:
- Death or Serious Injury: Report within 5 calendar days
- Malfunction that could cause death/injury: Report within 30 calendar days

Next Steps:
1. Immediately notify regulatory affairs team
2. Preserve all complaint records and device samples
3. Conduct root cause analysis
4. Prepare MDR submission with required information
5. Consider need for product recall or safety alert
"""
        
        return report
    
    def get_cost_summary(self) -> Dict[str, Any]:
        """Get detailed cost summary including injury tracking"""
        summary = self.cost_tracker.get_summary()
        summary['injury_tracking'] = self.get_injury_summary()
        return summary
    
    def __del__(self):
        """Cleanup resources"""
        if hasattr(self, 'executor'):
            self.executor.shutdown(wait=False)
        if hasattr(self, 'session') and self.session:
            self.session.close()

# Export all components
__all__ = [
    'EnhancedAIAnalyzer',
    'AIProvider',
    'MEDICAL_DEVICE_CATEGORIES',
    'FBA_REASON_MAP',
    'detect_injury_severity',
    'InjuryCase',
    'CostEstimate',
    'CostTracker',
    'estimate_tokens',
    'calculate_cost',
    'INJURY_RISK_CATEGORIES',
    'INJURY_KEYWORDS'
]
