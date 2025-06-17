"""
Enhanced AI Analysis Module - Universal File Understanding
Version: 10.0 - Dual AI (OpenAI + Claude) with Universal Import
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
from dataclasses import dataclass
import asyncio
from concurrent.futures import ThreadPoolExecutor

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

# Standard imports that should always work
import pandas as pd
import numpy as np

# Model configurations
MODEL_CONFIG = {
    'openai': {
        'primary': 'gpt-4o-mini',
        'vision': 'gpt-4-vision-preview',
        'fallback': 'gpt-3.5-turbo'
    },
    'claude': {
        'primary': 'claude-3-haiku-20240307',
        'vision': 'claude-3-haiku-20240307',
        'advanced': 'claude-3-sonnet-20240229'
    }
}

# Return reason categories - Universal
RETURN_CATEGORIES = {
    'QUALITY_DEFECTS': {
        'keywords': ['defective', 'broken', 'damaged', 'doesn\'t work', 'poor quality', 'fell apart', 
                    'stopped working', 'malfunction', 'structural failure', 'collapsed'],
        'priority': 'critical'
    },
    'SIZE_FIT_ISSUES': {
        'keywords': ['too small', 'too large', 'doesn\'t fit', 'wrong size', 'tight', 'loose',
                    'small seat', 'dimension'],
        'priority': 'high'
    },
    'FUNCTIONALITY_ISSUES': {
        'keywords': ['not comfortable', 'hard to use', 'unstable', 'difficult', 'complicated',
                    'brake issues', 'wheel problems', 'locking mechanism'],
        'priority': 'high'
    },
    'WRONG_PRODUCT': {
        'keywords': ['wrong item', 'not as described', 'different', 'not what ordered'],
        'priority': 'medium'
    },
    'BUYER_MISTAKE': {
        'keywords': ['bought by mistake', 'accidentally ordered', 'ordered wrong', 'my fault'],
        'priority': 'low'
    },
    'NO_LONGER_NEEDED': {
        'keywords': ['no longer needed', 'changed mind', 'don\'t need', 'found alternative'],
        'priority': 'low'
    },
    'COMPATIBILITY_ISSUES': {
        'keywords': ['doesn\'t fit', 'not compatible', 'incompatible', 'won\'t work with'],
        'priority': 'medium'
    },
    'MISSING_PARTS': {
        'keywords': ['missing', 'incomplete', 'parts missing', 'not included'],
        'priority': 'high'
    },
    'SHIPPING_DAMAGE': {
        'keywords': ['damaged in shipping', 'arrived damaged', 'package damaged'],
        'priority': 'medium'
    }
}

@dataclass
class FileAnalysis:
    """Results from file analysis"""
    file_type: str
    content_type: str  # 'returns', 'reviews', 'other'
    extracted_data: Dict[str, Any]
    confidence: float
    ai_provider: str
    needs_clarification: bool = False
    clarification_questions: List[str] = None

class UniversalAIAnalyzer:
    """Universal AI analyzer with dual AI support and file understanding"""
    
    def __init__(self):
        self.openai_key = self._get_api_key('openai')
        self.claude_key = self._get_api_key('claude')
        self.executor = ThreadPoolExecutor(max_workers=5)
        
        # Initialize Claude client if available
        self.claude_client = None
        if self.claude_key and has_anthropic:
            try:
                from anthropic import Anthropic
                self.claude_client = Anthropic(api_key=self.claude_key)
                logger.info("Claude API initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize Claude: {e}")
        
        # Track API usage
        self.api_calls = {'openai': 0, 'claude': 0}
        self.last_call_time = {'openai': 0, 'claude': 0}
        
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
            # Use AI to understand unknown file
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
            extracted_text = []
            extracted_data = {
                'returns': [],
                'summary': {},
                'raw_text': ''
            }
            
            with pdfplumber.open(BytesIO(content)) as pdf:
                for page_num, page in enumerate(pdf.pages):
                    text = page.extract_text()
                    extracted_text.append(text)
                    
                    # Look for Amazon return patterns
                    if self._is_amazon_return_page(text):
                        returns = self._extract_amazon_returns(text)
                        extracted_data['returns'].extend(returns)
            
            extracted_data['raw_text'] = '\n'.join(extracted_text)
            
            # Use AI to understand the content if patterns not found
            if not extracted_data['returns'] and extracted_data['raw_text']:
                ai_analysis = await self._ai_analyze_text(
                    extracted_data['raw_text'],
                    "Analyze this PDF content and extract any return information, product issues, or quality concerns."
                )
                extracted_data['ai_analysis'] = ai_analysis
            
            # Determine content type
            content_type = 'returns' if extracted_data['returns'] else 'other'
            
            return FileAnalysis(
                file_type='pdf',
                content_type=content_type,
                extracted_data=extracted_data,
                confidence=0.9 if extracted_data['returns'] else 0.7,
                ai_provider='pattern_matching' if extracted_data['returns'] else 'ai'
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
    
    def _is_amazon_return_page(self, text: str) -> bool:
        """Check if text appears to be from Amazon return page"""
        indicators = [
            'manage returns', 'return request', 'order id', 'asin', 
            'return reason', 'customer comments', 'refund', 'rma'
        ]
        text_lower = text.lower()
        matches = sum(1 for ind in indicators if ind in text_lower)
        return matches >= 3
    
    def _extract_amazon_returns(self, text: str) -> List[Dict]:
        """Extract return information from Amazon format"""
        returns = []
        
        # Pattern for order IDs (format: 123-1234567-1234567)
        order_pattern = r'\b\d{3}-\d{7}-\d{7}\b'
        
        # Pattern for ASINs (10 alphanumeric characters)
        asin_pattern = r'\b[A-Z0-9]{10}\b'
        
        # Extract sections that look like return entries
        lines = text.split('\n')
        current_return = {}
        
        for line in lines:
            # Check for order ID
            order_match = re.search(order_pattern, line)
            if order_match:
                if current_return:
                    returns.append(current_return)
                current_return = {'order_id': order_match.group()}
            
            # Check for ASIN
            asin_match = re.search(asin_pattern, line)
            if asin_match and current_return:
                current_return['asin'] = asin_match.group()
            
            # Look for return reasons
            if 'reason' in line.lower() and current_return:
                # Extract text after "reason"
                reason_text = line.split('reason', 1)[-1].strip(': ')
                current_return['return_reason'] = reason_text
            
            # Look for customer comments
            if 'comment' in line.lower() and current_return:
                comment_text = line.split('comment', 1)[-1].strip(': ')
                current_return['customer_comment'] = comment_text
        
        if current_return:
            returns.append(current_return)
        
        return returns
    
    async def _analyze_image(self, content: bytes, filename: str) -> FileAnalysis:
        """Analyze image files using OCR and vision AI"""
        if not has_pil:
            return FileAnalysis(
                file_type='image',
                content_type='error',
                extracted_data={'error': 'Image processing not available'},
                confidence=0.0,
                ai_provider='none'
            )
        
        try:
            # Perform OCR if available
            extracted_text = ""
            if has_tesseract:
                from PIL import Image
                image = Image.open(BytesIO(content))
                extracted_text = pytesseract.image_to_string(image)
            
            # Use vision AI for better understanding
            vision_analysis = await self._analyze_with_vision_ai(content, filename)
            
            return FileAnalysis(
                file_type='image',
                content_type='returns' if 'return' in extracted_text.lower() else 'other',
                extracted_data={
                    'ocr_text': extracted_text,
                    'vision_analysis': vision_analysis
                },
                confidence=0.8,
                ai_provider='vision_ai'
            )
            
        except Exception as e:
            logger.error(f"Image analysis error: {e}")
            return FileAnalysis(
                file_type='image',
                content_type='error',
                extracted_data={'error': str(e)},
                confidence=0.0,
                ai_provider='none'
            )
    
    async def _analyze_text_file(self, content: bytes, filename: str, 
                                file_type: str) -> FileAnalysis:
        """Analyze text-based files (CSV, TSV, TXT)"""
        try:
            # Detect encoding
            import chardet
            detection = chardet.detect(content[:10000])
            encoding = detection['encoding'] or 'utf-8'
            
            text = content.decode(encoding)
            
            # Parse based on file type
            if file_type in ['csv', 'tsv']:
                delimiter = '\t' if file_type == 'tsv' else ','
                df = pd.read_csv(BytesIO(content), delimiter=delimiter, encoding=encoding)
                
                # Check if it's an FBA return report
                if self._is_fba_return_report(df):
                    return self._analyze_fba_returns(df)
                else:
                    # General CSV/TSV analysis
                    return FileAnalysis(
                        file_type=file_type,
                        content_type='data',
                        extracted_data={
                            'columns': df.columns.tolist(),
                            'row_count': len(df),
                            'sample': df.head(10).to_dict('records')
                        },
                        confidence=0.9,
                        ai_provider='pandas'
                    )
            else:
                # Plain text analysis
                ai_analysis = await self._ai_analyze_text(
                    text,
                    "Analyze this text file and identify if it contains return data, reviews, or other e-commerce information."
                )
                
                return FileAnalysis(
                    file_type='txt',
                    content_type='text',
                    extracted_data={
                        'text': text[:5000],  # First 5000 chars
                        'ai_analysis': ai_analysis
                    },
                    confidence=0.7,
                    ai_provider='ai'
                )
                
        except Exception as e:
            logger.error(f"Text file analysis error: {e}")
            return FileAnalysis(
                file_type=file_type,
                content_type='error',
                extracted_data={'error': str(e)},
                confidence=0.0,
                ai_provider='none'
            )
    
    def _is_fba_return_report(self, df: pd.DataFrame) -> bool:
        """Check if DataFrame is an FBA return report"""
        expected_columns = ['return-date', 'order-id', 'sku', 'asin', 'reason', 'quantity']
        df_columns = [col.lower() for col in df.columns]
        matches = sum(1 for col in expected_columns if col in df_columns)
        return matches >= 4
    
    def _analyze_fba_returns(self, df: pd.DataFrame) -> FileAnalysis:
        """Analyze FBA return report"""
        try:
            # Categorize returns
            categorized_returns = []
            
            for _, row in df.iterrows():
                return_entry = {
                    'order_id': row.get('order-id', ''),
                    'asin': row.get('asin', ''),
                    'sku': row.get('sku', ''),
                    'return_date': row.get('return-date', ''),
                    'reason': row.get('reason', ''),
                    'customer_comments': row.get('customer-comments', ''),
                    'quantity': row.get('quantity', 1)
                }
                
                # Categorize the return
                category = self._categorize_return(
                    return_entry['reason'],
                    return_entry['customer_comments']
                )
                return_entry['category'] = category
                
                categorized_returns.append(return_entry)
            
            # Generate summary statistics
            summary = self._generate_return_summary(categorized_returns)
            
            return FileAnalysis(
                file_type='fba_returns',
                content_type='returns',
                extracted_data={
                    'returns': categorized_returns,
                    'summary': summary,
                    'total_returns': len(categorized_returns)
                },
                confidence=0.95,
                ai_provider='rule_based'
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
    
    def _categorize_return(self, reason: str, comment: str = "") -> Dict[str, Any]:
        """Categorize a return based on reason and comment"""
        text = f"{reason} {comment}".lower()
        
        # Check each category
        scores = {}
        for category, info in RETURN_CATEGORIES.items():
            score = sum(1 for keyword in info['keywords'] if keyword in text)
            if score > 0:
                scores[category] = score
        
        if scores:
            # Get category with highest score
            best_category = max(scores, key=scores.get)
            return {
                'category': best_category,
                'confidence': min(scores[best_category] / 3, 1.0),
                'priority': RETURN_CATEGORIES[best_category]['priority']
            }
        else:
            return {
                'category': 'OTHER',
                'confidence': 0.5,
                'priority': 'low'
            }
    
    def _generate_return_summary(self, returns: List[Dict]) -> Dict[str, Any]:
        """Generate summary statistics from returns"""
        summary = {
            'total_returns': len(returns),
            'by_category': Counter(r['category']['category'] for r in returns),
            'by_priority': Counter(r['category']['priority'] for r in returns),
            'by_asin': Counter(r['asin'] for r in returns if r['asin']),
            'by_reason': Counter(r['reason'] for r in returns if r['reason'])
        }
        
        # Calculate return rate if we have sales data
        # This would need to be enhanced with actual sales data
        
        return summary
    
    async def _ai_analyze_text(self, text: str, prompt: str) -> Dict[str, Any]:
        """Use AI to analyze text content"""
        providers = self.get_available_providers()
        if not providers:
            return {'error': 'No AI providers available'}
        
        # Try Claude first for better understanding
        if 'claude' in providers:
            result = await self._call_claude(text, prompt)
            if result['success']:
                return result
        
        # Fallback to OpenAI
        if 'openai' in providers:
            result = await self._call_openai(text, prompt)
            if result['success']:
                return result
        
        return {'error': 'AI analysis failed'}
    
    async def _call_claude(self, text: str, prompt: str) -> Dict[str, Any]:
        """Call Claude API"""
        if not self.claude_client:
            return {'success': False, 'error': 'Claude not configured'}
        
        try:
            message = self.claude_client.messages.create(
                model=MODEL_CONFIG['claude']['primary'],
                max_tokens=1000,
                temperature=0.3,
                system="You are an expert at analyzing e-commerce data, especially Amazon returns and reviews. Extract structured information and provide insights.",
                messages=[
                    {
                        "role": "user",
                        "content": f"{prompt}\n\nContent:\n{text[:4000]}"
                    }
                ]
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
        """Call OpenAI API"""
        if not self.openai_key:
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
                    "content": "You are an expert at analyzing e-commerce data, especially Amazon returns and reviews. Extract structured information and provide insights."
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
    
    async def _analyze_with_vision_ai(self, image_content: bytes, filename: str) -> Dict[str, Any]:
        """Analyze image using vision AI"""
        providers = self.get_available_providers()
        
        # Encode image
        base64_image = base64.b64encode(image_content).decode('utf-8')
        
        if 'claude' in providers and self.claude_client:
            try:
                message = self.claude_client.messages.create(
                    model=MODEL_CONFIG['claude']['vision'],
                    max_tokens=1000,
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "text",
                                    "text": "Analyze this image and extract any return information, order IDs, ASINs, or product issues. If this appears to be from Amazon's return management page, extract all visible return details."
                                },
                                {
                                    "type": "image",
                                    "source": {
                                        "type": "base64",
                                        "media_type": "image/jpeg",
                                        "data": base64_image
                                    }
                                }
                            ]
                        }
                    ]
                )
                
                return {
                    'success': True,
                    'analysis': message.content[0].text,
                    'provider': 'claude_vision'
                }
            except Exception as e:
                logger.error(f"Claude vision error: {e}")
        
        # Fallback to OpenAI vision
        if 'openai' in providers:
            # Implement OpenAI vision API call here
            pass
        
        return {'success': False, 'error': 'Vision analysis not available'}
    
    async def interactive_file_clarification(self, file_analysis: FileAnalysis,
                                           user_response: str = None) -> Dict[str, Any]:
        """Interactive clarification for unclear files"""
        if not file_analysis.needs_clarification:
            return {'status': 'no_clarification_needed'}
        
        if not user_response:
            # Generate clarification questions
            questions = [
                "What type of data does this file contain? (returns, reviews, inventory, etc.)",
                "Is this file from Amazon Seller Central?",
                "What specific analysis would you like me to perform?"
            ]
            
            return {
                'status': 'needs_input',
                'questions': questions
            }
        else:
            # Process user response and re-analyze
            enhanced_prompt = f"""
            User clarification: {user_response}
            
            Please analyze the file content with this additional context and extract relevant data.
            """
            
            # Re-analyze with context
            result = await self._ai_analyze_text(
                file_analysis.extracted_data.get('raw_text', ''),
                enhanced_prompt
            )
            
            return {
                'status': 'clarified',
                'analysis': result
            }
    
    def generate_return_analysis_report(self, asin: str, returns_data: List[Dict],
                                      reviews_data: List[Dict] = None) -> Dict[str, Any]:
        """Generate comprehensive return analysis report like the PDF example"""
        
        # Filter returns for specific ASIN
        asin_returns = [r for r in returns_data if r.get('asin') == asin]
        
        if not asin_returns:
            return {'error': f'No returns found for ASIN {asin}'}
        
        # Calculate metrics
        total_returns = len(asin_returns)
        
        # Categorize returns
        categorized = {}
        for return_item in asin_returns:
            category = return_item.get('category', {}).get('category', 'OTHER')
            if category not in categorized:
                categorized[category] = []
            categorized[category].append(return_item)
        
        # Calculate return rate (would need sales data)
        # For now, we'll use a placeholder
        
        # Generate report sections
        report = {
            'executive_summary': {
                'asin': asin,
                'total_returns': total_returns,
                'main_issues': self._identify_main_issues(categorized),
                'trend': self._calculate_trend(asin_returns)
            },
            'category_breakdown': {
                category: {
                    'count': len(returns),
                    'percentage': len(returns) / total_returns * 100,
                    'priority': RETURN_CATEGORIES.get(category, {}).get('priority', 'low'),
                    'sample_comments': [r.get('customer_comments', '') for r in returns[:3] if r.get('customer_comments')]
                }
                for category, returns in categorized.items()
            },
            'business_impact': self._assess_business_impact(categorized, total_returns),
            'action_items': self._generate_action_items(categorized),
            'quality_priorities': self._identify_quality_priorities(categorized)
        }
        
        # Add review correlation if available
        if reviews_data:
            report['review_correlation'] = self._correlate_reviews_returns(
                asin_returns, reviews_data
            )
        
        return report
    
    def _identify_main_issues(self, categorized: Dict) -> List[str]:
        """Identify main issues from categorized returns"""
        # Sort by count and priority
        issues = []
        
        for category, returns in sorted(categorized.items(), 
                                      key=lambda x: len(x[1]), reverse=True)[:3]:
            if RETURN_CATEGORIES.get(category, {}).get('priority') in ['critical', 'high']:
                issues.append(f"{category.replace('_', ' ').title()} ({len(returns)} returns)")
        
        return issues
    
    def _calculate_trend(self, returns: List[Dict]) -> str:
        """Calculate return trend"""
        # This would need date analysis
        # For now, return a placeholder
        return "Increasing" if len(returns) > 50 else "Stable"
    
    def _assess_business_impact(self, categorized: Dict, total: int) -> Dict[str, Any]:
        """Assess business impact of returns"""
        critical_returns = sum(
            len(returns) for cat, returns in categorized.items()
            if RETURN_CATEGORIES.get(cat, {}).get('priority') == 'critical'
        )
        
        return {
            'severity': 'High' if critical_returns / total > 0.3 else 'Medium',
            'critical_return_percentage': critical_returns / total * 100,
            'estimated_cost_impact': 'Significant' if critical_returns > 20 else 'Moderate',
            'risk_assessment': self._assess_risk(categorized)
        }
    
    def _assess_risk(self, categorized: Dict) -> str:
        """Assess risk level based on return categories"""
        if 'QUALITY_DEFECTS' in categorized and len(categorized['QUALITY_DEFECTS']) > 10:
            return "High - Potential safety/liability issues"
        elif any(cat in categorized for cat in ['FUNCTIONALITY_ISSUES', 'MISSING_PARTS']):
            return "Medium - Customer satisfaction at risk"
        else:
            return "Low - Mostly user-related issues"
    
    def _generate_action_items(self, categorized: Dict) -> List[Dict[str, str]]:
        """Generate prioritized action items"""
        actions = []
        
        # Quality defects - highest priority
        if 'QUALITY_DEFECTS' in categorized:
            actions.append({
                'priority': 'IMMEDIATE',
                'action': 'Conduct quality audit with manufacturer',
                'reason': f"{len(categorized['QUALITY_DEFECTS'])} quality-related returns"
            })
        
        # Size/fit issues
        if 'SIZE_FIT_ISSUES' in categorized:
            actions.append({
                'priority': 'HIGH',
                'action': 'Update product dimensions and sizing guide',
                'reason': 'Size-related returns impacting customer satisfaction'
            })
        
        # Functionality issues
        if 'FUNCTIONALITY_ISSUES' in categorized:
            actions.append({
                'priority': 'HIGH',
                'action': 'Improve product instructions and add video guides',
                'reason': 'Customers struggling with product usage'
            })
        
        return actions[:5]  # Top 5 actions
    
    def _identify_quality_priorities(self, categorized: Dict) -> List[str]:
        """Identify quality improvement priorities"""
        priorities = []
        
        if 'QUALITY_DEFECTS' in categorized:
            # Analyze specific defects mentioned
            defect_keywords = ['broken', 'cracked', 'damaged', 'defective']
            defect_counts = Counter()
            
            for return_item in categorized['QUALITY_DEFECTS']:
                comment = return_item.get('customer_comments', '').lower()
                reason = return_item.get('reason', '').lower()
                text = f"{comment} {reason}"
                
                for keyword in defect_keywords:
                    if keyword in text:
                        defect_counts[keyword] += 1
            
            for defect, count in defect_counts.most_common(3):
                priorities.append(f"Address {defect} issues ({count} occurrences)")
        
        return priorities
    
    def _correlate_reviews_returns(self, returns: List[Dict], 
                                  reviews: List[Dict]) -> Dict[str, Any]:
        """Correlate return data with review feedback"""
        # Extract themes from negative reviews
        negative_reviews = [r for r in reviews if r.get('rating', 5) <= 2]
        
        # Find common themes
        return {
            'matching_complaints': "Analysis of review-return correlation",
            'predicted_future_returns': "Based on recent reviews",
            'preventable_returns': "Returns that could be avoided with better product info"
        }
    
    def generate_chat_response(self, user_message: str, context: Dict[str, Any]) -> str:
        """Generate contextual chat responses for the AI assistant"""
        providers = self.get_available_providers()
        
        if not providers:
            return "AI chat is not available. Please configure your OpenAI or Claude API key to enable this feature."
        
        try:
            # Build context-aware system prompt
            system_prompt = """You are a helpful Amazon quality analysis assistant specializing in return analysis and product quality improvement. 
            Provide clear, actionable advice based on the user's question and the analysis context.
            Be concise but thorough. Focus on practical implementation steps for quality improvement."""
            
            # Add relevant context to the prompt
            context_info = []
            if context.get('has_analysis'):
                context_info.append("The user has completed an analysis of their Amazon returns and reviews.")
            
            if context.get('current_asin'):
                context_info.append(f"Currently analyzing ASIN: {context['current_asin']}")
            
            if context.get('file_count', 0) > 0:
                context_info.append(f"Files loaded: {context['file_count']}")
            
            if context_info:
                system_prompt += "\n\nContext:\n" + "\n".join(context_info)
            
            # Prepare the full prompt
            full_prompt = f"{system_prompt}\n\nUser question: {user_message}"
            
            # Try Claude first if available (often better for analysis)
            if 'claude' in providers:
                result = asyncio.run(self._call_claude(
                    user_message,
                    "You are a quality analysis expert. " + full_prompt
                ))
                if result['success']:
                    return result['response']
            
            # Fallback to OpenAI
            if 'openai' in providers:
                result = asyncio.run(self._call_openai(
                    user_message,
                    full_prompt
                ))
                if result['success']:
                    return result['response']
            
            return "I encountered an error processing your request. Please try again or rephrase your question."
            
        except Exception as e:
            logger.error(f"Chat error: {e}")
            return f"I'm having trouble processing your request: {str(e)}. Please try again."

# Export the enhanced analyzer
__all__ = ['UniversalAIAnalyzer', 'FileAnalysis', 'RETURN_CATEGORIES']
