"""
Enhanced AI Analysis Module - Amazon Listing Optimization
Version: 8.0 - Improved Reliability and User Experience
"""

import logging
import os
import json
import re
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple, Union
from collections import Counter
import time

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
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
MAX_RETRIES = 3
DEFAULT_MAX_TOKENS = 2500
RATE_LIMIT_DELAY = 2

# Model configuration
MODEL_CONFIG = {
    'primary': 'gpt-4o-mini',  # Cost-effective primary model
    'fallback': 'gpt-3.5-turbo',  # Fallback option
    'context_window': 16000,  # Safe context window
    'response_tokens': 2500  # Max response tokens
}

class APIClient:
    """Enhanced OpenAI API client with improved error handling and feedback"""
    
    def __init__(self):
        self.api_key = self._get_api_key()
        self.base_url = "https://api.openai.com/v1/chat/completions"
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}" if self.api_key else ""
        }
        self.last_call_time = 0
        self._test_result = None
        
        # Initialize status
        self._initialize_status()
    
    def _get_api_key(self) -> Optional[str]:
        """Get API key with clear feedback on source"""
        key_sources = []
        
        # Try Streamlit secrets first
        try:
            import streamlit as st
            if hasattr(st, 'secrets'):
                # Try multiple possible key names
                secret_keys = ['openai_api_key', 'OPENAI_API_KEY', 'openai', 'api_key']
                for key_name in secret_keys:
                    if key_name in st.secrets:
                        api_key = str(st.secrets[key_name]).strip()
                        if api_key and api_key.startswith('sk-'):
                            logger.info(f"API key found in Streamlit secrets under '{key_name}'")
                            return api_key
                    
                # Check nested secrets
                if "openai" in st.secrets and isinstance(st.secrets["openai"], dict):
                    if "api_key" in st.secrets["openai"]:
                        api_key = str(st.secrets["openai"]["api_key"]).strip()
                        if api_key and api_key.startswith('sk-'):
                            logger.info("API key found in nested Streamlit secrets")
                            return api_key
                
                key_sources.append("Streamlit secrets (not found)")
        except Exception as e:
            logger.debug(f"Streamlit secrets not available: {e}")
            key_sources.append("Streamlit secrets (not available)")
        
        # Try environment variables
        env_vars = ['OPENAI_API_KEY', 'OPENAI_API', 'API_KEY']
        for env_name in env_vars:
            api_key = os.environ.get(env_name, '').strip()
            if api_key and api_key.startswith('sk-'):
                logger.info(f"API key found in environment variable '{env_name}'")
                return api_key
        
        key_sources.append("Environment variables (not found)")
        
        # Log where we looked
        logger.warning(f"No valid OpenAI API key found. Searched in: {', '.join(key_sources)}")
        logger.info("To enable AI features, add your OpenAI API key to Streamlit secrets or environment variables")
        
        return None
    
    def _initialize_status(self):
        """Initialize API status with detailed information"""
        if self.api_key:
            # Mask API key for logging
            masked_key = f"{self.api_key[:7]}...{self.api_key[-4:]}"
            logger.info(f"API key configured: {masked_key}")
        else:
            logger.warning("No API key configured - AI features will be disabled")
    
    def is_available(self) -> bool:
        """Check if API is available"""
        return bool(self.api_key and has_requests)
    
    def _apply_rate_limiting(self):
        """Apply rate limiting to avoid 429 errors"""
        current_time = time.time()
        time_since_last = current_time - self.last_call_time
        
        if time_since_last < RATE_LIMIT_DELAY:
            sleep_time = RATE_LIMIT_DELAY - time_since_last
            logger.debug(f"Rate limiting: sleeping for {sleep_time:.2f} seconds")
            time.sleep(sleep_time)
        
        self.last_call_time = time.time()
    
    def test_connection(self) -> Dict[str, Any]:
        """Test API connection with minimal token usage"""
        if self._test_result is not None:
            return self._test_result
        
        test_response = self.call_api(
            messages=[{"role": "user", "content": "Hi"}],
            max_tokens=5,
            temperature=0
        )
        
        self._test_result = {
            'success': test_response['success'],
            'error': test_response.get('error'),
            'model': test_response.get('model')
        }
        
        return self._test_result
    
    def call_api(self, messages: List[Dict[str, str]], 
                model: str = None,
                temperature: float = 0.3,
                max_tokens: int = DEFAULT_MAX_TOKENS,
                fallback_enabled: bool = True) -> Dict[str, Any]:
        """
        Enhanced API call with better error handling and fallback
        """
        if not self.is_available():
            return {
                "success": False,
                "error": "API not configured",
                "result": self._get_configuration_help()
            }
        
        # Use default model if not specified
        model = model or MODEL_CONFIG['primary']
        
        # Apply rate limiting
        self._apply_rate_limiting()
        
        # Prepare request
        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens
        }
        
        # Try API call with retries
        last_error = None
        for attempt in range(MAX_RETRIES):
            try:
                logger.info(f"API call attempt {attempt + 1}/{MAX_RETRIES} to {model}")
                
                response = requests.post(
                    self.base_url,
                    headers=self.headers,
                    json=payload,
                    timeout=API_TIMEOUT
                )
                
                # Handle response
                if response.status_code == 200:
                    result = response.json()
                    content = result["choices"][0]["message"]["content"]
                    
                    logger.info(f"API call successful. Tokens used: {result.get('usage', {}).get('total_tokens', 'unknown')}")
                    
                    return {
                        "success": True,
                        "result": content,
                        "usage": result.get("usage", {}),
                        "model": model
                    }
                
                # Handle specific error codes
                elif response.status_code == 401:
                    return {
                        "success": False,
                        "error": "Invalid API key",
                        "result": "Your OpenAI API key is invalid. Please check your configuration and ensure you're using a valid key that starts with 'sk-'."
                    }
                
                elif response.status_code == 429:
                    # Rate limited
                    wait_time = 2 ** (attempt + 1)
                    logger.warning(f"Rate limited, waiting {wait_time} seconds before retry")
                    time.sleep(wait_time)
                    continue
                
                elif response.status_code == 404 and fallback_enabled and model != MODEL_CONFIG['fallback']:
                    # Model not found, try fallback
                    logger.warning(f"Model {model} not found, trying fallback model")
                    return self.call_api(
                        messages=messages,
                        model=MODEL_CONFIG['fallback'],
                        temperature=temperature,
                        max_tokens=max_tokens,
                        fallback_enabled=False  # Prevent infinite recursion
                    )
                
                else:
                    # Other errors
                    error_data = response.json() if response.text else {}
                    error_msg = error_data.get('error', {}).get('message', f'API error {response.status_code}')
                    last_error = error_msg
                    
                    if attempt < MAX_RETRIES - 1:
                        logger.warning(f"API error: {error_msg}, retrying...")
                        time.sleep(2 ** attempt)
                        continue
                    
            except requests.exceptions.Timeout:
                last_error = "Request timed out"
                logger.warning(f"Timeout on attempt {attempt + 1}")
                if attempt < MAX_RETRIES - 1:
                    time.sleep(2 ** attempt)
                    continue
                    
            except requests.exceptions.ConnectionError:
                last_error = "Connection failed - check your internet connection"
                logger.warning(f"Connection error on attempt {attempt + 1}")
                if attempt < MAX_RETRIES - 1:
                    time.sleep(2 ** attempt)
                    continue
                    
            except Exception as e:
                last_error = str(e)
                logger.error(f"Unexpected error: {e}")
                if attempt < MAX_RETRIES - 1:
                    time.sleep(2 ** attempt)
                    continue
        
        # All retries exhausted
        return {
            "success": False,
            "error": last_error or "Unknown error",
            "result": None
        }
    
    def _get_configuration_help(self) -> str:
        """Get helpful configuration message"""
        return """AI analysis requires an OpenAI API key. 

To enable AI features:
1. Get an API key from https://platform.openai.com/api-keys
2. Add it to your Streamlit secrets (recommended) or environment variables
3. For Streamlit secrets, create a file `.streamlit/secrets.toml` with:
   ```
   openai_api_key = "sk-your-key-here"
   ```
4. Restart the application

Without AI, you can still:
- View detailed metrics and statistics
- Export data for manual analysis
- Use the basic review insights"""

class EnhancedAIAnalyzer:
    """Enhanced AI analyzer with improved reliability and user feedback"""
    
    def __init__(self):
        self.api_client = APIClient()
        self.initialized = self.api_client.is_available()
        
        logger.info(f"AI Analyzer initialized - Available: {self.initialized}")
    
    def get_api_status(self) -> Dict[str, Any]:
        """Get detailed API status for user feedback"""
        status = {
            'available': False,
            'configured': bool(self.api_client.api_key),
            'tested': False,
            'message': '',
            'error': None
        }
        
        if not self.api_client.api_key:
            status['message'] = 'API key not configured'
            status['error'] = 'missing_key'
            return status
        
        if not has_requests:
            status['message'] = 'Requests library not available'
            status['error'] = 'missing_dependency'
            return status
        
        # Test the connection
        test_result = self.api_client.test_connection()
        
        status['tested'] = True
        status['available'] = test_result['success']
        
        if test_result['success']:
            status['message'] = f"AI ready using {test_result.get('model', 'OpenAI')}"
            status['model'] = test_result.get('model')
        else:
            status['message'] = f"API test failed: {test_result.get('error', 'Unknown error')}"
            status['error'] = test_result.get('error')
        
        return status
    
    def _prepare_review_summary(self, reviews: List[Dict]) -> Dict[str, Any]:
        """Prepare a structured summary of reviews for analysis"""
        summary = {
            'total_count': len(reviews),
            'rating_distribution': {1: 0, 2: 0, 3: 0, 4: 0, 5: 0},
            'verified_count': 0,
            'common_phrases': {
                'positive': Counter(),
                'negative': Counter()
            },
            'review_samples': {
                'positive': [],
                'negative': [],
                'detailed': []
            }
        }
        
        # Process reviews
        for review in reviews:
            rating = review.get('rating', 3)
            summary['rating_distribution'][rating] += 1
            
            if review.get('verified'):
                summary['verified_count'] += 1
            
            # Extract phrases
            body = review.get('body', '').lower()
            words = re.findall(r'\b[a-z]+\b', body)
            
            if rating >= 4:
                summary['common_phrases']['positive'].update(words)
                if len(summary['review_samples']['positive']) < 10:
                    summary['review_samples']['positive'].append(review)
            elif rating <= 2:
                summary['common_phrases']['negative'].update(words)
                if len(summary['review_samples']['negative']) < 10:
                    summary['review_samples']['negative'].append(review)
            
            # Collect detailed reviews
            if len(body) > 200 and len(summary['review_samples']['detailed']) < 5:
                summary['review_samples']['detailed'].append(review)
        
        return summary
    
    def _build_optimized_prompt(self, reviews: List[Dict], 
                               product_info: Dict,
                               listing_details: Optional[Dict],
                               metrics: Optional[Dict],
                               marketplace_data: Optional[Dict]) -> str:
        """Build an optimized prompt that fits within token limits"""
        
        # Start with review summary
        review_summary = self._prepare_review_summary(reviews)
        
        prompt_parts = []
        
        # Product context
        prompt_parts.append(f"""
PRODUCT ANALYSIS CONTEXT
Product ASIN: {product_info.get('asin', 'Unknown')}
Total Reviews: {product_info.get('total_reviews', len(reviews))}
Reviews Being Analyzed: {len(reviews)}
Average Rating: {sum(r['rating'] for r in reviews) / len(reviews):.2f}/5
""")
        
        # Current listing if available
        if listing_details and listing_details.get('title'):
            prompt_parts.append(f"""
CURRENT LISTING DETAILS
Title: {listing_details.get('title', 'Not provided')}
Brand: {listing_details.get('brand', 'Not provided')}
ASIN: {listing_details.get('asin', 'Not provided')}

Current Bullet Points:
{chr(10).join([f'• {b}' for b in listing_details.get('bullet_points', []) if b.strip()][:5])}
""")
        
        # Key metrics
        if metrics:
            prompt_parts.append(f"""
KEY METRICS
Health Score: {metrics.get('listing_health_score', {}).get('total_score', 'N/A')}/100
Sentiment: Positive {metrics.get('sentiment_breakdown', {}).get('positive', 0)}, Negative {metrics.get('sentiment_breakdown', {}).get('negative', 0)}
Top Issues: {', '.join([k for k, v in sorted(metrics.get('issue_categories', {}).items(), key=lambda x: x[1], reverse=True)[:3] if v > 0])}
""")
        
        # Marketplace insights
        if marketplace_data:
            returns_total = 0
            top_return_reason = "Unknown"
            
            if 'return_patterns' in marketplace_data:
                for pattern in marketplace_data['return_patterns'].values():
                    returns_total += pattern.get('count', 0)
                    if pattern.get('reasons'):
                        reason = max(pattern['reasons'].items(), key=lambda x: x[1])
                        top_return_reason = reason[0]
            
            prompt_parts.append(f"""
MARKETPLACE DATA
Total Returns: {returns_total}
Top Return Reason: {top_return_reason}
""")
        
        # Review samples - balanced selection
        prompt_parts.append("""
CUSTOMER REVIEWS (Sorted by relevance and rating):""")
        
        # Include negative reviews first (more important for improvement)
        negative_reviews = review_summary['review_samples']['negative'][:15]
        for i, review in enumerate(negative_reviews, 1):
            prompt_parts.append(f"""
Review {i} ({review['rating']}/5){' [Verified]' if review.get('verified') else ''}:
Title: {review.get('title', '')[:100]}
Body: {review.get('body', '')[:300]}""")
        
        # Then positive reviews
        positive_reviews = review_summary['review_samples']['positive'][:10]
        for i, review in enumerate(positive_reviews, len(negative_reviews) + 1):
            prompt_parts.append(f"""
Review {i} ({review['rating']}/5){' [Verified]' if review.get('verified') else ''}:
Title: {review.get('title', '')[:100]}
Body: {review.get('body', '')[:200]}""")
        
        # Analysis instructions
        prompt_parts.append("""

ANALYSIS INSTRUCTIONS:
Provide specific, actionable Amazon listing optimization recommendations that directly address the issues found in customer reviews. Focus on medical device quality and safety concerns.

Format your response EXACTLY as follows:

## TITLE OPTIMIZATION
[Identify specific keywords customers use that are missing from the current title]
[Provide a new optimized title that addresses main concerns and includes customer language]

## BULLET POINT OPTIMIZATION
[Create 5 bullet points that specifically address the issues and concerns from reviews]
• Bullet 1: [Address the #1 complaint]
• Bullet 2: [Address quality/safety concerns]
• Bullet 3: [Highlight features customers love]
• Bullet 4: [Address usability/training needs]
• Bullet 5: [Warranty/support information]

## BACKEND KEYWORDS
[List keywords extracted from customer language, max 250 characters]

## IMMEDIATE ACTION ITEMS
1. [Most critical change based on negative feedback]
2. [Second priority change]
3. [Third priority change]

## QUALITY IMPROVEMENT PRIORITIES
[Specific quality issues to address based on returns and reviews]

Be specific and use the exact language customers use in their reviews.""")
        
        return '\n'.join(prompt_parts)
    
    def analyze_reviews_for_listing_optimization(self, 
                                               reviews: List[Dict],
                                               product_info: Dict,
                                               listing_details: Optional[Dict] = None,
                                               metrics: Optional[Dict] = None,
                                               marketplace_data: Optional[Dict] = None) -> str:
        """
        Main analysis method with improved error handling and progress feedback
        """
        if not self.initialized:
            return """## AI Analysis Not Available

The AI analysis service is not configured. Please add your OpenAI API key to enable AI-powered recommendations.

### Manual Analysis Suggestions:
1. Review the metrics dashboard for key insights
2. Focus on addressing the most common complaints
3. Update your title with frequently mentioned keywords
4. Revise bullet points to address top concerns
5. Export the data for detailed manual analysis

💡 To enable AI analysis, add your OpenAI API key to the application configuration."""
        
        try:
            logger.info(f"Starting AI analysis for {len(reviews)} reviews")
            
            # Build optimized prompt
            prompt = self._build_optimized_prompt(
                reviews, product_info, listing_details, metrics, marketplace_data
            )
            
            # Check prompt size (rough estimate: 4 chars = 1 token)
            estimated_tokens = len(prompt) / 4
            if estimated_tokens > MODEL_CONFIG['context_window'] * 0.8:
                logger.warning(f"Prompt may be too long: ~{estimated_tokens} tokens")
                # Reduce review count if needed
                reviews = reviews[:30]
                prompt = self._build_optimized_prompt(
                    reviews, product_info, listing_details, metrics, marketplace_data
                )
            
            # Make API call
            response = self.api_client.call_api(
                messages=[
                    {
                        "role": "system", 
                        "content": "You are an expert Amazon listing optimization specialist with deep knowledge of medical devices and e-commerce best practices. Provide specific, actionable recommendations based on customer feedback."
                    },
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=MODEL_CONFIG['response_tokens']
            )
            
            if response['success']:
                logger.info("AI analysis completed successfully")
                
                # Add context footer
                analysis = response['result']
                
                footer_parts = ["\n\n---\n### Analysis Context"]
                
                if listing_details and listing_details.get('title'):
                    footer_parts.append(f"✅ Compared current listing with customer feedback")
                
                if marketplace_data:
                    footer_parts.append(f"✅ Incorporated return patterns and financial data")
                
                footer_parts.append(f"📊 Analyzed {len(reviews)} customer reviews")
                footer_parts.append("\n💡 **Next Steps**: Use the AI Chat to discuss implementation strategies or get alternative suggestions.")
                
                return analysis + '\n'.join(footer_parts)
            
            else:
                error_msg = response.get('error', 'Unknown error')
                logger.error(f"AI analysis failed: {error_msg}")
                
                return f"""## AI Analysis Error

{error_msg}

### Troubleshooting:
1. Verify your OpenAI API key is valid and has credits
2. Check your internet connection
3. Try again in a few moments (rate limits may apply)

### Alternative Actions:
- Use the metrics dashboard to identify key issues
- Export data for manual analysis
- Focus on addressing the most frequent complaints

💡 If the error persists, please check the application logs or contact support."""
                
        except Exception as e:
            logger.error(f"Unexpected error in analysis: {str(e)}", exc_info=True)
            
            return f"""## Analysis Error

An unexpected error occurred during analysis.

Error details: {str(e)}

Please try again or contact support if the issue persists.

💡 You can still use the metrics dashboard and export features while we resolve this issue."""
    
    def generate_chat_response(self, 
                             user_message: str,
                             context: Dict[str, Any]) -> str:
        """Generate contextual chat responses"""
        if not self.initialized:
            return "AI chat is not available. Please configure your OpenAI API key to enable this feature."
        
        try:
            # Build context-aware system prompt
            system_prompt = """You are a helpful Amazon listing optimization assistant specializing in medical devices. 
            Provide clear, actionable advice based on the user's question and the analysis context.
            Be concise but thorough. Focus on practical implementation steps."""
            
            # Add relevant context
            if context.get('has_analysis'):
                system_prompt += "\nThe user has completed an analysis of their Amazon reviews."
            
            if context.get('current_asin'):
                system_prompt += f"\nWorking with ASIN: {context['current_asin']}"
            
            # Make API call
            response = self.api_client.call_api(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message}
                ],
                max_tokens=500,
                temperature=0.7
            )
            
            if response['success']:
                return response['result']
            else:
                return f"I encountered an error: {response.get('error', 'Unknown error')}. Please try rephrasing your question."
                
        except Exception as e:
            logger.error(f"Chat error: {e}")
            return "I'm having trouble processing your request. Please try again."

# Export classes
__all__ = ['EnhancedAIAnalyzer', 'APIClient']
