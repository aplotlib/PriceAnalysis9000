"""
enhanced_ai_analysis.py - Enhanced AI Module for Medical Device Return Analysis
Supports both OpenAI and Anthropic with automatic fallback
Includes comprehensive injury detection and quality categorization
"""

import os
import logging
import json
import re
import traceback
from typing import Dict, List, Any, Optional, Tuple, Union
from datetime import datetime
import pandas as pd
import io
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

# Try to import PDF libraries
try:
    import pdfplumber
    PDFPLUMBER_AVAILABLE = True
except ImportError:
    PDFPLUMBER_AVAILABLE = False

try:
    import PyPDF2
    PYPDF2_AVAILABLE = True
except ImportError:
    PYPDF2_AVAILABLE = False

# Try to import Amazon PDF parser
try:
    from amazon_pdf_parser import AmazonPDFParser
    AMAZON_PARSER_AVAILABLE = True
except ImportError:
    AMAZON_PARSER_AVAILABLE = False
    logger.warning("Amazon PDF parser not available")

# Try to import Amazon PDF parser
try:
    from amazon_pdf_parser import AmazonPDFParser
    AMAZON_PARSER_AVAILABLE = True
except ImportError:
    AMAZON_PARSER_AVAILABLE = False
    logger.warning("Amazon PDF parser not available")

# Try to import simple PDF processor
try:
    from simple_pdf_processor import read_pdf_simple
    SIMPLE_PDF_AVAILABLE = True
except ImportError:
    SIMPLE_PDF_AVAILABLE = False
    logger.warning("Simple PDF processor not available")

# Medical device return categories
MEDICAL_DEVICE_CATEGORIES = [
    'QUALITY_DEFECTS',
    'FUNCTIONALITY_ISSUES', 
    'SIZE_FIT_ISSUES',
    'COMPATIBILITY_ISSUES',
    'WRONG_PRODUCT',
    'BUYER_MISTAKE',
    'NO_LONGER_NEEDED',
    'INJURY_RISK',
    'OTHER'
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

class FileProcessor:
    """Process various file types for return analysis"""
    
    @staticmethod
    def read_file(file, file_type: str) -> pd.DataFrame:
        """Read file and return DataFrame"""
        try:
            file_name = file.name if hasattr(file, 'name') else 'unknown'
            logger.info(f"Reading file: {file_name}, type: {file_type}")
            
            # PDF files
            if 'pdf' in file_type.lower() or file_name.lower().endswith('.pdf'):
                return FileProcessor.read_pdf(file)
            
            # Excel files
            elif any(ext in file_name.lower() for ext in ['.xlsx', '.xls']) or 'excel' in file_type:
                return pd.read_excel(file)
            
            # TSV/TXT files (FBA returns format)
            elif file_name.lower().endswith('.txt') or file_name.lower().endswith('.tsv'):
                # Read file content
                if hasattr(file, 'read'):
                    content = file.read()
                    if isinstance(content, bytes):
                        content = content.decode('utf-8')
                    file.seek(0)  # Reset file pointer
                else:
                    content = str(file)
                
                # Check if it's tab-delimited
                if '\t' in content[:1000]:
                    return pd.read_csv(file, sep='\t')
                else:
                    return pd.read_csv(file)
            
            # Default to CSV
            else:
                return pd.read_csv(file)
                
        except Exception as e:
            logger.error(f"Error reading file: {e}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            raise
    
    @staticmethod
    def read_pdf(file) -> pd.DataFrame:
        """Extract data from PDF file with multiple fallback options"""
        # Try simple PDF processor first (most robust)
        if SIMPLE_PDF_AVAILABLE:
            try:
                logger.info("Trying simple PDF processor...")
                df = read_pdf_simple(file)
                if not df.empty:
                    logger.info(f"Simple PDF processor succeeded: {df.shape}")
                    return df
            except Exception as e:
                logger.error(f"Simple PDF processor failed: {e}")
        
        # Try pdfplumber with enhanced parsing
        if PDFPLUMBER_AVAILABLE:
            try:
                logger.info("Trying pdfplumber...")
                return FileProcessor._read_pdf_with_pdfplumber(file)
            except Exception as e:
                logger.error(f"PDFPlumber failed: {e}")
                if PYPDF2_AVAILABLE:
                    logger.info("Falling back to PyPDF2")
                    try:
                        return FileProcessor._read_pdf_with_pypdf2(file)
                    except Exception as e2:
                        logger.error(f"PyPDF2 also failed: {e2}")
                        # Return empty DataFrame instead of raising
                        return pd.DataFrame(columns=['order-id', 'asin', 'reason', 'customer-comments'])
        
        # Try PyPDF2
        elif PYPDF2_AVAILABLE:
            try:
                return FileProcessor._read_pdf_with_pypdf2(file)
            except Exception as e:
                logger.error(f"PyPDF2 failed: {e}")
                return pd.DataFrame(columns=['order-id', 'asin', 'reason', 'customer-comments'])
        
        else:
            logger.error("No PDF library available")
            return pd.DataFrame(columns=['order-id', 'asin', 'reason', 'customer-comments'])
    
    @staticmethod
    def _read_pdf_with_pdfplumber(file) -> pd.DataFrame:
        """Read PDF using pdfplumber with Amazon-specific parsing"""
        import pdfplumber
        
        # Try Amazon-specific parser first
        if AMAZON_PARSER_AVAILABLE:
            parser = AmazonPDFParser()
        else:
            parser = None
        
        all_tables = []
        all_text = []
        
        try:
            with pdfplumber.open(file) as pdf:
                for page_num, page in enumerate(pdf.pages):
                    try:
                        # Extract text
                        text = page.extract_text()
                        if text:
                            all_text.append(text)
                        
                        # Extract tables with better settings
                        tables = page.extract_tables(
                            table_settings={
                                "vertical_strategy": "lines",
                                "horizontal_strategy": "lines",
                                "snap_tolerance": 3,
                                "join_tolerance": 3,
                                "edge_min_length": 3,
                                "min_words_vertical": 1,
                                "min_words_horizontal": 1
                            }
                        )
                        
                        if tables:
                            all_tables.extend(tables)
                    
                    except Exception as e:
                        logger.warning(f"Error processing page {page_num}: {e}")
                        continue
            
            # If we have the Amazon parser, use it
            if parser:
                # Parse text
                full_text = '\n'.join(all_text) if all_text else ''
                text_returns = parser.parse_pdf_text(full_text)
                
                # Parse tables
                table_df = parser.parse_pdf_tables(all_tables)
                
                # Combine results
                result_df = parser.combine_results(text_returns, table_df)
                
                if not result_df.empty:
                    return result_df
            
            # Fallback to generic parsing
            # Process tables first
            all_data = []
            for table_idx, table in enumerate(all_tables):
                if table and len(table) > 1:
                    try:
                        # Handle potential duplicate column names
                        headers = table[0]
                        if headers:
                            # Make column names unique
                            seen = {}
                            unique_headers = []
                            for header in headers:
                                header_str = str(header).strip() if header else f"Column_{len(unique_headers)}"
                                if not header_str:
                                    header_str = f"Column_{len(unique_headers)}"
                                
                                if header_str in seen:
                                    seen[header_str] += 1
                                    unique_headers.append(f"{header_str}_{seen[header_str]}")
                                else:
                                    seen[header_str] = 0
                                    unique_headers.append(header_str)
                            
                            # Create DataFrame with unique columns
                            df = pd.DataFrame(table[1:], columns=unique_headers)
                            
                            # Remove empty rows
                            df = df.dropna(how='all')
                            df = df[df.astype(str).ne('').any(axis=1)]
                            
                            if not df.empty:
                                all_data.append(df)
                    except Exception as e:
                        logger.warning(f"Error processing table {table_idx}: {e}")
                        continue
            
            # If we have table data, use it
            if all_data:
                try:
                    # Concatenate all dataframes
                    combined_df = pd.concat(all_data, ignore_index=True, sort=False)
                    return combined_df
                except Exception as e:
                    logger.error(f"Error combining table data: {e}")
            
            # Otherwise, try to parse text
            if all_text:
                full_text = '\n'.join(all_text)
                lines = full_text.split('\n')
                # Parse Amazon return format
                data = FileProcessor._parse_amazon_return_text(lines)
                if data:
                    return pd.DataFrame(data)
            
            return pd.DataFrame()
            
        except Exception as e:
            logger.error(f"PDFPlumber error: {e}")
            raise
    
    @staticmethod
    def _read_pdf_with_pypdf2(file) -> pd.DataFrame:
        """Read PDF using PyPDF2"""
        import PyPDF2
        
        all_text = []
        
        pdf_reader = PyPDF2.PdfReader(file)
        for page in pdf_reader.pages:
            text = page.extract_text()
            all_text.append(text)
        
        # Parse combined text
        full_text = '\n'.join(all_text)
        lines = full_text.split('\n')
        
        data = FileProcessor._parse_amazon_return_text(lines)
        if data:
            return pd.DataFrame(data)
        else:
            return pd.DataFrame()
    
    @staticmethod
    def _parse_amazon_return_text(lines: List[str]) -> List[Dict[str, Any]]:
        """Parse Amazon return text format"""
        returns = []
        current_return = {}
        
        # Patterns for Amazon return data
        order_pattern = r'(\d{3}-\d{7}-\d{7})'
        asin_pattern = r'(B[A-Z0-9]{9})'
        
        # Also look for common headers/labels
        return_indicators = ['return reason', 'reason:', 'customer comment', 'buyer comment', 
                           'return request', 'order id', 'asin:', 'sku:']
        
        for i, line in enumerate(lines):
            line = line.strip()
            if not line:
                continue
            
            # Check for order ID
            order_match = re.search(order_pattern, line)
            if order_match:
                # Save previous return if exists
                if current_return and 'order-id' in current_return:
                    returns.append(current_return)
                
                current_return = {
                    'order-id': order_match.group(1),
                    'reason': '',
                    'customer-comments': ''
                }
            
            # Check for ASIN
            asin_match = re.search(asin_pattern, line)
            if asin_match and current_return:
                current_return['asin'] = asin_match.group(1)
            
            # Look for return reason and comments
            if current_return:
                line_lower = line.lower()
                
                # Check if this line contains a label
                for indicator in return_indicators:
                    if indicator in line_lower:
                        # Try to extract value after the indicator
                        if ':' in line:
                            parts = line.split(':', 1)
                            if len(parts) > 1:
                                value = parts[1].strip()
                                
                                if 'reason' in indicator and value:
                                    current_return['reason'] = value
                                elif 'comment' in indicator and value:
                                    current_return['customer-comments'] = value
                        
                        # Also check next line for value
                        elif i + 1 < len(lines):
                            next_line = lines[i + 1].strip()
                            if next_line and not any(ind in next_line.lower() for ind in return_indicators):
                                if 'reason' in indicator:
                                    current_return['reason'] = next_line
                                elif 'comment' in indicator:
                                    current_return['customer-comments'] = next_line
        
        # Don't forget the last return
        if current_return and 'order-id' in current_return:
            returns.append(current_return)
        
        return returns

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
        try:
            import streamlit as st
            if hasattr(st, 'secrets') and 'OPENAI_API_KEY' in st.secrets:
                return True
        except ImportError:
            pass
        return bool(os.getenv('OPENAI_API_KEY'))
    
    def _check_anthropic_key(self) -> bool:
        """Check if Anthropic API key is available"""
        try:
            import streamlit as st
            if hasattr(st, 'secrets') and 'ANTHROPIC_API_KEY' in st.secrets:
                return True
        except ImportError:
            pass
        return bool(os.getenv('ANTHROPIC_API_KEY'))
    
    def _initialize_ai(self):
        """Initialize AI provider with proper error handling"""
        try:
            import streamlit as st
            
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
        """Test AI connection"""
        if self.ai_available and self.ai_client:
            try:
                # Test with a simple categorization
                test_result = self.categorize_return("Product is defective", "Item broke on first use")
                
                if test_result and 'category' in test_result:
                    return {
                        'status': 'AI connection successful',
                        'provider': self.provider,
                        'model': self.model,
                        'test_category': test_result['category']
                    }
                else:
                    return {
                        'status': 'AI test failed',
                        'provider': self.provider,
                        'model': self.model
                    }
            except Exception as e:
                logger.error(f"AI test failed: {e}")
                return {
                    'status': f'AI error: {str(e)}',
                    'provider': self.provider,
                    'model': self.model
                }
        else:
            return {
                'status': 'Pattern matching mode (no AI)',
                'provider': 'pattern',
                'model': None
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
                        {"role": "system", "content": "You are a medical device quality analyst specializing in return categorization. Categorize returns into these exact categories: QUALITY_DEFECTS, FUNCTIONALITY_ISSUES, SIZE_FIT_ISSUES, COMPATIBILITY_ISSUES, WRONG_PRODUCT, BUYER_MISTAKE, NO_LONGER_NEEDED, INJURY_RISK, OTHER."},
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
    
    def categorize_return(self, complaint: str = "", return_reason: str = "", 
                         fba_reason: str = "", return_data: Dict = None) -> Dict[str, Any]:
        """Categorize return with all available information"""
        # Combine all text
        full_text = f"{complaint} {return_reason} {fba_reason}".strip()
        
        if not full_text:
            return {'category': 'OTHER', 'confidence': 0.0}
        
        # Check for injuries first
        injury_analysis = self.detect_injuries(full_text)
        
        # Get category
        if self.ai_available:
            category = self._ai_categorize(full_text)
        else:
            category = self._pattern_categorize(full_text)
        
        # Override if injury detected
        if injury_analysis['has_injury'] and injury_analysis['severity'] in ['CRITICAL', 'HIGH']:
            category = 'INJURY_RISK'
        
        return {
            'category': category,
            'has_injury': injury_analysis['has_injury'],
            'severity': injury_analysis['severity'],
            'confidence': 0.95 if self.ai_available else 0.7
        }
    
    def _ai_categorize(self, text: str) -> str:
        """Use AI to categorize return"""
        if not text:
            return 'OTHER'
        
        prompt = f"""Categorize this medical device return into EXACTLY ONE of these categories:
- QUALITY_DEFECTS: defective, broken, damaged, doesn't work, poor quality
- FUNCTIONALITY_ISSUES: not comfortable, hard to use, unstable
- SIZE_FIT_ISSUES: too small, too large, doesn't fit, wrong size
- COMPATIBILITY_ISSUES: doesn't fit toilet, not compatible
- WRONG_PRODUCT: wrong item, not as described
- BUYER_MISTAKE: bought by mistake, accidentally ordered
- NO_LONGER_NEEDED: no longer needed, changed mind
- INJURY_RISK: injury, hurt, pain, hospital
- OTHER: anything else

Return text: "{text}"

Respond with ONLY the category name, nothing else."""
        
        response = self._call_ai(prompt, max_tokens=30)
        
        if response:
            # Validate response
            response_upper = response.upper().strip()
            valid_categories = ['QUALITY_DEFECTS', 'FUNCTIONALITY_ISSUES', 'SIZE_FIT_ISSUES', 
                               'COMPATIBILITY_ISSUES', 'WRONG_PRODUCT', 'BUYER_MISTAKE', 
                               'NO_LONGER_NEEDED', 'INJURY_RISK', 'OTHER']
            
            for category in valid_categories:
                if category in response_upper:
                    return category
        
        # Fallback to pattern matching
        return self._pattern_categorize(text)
    
    def _pattern_categorize(self, text: str) -> str:
        """Pattern-based categorization"""
        text_lower = text.lower()
        
        # Priority order categorization
        if any(word in text_lower for word in ['injury', 'injured', 'hurt', 'hospital', 'emergency', 'pain', 'bleeding']):
            return 'INJURY_RISK'
        
        elif any(word in text_lower for word in ['defect', 'broken', 'damaged', 'malfunction', 'not working', 'poor quality', 'faulty']):
            return 'QUALITY_DEFECTS'
        
        elif any(word in text_lower for word in ['too small', 'too large', 'too big', 'size', 'fit', "doesn't fit", 'wrong size']):
            return 'SIZE_FIT_ISSUES'
        
        elif any(word in text_lower for word in ['not comfortable', 'uncomfortable', 'hard to use', 'difficult', 'unstable', 'wobble']):
            return 'FUNCTIONALITY_ISSUES'
        
        elif any(word in text_lower for word in ['compatible', 'compatibility', "doesn't fit toilet", "won't fit", 'not compatible']):
            return 'COMPATIBILITY_ISSUES'
        
        elif any(word in text_lower for word in ['wrong', 'incorrect', 'not as described', 'different']):
            return 'WRONG_PRODUCT'
        
        elif any(word in text_lower for word in ['mistake', 'accident', 'accidentally', 'error', 'my fault']):
            return 'BUYER_MISTAKE'
        
        elif any(word in text_lower for word in ['no longer needed', 'changed mind', "don't need", 'not needed']):
            return 'NO_LONGER_NEEDED'
        
        else:
            return 'OTHER'
    
    def detect_injuries(self, text: str) -> Dict[str, Any]:
        """Detect potential injuries in text"""
        if not text:
            return {
                'has_injury': False,
                'injury_type': None,
                'severity': None,
                'fda_reportable': False
            }
        
        text_lower = text.lower()
        injuries_found = []
        max_severity = None
        fda_reportable = False
        
        # Check each injury pattern
        for injury_type, pattern_data in INJURY_PATTERNS.items():
            if any(keyword in text_lower for keyword in pattern_data['keywords']):
                injuries_found.append(injury_type)
                
                # Update severity
                if not max_severity or self._compare_severity(pattern_data['severity'], max_severity) > 0:
                    max_severity = pattern_data['severity']
                
                if pattern_data['fda_reportable']:
                    fda_reportable = True
        
        return {
            'has_injury': len(injuries_found) > 0,
            'injury_type': injuries_found[0] if injuries_found else None,
            'severity': max_severity,
            'fda_reportable': fda_reportable
        }
    
    def _compare_severity(self, sev1: str, sev2: str) -> int:
        """Compare severity levels"""
        severity_order = {'LOW': 0, 'MODERATE': 1, 'HIGH': 2, 'CRITICAL': 3}
        return severity_order.get(sev1, 0) - severity_order.get(sev2, 0)
    
    def check_for_injury(self, text: str) -> bool:
        """Simple injury check"""
        result = self.detect_injuries(text)
        return result['has_injury']
    
    def generate_insights(self, category_analysis: Dict, product_analysis: Dict) -> str:
        """Generate insights from analysis"""
        insights = []
        
        if category_analysis:
            total = sum(category_analysis.values())
            
            # Find top categories
            top_categories = sorted(category_analysis.items(), key=lambda x: x[1], reverse=True)[:3]
            
            insights.append("## Return Analysis Summary\n")
            insights.append(f"**Total Returns Analyzed:** {total}\n")
            
            insights.append("### Top Return Categories:")
            for cat, count in top_categories:
                pct = (count / total * 100) if total > 0 else 0
                insights.append(f"- **{cat}**: {count} returns ({pct:.1f}%)")
            
            # Quality concerns
            quality_issues = category_analysis.get('QUALITY_DEFECTS', 0)
            if quality_issues > 0:
                quality_pct = (quality_issues / total * 100)
                insights.append(f"\n### ⚠️ Quality Alert")
                insights.append(f"{quality_issues} quality defects detected ({quality_pct:.1f}% of returns)")
            
            # Injury concerns
            injury_issues = category_analysis.get('INJURY_RISK', 0)
            if injury_issues > 0:
                insights.append(f"\n### 🚨 Safety Alert")
                insights.append(f"{injury_issues} potential injury cases requiring immediate review")
        
        return '\n'.join(insights)
    
    def get_cost_summary(self) -> Dict[str, Any]:
        """Get API usage summary"""
        return {
            'api_calls': self.api_calls,
            'total_cost': self.total_cost,
            'provider': self.provider,
            'model': self.model
        }

# Legacy function for compatibility
def categorize_amazon_return(reason: str, comment: str = "") -> str:
    """Legacy function for categorization"""
    analyzer = EnhancedAIAnalyzer(AIProvider.AUTO)
    result = analyzer.categorize_return(reason, comment)
    return result['category']

# Export all necessary components
__all__ = [
    'EnhancedAIAnalyzer',
    'AIProvider',
    'FileProcessor',
    'MEDICAL_DEVICE_CATEGORIES',
    'INJURY_PATTERNS',
    'categorize_amazon_return'
]
