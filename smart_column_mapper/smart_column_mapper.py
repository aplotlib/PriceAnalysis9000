"""
Smart Column Mapper - AI-Powered Column Detection and Mapping
Intelligently detects column types from various file formats
"""

import pandas as pd
import numpy as np
import re
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

class SmartColumnMapper:
    """Intelligently detect and map columns from various file formats"""
    
    # Column patterns and indicators
    COLUMN_PATTERNS = {
        'date': {
            'header_patterns': [
                r'date', r'when', r'time', r'created', r'made', r'received',
                r'complaint.*date', r'return.*date', r'order.*date'
            ],
            'content_patterns': [
                r'\d{1,2}[/-]\d{1,2}[/-]\d{2,4}',  # MM/DD/YYYY or MM-DD-YYYY
                r'\d{4}[/-]\d{1,2}[/-]\d{1,2}',    # YYYY-MM-DD
                r'\w+ \d{1,2}, \d{4}',             # Month DD, YYYY
                r'\d{1,2}\s+\w+\s+\d{4}'           # DD Month YYYY
            ],
            'priority': 1
        },
        'product_id': {
            'header_patterns': [
                r'product', r'sku', r'item', r'identifier', r'tag', r'model',
                r'product.*id', r'product.*code', r'product.*tag'
            ],
            'content_patterns': [
                r'^[A-Z]{2,4}[\-\d]+',           # Like SUP1077, MOB-001
                r'^[A-Z0-9]{4,}[\-][A-Z0-9]+',   # Generic product codes
                r'\w+\s*-\s*[A-Z0-9]+',          # Product name - code
            ],
            'priority': 2
        },
        'asin': {
            'header_patterns': [
                r'asin', r'amazon.*id', r'amazon.*code'
            ],
            'content_patterns': [
                r'^B[A-Z0-9]{9}$',  # Standard ASIN format
                r'\bB[A-Z0-9]{9}\b'  # ASIN within text
            ],
            'priority': 3
        },
        'order_id': {
            'header_patterns': [
                r'order', r'order.*id', r'order.*#', r'order.*number',
                r'transaction', r'purchase'
            ],
            'content_patterns': [
                r'\d{3}-\d{7}-\d{7}',  # Amazon order format
                r'\d{10,}',            # Long numeric IDs
                r'[A-Z0-9]{8,}'        # Alphanumeric order IDs
            ],
            'priority': 4
        },
        'complaint': {
            'header_patterns': [
                r'complaint', r'comment', r'reason', r'feedback', r'issue',
                r'description', r'problem', r'concern', r'return.*reason',
                r'customer.*comment', r'investigating.*complaint'
            ],
            'content_patterns': [
                r'.{20,}',  # Long text (20+ chars)
                r'.*[.!?].*',  # Contains punctuation
                r'.*(too|not|bad|broken|defect|wrong|missing).*'  # Common complaint words
            ],
            'priority': 5
        },
        'source': {
            'header_patterns': [
                r'source', r'channel', r'origin', r'from', r'type',
                r'complaint.*source', r'return.*source'
            ],
            'content_patterns': [
                r'amazon', r'email', r'phone', r'chat', r'return',
                r'customer.*service', r'online', r'website'
            ],
            'priority': 6
        },
        'agent': {
            'header_patterns': [
                r'agent', r'investigator', r'assigned', r'handled.*by',
                r'categorizing.*agent', r'investigating.*agent', r'analyst'
            ],
            'content_patterns': [
                r'^[A-Z][a-z]+$',  # Single name
                r'^[A-Z][a-z]+\s+[A-Z][a-z]+$',  # Full name
            ],
            'priority': 7
        },
        'ticket_id': {
            'header_patterns': [
                r'ticket', r'case', r'reference', r'cs.*ticket', r'support.*id'
            ],
            'content_patterns': [
                r'[A-Z]{2,}\d+',  # Like CS12345
                r'\d{5,}'         # Numeric ticket
            ],
            'priority': 8
        },
        'udi': {
            'header_patterns': [
                r'udi', r'device.*identifier', r'unique.*device'
            ],
            'content_patterns': [
                r'\(\d+\)',  # UDI format variations
                r'[A-Z0-9]{10,}'
            ],
            'priority': 9
        }
    }
    
    def __init__(self, ai_analyzer=None):
        self.ai_analyzer = ai_analyzer
        self.mapping_cache = {}
    
    def detect_columns(self, df: pd.DataFrame, sample_size: int = 100) -> Dict[str, str]:
        """
        Detect column types using header names and content analysis
        Returns mapping of detected_type -> actual_column_name
        """
        column_mapping = {}
        confidence_scores = {}
        
        # Sample rows for analysis
        sample_df = df.head(min(sample_size, len(df)))
        
        for col in df.columns:
            col_lower = str(col).lower().strip()
            
            # Skip if already mapped
            if col in column_mapping.values():
                continue
            
            # Score each column type
            type_scores = {}
            
            for col_type, patterns in self.COLUMN_PATTERNS.items():
                score = 0
                
                # Check header patterns
                for pattern in patterns['header_patterns']:
                    if re.search(pattern, col_lower):
                        score += 10 * patterns['priority']
                
                # Check content patterns
                non_null_values = sample_df[col].dropna()
                if len(non_null_values) > 0:
                    matches = 0
                    for value in non_null_values.astype(str).head(10):
                        for pattern in patterns['content_patterns']:
                            if re.search(pattern, value, re.IGNORECASE):
                                matches += 1
                                break
                    
                    # Score based on match percentage
                    if matches > 0:
                        match_rate = matches / min(10, len(non_null_values))
                        score += match_rate * 20 * patterns['priority']
                
                if score > 0:
                    type_scores[col_type] = score
            
            # Assign highest scoring type
            if type_scores:
                best_type = max(type_scores.items(), key=lambda x: x[1])
                if best_type[1] > 10:  # Minimum confidence threshold
                    column_mapping[best_type[0]] = col
                    confidence_scores[best_type[0]] = best_type[1]
        
        # Use AI for ambiguous columns if available
        if self.ai_analyzer and len(column_mapping) < len(self.COLUMN_PATTERNS):
            unmapped_cols = [col for col in df.columns if col not in column_mapping.values()]
            if unmapped_cols:
                ai_mapping = self._ai_column_detection(df, unmapped_cols, column_mapping)
                column_mapping.update(ai_mapping)
        
        logger.info(f"Column mapping detected: {column_mapping}")
        logger.info(f"Confidence scores: {confidence_scores}")
        
        return column_mapping
    
    def _ai_column_detection(self, df: pd.DataFrame, unmapped_cols: List[str], 
                           existing_mapping: Dict[str, str]) -> Dict[str, str]:
        """Use AI to detect ambiguous columns"""
        
        if not self.ai_analyzer:
            return {}
        
        # Prepare sample data for AI
        sample_data = []
        for col in unmapped_cols[:5]:  # Limit to 5 columns for API efficiency
            non_null = df[col].dropna().head(5)
            if len(non_null) > 0:
                sample_data.append({
                    'column_name': col,
                    'samples': non_null.tolist()
                })
        
        if not sample_data:
            return {}
        
        # Build prompt
        prompt = f"""Analyze these columns and identify their types based on the column names and sample data.

Already mapped columns: {existing_mapping}

Unmapped columns to analyze:
{self._format_sample_data(sample_data)}

Possible column types to identify:
- date: Date when complaint/return was made
- product_id: Product SKU or identifier
- asin: Amazon ASIN (10-character code starting with B)
- order_id: Order number (often format XXX-XXXXXXX-XXXXXXX)
- complaint: Customer complaint/comment text
- source: Where the complaint came from
- agent: Person handling the complaint
- ticket_id: Support ticket number
- udi: Unique Device Identifier (medical devices)

Respond with a JSON mapping of column_type: column_name for any columns you can confidently identify.
Only include columns you're confident about."""
        
        try:
            # Call AI (using the enhanced_ai_analysis module pattern)
            response = self.ai_analyzer._call_openai_sync(
                prompt,
                "You are an expert at analyzing data structures and identifying column types.",
                'standard'
            )
            
            if response:
                # Parse JSON response
                import json
                try:
                    ai_mapping = json.loads(response)
                    # Validate mapping
                    validated_mapping = {}
                    for col_type, col_name in ai_mapping.items():
                        if col_type in self.COLUMN_PATTERNS and col_name in unmapped_cols:
                            validated_mapping[col_type] = col_name
                    return validated_mapping
                except json.JSONDecodeError:
                    logger.warning("AI response was not valid JSON")
                    
        except Exception as e:
            logger.error(f"AI column detection error: {e}")
        
        return {}
    
    def _format_sample_data(self, sample_data: List[Dict]) -> str:
        """Format sample data for AI prompt"""
        formatted = []
        for item in sample_data:
            samples_str = ', '.join([f'"{str(s)}"' for s in item['samples'][:3]])
            formatted.append(f"Column '{item['column_name']}': [{samples_str}]")
        return '\n'.join(formatted)
    
    def map_dataframe(self, df: pd.DataFrame, column_mapping: Dict[str, str]) -> pd.DataFrame:
        """
        Create standardized dataframe with mapped columns
        """
        standardized_data = {}
        
        # Core fields we want to extract
        core_fields = {
            'date': 'return_date',
            'product_id': 'sku',
            'asin': 'asin',
            'order_id': 'order_id',
            'complaint': 'customer_comment',
            'source': 'source',
            'agent': 'agent',
            'ticket_id': 'ticket_id',
            'udi': 'udi'
        }
        
        # Map detected columns to standard names
        for detected_type, actual_col in column_mapping.items():
            if detected_type in core_fields and actual_col in df.columns:
                standard_name = core_fields[detected_type]
                standardized_data[standard_name] = df[actual_col]
        
        # Include unmapped columns with original names
        for col in df.columns:
            if col not in column_mapping.values():
                # Sanitize column name
                safe_name = re.sub(r'[^\w\s]', '', col).replace(' ', '_').lower()
                if safe_name and safe_name not in standardized_data:
                    standardized_data[safe_name] = df[col]
        
        # Create new dataframe
        result_df = pd.DataFrame(standardized_data)
        
        # Ensure essential columns exist
        essential_cols = ['return_date', 'sku', 'asin', 'order_id', 'customer_comment']
        for col in essential_cols:
            if col not in result_df.columns:
                result_df[col] = ''
        
        # Clean and validate data
        result_df = self._clean_mapped_data(result_df)
        
        return result_df
    
    def _clean_mapped_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and validate mapped data"""
        
        # Clean dates
        if 'return_date' in df.columns:
            df['return_date'] = pd.to_datetime(df['return_date'], errors='coerce')
            df['return_date'] = df['return_date'].dt.strftime('%Y-%m-%d')
        
        # Clean ASIN
        if 'asin' in df.columns:
            # Extract ASIN from text if needed
            df['asin'] = df['asin'].apply(self._extract_asin)
        
        # Clean order IDs
        if 'order_id' in df.columns:
            df['order_id'] = df['order_id'].astype(str).str.strip()
        
        # Ensure complaint text is string
        if 'customer_comment' in df.columns:
            df['customer_comment'] = df['customer_comment'].fillna('').astype(str)
        
        return df
    
    def _extract_asin(self, value: Any) -> str:
        """Extract ASIN from value"""
        if pd.isna(value):
            return ''
        
        value_str = str(value)
        # Look for ASIN pattern
        asin_match = re.search(r'\b(B[A-Z0-9]{9})\b', value_str)
        if asin_match:
            return asin_match.group(1)
        
        # If it's exactly 10 alphanumeric chars starting with B
        if re.match(r'^B[A-Z0-9]{9}$', value_str.strip()):
            return value_str.strip()
        
        return ''
    
    def validate_mapping(self, df: pd.DataFrame, mapping: Dict[str, str]) -> Dict[str, Any]:
        """Validate the quality of column mapping"""
        
        validation_results = {
            'is_valid': True,
            'missing_required': [],
            'confidence_scores': {},
            'warnings': []
        }
        
        # Required columns for return analysis
        required_columns = ['complaint', 'date']
        
        for req_col in required_columns:
            if req_col not in mapping:
                validation_results['missing_required'].append(req_col)
                validation_results['is_valid'] = False
        
        # Check data quality for mapped columns
        for col_type, col_name in mapping.items():
            if col_name in df.columns:
                non_null_pct = (df[col_name].notna().sum() / len(df)) * 100
                validation_results['confidence_scores'][col_type] = non_null_pct
                
                if non_null_pct < 50:
                    validation_results['warnings'].append(
                        f"Column '{col_name}' (detected as {col_type}) has {non_null_pct:.1f}% non-null values"
                    )
        
        return validation_results

# Export the class
__all__ = ['SmartColumnMapper']
