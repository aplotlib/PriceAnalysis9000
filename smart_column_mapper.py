"""
Smart Column Mapper - AI-powered column detection and mapping
For automatic detection of Amazon return data columns
"""

import pandas as pd
import numpy as np
import re
import logging
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import json

logger = logging.getLogger(__name__)

class SmartColumnMapper:
    """Intelligent column mapping for Amazon return data"""
    
    def __init__(self, ai_analyzer=None):
        """Initialize with optional AI analyzer for enhanced detection"""
        self.ai_analyzer = ai_analyzer
        
        # Define column patterns for Amazon return data
        self.COLUMN_PATTERNS = {
            'order_id': {
                'patterns': [
                    r'order[\s\-_]?id',
                    r'order[\s\-_]?number',
                    r'order',
                    r'purchase[\s\-_]?id'
                ],
                'content_patterns': [
                    r'^\d{3}-\d{7}-\d{7}$',  # Amazon order format
                    r'^\d{3}-\d{10}$'         # Alternative format
                ],
                'priority': 10
            },
            'asin': {
                'patterns': [
                    r'asin',
                    r'product[\s\-_]?asin',
                    r'item[\s\-_]?asin'
                ],
                'content_patterns': [
                    r'^B[A-Z0-9]{9}$',        # Standard ASIN
                    r'\bB[A-Z0-9]{9}\b'       # ASIN within text
                ],
                'priority': 9
            },
            'sku': {
                'patterns': [
                    r'sku',
                    r'product[\s\-_]?sku',
                    r'seller[\s\-_]?sku',
                    r'merchant[\s\-_]?sku'
                ],
                'content_patterns': [
                    r'^[A-Z0-9\-_]+$'         # Typical SKU format
                ],
                'priority': 8
            },
            'return_date': {
                'patterns': [
                    r'return[\s\-_]?date',
                    r'returned[\s\-_]?date',
                    r'date[\s\-_]?returned',
                    r'return[\s\-_]?request[\s\-_]?date',
                    r'date'
                ],
                'content_patterns': [
                    r'\d{4}-\d{2}-\d{2}',     # ISO format
                    r'\d{1,2}[/-]\d{1,2}[/-]\d{2,4}',  # US format
                    r'\d{2}[/-]\d{2}[/-]\d{4}' # European format
                ],
                'priority': 7
            },
            'return_reason': {
                'patterns': [
                    r'return[\s\-_]?reason',
                    r'reason[\s\-_]?for[\s\-_]?return',
                    r'reason',
                    r'return[\s\-_]?type',
                    r'issue'
                ],
                'content_patterns': [
                    r'defect|broken|damaged|wrong|size|fit|comfort|quality',
                    r'NOT_COMPATIBLE|DEFECTIVE|DAMAGED|WRONG_ITEM'  # FBA codes
                ],
                'priority': 9
            },
            'buyer_comment': {
                'patterns': [
                    r'customer[\s\-_]?comment',
                    r'buyer[\s\-_]?comment',
                    r'comment',
                    r'feedback',
                    r'notes',
                    r'description',
                    r'customer[\s\-_]?feedback'
                ],
                'content_patterns': [
                    r'.{20,}',  # Longer text entries
                    r'[a-zA-Z\s]{10,}'  # Text with words
                ],
                'priority': 8
            },
            'quantity': {
                'patterns': [
                    r'quantity',
                    r'qty',
                    r'amount',
                    r'units'
                ],
                'content_patterns': [
                    r'^\d+$',  # Integer
                    r'^\d+\.\d+$'  # Decimal
                ],
                'priority': 5
            },
            'product_name': {
                'patterns': [
                    r'product[\s\-_]?name',
                    r'item[\s\-_]?name',
                    r'product[\s\-_]?title',
                    r'description',
                    r'title'
                ],
                'content_patterns': [
                    r'[a-zA-Z\s\-,]{10,}',  # Product descriptions
                ],
                'priority': 6
            },
            'fnsku': {
                'patterns': [
                    r'fnsku',
                    r'fulfillment[\s\-_]?sku'
                ],
                'content_patterns': [
                    r'^[A-Z0-9]{10}$'  # Amazon FNSKU format
                ],
                'priority': 7
            }
        }
        
        # Common column name variations
        self.COLUMN_ALIASES = {
            'order-id': 'order_id',
            'order id': 'order_id',
            'orderId': 'order_id',
            'customer-comments': 'buyer_comment',
            'customer comments': 'buyer_comment',
            'return-date': 'return_date',
            'return date': 'return_date',
            'product-name': 'product_name',
            'product name': 'product_name'
        }
    
    def detect_columns(self, df: pd.DataFrame, sample_size: int = 100) -> Dict[str, str]:
        """
        Automatically detect column types using pattern matching and AI
        
        Args:
            df: DataFrame to analyze
            sample_size: Number of rows to sample for analysis
            
        Returns:
            Dictionary mapping column types to column names
        """
        column_mapping = {}
        confidence_scores = {}
        
        # Normalize column names
        df_columns_lower = [str(col).lower().strip() for col in df.columns]
        
        # First pass: Direct alias matching
        for idx, col in enumerate(df.columns):
            col_lower = df_columns_lower[idx]
            if col_lower in self.COLUMN_ALIASES:
                mapped_type = self.COLUMN_ALIASES[col_lower]
                column_mapping[mapped_type] = col
                confidence_scores[mapped_type] = 100
        
        # Second pass: Pattern matching
        sample_df = df.head(min(sample_size, len(df)))
        
        for col in df.columns:
            if col in column_mapping.values():
                continue
                
            col_lower = str(col).lower().strip()
            
            # Score each column against each pattern
            type_scores = {}
            
            for col_type, patterns in self.COLUMN_PATTERNS.items():
                if col_type in column_mapping:
                    continue
                
                score = 0
                
                # Check column name patterns
                for pattern in patterns['patterns']:
                    if re.search(pattern, col_lower):
                        score += 50 * patterns['priority']
                        break
                
                # Check content patterns
                non_null_values = sample_df[col].dropna()
                if len(non_null_values) > 0:
                    matches = 0
                    for value in non_null_values.astype(str).head(20):
                        for pattern in patterns['content_patterns']:
                            if re.search(pattern, value, re.IGNORECASE):
                                matches += 1
                                break
                    
                    if matches > 0:
                        match_rate = matches / min(20, len(non_null_values))
                        score += match_rate * 30 * patterns['priority']
                
                if score > 0:
                    type_scores[col_type] = score
            
            # Assign highest scoring type
            if type_scores:
                best_type = max(type_scores.items(), key=lambda x: x[1])
                if best_type[1] > 50:  # Confidence threshold
                    column_mapping[best_type[0]] = col
                    confidence_scores[best_type[0]] = best_type[1]
        
        # Third pass: Use AI for ambiguous columns if available
        if self.ai_analyzer and len(column_mapping) < 5:  # If we haven't found key columns
            unmapped_cols = [col for col in df.columns if col not in column_mapping.values()]
            if unmapped_cols:
                try:
                    ai_mapping = self._ai_column_detection(df, unmapped_cols, column_mapping)
                    column_mapping.update(ai_mapping)
                except Exception as e:
                    logger.error(f"AI column detection failed: {e}")
        
        # Log results
        logger.info(f"Column mapping detected: {column_mapping}")
        logger.info(f"Confidence scores: {confidence_scores}")
        
        return column_mapping
    
    def _ai_column_detection(self, df: pd.DataFrame, unmapped_cols: List[str], 
                           existing_mapping: Dict[str, str]) -> Dict[str, str]:
        """Use AI to detect ambiguous columns"""
        
        if not self.ai_analyzer:
            return {}
        
        # Prepare sample data
        sample_data = []
        for col in unmapped_cols[:10]:  # Limit to 10 columns
            non_null = df[col].dropna().head(5)
            if len(non_null) > 0:
                sample_data.append({
                    'column_name': col,
                    'samples': non_null.astype(str).tolist()
                })
        
        if not sample_data:
            return {}
        
        # Build prompt
        prompt = f"""Analyze these columns from an Amazon returns dataset and identify their types.

Existing mappings: {json.dumps(existing_mapping, indent=2)}

Columns to analyze:
{json.dumps(sample_data, indent=2)}

For each column, determine if it matches one of these types:
- order_id: Amazon order IDs (format: 123-1234567-1234567)
- asin: Product ASINs (format: B followed by 9 alphanumeric)
- sku: Product SKUs
- return_date: Date of return
- return_reason: Reason for return (often short phrases or codes)
- buyer_comment: Customer comments (usually longer text)
- quantity: Number of items
- product_name: Product name/title

Respond with a JSON object mapping column types to column names.
Only include columns you're confident about."""

        try:
            # Call AI (this is a simplified version - actual implementation depends on AI module)
            response = self.ai_analyzer._call_ai_for_analysis(prompt)
            
            if response:
                # Parse AI response
                ai_mapping = json.loads(response)
                
                # Validate AI suggestions
                validated_mapping = {}
                for col_type, col_name in ai_mapping.items():
                    if col_name in unmapped_cols and col_type not in existing_mapping:
                        validated_mapping[col_type] = col_name
                
                return validated_mapping
                
        except Exception as e:
            logger.error(f"AI column detection error: {e}")
        
        return {}
    
    def validate_mapping(self, df: pd.DataFrame, mapping: Dict[str, str]) -> Dict[str, Any]:
        """Validate the quality of column mapping"""
        
        validation_results = {
            'is_valid': True,
            'missing_required': [],
            'confidence_scores': {},
            'warnings': [],
            'suggestions': []
        }
        
        # Required columns for return analysis
        required_columns = ['order_id', 'return_reason', 'buyer_comment']
        
        # Check for required columns
        for req_col in required_columns:
            if req_col not in mapping:
                validation_results['missing_required'].append(req_col)
                validation_results['is_valid'] = False
                
                # Try to suggest alternatives
                for col in df.columns:
                    col_lower = str(col).lower()
                    if req_col.replace('_', '') in col_lower.replace(' ', '').replace('-', ''):
                        validation_results['suggestions'].append(
                            f"Consider mapping '{col}' to '{req_col}'"
                        )
        
        # Check data quality for mapped columns
        for col_type, col_name in mapping.items():
            if col_name in df.columns:
                non_null_pct = (df[col_name].notna().sum() / len(df)) * 100
                validation_results['confidence_scores'][col_type] = non_null_pct
                
                if non_null_pct < 30:
                    validation_results['warnings'].append(
                        f"Column '{col_name}' (mapped to {col_type}) has only {non_null_pct:.1f}% data"
                    )
                
                # Type-specific validation
                if col_type == 'order_id' and non_null_pct > 50:
                    # Check if values match Amazon order format
                    sample = df[col_name].dropna().astype(str).head(10)
                    valid_format = sample.str.match(r'^\d{3}-\d{7}-\d{7}$').sum()
                    if valid_format < len(sample) * 0.5:
                        validation_results['warnings'].append(
                            f"Order IDs in '{col_name}' don't match Amazon format"
                        )
                
                elif col_type == 'asin' and non_null_pct > 50:
                    # Check ASIN format
                    sample = df[col_name].dropna().astype(str).head(10)
                    valid_format = sample.str.match(r'^B[A-Z0-9]{9}$').sum()
                    if valid_format < len(sample) * 0.5:
                        validation_results['warnings'].append(
                            f"ASINs in '{col_name}' don't match standard format"
                        )
        
        return validation_results
    
    def map_dataframe(self, df: pd.DataFrame, column_mapping: Dict[str, str]) -> pd.DataFrame:
        """Apply column mapping to create standardized dataframe"""
        
        # Create new dataframe with standardized columns
        mapped_data = {}
        
        # Map existing columns
        for col_type, col_name in column_mapping.items():
            if col_name in df.columns:
                mapped_data[col_type] = df[col_name]
        
        # Add any unmapped columns with original names
        for col in df.columns:
            if col not in column_mapping.values():
                mapped_data[f'unmapped_{col}'] = df[col]
        
        # Create standardized dataframe
        result_df = pd.DataFrame(mapped_data)
        
        # Ensure required columns exist (even if empty)
        required_cols = ['order_id', 'asin', 'sku', 'return_date', 
                        'return_reason', 'buyer_comment']
        
        for col in required_cols:
            if col not in result_df.columns:
                result_df[col] = ''
        
        # Clean and standardize data
        result_df = self._clean_mapped_data(result_df)
        
        return result_df
    
    def _clean_mapped_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and standardize mapped data"""
        
        # Clean order IDs
        if 'order_id' in df.columns:
            df['order_id'] = df['order_id'].astype(str).str.strip()
        
        # Clean ASINs
        if 'asin' in df.columns:
            df['asin'] = df['asin'].apply(self._extract_asin)
        
        # Parse dates
        if 'return_date' in df.columns:
            try:
                df['return_date'] = pd.to_datetime(df['return_date'], errors='coerce')
            except:
                pass
        
        # Clean text fields
        text_fields = ['return_reason', 'buyer_comment', 'product_name']
        for field in text_fields:
            if field in df.columns:
                df[field] = df[field].fillna('').astype(str).str.strip()
        
        return df
    
    def _extract_asin(self, value: Any) -> str:
        """Extract ASIN from value"""
        if pd.isna(value):
            return ''
        
        value_str = str(value).strip()
        
        # Look for ASIN pattern
        asin_match = re.search(r'\b(B[A-Z0-9]{9})\b', value_str)
        if asin_match:
            return asin_match.group(1)
        
        # If it's exactly 10 alphanumeric chars starting with B
        if re.match(r'^B[A-Z0-9]{9}$', value_str):
            return value_str
        
        return value_str if len(value_str) == 10 else ''
    
    def suggest_mapping_improvements(self, df: pd.DataFrame, 
                                   current_mapping: Dict[str, str]) -> List[str]:
        """Suggest improvements to current mapping"""
        
        suggestions = []
        
        # Check for potential date columns
        for col in df.columns:
            if col not in current_mapping.values():
                # Check if column contains dates
                try:
                    parsed = pd.to_datetime(df[col], errors='coerce')
                    if parsed.notna().sum() > len(df) * 0.5:
                        suggestions.append(f"Column '{col}' appears to contain dates")
                except:
                    pass
        
        # Check for potential comment columns (long text)
        for col in df.columns:
            if col not in current_mapping.values():
                non_null = df[col].dropna()
                if len(non_null) > 0:
                    avg_length = non_null.astype(str).str.len().mean()
                    if avg_length > 50:
                        suggestions.append(f"Column '{col}' contains long text (avg {avg_length:.0f} chars)")
        
        return suggestions

# Export the class
__all__ = ['SmartColumnMapper']
