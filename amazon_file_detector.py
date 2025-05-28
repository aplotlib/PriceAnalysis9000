"""
Amazon Marketplace File Detection and Processing Module
Version: 2.0 - Enhanced with Better Error Handling and User Feedback
"""

import pandas as pd
import logging
from typing import Dict, Any, Optional, Tuple, List, Set
import chardet
from io import StringIO
import re
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class AmazonFileDetector:
    """Enhanced detector for Amazon marketplace data files"""
    
    # Expected column patterns for each file type
    REIMBURSEMENT_PATTERNS = {
        'required': ['reimbursement-id', 'amount-total', 'quantity-reimbursed'],
        'optional': ['approval-date', 'case-id', 'amazon-order-id', 'reason', 'sku', 
                    'fnsku', 'asin', 'product-name', 'condition', 'currency-unit',
                    'amount-per-unit', 'quantity-reimbursed-cash', 
                    'quantity-reimbursed-inventory', 'original-reimbursement-id']
    }
    
    FBA_RETURN_PATTERNS = {
        'required': ['return-date', 'quantity', 'reason'],
        'optional': ['order-id', 'sku', 'asin', 'fnsku', 'product-name',
                    'fulfillment-center-id', 'detailed-disposition', 'status',
                    'license-plate-number', 'customer-comments']
    }
    
    FBM_RETURN_PATTERNS = {
        'required': ['Order ID', 'Return quantity', 'Return Reason'],
        'optional': ['Order date', 'Return request date', 'Return request status',
                    'Amazon RMA ID', 'Merchant RMA ID', 'Label type', 'Label cost',
                    'Currency code', 'Return carrier', 'Tracking ID', 
                    'A-to-Z Claim', 'Is prime', 'ASIN', 'Merchant SKU', 
                    'Item Name', 'In policy', 'Return type', 'Resolution',
                    'SafeT claim id', 'SafeT claim state', 'Refunded Amount']
    }
    
    # Common return reasons for validation
    COMMON_RETURN_REASONS = {
        'DEFECTIVE', 'DAMAGED_BY_CARRIER', 'CUSTOMER_DAMAGED', 'WRONG_ITEM',
        'SWITCHEROO', 'UNWANTED', 'NOT_AS_DESCRIBED', 'QUALITY_ISSUE',
        'MISSING_PARTS', 'EXPIRED', 'NO_REASON', 'OTHER'
    }
    
    @staticmethod
    def detect_delimiter(content: str, filename: str = "") -> str:
        """Intelligently detect the delimiter"""
        # Sample first few lines
        lines = content.split('\n')[:5]
        sample = '\n'.join(lines)
        
        # Count occurrences
        delimiters = {
            ',': sample.count(','),
            '\t': sample.count('\t'),
            '|': sample.count('|'),
            ';': sample.count(';')
        }
        
        # Choose delimiter with most consistent count
        if delimiters['\t'] > delimiters[','] and filename.lower().endswith(('.tsv', '.txt')):
            return '\t'
        
        return max(delimiters.items(), key=lambda x: x[1])[0]
    
    @staticmethod
    def normalize_column_names(columns: List[str]) -> List[str]:
        """Normalize column names for consistent detection"""
        return [col.strip().lower().replace(' ', '-').replace('_', '-') for col in columns]
    
    @staticmethod
    def calculate_match_score(df_columns: List[str], pattern_columns: Dict[str, List[str]]) -> Tuple[float, List[str]]:
        """Calculate how well columns match expected patterns"""
        normalized_df_cols = set(AmazonFileDetector.normalize_column_names(df_columns))
        
        required_matches = 0
        optional_matches = 0
        missing_required = []
        
        # Check required columns
        for col in pattern_columns['required']:
            if col in normalized_df_cols:
                required_matches += 1
            else:
                missing_required.append(col)
        
        # Check optional columns
        for col in pattern_columns['optional']:
            if col in normalized_df_cols:
                optional_matches += 1
        
        # Calculate score
        if len(pattern_columns['required']) > 0:
            required_score = required_matches / len(pattern_columns['required'])
        else:
            required_score = 1.0
            
        optional_score = optional_matches / len(pattern_columns['optional']) if pattern_columns['optional'] else 0
        
        # Weight required columns more heavily
        total_score = (required_score * 0.8) + (optional_score * 0.2)
        
        return total_score, missing_required
    
    @staticmethod
    def detect_file_type(df: pd.DataFrame, filename: str = "") -> Tuple[Optional[str], float, str]:
        """
        Enhanced file type detection with confidence score
        
        Returns:
            Tuple of (file_type, confidence_score, detection_method)
        """
        columns = df.columns.tolist()
        normalized_cols = AmazonFileDetector.normalize_column_names(columns)
        col_count = len(columns)
        
        logger.info(f"Detecting file type: {col_count} columns, filename: {filename}")
        
        # Calculate match scores for each file type
        scores = {}
        missing_cols = {}
        
        # Check reimbursements
        reimb_score, reimb_missing = AmazonFileDetector.calculate_match_score(
            columns, AmazonFileDetector.REIMBURSEMENT_PATTERNS
        )
        scores['reimbursements'] = reimb_score
        missing_cols['reimbursements'] = reimb_missing
        
        # Check FBA returns
        fba_score, fba_missing = AmazonFileDetector.calculate_match_score(
            columns, AmazonFileDetector.FBA_RETURN_PATTERNS
        )
        scores['fba_returns'] = fba_score
        missing_cols['fba_returns'] = fba_missing
        
        # Check FBM returns
        fbm_score, fbm_missing = AmazonFileDetector.calculate_match_score(
            columns, AmazonFileDetector.FBM_RETURN_PATTERNS
        )
        scores['fbm_returns'] = fbm_score
        missing_cols['fbm_returns'] = fbm_missing
        
        # Get best match
        best_type = max(scores.items(), key=lambda x: x[1])
        file_type, confidence = best_type
        
        # Log detection results
        logger.info(f"Detection scores: {scores}")
        logger.info(f"Best match: {file_type} with confidence {confidence:.2f}")
        
        # Require minimum confidence
        if confidence < 0.5:
            logger.warning(f"Low confidence detection: {confidence:.2f}")
            return None, confidence, "low_confidence"
        
        # Additional validation based on content
        detection_method = "pattern_matching"
        
        # Check for specific indicators
        if 'reimbursement-id' in normalized_cols and confidence > 0.7:
            return 'reimbursements', confidence, detection_method
        elif 'fulfillment-center-id' in normalized_cols and 'return-date' in normalized_cols:
            return 'fba_returns', max(confidence, 0.8), detection_method
        elif 'safet-claim-id' in normalized_cols or 'Order ID' in columns:
            return 'fbm_returns', max(confidence, 0.8), detection_method
        
        # Return best match if confidence is sufficient
        if confidence >= 0.6:
            return file_type, confidence, detection_method
        
        return None, confidence, "insufficient_match"
    
    @staticmethod
    def read_file_with_encoding(file_content: bytes, delimiter: str = None) -> Tuple[pd.DataFrame, str, str]:
        """
        Enhanced file reading with better encoding detection
        
        Returns:
            Tuple of (dataframe, detected_encoding, detected_delimiter)
        """
        # Detect encoding with larger sample
        sample_size = min(len(file_content), 10000)
        detection = chardet.detect(file_content[:sample_size])
        encoding = detection['encoding'] or 'utf-8'
        confidence = detection.get('confidence', 0)
        
        logger.info(f"Detected encoding: {encoding} (confidence: {confidence:.2f})")
        
        # Try multiple encodings
        encodings_to_try = [encoding]
        if encoding.lower() != 'utf-8':
            encodings_to_try.append('utf-8')
        encodings_to_try.extend(['cp1252', 'latin-1', 'iso-8859-1'])
        
        # Remove duplicates
        encodings_to_try = list(dict.fromkeys(encodings_to_try))
        
        last_error = None
        for enc in encodings_to_try:
            try:
                text = file_content.decode(enc)
                
                # Detect delimiter if not provided
                if delimiter is None:
                    delimiter = AmazonFileDetector.detect_delimiter(text)
                    logger.info(f"Detected delimiter: {repr(delimiter)}")
                
                # Read CSV with error handling
                df = pd.read_csv(
                    StringIO(text), 
                    delimiter=delimiter,
                    low_memory=False,
                    na_values=['', 'NA', 'N/A', 'null', 'NULL'],
                    keep_default_na=True
                )
                
                # Validate dataframe
                if len(df.columns) > 1 and len(df) > 0:
                    logger.info(f"Successfully read file with {enc} encoding: {len(df)} rows, {len(df.columns)} columns")
                    return df, enc, delimiter
                    
            except Exception as e:
                last_error = e
                logger.debug(f"Failed with {enc}: {str(e)}")
                continue
        
        # Last resort: try with error handling
        logger.warning("All encodings failed, using UTF-8 with error replacement")
        text = file_content.decode('utf-8', errors='replace')
        delimiter = delimiter or AmazonFileDetector.detect_delimiter(text)
        df = pd.read_csv(StringIO(text), delimiter=delimiter, low_memory=False)
        
        return df, 'utf-8 (with replacements)', delimiter
    
    @staticmethod
    def validate_and_clean_data(df: pd.DataFrame, file_type: str) -> Tuple[pd.DataFrame, List[str]]:
        """
        Validate and clean data with detailed warnings
        
        Returns:
            Tuple of (cleaned_dataframe, list_of_warnings)
        """
        warnings_list = []
        original_rows = len(df)
        
        # Remove completely empty rows
        df = df.dropna(how='all')
        if len(df) < original_rows:
            warnings_list.append(f"Removed {original_rows - len(df)} empty rows")
        
        # Type-specific validation
        if file_type == 'reimbursements':
            # Ensure numeric columns
            numeric_cols = ['amount-total', 'amount-per-unit', 'quantity-reimbursed-cash',
                          'quantity-reimbursed-inventory', 'quantity-reimbursed-total']
            
            for col in numeric_cols:
                if col in df.columns:
                    try:
                        df[col] = pd.to_numeric(df[col], errors='coerce')
                        invalid_count = df[col].isna().sum() - df[col].isna().sum()
                        if invalid_count > 0:
                            warnings_list.append(f"Found {invalid_count} invalid values in {col}")
                    except Exception as e:
                        warnings_list.append(f"Could not convert {col} to numeric: {str(e)}")
        
        elif file_type == 'fba_returns':
            # Validate quantity
            if 'quantity' in df.columns:
                df['quantity'] = pd.to_numeric(df['quantity'], errors='coerce')
                df['quantity'] = df['quantity'].fillna(1)  # Default to 1 if missing
            
            # Validate return reasons
            if 'reason' in df.columns:
                invalid_reasons = df[~df['reason'].str.upper().isin(AmazonFileDetector.COMMON_RETURN_REASONS)]
                if len(invalid_reasons) > 0:
                    warnings_list.append(f"Found {len(invalid_reasons)} non-standard return reasons")
        
        elif file_type == 'fbm_returns':
            # Validate return quantity
            if 'Return quantity' in df.columns:
                df['Return quantity'] = pd.to_numeric(df['Return quantity'], errors='coerce')
                df['Return quantity'] = df['Return quantity'].fillna(1)
            
            # Validate monetary columns
            money_cols = ['Label cost', 'Order Amount', 'Refunded Amount', 
                         'SafeT claim reimbursement amount']
            for col in money_cols:
                if col in df.columns:
                    # Remove currency symbols
                    df[col] = df[col].astype(str).str.replace('[$,€£]', '', regex=True)
                    df[col] = pd.to_numeric(df[col], errors='coerce')
        
        return df, warnings_list
    
    @staticmethod
    def process_dates(df: pd.DataFrame, file_type: str) -> pd.DataFrame:
        """Enhanced date processing with multiple format support"""
        date_columns = {
            'reimbursements': ['approval-date'],
            'fba_returns': ['return-date'],
            'fbm_returns': ['Order date', 'Return request date', 'Return delivery date', 
                          'SafeT claim creation time']
        }
        
        date_formats = [
            '%Y-%m-%d',
            '%m/%d/%Y',
            '%d/%m/%Y',
            '%B %d, %Y',
            '%b %d, %Y',
            '%d-%b-%Y',
            '%Y-%m-%d %H:%M:%S',
            '%m/%d/%Y %H:%M:%S'
        ]
        
        columns_to_parse = date_columns.get(file_type, [])
        
        for col in columns_to_parse:
            if col in df.columns:
                # Try pandas automatic parsing first
                df[f'{col}_parsed'] = pd.to_datetime(df[col], errors='coerce')
                
                # If many failed, try specific formats
                failed_count = df[f'{col}_parsed'].isna().sum()
                total_count = len(df[col].dropna())
                
                if failed_count > total_count * 0.5 and total_count > 0:
                    logger.info(f"Many dates failed parsing in {col}, trying specific formats")
                    
                    for fmt in date_formats:
                        try:
                            temp_parsed = pd.to_datetime(df[col], format=fmt, errors='coerce')
                            # Update only the previously failed ones
                            mask = df[f'{col}_parsed'].isna() & temp_parsed.notna()
                            df.loc[mask, f'{col}_parsed'] = temp_parsed[mask]
                            
                            if df[f'{col}_parsed'].notna().sum() > total_count * 0.8:
                                break
                        except:
                            continue
                
                # Log results
                success_rate = (df[f'{col}_parsed'].notna().sum() / total_count * 100) if total_count > 0 else 0
                logger.info(f"Date parsing for {col}: {success_rate:.1f}% success rate")
        
        return df
    
    @staticmethod
    def generate_summary_statistics(df: pd.DataFrame, file_type: str) -> Dict[str, Any]:
        """Generate comprehensive summary statistics"""
        summary = {
            'row_count': len(df),
            'file_type': file_type,
            'columns': df.columns.tolist(),
            'date_range': {},
            'data_quality': {
                'missing_values': {},
                'completeness_score': 0
            }
        }
        
        # Calculate completeness
        total_cells = len(df) * len(df.columns)
        missing_cells = df.isna().sum().sum()
        summary['data_quality']['completeness_score'] = round((1 - missing_cells / total_cells) * 100, 1)
        
        # Column-specific missing values
        for col in df.columns:
            missing_pct = (df[col].isna().sum() / len(df) * 100)
            if missing_pct > 0:
                summary['data_quality']['missing_values'][col] = round(missing_pct, 1)
        
        # Type-specific summaries
        if file_type == 'reimbursements':
            if 'amount-total' in df.columns:
                summary['financial_summary'] = {
                    'total_amount': df['amount-total'].sum(),
                    'average_amount': df['amount-total'].mean(),
                    'max_amount': df['amount-total'].max(),
                    'transactions': len(df)
                }
            
            if 'reason' in df.columns:
                summary['top_reasons'] = df['reason'].value_counts().head(5).to_dict()
            
            if 'asin' in df.columns:
                summary['unique_asins'] = df['asin'].nunique()
                summary['top_asins'] = df['asin'].value_counts().head(5).to_dict()
        
        elif file_type == 'fba_returns':
            if 'quantity' in df.columns:
                summary['return_summary'] = {
                    'total_units': df['quantity'].sum(),
                    'total_orders': len(df),
                    'avg_units_per_return': df['quantity'].mean()
                }
            
            if 'reason' in df.columns:
                summary['return_reasons'] = df['reason'].value_counts().to_dict()
            
            if 'fulfillment-center-id' in df.columns:
                summary['by_fulfillment_center'] = df['fulfillment-center-id'].value_counts().head(10).to_dict()
            
            if 'customer-comments' in df.columns:
                comments = df['customer-comments'].dropna()
                summary['customer_feedback'] = {
                    'total_comments': len(comments),
                    'sample_comments': comments.head(5).tolist() if len(comments) > 0 else []
                }
        
        elif file_type == 'fbm_returns':
            if 'Return quantity' in df.columns:
                summary['return_summary'] = {
                    'total_units': df['Return quantity'].sum(),
                    'total_orders': len(df)
                }
            
            if 'Refunded Amount' in df.columns:
                summary['financial_summary'] = {
                    'total_refunded': df['Refunded Amount'].sum(),
                    'average_refund': df['Refunded Amount'].mean()
                }
            
            if 'A-to-Z Claim' in df.columns:
                summary['a_to_z_claims'] = {
                    'total': len(df[df['A-to-Z Claim'] == 'Y']),
                    'percentage': round(len(df[df['A-to-Z Claim'] == 'Y']) / len(df) * 100, 1)
                }
            
            if 'Is prime' in df.columns:
                summary['prime_orders'] = {
                    'total': len(df[df['Is prime'] == 'Y']),
                    'percentage': round(len(df[df['Is prime'] == 'Y']) / len(df) * 100, 1)
                }
        
        # Add date ranges
        for col in df.columns:
            if col.endswith('_parsed') and df[col].notna().any():
                summary['date_range'][col.replace('_parsed', '')] = {
                    'start': df[col].min().strftime('%Y-%m-%d') if pd.notna(df[col].min()) else None,
                    'end': df[col].max().strftime('%Y-%m-%d') if pd.notna(df[col].max()) else None
                }
        
        return summary
    
    @staticmethod
    def process_file(file_content: bytes, filename: str = "") -> Dict[str, Any]:
        """
        Main processing function with enhanced error handling and feedback
        """
        try:
            # Step 1: Read file
            df, encoding, delimiter = AmazonFileDetector.read_file_with_encoding(file_content)
            
            if df.empty:
                return {
                    'success': False,
                    'error': 'File appears to be empty or could not be read properly'
                }
            
            # Step 2: Detect file type
            file_type, confidence, method = AmazonFileDetector.detect_file_type(df, filename)
            
            if not file_type:
                # Provide helpful error message
                column_list = ", ".join(df.columns[:10])
                if len(df.columns) > 10:
                    column_list += f"... ({len(df.columns)} total)"
                
                return {
                    'success': False,
                    'error': f'Could not determine file type. Found {len(df.columns)} columns: {column_list}',
                    'hint': 'Please ensure this is a valid Amazon seller report (Reimbursements, FBA Returns, or FBM Returns)',
                    'confidence': confidence
                }
            
            # Step 3: Standardize columns
            df = AmazonFileDetector.standardize_columns(df, file_type)
            
            # Step 4: Validate and clean
            df, warnings = AmazonFileDetector.validate_and_clean_data(df, file_type)
            
            # Step 5: Process dates
            df = AmazonFileDetector.process_dates(df, file_type)
            
            # Step 6: Generate summary
            summary = AmazonFileDetector.generate_summary_statistics(df, file_type)
            
            # Add processing metadata
            summary['processing_info'] = {
                'filename': filename,
                'encoding': encoding,
                'delimiter': repr(delimiter),
                'detection_confidence': round(confidence, 2),
                'detection_method': method,
                'warnings': warnings
            }
            
            logger.info(f"Successfully processed {file_type} file: {len(df)} rows")
            
            return {
                'success': True,
                'file_type': file_type,
                'dataframe': df,
                'summary': summary
            }
            
        except Exception as e:
            logger.error(f"Error processing file: {str(e)}", exc_info=True)
            return {
                'success': False,
                'error': f'Error processing file: {str(e)}',
                'hint': 'Please check that the file is not corrupted and is in CSV/TSV format'
            }
    
    @staticmethod
    def standardize_columns(df: pd.DataFrame, file_type: str) -> pd.DataFrame:
        """Enhanced column standardization"""
        # Common mappings
        common_mappings = {
            'amazon-order-id': 'order_id',
            'order-id': 'order_id',
            'Order ID': 'order_id',
            'sku': 'sku',
            'Merchant SKU': 'sku',
            'asin': 'asin',
            'ASIN': 'asin',
            'product-name': 'product_name',
            'Item Name': 'product_name',
            'reason': 'return_reason',
            'Return Reason': 'return_reason'
        }
        
        # Apply mappings
        df = df.rename(columns=common_mappings)
        
        # Ensure critical columns exist
        if file_type == 'reimbursements':
            if 'amount-total' not in df.columns and 'amount-per-unit' in df.columns and 'quantity-reimbursed-total' in df.columns:
                df['amount-total'] = df['amount-per-unit'] * df['quantity-reimbursed-total']
        
        return df
    
    @staticmethod
    def correlate_with_asin(marketplace_data: Dict[str, pd.DataFrame], 
                           target_asin: str,
                           include_related: bool = True) -> Dict[str, Any]:
        """
        Enhanced ASIN correlation with better insights
        """
        correlation_results = {
            'target_asin': target_asin,
            'found_data': False,
            'related_products': {},
            'return_patterns': {},
            'financial_impact': {},
            'time_series': {},
            'insights': []
        }
        
        # Validate ASIN format
        if not target_asin or not re.match(r'^[A-Z0-9]{10}$', target_asin.upper()):
            correlation_results['error'] = 'Invalid ASIN format'
            return correlation_results
        
        target_asin = target_asin.upper()
        
        # Process each data source
        for data_type, df in marketplace_data.items():
            if df is None or df.empty:
                continue
            
            # Find ASIN column
            asin_col = None
            for col in ['asin', 'ASIN']:
                if col in df.columns:
                    asin_col = col
                    break
            
            if not asin_col:
                logger.warning(f"No ASIN column found in {data_type}")
                continue
            
            # Filter for target ASIN
            asin_data = df[df[asin_col] == target_asin]
            
            if len(asin_data) > 0:
                correlation_results['found_data'] = True
                
                # Process based on type
                if data_type == 'reimbursements':
                    correlation_results['financial_impact']['reimbursements'] = {
                        'count': len(asin_data),
                        'total_amount': asin_data['amount-total'].sum() if 'amount-total' in asin_data else 0,
                        'reasons': asin_data['reason'].value_counts().head(5).to_dict() if 'reason' in asin_data else {}
                    }
                
                elif data_type == 'fba_returns':
                    correlation_results['return_patterns']['fba'] = {
                        'count': len(asin_data),
                        'quantity': asin_data['quantity'].sum() if 'quantity' in asin_data else len(asin_data),
                        'reasons': asin_data['return_reason'].value_counts().to_dict() if 'return_reason' in asin_data else {},
                        'customer_comments': asin_data['customer-comments'].dropna().tolist()[:10] if 'customer-comments' in asin_data else []
                    }
                
                elif data_type == 'fbm_returns':
                    correlation_results['return_patterns']['fbm'] = {
                        'count': len(asin_data),
                        'quantity': asin_data['Return quantity'].sum() if 'Return quantity' in asin_data else len(asin_data),
                        'reasons': asin_data['return_reason'].value_counts().to_dict() if 'return_reason' in asin_data else {},
                        'refund_amount': asin_data['Refunded Amount'].sum() if 'Refunded Amount' in asin_data else 0
                    }
            
            # Find related products if requested
            if include_related and 'product_name' in df.columns and len(asin_data) > 0:
                try:
                    # Get target product name
                    target_name = asin_data.iloc[0]['product_name']
                    if pd.notna(target_name):
                        # Find similar products
                        name_parts = str(target_name).split()[:3]
                        pattern = '|'.join([re.escape(part) for part in name_parts if len(part) > 3])
                        
                        if pattern:
                            similar = df[df['product_name'].str.contains(pattern, case=False, na=False, regex=True)]
                            related_asins = similar[asin_col].unique()
                            related_asins = [a for a in related_asins if a != target_asin and pd.notna(a)][:5]
                            
                            if related_asins:
                                correlation_results['related_products'][data_type] = related_asins
                except Exception as e:
                    logger.warning(f"Error finding related products: {e}")
        
        # Generate insights
        if correlation_results['found_data']:
            insights = AmazonFileDetector.generate_correlation_insights(correlation_results)
            correlation_results['insights'] = insights
        
        return correlation_results
    
    @staticmethod
    def generate_correlation_insights(correlation_results: Dict[str, Any]) -> List[str]:
        """Generate actionable insights from correlation data"""
        insights = []
        
        # Return rate insights
        total_returns = 0
        return_reasons = {}
        
        for source in ['fba', 'fbm']:
            if source in correlation_results.get('return_patterns', {}):
                data = correlation_results['return_patterns'][source]
                total_returns += data.get('quantity', 0)
                
                # Aggregate reasons
                for reason, count in data.get('reasons', {}).items():
                    return_reasons[reason] = return_reasons.get(reason, 0) + count
        
        if total_returns > 0:
            insights.append(f"📦 Total returns: {total_returns} units")
            
            if return_reasons:
                top_reason = max(return_reasons.items(), key=lambda x: x[1])
                reason_pct = (top_reason[1] / sum(return_reasons.values()) * 100)
                insights.append(f"🎯 Top return reason: '{top_reason[0]}' ({reason_pct:.0f}% of returns)")
        
        # Financial insights
        if 'reimbursements' in correlation_results.get('financial_impact', {}):
            reimb_data = correlation_results['financial_impact']['reimbursements']
            if reimb_data['count'] > 0:
                insights.append(f"💰 Reimbursements: {reimb_data['count']} cases, ${reimb_data['total_amount']:.2f} total")
        
        # Customer feedback insights
        all_comments = []
        if 'fba' in correlation_results.get('return_patterns', {}):
            all_comments.extend(correlation_results['return_patterns']['fba'].get('customer_comments', []))
        
        if all_comments:
            insights.append(f"💬 {len(all_comments)} customer comments available for review")
        
        # Quality indicators
        if total_returns > 50:
            insights.append("⚠️ High return volume - consider quality review")
        
        return insights

# Export the enhanced class
__all__ = ['AmazonFileDetector']
