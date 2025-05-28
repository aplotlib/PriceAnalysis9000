"""
Amazon Marketplace File Detection and Processing Module
Handles Reimbursements, FBA Returns, and FBM Returns files
Version: 1.0 - Complete Implementation
"""

import pandas as pd
import logging
from typing import Dict, Any, Optional, Tuple, List
import chardet
from io import StringIO
import re
from datetime import datetime

logger = logging.getLogger(__name__)

class AmazonFileDetector:
    """Detect and process Amazon marketplace data files"""
    
    # Expected column structures for each file type
    REIMBURSEMENT_COLS = [
        'approval-date', 'reimbursement-id', 'case-id', 'amazon-order-id', 
        'reason', 'sku', 'fnsku', 'asin', 'product-name', 'condition', 
        'currency-unit', 'amount-per-unit', 'amount-total', 
        'quantity-reimbursed-cash', 'quantity-reimbursed-inventory', 
        'quantity-reimbursed-total', 'original-reimbursement-id', 
        'original-reimbursement-type'
    ]
    
    FBA_RETURN_COLS = [
        'return-date', 'order-id', 'sku', 'asin', 'fnsku', 'product-name', 
        'quantity', 'fulfillment-center-id', 'detailed-disposition', 
        'reason', 'status', 'license-plate-number', 'customer-comments'
    ]
    
    FBM_RETURN_COLS = [
        'Order ID', 'Order date', 'Return request date', 'Return request status',
        'Amazon RMA ID', 'Merchant RMA ID', 'Label type', 'Label cost',
        'Currency code', 'Return carrier', 'Tracking ID', 'Label to be paid by',
        'A-to-Z Claim', 'Is prime', 'ASIN', 'Merchant SKU', 'Item Name',
        'Return quantity', 'Return Reason', 'In policy', 'Return type',
        'Resolution', 'Invoice number', 'Return delivery date', 'Order Amount',
        'Order quantity', 'SafeT Action reason', 'SafeT claim id',
        'SafeT claim state', 'SafeT claim creation time',
        'SafeT claim reimbursement amount', 'Refunded Amount'
    ]
    
    @staticmethod
    def detect_file_type(df: pd.DataFrame, filename: str = "") -> Optional[str]:
        """
        Detect file type based on structure, not filename
        
        Args:
            df: DataFrame to analyze
            filename: Original filename (optional, for logging)
            
        Returns:
            File type: 'reimbursements', 'fba_returns', 'fbm_returns', or None
        """
        
        # Get column count and names
        col_count = len(df.columns)
        columns = df.columns.tolist()
        columns_lower = [col.lower() for col in columns]
        
        logger.info(f"Detecting file type for {filename}: {col_count} columns")
        
        # Check for exact column count matches first
        if col_count == 18:
            # Check for reimbursement-specific columns
            if any('reimbursement' in col for col in columns_lower):
                logger.info("Detected as reimbursements file (18 columns with 'reimbursement')")
                return 'reimbursements'
                
        elif col_count == 13:
            # Check for FBA return-specific columns
            if any('fulfillment-center' in col for col in columns_lower):
                logger.info("Detected as FBA returns file (13 columns with 'fulfillment-center')")
                return 'fba_returns'
                
        elif col_count == 34:
            # Check for FBM return-specific columns
            if any('safet' in col for col in columns_lower):
                logger.info("Detected as FBM returns file (34 columns with 'safet')")
                return 'fbm_returns'
        
        # Fallback: Check for key column combinations
        if 'reimbursement-id' in columns_lower or 'quantity-reimbursed-cash' in columns_lower:
            logger.info("Detected as reimbursements file (key columns found)")
            return 'reimbursements'
        elif 'fulfillment-center-id' in columns_lower and 'license-plate-number' in columns_lower:
            logger.info("Detected as FBA returns file (key columns found)")
            return 'fba_returns'
        elif 'Order ID' in columns and 'SafeT claim id' in columns:
            logger.info("Detected as FBM returns file (key columns found)")
            return 'fbm_returns'
        
        # Additional checks based on content patterns
        if 'Return Reason' in columns or 'return-date' in columns_lower:
            if col_count > 20:
                logger.info("Detected as FBM returns file (many columns with return data)")
                return 'fbm_returns'
            else:
                logger.info("Detected as FBA returns file (fewer columns with return data)")
                return 'fba_returns'
        
        logger.warning(f"Could not determine file type for {filename}")
        return None
    
    @staticmethod
    def read_file_with_encoding(file_content: bytes, delimiter: str = ',') -> pd.DataFrame:
        """
        Read file with automatic encoding detection
        
        Args:
            file_content: Raw file bytes
            delimiter: CSV delimiter (default comma, but can be tab)
            
        Returns:
            DataFrame with properly decoded content
        """
        
        # Detect encoding
        detection = chardet.detect(file_content)
        encoding = detection['encoding'] or 'utf-8'
        confidence = detection.get('confidence', 0)
        
        logger.info(f"Detected encoding: {encoding} (confidence: {confidence:.2f})")
        
        # Try multiple encodings if the detected one fails
        encodings_to_try = [encoding, 'utf-8', 'cp1252', 'latin-1', 'iso-8859-1', 'utf-16']
        # Remove duplicates while preserving order
        encodings_to_try = list(dict.fromkeys(encodings_to_try))
        
        for enc in encodings_to_try:
            try:
                text = file_content.decode(enc)
                # Try to read as CSV
                df = pd.read_csv(StringIO(text), delimiter=delimiter)
                logger.info(f"Successfully read file with {enc} encoding")
                return df
            except Exception as e:
                logger.debug(f"Failed to read with {enc}: {str(e)}")
                continue
        
        # If all fail, use errors='replace' to at least get something
        logger.warning("All encodings failed, using UTF-8 with error replacement")
        text = file_content.decode('utf-8', errors='replace')
        return pd.read_csv(StringIO(text), delimiter=delimiter)
    
    @staticmethod
    def standardize_columns(df: pd.DataFrame, file_type: str) -> pd.DataFrame:
        """
        Standardize column names for easier processing
        
        Args:
            df: DataFrame to standardize
            file_type: Type of file for specific mappings
            
        Returns:
            DataFrame with standardized column names
        """
        
        # Common mappings across all file types
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
        
        # File-specific standardizations
        if file_type == 'reimbursements':
            # Ensure numeric columns are properly typed
            numeric_cols = ['amount-per-unit', 'amount-total', 'quantity-reimbursed-cash',
                          'quantity-reimbursed-inventory', 'quantity-reimbursed-total']
            for col in numeric_cols:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                    
        elif file_type == 'fba_returns':
            if 'quantity' in df.columns:
                df['quantity'] = pd.to_numeric(df['quantity'], errors='coerce')
                
        elif file_type == 'fbm_returns':
            numeric_cols = ['Label cost', 'Order Amount', 'Return quantity', 
                          'SafeT claim reimbursement amount', 'Refunded Amount']
            for col in numeric_cols:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
        
        return df
    
    @staticmethod
    def process_dates(df: pd.DataFrame, file_type: str) -> pd.DataFrame:
        """
        Process and standardize date columns
        
        Args:
            df: DataFrame with date columns
            file_type: Type of file for specific date columns
            
        Returns:
            DataFrame with parsed date columns
        """
        
        date_columns = []
        
        if file_type == 'reimbursements':
            date_columns = ['approval-date']
        elif file_type == 'fba_returns':
            date_columns = ['return-date']
        elif file_type == 'fbm_returns':
            date_columns = ['Order date', 'Return request date', 'Return delivery date',
                          'SafeT claim creation time']
        
        for col in date_columns:
            if col in df.columns:
                # Try multiple date formats
                df[f'{col}_parsed'] = pd.to_datetime(df[col], errors='coerce')
                
                # If many failed, try additional formats
                if df[f'{col}_parsed'].isna().sum() > len(df) * 0.5:
                    # Try common Amazon date formats
                    for fmt in ['%d-%b-%Y', '%Y-%m-%d', '%m/%d/%Y', '%B %d, %Y']:
                        try:
                            df[f'{col}_parsed'] = pd.to_datetime(df[col], format=fmt, errors='coerce')
                            if df[f'{col}_parsed'].notna().sum() > len(df) * 0.5:
                                break
                        except:
                            continue
        
        return df
    
    @staticmethod
    def process_file(file_content: bytes, filename: str = "") -> Dict[str, Any]:
        """
        Process an uploaded file and return structured data
        
        Args:
            file_content: Raw file bytes
            filename: Original filename
            
        Returns:
            Dictionary with success status, file type, dataframe, and summary
        """
        
        try:
            # Try comma-separated first
            df = AmazonFileDetector.read_file_with_encoding(file_content, delimiter=',')
            
            # If very few columns, might be TSV
            if len(df.columns) < 5:
                logger.info("Few columns detected, trying tab delimiter")
                df = AmazonFileDetector.read_file_with_encoding(file_content, delimiter='\t')
            
            # Detect file type
            file_type = AmazonFileDetector.detect_file_type(df, filename)
            
            if not file_type:
                return {
                    'success': False,
                    'error': 'Could not determine file type. Please ensure you are uploading a valid Amazon marketplace file.'
                }
            
            # Standardize columns
            df = AmazonFileDetector.standardize_columns(df, file_type)
            
            # Process dates
            df = AmazonFileDetector.process_dates(df, file_type)
            
            # Get summary statistics
            summary = {
                'row_count': len(df),
                'file_type': file_type,
                'columns': df.columns.tolist(),
                'date_range': {}
            }
            
            # Type-specific processing and summary
            if file_type == 'reimbursements':
                summary['total_cash_reimbursed'] = df['amount-total'].sum() if 'amount-total' in df.columns else 0
                summary['unique_asins'] = df['asin'].nunique() if 'asin' in df else 0
                summary['unique_skus'] = df['sku'].nunique() if 'sku' in df else 0
                summary['reimbursement_reasons'] = df['reason'].value_counts().to_dict() if 'reason' in df else {}
                
                # Date range
                if 'approval-date_parsed' in df.columns:
                    valid_dates = df['approval-date_parsed'].dropna()
                    if len(valid_dates) > 0:
                        summary['date_range'] = {
                            'start': valid_dates.min().strftime('%Y-%m-%d'),
                            'end': valid_dates.max().strftime('%Y-%m-%d')
                        }
                
            elif file_type == 'fba_returns':
                summary['total_returns'] = df['quantity'].sum() if 'quantity' in df else len(df)
                summary['unique_asins'] = df['asin'].nunique() if 'asin' in df else 0
                summary['unique_skus'] = df['sku'].nunique() if 'sku' in df else 0
                summary['return_reasons'] = df['return_reason'].value_counts().to_dict() if 'return_reason' in df else {}
                summary['fulfillment_centers'] = df['fulfillment-center-id'].value_counts().to_dict() if 'fulfillment-center-id' in df else {}
                
                # Customer comments preview
                if 'customer-comments' in df.columns:
                    comments = df['customer-comments'].dropna()
                    if len(comments) > 0:
                        summary['sample_comments'] = comments.head(5).tolist()
                
                # Date range
                if 'return-date_parsed' in df.columns:
                    valid_dates = df['return-date_parsed'].dropna()
                    if len(valid_dates) > 0:
                        summary['date_range'] = {
                            'start': valid_dates.min().strftime('%Y-%m-%d'),
                            'end': valid_dates.max().strftime('%Y-%m-%d')
                        }
                
            elif file_type == 'fbm_returns':
                summary['total_returns'] = df['Return quantity'].sum() if 'Return quantity' in df else len(df)
                summary['unique_asins'] = df['asin'].nunique() if 'asin' in df else 0
                summary['unique_skus'] = df['sku'].nunique() if 'sku' in df else 0
                summary['return_reasons'] = df['return_reason'].value_counts().to_dict() if 'return_reason' in df else {}
                summary['a_to_z_claims'] = len(df[df['A-to-Z Claim'] == 'Y']) if 'A-to-Z Claim' in df else 0
                summary['total_refunded'] = df['Refunded Amount'].sum() if 'Refunded Amount' in df else 0
                summary['prime_orders'] = len(df[df['Is prime'] == 'Y']) if 'Is prime' in df else 0
                
                # SafeT claims summary
                if 'SafeT claim state' in df.columns:
                    safet_claims = df[df['SafeT claim state'].notna()]
                    if len(safet_claims) > 0:
                        summary['safet_claims'] = {
                            'count': len(safet_claims),
                            'states': safet_claims['SafeT claim state'].value_counts().to_dict(),
                            'total_amount': safet_claims['SafeT claim reimbursement amount'].sum() if 'SafeT claim reimbursement amount' in safet_claims else 0
                        }
                
                # Date range
                if 'Return request date_parsed' in df.columns:
                    valid_dates = df['Return request date_parsed'].dropna()
                    if len(valid_dates) > 0:
                        summary['date_range'] = {
                            'start': valid_dates.min().strftime('%Y-%m-%d'),
                            'end': valid_dates.max().strftime('%Y-%m-%d')
                        }
            
            logger.info(f"Successfully processed {file_type} file: {len(df)} rows")
            
            return {
                'success': True,
                'file_type': file_type,
                'dataframe': df,
                'summary': summary
            }
            
        except Exception as e:
            logger.error(f"Error processing file: {str(e)}")
            return {
                'success': False,
                'error': f'Error processing file: {str(e)}'
            }
    
    @staticmethod
    def correlate_with_asin(marketplace_data: Dict[str, pd.DataFrame], target_asin: str) -> Dict[str, Any]:
        """
        Correlate marketplace data with a specific ASIN
        
        Args:
            marketplace_data: Dictionary of dataframes by file type
            target_asin: ASIN to analyze
            
        Returns:
            Dictionary with correlation results and insights
        """
        
        correlation_results = {
            'target_asin': target_asin,
            'related_products': {},
            'return_patterns': {},
            'financial_impact': {},
            'time_series': {}
        }
        
        # Process reimbursements
        if 'reimbursements' in marketplace_data:
            df = marketplace_data['reimbursements']
            asin_data = df[df['asin'] == target_asin] if 'asin' in df.columns else pd.DataFrame()
            
            if len(asin_data) > 0:
                correlation_results['financial_impact']['reimbursements'] = {
                    'count': len(asin_data),
                    'total_amount': asin_data['amount-total'].sum() if 'amount-total' in asin_data else 0,
                    'avg_amount': asin_data['amount-total'].mean() if 'amount-total' in asin_data else 0,
                    'reasons': asin_data['reason'].value_counts().to_dict() if 'reason' in asin_data else {},
                    'by_type': {
                        'cash': asin_data['quantity-reimbursed-cash'].sum() if 'quantity-reimbursed-cash' in asin_data else 0,
                        'inventory': asin_data['quantity-reimbursed-inventory'].sum() if 'quantity-reimbursed-inventory' in asin_data else 0
                    }
                }
                
                # Time series for reimbursements
                if 'approval-date_parsed' in asin_data.columns:
                    asin_data['month'] = asin_data['approval-date_parsed'].dt.to_period('M')
                    monthly_reimb = asin_data.groupby('month')['amount-total'].agg(['sum', 'count'])
                    correlation_results['time_series']['reimbursements'] = {
                        'monthly_amount': monthly_reimb['sum'].to_dict(),
                        'monthly_count': monthly_reimb['count'].to_dict()
                    }
            
            # Find related products (same SKU prefix or similar names)
            if 'product_name' in df.columns and len(asin_data) > 0:
                target_name = asin_data.iloc[0]['product_name'] if 'product_name' in asin_data else ""
                if target_name:
                    # Extract key words from product name
                    name_parts = target_name.split()[:3]  # First 3 words
                    pattern = '|'.join(name_parts)
                    
                    # Find products with similar names
                    similar_products = df[df['product_name'].str.contains(pattern, case=False, na=False, regex=True)]
                    related_asins = similar_products['asin'].unique().tolist() if 'asin' in similar_products else []
                    
                    # Remove target ASIN from related
                    related_asins = [a for a in related_asins if a != target_asin]
                    
                    correlation_results['related_products']['reimbursements'] = related_asins[:10]  # Top 10
        
        # Process FBA returns
        if 'fba_returns' in marketplace_data:
            df = marketplace_data['fba_returns']
            asin_data = df[df['asin'] == target_asin] if 'asin' in df.columns else pd.DataFrame()
            
            if len(asin_data) > 0:
                correlation_results['return_patterns']['fba'] = {
                    'count': len(asin_data),
                    'quantity': asin_data['quantity'].sum() if 'quantity' in asin_data else len(asin_data),
                    'reasons': asin_data['return_reason'].value_counts().to_dict() if 'return_reason' in asin_data else {},
                    'by_status': asin_data['status'].value_counts().to_dict() if 'status' in asin_data else {},
                    'by_disposition': asin_data['detailed-disposition'].value_counts().to_dict() if 'detailed-disposition' in asin_data else {},
                    'customer_comments': asin_data['customer-comments'].dropna().tolist()[:20] if 'customer-comments' in asin_data else []
                }
                
                # Time series for returns
                if 'return-date_parsed' in asin_data.columns:
                    asin_data['month'] = asin_data['return-date_parsed'].dt.to_period('M')
                    monthly_returns = asin_data.groupby('month')['quantity'].agg(['sum', 'count'])
                    correlation_results['time_series']['fba_returns'] = {
                        'monthly_quantity': monthly_returns['sum'].to_dict(),
                        'monthly_count': monthly_returns['count'].to_dict()
                    }
                
                # Fulfillment center analysis
                if 'fulfillment-center-id' in asin_data.columns:
                    fc_analysis = asin_data['fulfillment-center-id'].value_counts().to_dict()
                    correlation_results['return_patterns']['fba']['by_fulfillment_center'] = fc_analysis
        
        # Process FBM returns
        if 'fbm_returns' in marketplace_data:
            df = marketplace_data['fbm_returns']
            asin_data = df[df['asin'] == target_asin] if 'asin' in df.columns else pd.DataFrame()
            
            if len(asin_data) > 0:
                correlation_results['return_patterns']['fbm'] = {
                    'count': len(asin_data),
                    'quantity': asin_data['Return quantity'].sum() if 'Return quantity' in asin_data else len(asin_data),
                    'reasons': asin_data['return_reason'].value_counts().to_dict() if 'return_reason' in asin_data else {},
                    'refund_amount': asin_data['Refunded Amount'].sum() if 'Refunded Amount' in asin_data else 0,
                    'a_to_z_claims': len(asin_data[asin_data['A-to-Z Claim'] == 'Y']) if 'A-to-Z Claim' in asin_data else 0,
                    'prime_returns': len(asin_data[asin_data['Is prime'] == 'Y']) if 'Is prime' in asin_data else 0,
                    'in_policy_rate': (len(asin_data[asin_data['In policy'] == 'Y']) / len(asin_data) * 100) if 'In policy' in asin_data and len(asin_data) > 0 else 0
                }
                
                # SafeT claims analysis
                if 'SafeT claim state' in asin_data.columns:
                    safet_claims = asin_data[asin_data['SafeT claim state'].notna()]
                    if len(safet_claims) > 0:
                        correlation_results['return_patterns']['fbm']['safet_claims'] = {
                            'count': len(safet_claims),
                            'states': safet_claims['SafeT claim state'].value_counts().to_dict(),
                            'total_amount': safet_claims['SafeT claim reimbursement amount'].sum() if 'SafeT claim reimbursement amount' in safet_claims else 0
                        }
                
                # Time series for FBM returns
                if 'Return request date_parsed' in asin_data.columns:
                    asin_data['month'] = asin_data['Return request date_parsed'].dt.to_period('M')
                    monthly_returns = asin_data.groupby('month').agg({
                        'Return quantity': 'sum',
                        'Refunded Amount': 'sum',
                        'order_id': 'count'
                    })
                    correlation_results['time_series']['fbm_returns'] = {
                        'monthly_quantity': monthly_returns['Return quantity'].to_dict() if 'Return quantity' in monthly_returns else {},
                        'monthly_refunds': monthly_returns['Refunded Amount'].to_dict() if 'Refunded Amount' in monthly_returns else {},
                        'monthly_count': monthly_returns['order_id'].to_dict()
                    }
        
        # Calculate overall metrics
        total_returns = 0
        total_return_cost = 0
        
        if 'fba' in correlation_results['return_patterns']:
            total_returns += correlation_results['return_patterns']['fba'].get('quantity', 0)
        
        if 'fbm' in correlation_results['return_patterns']:
            total_returns += correlation_results['return_patterns']['fbm'].get('quantity', 0)
            total_return_cost += correlation_results['return_patterns']['fbm'].get('refund_amount', 0)
        
        if 'reimbursements' in correlation_results['financial_impact']:
            total_return_cost += correlation_results['financial_impact']['reimbursements'].get('total_amount', 0)
        
        correlation_results['overall_metrics'] = {
            'total_returns': total_returns,
            'total_financial_impact': total_return_cost,
            'return_rate_estimate': None  # Would need sales data to calculate
        }
        
        logger.info(f"Correlation analysis complete for ASIN {target_asin}")
        
        return correlation_results
    
    @staticmethod
    def generate_insights(correlation_results: Dict[str, Any]) -> List[str]:
        """
        Generate actionable insights from correlation results
        
        Args:
            correlation_results: Results from correlate_with_asin
            
        Returns:
            List of insight strings
        """
        
        insights = []
        
        # Return pattern insights
        if 'return_patterns' in correlation_results:
            all_reasons = {}
            
            # Combine return reasons from all sources
            for source in ['fba', 'fbm']:
                if source in correlation_results['return_patterns']:
                    reasons = correlation_results['return_patterns'][source].get('reasons', {})
                    for reason, count in reasons.items():
                        all_reasons[reason] = all_reasons.get(reason, 0) + count
            
            if all_reasons:
                top_reason = max(all_reasons.items(), key=lambda x: x[1])
                insights.append(f"Top return reason: '{top_reason[0]}' ({top_reason[1]} returns)")
                
                # Calculate percentage
                total_returns = sum(all_reasons.values())
                if total_returns > 0:
                    top_percentage = (top_reason[1] / total_returns) * 100
                    insights.append(f"This accounts for {top_percentage:.1f}% of all returns")
        
        # Financial impact insights
        if 'overall_metrics' in correlation_results:
            impact = correlation_results['overall_metrics'].get('total_financial_impact', 0)
            if impact > 0:
                insights.append(f"Total financial impact from returns/reimbursements: ${impact:,.2f}")
        
        # A-to-Z claims insight
        a_to_z_total = 0
        if 'fbm' in correlation_results.get('return_patterns', {}):
            a_to_z_total = correlation_results['return_patterns']['fbm'].get('a_to_z_claims', 0)
        
        if a_to_z_total > 0:
            insights.append(f"⚠️ {a_to_z_total} A-to-Z claims detected - review customer service processes")
        
        # Customer comments insight
        all_comments = []
        if 'fba' in correlation_results.get('return_patterns', {}):
            all_comments.extend(correlation_results['return_patterns']['fba'].get('customer_comments', []))
        
        if all_comments:
            insights.append(f"Found {len(all_comments)} customer return comments for detailed analysis")
        
        # Time series insights
        if 'time_series' in correlation_results:
            # Check for trends
            for return_type in ['fba_returns', 'fbm_returns']:
                if return_type in correlation_results['time_series']:
                    monthly_data = correlation_results['time_series'][return_type].get('monthly_count', {})
                    if len(monthly_data) >= 3:
                        values = list(monthly_data.values())
                        if values[-1] > values[0] * 1.5:
                            insights.append(f"⚠️ {return_type.replace('_', ' ').title()} are trending upward")
        
        return insights

# Export the class
__all__ = ['AmazonFileDetector']
