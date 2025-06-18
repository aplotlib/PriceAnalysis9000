"""
amazon_pdf_parser.py - Specialized parser for Amazon Seller Central PDFs
Handles the specific format of Amazon return exports
"""

import re
import pandas as pd
import logging
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)

class AmazonPDFParser:
    """Parse Amazon Seller Central return PDFs"""
    
    def __init__(self):
        # Amazon-specific patterns
        self.ORDER_PATTERN = r'(\d{3}-\d{7}-\d{7})'
        self.ASIN_PATTERN = r'(B[A-Z0-9]{9})'
        self.DATE_PATTERN = r'(\d{1,2}/\d{1,2}/\d{2,4})'
        
        # Return reason codes from Amazon
        self.REASON_CODES = {
            'DEFECTIVE': 'Product Defective/Does not work',
            'DAMAGED': 'Product/Package damaged',
            'MISSING_PARTS': 'Missing parts or accessories',
            'NOT_AS_DESCRIBED': 'Not as described on website',
            'WRONG_ITEM': 'Received wrong item',
            'QUALITY': 'Quality not as expected',
            'INCOMPATIBLE': 'Not compatible',
            'PERFORMANCE': 'Did not perform as expected',
            'NOT_NEEDED': 'No longer needed',
            'UNWANTED': 'Bought by mistake',
            'UNAUTHORIZED': 'Unauthorized purchase',
            'INACCURATE': 'Inaccurate website description',
            'BETTER_PRICE': 'Found better price elsewhere'
        }
    
    def parse_pdf_text(self, text: str) -> List[Dict[str, Any]]:
        """Parse PDF text and extract returns"""
        returns = []
        
        # Split by order IDs to segment returns
        order_segments = re.split(f'({self.ORDER_PATTERN})', text)
        
        # Process each segment
        for i in range(1, len(order_segments), 2):
            if i + 1 < len(order_segments):
                order_id = order_segments[i]
                segment_text = order_segments[i + 1]
                
                return_data = self._extract_return_data(order_id, segment_text)
                if return_data:
                    returns.append(return_data)
        
        return returns
    
    def _extract_return_data(self, order_id: str, text: str) -> Optional[Dict[str, Any]]:
        """Extract return data from a text segment"""
        return_data = {
            'order-id': order_id,
            'asin': '',
            'sku': '',
            'product-name': '',
            'reason': '',
            'customer-comments': '',
            'return-date': '',
            'quantity': 1
        }
        
        # Extract ASIN
        asin_match = re.search(self.ASIN_PATTERN, text)
        if asin_match:
            return_data['asin'] = asin_match.group(1)
        
        # Extract date
        date_match = re.search(self.DATE_PATTERN, text)
        if date_match:
            return_data['return-date'] = date_match.group(1)
        
        # Extract SKU (usually after "SKU:" or similar)
        sku_match = re.search(r'SKU[:\s]+([A-Z0-9\-_]+)', text, re.IGNORECASE)
        if sku_match:
            return_data['sku'] = sku_match.group(1)
        
        # Extract product name (usually between ASIN and return reason)
        lines = text.split('\n')
        for i, line in enumerate(lines):
            line = line.strip()
            
            # Look for product name patterns
            if self.ASIN_PATTERN in line and i + 1 < len(lines):
                # Next line might be product name
                next_line = lines[i + 1].strip()
                if next_line and not any(pattern in next_line.lower() for pattern in ['reason', 'comment', 'return']):
                    return_data['product-name'] = next_line
            
            # Look for return reason
            if 'return reason' in line.lower() or 'reason:' in line.lower():
                # Extract reason
                if ':' in line:
                    reason = line.split(':', 1)[1].strip()
                    return_data['reason'] = self._normalize_reason(reason)
                elif i + 1 < len(lines):
                    return_data['reason'] = self._normalize_reason(lines[i + 1].strip())
            
            # Look for customer comments
            if any(phrase in line.lower() for phrase in ['customer comment', 'buyer comment', 'comment:']):
                # Extract comment
                if ':' in line:
                    comment = line.split(':', 1)[1].strip()
                    return_data['customer-comments'] = comment
                elif i + 1 < len(lines):
                    # Collect multi-line comments
                    comment_lines = []
                    for j in range(i + 1, min(i + 5, len(lines))):
                        next_line = lines[j].strip()
                        if next_line and not any(label in next_line.lower() for label in ['reason', 'asin', 'order']):
                            comment_lines.append(next_line)
                        else:
                            break
                    if comment_lines:
                        return_data['customer-comments'] = ' '.join(comment_lines)
        
        return return_data if return_data['asin'] or return_data['reason'] else None
    
    def _normalize_reason(self, reason: str) -> str:
        """Normalize return reason"""
        reason = reason.strip()
        
        # Check if it's a known code
        for code, description in self.REASON_CODES.items():
            if code in reason.upper():
                return description
        
        return reason
    
    def parse_pdf_tables(self, tables: List[List[List[str]]]) -> pd.DataFrame:
        """Parse tables extracted from PDF"""
        all_returns = []
        
        for table in tables:
            if not table or len(table) < 2:
                continue
            
            # Process as Amazon return table
            df = self._process_return_table(table)
            if df is not None and not df.empty:
                all_returns.append(df)
        
        if all_returns:
            return pd.concat(all_returns, ignore_index=True)
        
        return pd.DataFrame()
    
    def _process_return_table(self, table: List[List[str]]) -> Optional[pd.DataFrame]:
        """Process a single return table"""
        try:
            # Find header row
            header_idx = 0
            for i, row in enumerate(table):
                if any('order' in str(cell).lower() for cell in row):
                    header_idx = i
                    break
            
            if header_idx >= len(table) - 1:
                return None
            
            # Extract headers and data
            headers = [str(h).strip() if h else f"Col_{i}" for i, h in enumerate(table[header_idx])]
            data_rows = table[header_idx + 1:]
            
            # Create DataFrame
            df = pd.DataFrame(data_rows, columns=headers)
            
            # Standardize column names
            column_mapping = {
                'Order ID': 'order-id',
                'Order': 'order-id',
                'ASIN': 'asin',
                'SKU': 'sku',
                'Product': 'product-name',
                'Product Name': 'product-name',
                'Return Reason': 'reason',
                'Reason': 'reason',
                'Customer Comments': 'customer-comments',
                'Comments': 'customer-comments',
                'Return Date': 'return-date',
                'Date': 'return-date',
                'Quantity': 'quantity',
                'Qty': 'quantity'
            }
            
            # Rename columns
            rename_dict = {}
            for col in df.columns:
                for standard, target in column_mapping.items():
                    if standard.lower() in col.lower():
                        rename_dict[col] = target
                        break
            
            if rename_dict:
                df = df.rename(columns=rename_dict)
            
            # Clean data
            df = df.dropna(how='all')
            df = df[df.astype(str).ne('').any(axis=1)]
            
            return df if not df.empty else None
            
        except Exception as e:
            logger.error(f"Error processing return table: {e}")
            return None
    
    def combine_results(self, text_returns: List[Dict], table_df: pd.DataFrame) -> pd.DataFrame:
        """Combine text and table extraction results"""
        # Convert text returns to DataFrame
        if text_returns:
            text_df = pd.DataFrame(text_returns)
        else:
            text_df = pd.DataFrame()
        
        # Combine both sources
        if not table_df.empty and not text_df.empty:
            # Merge on order ID to avoid duplicates
            combined = pd.concat([table_df, text_df], ignore_index=True)
            
            # Remove duplicates based on order ID
            if 'order-id' in combined.columns:
                combined = combined.drop_duplicates(subset=['order-id'], keep='first')
            
            return combined
        elif not table_df.empty:
            return table_df
        elif not text_df.empty:
            return text_df
        else:
            return pd.DataFrame()

# Export the parser
__all__ = ['AmazonPDFParser']
