"""
simple_pdf_processor.py - Simple and robust PDF processor for Amazon returns
This handles problematic PDFs with duplicate columns or irregular formatting
"""

import pandas as pd
import logging
import re
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)

def process_amazon_pdf_simple(pdf_path) -> pd.DataFrame:
    """
    Simple processor that extracts key information from Amazon PDFs
    even if they have formatting issues
    """
    try:
        import pdfplumber
    except ImportError:
        raise ImportError("pdfplumber is required. Install with: pip install pdfplumber")
    
    all_returns = []
    
    # Patterns to look for
    order_pattern = r'(\d{3}-\d{7}-\d{7})'
    asin_pattern = r'(B[A-Z0-9]{9})'
    
    with pdfplumber.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf.pages):
            try:
                # Extract all text from page
                text = page.extract_text()
                if not text:
                    continue
                
                # Method 1: Try to extract from tables (if any)
                tables = page.extract_tables()
                for table in tables:
                    if table and len(table) > 1:
                        # Process each row (skip header)
                        for row_idx in range(1, len(table)):
                            row = table[row_idx]
                            if not row:
                                continue
                            
                            # Convert row to string for pattern matching
                            row_text = ' '.join([str(cell) if cell else '' for cell in row])
                            
                            # Look for order ID and ASIN
                            order_match = re.search(order_pattern, row_text)
                            asin_match = re.search(asin_pattern, row_text)
                            
                            if order_match or asin_match:
                                return_data = {
                                    'order-id': order_match.group(1) if order_match else '',
                                    'asin': asin_match.group(1) if asin_match else '',
                                    'row_data': row_text,
                                    'source': f'table_page_{page_num + 1}'
                                }
                                
                                # Try to extract reason and comments from the row
                                # Look for common keywords
                                reason_keywords = ['defect', 'broken', 'damaged', 'wrong', 'not working', 
                                                 'quality', 'size', 'fit', 'compatible']
                                
                                for keyword in reason_keywords:
                                    if keyword in row_text.lower():
                                        return_data['detected_issue'] = keyword
                                        break
                                
                                all_returns.append(return_data)
                
                # Method 2: Extract from text using patterns
                if not tables or len(tables) == 0:
                    # Split text into segments by order IDs
                    segments = re.split(f'({order_pattern})', text)
                    
                    for i in range(1, len(segments), 2):
                        if i + 1 < len(segments):
                            order_id = segments[i]
                            segment_text = segments[i + 1]
                            
                            # Look for ASIN in segment
                            asin_match = re.search(asin_pattern, segment_text)
                            
                            return_data = {
                                'order-id': order_id,
                                'asin': asin_match.group(1) if asin_match else '',
                                'segment_text': segment_text[:500],  # First 500 chars
                                'source': f'text_page_{page_num + 1}'
                            }
                            
                            # Extract potential reason/comment
                            lines = segment_text.split('\n')
                            for line in lines:
                                line_lower = line.lower()
                                if 'reason' in line_lower or 'comment' in line_lower:
                                    return_data['extracted_text'] = line.strip()
                                    break
                            
                            all_returns.append(return_data)
                
            except Exception as e:
                logger.error(f"Error processing page {page_num + 1}: {e}")
                continue
    
    # Convert to DataFrame
    if all_returns:
        df = pd.DataFrame(all_returns)
        
        # Remove duplicates
        if 'order-id' in df.columns:
            df = df.drop_duplicates(subset=['order-id'], keep='first')
        
        # Add placeholder columns if needed
        for col in ['reason', 'customer-comments', 'sku', 'product-name']:
            if col not in df.columns:
                df[col] = ''
        
        return df
    else:
        # Return empty DataFrame with expected columns
        return pd.DataFrame(columns=['order-id', 'asin', 'sku', 'product-name', 
                                    'reason', 'customer-comments'])

def fix_duplicate_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Fix DataFrame with duplicate column names"""
    # Get column names
    cols = df.columns.tolist()
    
    # Make unique
    seen = {}
    unique_cols = []
    
    for col in cols:
        col_str = str(col)
        if col_str in seen:
            seen[col_str] += 1
            unique_cols.append(f"{col_str}_{seen[col_str]}")
        else:
            seen[col_str] = 0
            unique_cols.append(col_str)
    
    # Rename columns
    df.columns = unique_cols
    
    return df

# Simple function to use in FileProcessor
def read_pdf_simple(file) -> pd.DataFrame:
    """Simple PDF reader that handles problematic files"""
    try:
        import pdfplumber
        
        all_data = []
        
        with pdfplumber.open(file) as pdf:
            for page in pdf.pages:
                # Extract tables with error handling
                try:
                    tables = page.extract_tables()
                    for table in tables:
                        if table and len(table) > 1:
                            # Create DataFrame safely
                            try:
                                # Use first row as headers
                                headers = [str(h) if h else f"Col_{i}" for i, h in enumerate(table[0])]
                                
                                # Ensure unique headers
                                seen = {}
                                unique_headers = []
                                for h in headers:
                                    if h in seen:
                                        seen[h] += 1
                                        unique_headers.append(f"{h}_{seen[h]}")
                                    else:
                                        seen[h] = 0
                                        unique_headers.append(h)
                                
                                # Create DataFrame
                                df = pd.DataFrame(table[1:], columns=unique_headers)
                                all_data.append(df)
                            except:
                                # If table parsing fails, skip
                                continue
                except:
                    # If table extraction fails, continue
                    continue
        
        if all_data:
            # Combine all data
            result = pd.concat(all_data, ignore_index=True, sort=False)
            return result
        else:
            return pd.DataFrame()
            
    except Exception as e:
        logger.error(f"Simple PDF reading failed: {e}")
        return pd.DataFrame()

# Export functions
__all__ = ['process_amazon_pdf_simple', 'fix_duplicate_columns', 'read_pdf_simple']
