"""
PDF Analyzer Module - Extract Amazon Return Data from PDFs
Specialized for medical device returns from Amazon Seller Central
"""

import logging
import re
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
import io
import json

# Optional imports with fallbacks
try:
    import pdfplumber
    HAS_PDFPLUMBER = True
except ImportError:
    HAS_PDFPLUMBER = False

try:
    import PyPDF2
    HAS_PYPDF2 = True  
except ImportError:
    HAS_PYPDF2 = False

try:
    from PIL import Image
    import pytesseract
    HAS_OCR = True
except ImportError:
    HAS_OCR = False

logger = logging.getLogger(__name__)

class PDFAnalyzer:
    """Extract and analyze Amazon return data from PDF files"""
    
    def __init__(self):
        """Initialize PDF analyzer with available libraries"""
        self.has_pdfplumber = HAS_PDFPLUMBER
        self.has_pypdf2 = HAS_PYPDF2
        self.has_ocr = HAS_OCR
        
        if not self.has_pdfplumber and not self.has_pypdf2:
            logger.error("No PDF processing libraries available. Install pdfplumber or PyPDF2")
        
        # Amazon-specific patterns
        self.PATTERNS = {
            'order_id': r'\b(\d{3}-\d{7}-\d{7})\b',
            'asin': r'\b(B[A-Z0-9]{9})\b',
            'sku': r'SKU[:\s]+([A-Z0-9\-_]+)',
            'date': [
                r'(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})',
                r'(\w+ \d{1,2}, \d{4})',
                r'(\d{4}-\d{2}-\d{2})'
            ],
            'return_reasons': [
                'defective', 'broken', 'damaged', 'wrong item', 'not as described',
                'missing parts', 'doesn\'t work', 'poor quality', 'uncomfortable',
                'too small', 'too large', 'doesn\'t fit', 'incompatible',
                'changed mind', 'no longer needed', 'ordered by mistake'
            ]
        }
        
        # Common section headers in Amazon PDFs
        self.SECTION_HEADERS = [
            'return request', 'return reason', 'customer comments',
            'buyer comments', 'return details', 'order details',
            'product information', 'reason for return'
        ]
    
    def extract_returns_from_pdf(self, pdf_content: bytes, filename: str) -> Dict[str, Any]:
        """
        Extract return data from Amazon PDF
        
        Args:
            pdf_content: PDF file content as bytes
            filename: Name of the PDF file
            
        Returns:
            Dictionary containing extracted returns and metadata
        """
        
        if self.has_pdfplumber:
            return self._extract_with_pdfplumber(pdf_content, filename)
        elif self.has_pypdf2:
            return self._extract_with_pypdf2(pdf_content, filename)
        else:
            return {
                'error': 'No PDF processing library available',
                'returns': [],
                'filename': filename
            }
    
    def _extract_with_pdfplumber(self, pdf_content: bytes, filename: str) -> Dict[str, Any]:
        """Extract using pdfplumber (preferred method)"""
        try:
            returns = []
            all_text = []
            extracted_tables = []
            
            with pdfplumber.open(io.BytesIO(pdf_content)) as pdf:
                total_pages = len(pdf.pages)
                
                for page_num, page in enumerate(pdf.pages):
                    # Extract text
                    text = page.extract_text()
                    if text:
                        all_text.append(text)
                        
                        # Try to extract structured returns
                        page_returns = self._parse_amazon_return_text(text, page_num)
                        returns.extend(page_returns)
                    
                    # Extract tables
                    tables = page.extract_tables()
                    for table_idx, table in enumerate(tables):
                        if table and len(table) > 1:  # Has header and data
                            extracted_tables.append({
                                'page': page_num,
                                'table_index': table_idx,
                                'data': table
                            })
                            
                            # Parse table for returns
                            table_returns = self._parse_amazon_return_table(table, page_num)
                            returns.extend(table_returns)
            
            # If no structured returns found, try comprehensive parsing
            if not returns and all_text:
                full_text = '\n'.join(all_text)
                returns = self._comprehensive_text_extraction(full_text)
            
            # Deduplicate returns
            returns = self._deduplicate_returns(returns)
            
            return {
                'filename': filename,
                'pages': total_pages,
                'returns': returns,
                'tables_found': len(extracted_tables),
                'raw_text_preview': '\n'.join(all_text)[:2000] if all_text else '',
                'extraction_method': 'pdfplumber',
                'success': True
            }
            
        except Exception as e:
            logger.error(f"PDFPlumber extraction error: {e}")
            return {
                'error': str(e),
                'returns': [],
                'filename': filename,
                'success': False
            }
    
    def _extract_with_pypdf2(self, pdf_content: bytes, filename: str) -> Dict[str, Any]:
        """Extract using PyPDF2 (fallback method)"""
        try:
            returns = []
            all_text = []
            
            pdf_reader = PyPDF2.PdfReader(io.BytesIO(pdf_content))
            total_pages = len(pdf_reader.pages)
            
            for page_num, page in enumerate(pdf_reader.pages):
                text = page.extract_text()
                if text:
                    all_text.append(text)
                    
                    # Extract returns from this page
                    page_returns = self._parse_amazon_return_text(text, page_num)
                    returns.extend(page_returns)
            
            # Comprehensive parsing if needed
            if not returns and all_text:
                full_text = '\n'.join(all_text)
                returns = self._comprehensive_text_extraction(full_text)
            
            # Deduplicate
            returns = self._deduplicate_returns(returns)
            
            return {
                'filename': filename,
                'pages': total_pages,
                'returns': returns,
                'raw_text_preview': '\n'.join(all_text)[:2000] if all_text else '',
                'extraction_method': 'pypdf2',
                'success': True
            }
            
        except Exception as e:
            logger.error(f"PyPDF2 extraction error: {e}")
            return {
                'error': str(e),
                'returns': [],
                'filename': filename,
                'success': False
            }
    
    def _parse_amazon_return_text(self, text: str, page_num: int) -> List[Dict[str, Any]]:
        """Parse return data from text using Amazon-specific patterns"""
        returns = []
        
        # Normalize text
        text = re.sub(r'\s+', ' ', text)
        
        # Find all order IDs as anchors
        order_matches = list(re.finditer(self.PATTERNS['order_id'], text))
        
        for i, match in enumerate(order_matches):
            order_id = match.group(1)
            start_pos = match.start()
            
            # Determine end position (next order or end of text)
            end_pos = order_matches[i + 1].start() if i + 1 < len(order_matches) else len(text)
            
            # Extract section for this order
            order_section = text[start_pos:end_pos]
            
            return_entry = {
                'order_id': order_id,
                'page': page_num,
                'source': 'text_parsing'
            }
            
            # Extract ASIN
            asin_match = re.search(self.PATTERNS['asin'], order_section)
            if asin_match:
                return_entry['asin'] = asin_match.group(1)
            
            # Extract dates
            for date_pattern in self.PATTERNS['date']:
                date_match = re.search(date_pattern, order_section)
                if date_match:
                    return_entry['return_date'] = date_match.group(1)
                    break
            
            # Extract SKU
            sku_match = re.search(self.PATTERNS['sku'], order_section)
            if sku_match:
                return_entry['sku'] = sku_match.group(1)
            
            # Extract return reason and comments
            reason_comment = self._extract_reason_and_comment(order_section)
            return_entry.update(reason_comment)
            
            returns.append(return_entry)
        
        return returns
    
    def _parse_amazon_return_table(self, table: List[List[str]], page_num: int) -> List[Dict[str, Any]]:
        """Parse returns from table data"""
        if not table or len(table) < 2:
            return []
        
        returns = []
        
        # Normalize headers
        headers = [str(h).lower().strip() if h else '' for h in table[0]]
        
        # Map headers to fields
        column_map = self._map_table_columns(headers)
        
        # Extract data rows
        for row_idx, row in enumerate(table[1:]):
            if not row or all(not cell for cell in row):
                continue
            
            return_entry = {
                'page': page_num,
                'source': 'table',
                'row_index': row_idx
            }
            
            # Extract mapped fields
            for field, col_idx in column_map.items():
                if col_idx < len(row) and row[col_idx]:
                    value = str(row[col_idx]).strip()
                    if value and value.lower() not in ['n/a', 'none', '-']:
                        return_entry[field] = value
            
            # Only add if we have meaningful data
            if 'order_id' in return_entry or 'asin' in return_entry:
                returns.append(return_entry)
        
        return returns
    
    def _map_table_columns(self, headers: List[str]) -> Dict[str, int]:
        """Map table headers to standardized field names"""
        column_map = {}
        
        # Define header mappings
        header_mappings = {
            'order_id': ['order id', 'order-id', 'order number', 'order#'],
            'asin': ['asin', 'product asin'],
            'sku': ['sku', 'seller sku', 'merchant sku'],
            'return_date': ['return date', 'returned date', 'date returned', 'date'],
            'return_reason': ['return reason', 'reason', 'return type', 'issue'],
            'buyer_comment': ['customer comment', 'buyer comment', 'comment', 'feedback', 'notes'],
            'product_name': ['product name', 'product', 'item name', 'description'],
            'quantity': ['quantity', 'qty', 'units']
        }
        
        # Map headers
        for field, patterns in header_mappings.items():
            for idx, header in enumerate(headers):
                if header in patterns:
                    column_map[field] = idx
                    break
        
        return column_map
    
    def _extract_reason_and_comment(self, text: str) -> Dict[str, str]:
        """Extract return reason and customer comments from text"""
        result = {
            'return_reason': '',
            'buyer_comment': ''
        }
        
        # Look for section headers
        text_lower = text.lower()
        
        # Find return reason
        reason_markers = ['return reason:', 'reason for return:', 'reason:', 'issue:']
        for marker in reason_markers:
            if marker in text_lower:
                start = text_lower.find(marker) + len(marker)
                # Extract next 100 chars or until next section
                reason_text = text[start:start+100].strip()
                # Clean up - stop at next label or line break
                reason_text = re.split(r'[\n\r]|(?:[A-Z][a-z]*:)', reason_text)[0].strip()
                if reason_text:
                    result['return_reason'] = reason_text
                    break
        
        # Find customer comments
        comment_markers = ['customer comment:', 'buyer comment:', 'comment:', 'feedback:', 'notes:']
        for marker in comment_markers:
            if marker in text_lower:
                start = text_lower.find(marker) + len(marker)
                # Extract next 500 chars or until next section
                comment_text = text[start:start+500].strip()
                # Clean up
                comment_text = re.split(r'(?:[A-Z][a-z]*:)', comment_text)[0].strip()
                if comment_text:
                    result['buyer_comment'] = comment_text
                    break
        
        # If no structured comments found, look for complaint keywords
        if not result['buyer_comment']:
            complaint_keywords = ['broken', 'damaged', 'defective', 'hurt', 'injured', 
                                'pain', 'uncomfortable', 'wrong', 'missing', 'doesn\'t work']
            
            sentences = re.split(r'[.!?]', text)
            relevant_sentences = []
            
            for sentence in sentences:
                if any(keyword in sentence.lower() for keyword in complaint_keywords):
                    relevant_sentences.append(sentence.strip())
            
            if relevant_sentences:
                result['buyer_comment'] = ' '.join(relevant_sentences[:3])  # First 3 relevant sentences
        
        return result
    
    def _comprehensive_text_extraction(self, full_text: str) -> List[Dict[str, Any]]:
        """Comprehensive extraction when structured parsing fails"""
        returns = []
        
        # Split by order IDs
        order_sections = re.split(self.PATTERNS['order_id'], full_text)
        
        # Process each section
        for i in range(1, len(order_sections), 2):
            if i >= len(order_sections):
                break
                
            order_id = order_sections[i]
            content = order_sections[i + 1] if i + 1 < len(order_sections) else ''
            
            return_entry = {
                'order_id': order_id,
                'source': 'comprehensive_extraction'
            }
            
            # Extract all available data
            # ASIN
            asin_match = re.search(self.PATTERNS['asin'], content)
            if asin_match:
                return_entry['asin'] = asin_match.group(1)
            
            # Date
            for date_pattern in self.PATTERNS['date']:
                date_match = re.search(date_pattern, content)
                if date_match:
                    return_entry['return_date'] = date_match.group(1)
                    break
            
            # Return reason - look for keywords
            content_lower = content.lower()
            for reason in self.PATTERNS['return_reasons']:
                if reason in content_lower:
                    return_entry['return_reason'] = reason
                    break
            
            # Extract any substantial text as potential comment
            sentences = re.split(r'[.!?]', content)
            long_sentences = [s.strip() for s in sentences if len(s.strip()) > 20]
            if long_sentences:
                return_entry['buyer_comment'] = ' '.join(long_sentences[:2])
            
            returns.append(return_entry)
        
        return returns
    
    def _deduplicate_returns(self, returns: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Remove duplicate returns based on order ID"""
        seen_orders = set()
        unique_returns = []
        
        for return_item in returns:
            order_id = return_item.get('order_id', '')
            
            if order_id and order_id not in seen_orders:
                seen_orders.add(order_id)
                unique_returns.append(return_item)
            elif not order_id:
                # Keep returns without order IDs (might be valuable)
                unique_returns.append(return_item)
        
        return unique_returns
    
    def extract_with_ocr(self, pdf_content: bytes, filename: str) -> Dict[str, Any]:
        """Use OCR for scanned PDFs (if image processing is available)"""
        if not self.has_ocr:
            return {
                'error': 'OCR not available. Install Pillow and pytesseract.',
                'returns': [],
                'filename': filename
            }
        
        # This would implement OCR extraction for scanned PDFs
        # For now, return not implemented
        return {
            'error': 'OCR extraction not yet implemented',
            'returns': [],
            'filename': filename
        }

# Export class
__all__ = ['PDFAnalyzer']
