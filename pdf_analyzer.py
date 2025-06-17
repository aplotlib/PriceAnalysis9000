"""
PDF Analyzer Module - Extract Amazon Return Data from PDFs
Focused on medical device returns and safety analysis
"""

import logging
import re
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import io

# Safe imports
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

logger = logging.getLogger(__name__)

@dataclass
class InjuryAnalysis:
    """Results from injury analysis"""
    total_injuries: int
    critical_cases: List[Dict[str, Any]]
    severity_breakdown: Dict[str, int]
    injury_types: Any  # Counter object
    risk_assessment: str
    recommendations: List[str]

class PDFAnalyzer:
    """Specialized PDF analyzer for Amazon return reports"""
    
    def __init__(self):
        self.has_pdfplumber = HAS_PDFPLUMBER
        self.has_pypdf2 = HAS_PYPDF2
        
        if not self.has_pdfplumber and not self.has_pypdf2:
            logger.warning("No PDF libraries available. Install pdfplumber or PyPDF2")
    
    def extract_returns_from_pdf(self, pdf_content: bytes, filename: str) -> Dict[str, Any]:
        """Extract return data from Amazon PDF"""
        
        if self.has_pdfplumber:
            return self._extract_with_pdfplumber(pdf_content, filename)
        elif self.has_pypdf2:
            return self._extract_with_pypdf2(pdf_content, filename)
        else:
            return {
                'error': 'No PDF processing library available',
                'returns': []
            }
    
    def _extract_with_pdfplumber(self, pdf_content: bytes, filename: str) -> Dict[str, Any]:
        """Extract using pdfplumber (preferred method)"""
        try:
            returns = []
            all_text = []
            
            with pdfplumber.open(io.BytesIO(pdf_content)) as pdf:
                for page_num, page in enumerate(pdf.pages):
                    # Extract text
                    text = page.extract_text()
                    if text:
                        all_text.append(text)
                        
                        # Extract returns from this page
                        page_returns = self._parse_amazon_return_text(text, page_num)
                        returns.extend(page_returns)
                    
                    # Also try table extraction
                    tables = page.extract_tables()
                    for table in tables:
                        table_returns = self._parse_amazon_return_table(table, page_num)
                        returns.extend(table_returns)
            
            # If no structured returns found, try pattern matching on all text
            if not returns and all_text:
                full_text = '\n'.join(all_text)
                returns = self._extract_returns_from_text(full_text)
            
            return {
                'filename': filename,
                'pages': len(all_text),
                'returns': returns,
                'raw_text': '\n'.join(all_text)[:5000],  # First 5000 chars for reference
                'extraction_method': 'pdfplumber'
            }
            
        except Exception as e:
            logger.error(f"PDFPlumber extraction error: {e}")
            return {
                'error': str(e),
                'returns': []
            }
    
    def _extract_with_pypdf2(self, pdf_content: bytes, filename: str) -> Dict[str, Any]:
        """Extract using PyPDF2 (fallback method)"""
        try:
            returns = []
            all_text = []
            
            pdf_reader = PyPDF2.PdfReader(io.BytesIO(pdf_content))
            
            for page_num, page in enumerate(pdf_reader.pages):
                text = page.extract_text()
                if text:
                    all_text.append(text)
                    
                    # Extract returns from this page
                    page_returns = self._parse_amazon_return_text(text, page_num)
                    returns.extend(page_returns)
            
            # If no structured returns found, try pattern matching
            if not returns and all_text:
                full_text = '\n'.join(all_text)
                returns = self._extract_returns_from_text(full_text)
            
            return {
                'filename': filename,
                'pages': len(pdf_reader.pages),
                'returns': returns,
                'raw_text': '\n'.join(all_text)[:5000],
                'extraction_method': 'pypdf2'
            }
            
        except Exception as e:
            logger.error(f"PyPDF2 extraction error: {e}")
            return {
                'error': str(e),
                'returns': []
            }
    
    def _parse_amazon_return_text(self, text: str, page_num: int) -> List[Dict[str, Any]]:
        """Parse Amazon return data from text"""
        returns = []
        
        # Common patterns in Amazon return PDFs
        # Pattern for order IDs (format: 123-1234567-1234567)
        order_pattern = r'\b(\d{3}-\d{7}-\d{7})\b'
        
        # Pattern for ASINs (10 alphanumeric characters)
        asin_pattern = r'\b([A-Z0-9]{10})\b'
        
        # Pattern for dates
        date_patterns = [
            r'(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})',
            r'(\w+ \d{1,2}, \d{4})',
            r'(\d{4}-\d{2}-\d{2})'
        ]
        
        # Split text into potential return entries
        lines = text.split('\n')
        
        current_return = {}
        capture_comment = False
        comment_lines = []
        
        for line in lines:
            line = line.strip()
            
            # Skip empty lines
            if not line:
                if capture_comment and comment_lines:
                    current_return['customer_comment'] = ' '.join(comment_lines)
                    comment_lines = []
                    capture_comment = False
                continue
            
            # Check for order ID
            order_match = re.search(order_pattern, line)
            if order_match:
                # Save previous return if exists
                if current_return and 'order_id' in current_return:
                    returns.append(current_return)
                
                current_return = {
                    'order_id': order_match.group(1),
                    'page': page_num
                }
                capture_comment = False
                comment_lines = []
            
            # Check for ASIN
            asin_match = re.search(asin_pattern, line)
            if asin_match and current_return:
                # Verify it's likely an ASIN (not just any 10-char string)
                potential_asin = asin_match.group(1)
                if potential_asin.startswith('B'):  # Most ASINs start with B
                    current_return['asin'] = potential_asin
            
            # Check for dates
            for date_pattern in date_patterns:
                date_match = re.search(date_pattern, line)
                if date_match and current_return:
                    try:
                        # Try to parse the date
                        date_str = date_match.group(1)
                        current_return['return_date'] = date_str
                        break
                    except:
                        pass
            
            # Look for SKU patterns
            if 'sku' not in current_return and current_return:
                # Common SKU patterns
                sku_patterns = [
                    r'SKU[:\s]+([A-Z0-9\-]+)',
                    r'\b([A-Z]{2,4}[\-\d]+)\b',  # Like MOB-001, LVA1004
                ]
                for sku_pattern in sku_patterns:
                    sku_match = re.search(sku_pattern, line)
                    if sku_match:
                        current_return['sku'] = sku_match.group(1)
                        break
            
            # Capture return reasons and comments
            reason_indicators = [
                'return reason', 'reason:', 'customer comment', 'comment:',
                'feedback:', 'issue:', 'problem:', 'complaint:'
            ]
            
            if any(indicator in line.lower() for indicator in reason_indicators):
                capture_comment = True
                # Check if reason is on same line
                for indicator in reason_indicators:
                    if indicator in line.lower():
                        parts = line.lower().split(indicator, 1)
                        if len(parts) > 1 and parts[1].strip():
                            comment_lines.append(parts[1].strip())
                        break
            elif capture_comment:
                # Continue capturing multi-line comments
                comment_lines.append(line)
        
        # Don't forget the last return
        if current_return and 'order_id' in current_return:
            if comment_lines:
                current_return['customer_comment'] = ' '.join(comment_lines)
            returns.append(current_return)
        
        return returns
    
    def _parse_amazon_return_table(self, table: List[List[str]], page_num: int) -> List[Dict[str, Any]]:
        """Parse returns from table data"""
        returns = []
        
        if not table or len(table) < 2:
            return returns
        
        # Try to identify headers
        headers = [str(h).lower().strip() for h in table[0]]
        
        # Map headers to our fields
        field_mapping = {
            'order': 'order_id',
            'order id': 'order_id',
            'order-id': 'order_id',
            'asin': 'asin',
            'sku': 'sku',
            'date': 'return_date',
            'return date': 'return_date',
            'reason': 'return_reason',
            'comment': 'customer_comment',
            'customer comment': 'customer_comment',
            'feedback': 'customer_comment'
        }
        
        # Find column indices
        column_map = {}
        for idx, header in enumerate(headers):
            for key, field in field_mapping.items():
                if key in header:
                    column_map[field] = idx
                    break
        
        # Extract data rows
        for row in table[1:]:
            if not row or all(not cell for cell in row):
                continue
            
            return_entry = {'page': page_num}
            
            for field, idx in column_map.items():
                if idx < len(row) and row[idx]:
                    return_entry[field] = str(row[idx]).strip()
            
            # Only add if we have at least an order ID or substantial data
            if 'order_id' in return_entry or len(return_entry) > 2:
                returns.append(return_entry)
        
        return returns
    
    def _extract_returns_from_text(self, text: str) -> List[Dict[str, Any]]:
        """Extract returns using pattern matching as last resort"""
        returns = []
        
        # Split by potential return boundaries
        # Look for order ID patterns as boundaries
        order_pattern = r'(\d{3}-\d{7}-\d{7})'
        
        segments = re.split(order_pattern, text)
        
        for i in range(1, len(segments), 2):
            if i + 1 < len(segments):
                order_id = segments[i]
                content = segments[i + 1]
                
                return_entry = {
                    'order_id': order_id,
                    'raw_content': content[:500]  # First 500 chars
                }
                
                # Try to extract ASIN
                asin_match = re.search(r'\b(B[A-Z0-9]{9})\b', content)
                if asin_match:
                    return_entry['asin'] = asin_match.group(1)
                
                # Try to extract date
                date_match = re.search(r'(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})', content)
                if date_match:
                    return_entry['return_date'] = date_match.group(1)
                
                # Extract any text that might be a comment
                # Look for sentences that contain complaint-like words
                complaint_keywords = ['defect', 'broken', 'hurt', 'injured', 'pain', 
                                    'hospital', 'bleeding', 'dangerous', 'unsafe']
                
                sentences = re.split(r'[.!?]', content)
                relevant_sentences = []
                
                for sentence in sentences:
                    if any(keyword in sentence.lower() for keyword in complaint_keywords):
                        relevant_sentences.append(sentence.strip())
                
                if relevant_sentences:
                    return_entry['customer_comment'] = ' '.join(relevant_sentences)
                elif len(content) < 500:
                    # If short enough, just use the whole content
                    return_entry['customer_comment'] = content.strip()
                
                returns.append(return_entry)
        
        return returns
    
    def enhance_with_ai(self, returns: List[Dict], ai_client=None) -> List[Dict]:
        """Enhance extracted data with AI if available"""
        # This would be implemented if AI enhancement is needed
        # For now, returns data as-is
        return returns

# Export classes
__all__ = ['PDFAnalyzer', 'InjuryAnalysis']
