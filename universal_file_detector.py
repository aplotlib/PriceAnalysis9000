"""
Universal File Detection and Processing Module
Version: 3.0 - Handles PDF, Images, and all Amazon formats
"""

import pandas as pd
import logging
from typing import Dict, Any, Optional, Tuple, List, Union
import chardet
from io import StringIO, BytesIO
import re
from datetime import datetime
import json
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

@dataclass
class ProcessedFile:
    """Structured file processing result"""
    file_type: str
    format: str  # pdf, csv, image, etc.
    content_category: str  # returns, reviews, mixed, unknown
    data: pd.DataFrame = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    raw_content: Any = None
    confidence: float = 0.0
    extraction_method: str = ""
    warnings: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            'file_type': self.file_type,
            'format': self.format,
            'content_category': self.content_category,
            'data_shape': self.data.shape if self.data is not None else None,
            'metadata': self.metadata,
            'confidence': self.confidence,
            'extraction_method': self.extraction_method,
            'warnings': self.warnings
        }

class UniversalFileDetector:
    """Enhanced file detector for all Amazon marketplace formats"""
    
    # Amazon return report patterns
    RETURN_PATTERNS = {
        'manage_returns_pdf': {
            'indicators': ['booking id', 'return request', 'defect names', 'arrival time'],
            'confidence': 0.9
        },
        'fba_returns': {
            'columns': ['return-date', 'order-id', 'sku', 'asin', 'reason', 'quantity'],
            'confidence': 0.95
        },
        'seller_central_returns': {
            'indicators': ['order id', 'return reason', 'asin', 'refund amount'],
            'confidence': 0.85
        }
    }
    
    # Review file patterns
    REVIEW_PATTERNS = {
        'helium10': {
            'columns': ['Title', 'Body', 'Rating', 'Date', 'Verified'],
            'confidence': 0.95
        },
        'amazon_reviews': {
            'indicators': ['customer review', 'star rating', 'verified purchase'],
            'confidence': 0.85
        }
    }
    
    @staticmethod
    def detect_file_format(filename: str, content: bytes = None) -> str:
        """Detect file format from extension or content"""
        ext = filename.lower().split('.')[-1]
        
        # Direct extension mapping
        format_map = {
            'pdf': 'pdf',
            'csv': 'csv',
            'tsv': 'tsv',
            'txt': 'txt',
            'xlsx': 'excel',
            'xls': 'excel',
            'jpg': 'image',
            'jpeg': 'image',
            'png': 'image'
        }
        
        return format_map.get(ext, 'unknown')
    
    @staticmethod
    def process_file(file_content: bytes, filename: str, 
                    target_asin: str = None) -> ProcessedFile:
        """Main entry point for file processing"""
        
        format_type = UniversalFileDetector.detect_file_format(filename)
        
        try:
            if format_type == 'pdf':
                return UniversalFileDetector._process_pdf(file_content, filename)
            elif format_type == 'image':
                return UniversalFileDetector._process_image(file_content, filename)
            elif format_type in ['csv', 'tsv', 'txt']:
                return UniversalFileDetector._process_text_file(
                    file_content, filename, format_type
                )
            elif format_type == 'excel':
                return UniversalFileDetector._process_excel(file_content, filename)
            else:
                return ProcessedFile(
                    file_type='unknown',
                    format=format_type,
                    content_category='unknown',
                    warnings=['Unsupported file format']
                )
                
        except Exception as e:
            logger.error(f"File processing error: {e}")
            return ProcessedFile(
                file_type='error',
                format=format_type,
                content_category='error',
                warnings=[str(e)]
            )
    
    @staticmethod
    def _process_pdf(content: bytes, filename: str) -> ProcessedFile:
        """Process PDF files"""
        # This will be handled by the AI analyzer
        # Return a placeholder for now
        return ProcessedFile(
            file_type='pdf',
            format='pdf',
            content_category='pending_ai_analysis',
            raw_content=content,
            metadata={'filename': filename, 'size': len(content)},
            extraction_method='ai_required'
        )
    
    @staticmethod
    def _process_image(content: bytes, filename: str) -> ProcessedFile:
        """Process image files"""
        # Images will be processed by vision AI
        return ProcessedFile(
            file_type='image',
            format='image',
            content_category='pending_vision_analysis',
            raw_content=content,
            metadata={'filename': filename, 'size': len(content)},
            extraction_method='vision_ai_required'
        )
    
    @staticmethod
    def _process_text_file(content: bytes, filename: str, 
                          format_type: str) -> ProcessedFile:
        """Process text-based files (CSV, TSV, TXT)"""
        
        # Detect encoding
        encoding = UniversalFileDetector._detect_encoding(content)
        
        try:
            text = content.decode(encoding)
            
            # Determine delimiter
            if format_type == 'tsv':
                delimiter = '\t'
            elif format_type == 'csv':
                delimiter = ','
            else:
                # Try to detect delimiter for txt files
                delimiter = UniversalFileDetector._detect_delimiter(text)
            
            # Try to parse as structured data
            try:
                df = pd.read_csv(StringIO(text), delimiter=delimiter)
                
                # Detect content type
                content_category = UniversalFileDetector._detect_content_type(df)
                
                # Clean column names
                df.columns = [col.strip() for col in df.columns]
                
                return ProcessedFile(
                    file_type=content_category,
                    format=format_type,
                    content_category=content_category,
                    data=df,
                    metadata={
                        'filename': filename,
                        'encoding': encoding,
                        'delimiter': delimiter,
                        'columns': df.columns.tolist(),
                        'row_count': len(df)
                    },
                    confidence=0.9,
                    extraction_method='structured_parsing'
                )
                
            except Exception as parse_error:
                # If parsing fails, treat as unstructured text
                return ProcessedFile(
                    file_type='text',
                    format=format_type,
                    content_category='unstructured',
                    raw_content=text,
                    metadata={
                        'filename': filename,
                        'encoding': encoding,
                        'parse_error': str(parse_error)
                    },
                    extraction_method='text_extraction',
                    warnings=['Could not parse as structured data']
                )
                
        except Exception as e:
            logger.error(f"Text file processing error: {e}")
            return ProcessedFile(
                file_type='error',
                format=format_type,
                content_category='error',
                warnings=[f'Decoding error: {str(e)}']
            )
    
    @staticmethod
    def _process_excel(content: bytes, filename: str) -> ProcessedFile:
        """Process Excel files"""
        try:
            # Read all sheets
            excel_file = pd.ExcelFile(BytesIO(content))
            
            # If single sheet, process directly
            if len(excel_file.sheet_names) == 1:
                df = pd.read_excel(BytesIO(content))
                content_category = UniversalFileDetector._detect_content_type(df)
                
                return ProcessedFile(
                    file_type=content_category,
                    format='excel',
                    content_category=content_category,
                    data=df,
                    metadata={
                        'filename': filename,
                        'sheet_count': 1,
                        'columns': df.columns.tolist(),
                        'row_count': len(df)
                    },
                    confidence=0.9,
                    extraction_method='excel_parsing'
                )
            else:
                # Multiple sheets - need user guidance
                return ProcessedFile(
                    file_type='multi_sheet_excel',
                    format='excel',
                    content_category='needs_clarification',
                    raw_content=content,
                    metadata={
                        'filename': filename,
                        'sheet_names': excel_file.sheet_names,
                        'sheet_count': len(excel_file.sheet_names)
                    },
                    extraction_method='needs_user_input',
                    warnings=['Multiple sheets detected. Please specify which sheet to analyze.']
                )
                
        except Exception as e:
            logger.error(f"Excel processing error: {e}")
            return ProcessedFile(
                file_type='error',
                format='excel',
                content_category='error',
                warnings=[f'Excel processing error: {str(e)}']
            )
    
    @staticmethod
    def _detect_encoding(content: bytes) -> str:
        """Detect file encoding"""
        # Sample for detection
        sample_size = min(len(content), 10000)
        detection = chardet.detect(content[:sample_size])
        
        encoding = detection.get('encoding', 'utf-8')
        confidence = detection.get('confidence', 0)
        
        # Fallback for low confidence
        if confidence < 0.7:
            # Try common encodings
            for enc in ['utf-8', 'cp1252', 'latin-1']:
                try:
                    content.decode(enc)
                    return enc
                except:
                    continue
        
        return encoding or 'utf-8'
    
    @staticmethod
    def _detect_delimiter(text: str) -> str:
        """Detect delimiter in text"""
        # Sample first few lines
        lines = text.split('\n')[:5]
        sample = '\n'.join(lines)
        
        # Count occurrences
        delimiters = {
            ',': sample.count(','),
            '\t': sample.count('\t'),
            '|': sample.count('|'),
            ';': sample.count(';')
        }
        
        # Return most common
        return max(delimiters.items(), key=lambda x: x[1])[0]
    
    @staticmethod
    def _detect_content_type(df: pd.DataFrame) -> str:
        """Detect type of content in DataFrame"""
        columns_lower = [col.lower() for col in df.columns]
        
        # Check for return data patterns
        return_indicators = ['return', 'reason', 'refund', 'rma', 'defect']
        return_score = sum(1 for ind in return_indicators 
                         if any(ind in col for col in columns_lower))
        
        # Check for review data patterns
        review_indicators = ['rating', 'review', 'title', 'body', 'comment', 'feedback']
        review_score = sum(1 for ind in review_indicators 
                         if any(ind in col for col in columns_lower))
        
        # Check specific patterns
        if UniversalFileDetector._is_fba_return_report(df):
            return 'fba_returns'
        elif UniversalFileDetector._is_helium10_reviews(df):
            return 'helium10_reviews'
        elif return_score > review_score:
            return 'returns'
        elif review_score > return_score:
            return 'reviews'
        else:
            return 'data'
    
    @staticmethod
    def _is_fba_return_report(df: pd.DataFrame) -> bool:
        """Check if DataFrame is FBA return report"""
        expected = ['return-date', 'order-id', 'sku', 'asin', 'reason']
        columns_lower = [col.lower() for col in df.columns]
        matches = sum(1 for exp in expected if exp in columns_lower)
        return matches >= 3
    
    @staticmethod
    def _is_helium10_reviews(df: pd.DataFrame) -> bool:
        """Check if DataFrame is Helium 10 review export"""
        expected = ['Title', 'Body', 'Rating']
        matches = sum(1 for exp in expected if exp in df.columns)
        return matches == 3
    
    @staticmethod
    def filter_by_asin(df: pd.DataFrame, target_asin: str) -> pd.DataFrame:
        """Filter DataFrame by ASIN"""
        if not target_asin:
            return df
        
        # Find ASIN column
        asin_columns = [col for col in df.columns 
                       if 'asin' in col.lower()]
        
        if not asin_columns:
            return df
        
        # Filter by ASIN
        asin_col = asin_columns[0]
        return df[df[asin_col].str.upper() == target_asin.upper()]
    
    @staticmethod
    def extract_return_metrics(df: pd.DataFrame, 
                             date_column: str = None) -> Dict[str, Any]:
        """Extract return metrics from DataFrame"""
        metrics = {
            'total_returns': len(df),
            'by_reason': {},
            'by_date': {},
            'by_sku': {}
        }
        
        # Reason analysis
        reason_cols = [col for col in df.columns 
                      if 'reason' in col.lower()]
        if reason_cols:
            reason_col = reason_cols[0]
            metrics['by_reason'] = df[reason_col].value_counts().to_dict()
        
        # SKU analysis
        sku_cols = [col for col in df.columns 
                   if 'sku' in col.lower()]
        if sku_cols:
            sku_col = sku_cols[0]
            metrics['by_sku'] = df[sku_col].value_counts().head(10).to_dict()
        
        # Date analysis
        if date_column and date_column in df.columns:
            try:
                df['parsed_date'] = pd.to_datetime(df[date_column])
                df['month'] = df['parsed_date'].dt.to_period('M')
                metrics['by_date'] = df.groupby('month').size().to_dict()
            except:
                pass
        
        return metrics
    
    @staticmethod
    def merge_multiple_sources(files: List[ProcessedFile]) -> ProcessedFile:
        """Merge data from multiple file sources"""
        # Separate by type
        returns_data = []
        reviews_data = []
        other_data = []
        
        for file in files:
            if file.content_category in ['returns', 'fba_returns']:
                if file.data is not None:
                    returns_data.append(file.data)
            elif file.content_category in ['reviews', 'helium10_reviews']:
                if file.data is not None:
                    reviews_data.append(file.data)
            else:
                other_data.append(file)
        
        # Merge returns
        merged_returns = None
        if returns_data:
            merged_returns = pd.concat(returns_data, ignore_index=True)
        
        # Merge reviews
        merged_reviews = None
        if reviews_data:
            merged_reviews = pd.concat(reviews_data, ignore_index=True)
        
        # Create merged result
        return ProcessedFile(
            file_type='merged',
            format='multiple',
            content_category='merged_data',
            data=merged_returns,  # Primary data
            metadata={
                'source_count': len(files),
                'returns_count': len(returns_data),
                'reviews_count': len(reviews_data),
                'merged_reviews': merged_reviews,
                'other_files': other_data
            },
            confidence=0.95,
            extraction_method='file_merge'
        )

# Export
__all__ = ['UniversalFileDetector', 'ProcessedFile']
