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
                    file_content, filename, format_type, target_asin
                )
            elif format_type == 'excel':
                return UniversalFileDetector._process_excel(file_content, filename, target_asin)
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
                          format_type: str, target_asin: str = None) -> ProcessedFile:
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
                
                # Clean column names (remove any BOM or special characters)
                df.columns = [col.strip().replace('\ufeff', '') for col in df.columns]
                
                # Detect content type
                content_category = UniversalFileDetector._detect_content_type(df)
                
                # Filter by ASIN if provided
                if target_asin:
                    df = UniversalFileDetector.filter_by_asin(df, target_asin)
                
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
    def _process_excel(content: bytes, filename: str, target_asin: str = None) -> ProcessedFile:
        """Process Excel files"""
        try:
            # Read all sheets
            excel_file = pd.ExcelFile(BytesIO(content))
            
            # If single sheet, process directly
            if len(excel_file.sheet_names) == 1:
                df = pd.read_excel(BytesIO(content))
                
                # Clean column names
                df.columns = [col.strip() for col in df.columns]
                
                content_category = UniversalFileDetector._detect_content_type(df)
                
                # Filter by ASIN if provided
                if target_asin:
                    df = UniversalFileDetector.filter_by_asin(df, target_asin)
                
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
                # Multiple sheets - process all and combine if they're similar
                all_data = []
                sheet_info = []
                
                for sheet_name in excel_file.sheet_names:
                    try:
                        df = pd.read_excel(BytesIO(content), sheet_name=sheet_name)
                        content_type = UniversalFileDetector._detect_content_type(df)
                        
                        if target_asin:
                            df = UniversalFileDetector.filter_by_asin(df, target_asin)
                        
                        if len(df) > 0:
                            all_data.append(df)
                            sheet_info.append({
                                'name': sheet_name,
                                'type': content_type,
                                'rows': len(df)
                            })
                    except Exception as e:
                        logger.warning(f"Error processing sheet {sheet_name}: {e}")
                
                # Combine similar sheets
                if all_data:
                    combined_df = pd.concat(all_data, ignore_index=True)
                    content_category = UniversalFileDetector._detect_content_type(combined_df)
                    
                    return ProcessedFile(
                        file_type=content_category,
                        format='excel',
                        content_category=content_category,
                        data=combined_df,
                        metadata={
                            'filename': filename,
                            'sheet_names': excel_file.sheet_names,
                            'sheet_count': len(excel_file.sheet_names),
                            'sheet_info': sheet_info,
                            'columns': combined_df.columns.tolist(),
                            'row_count': len(combined_df)
                        },
                        extraction_method='excel_multi_sheet',
                        warnings=[f'Combined {len(all_data)} sheets']
                    )
                else:
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
                        warnings=['No data found after filtering']
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
            for enc in ['utf-8', 'cp1252', 'latin-1', 'iso-8859-1']:
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
        if df.empty:
            return 'empty'
            
        columns_lower = [str(col).lower() for col in df.columns]
        
        # Check for return data patterns - FBA specific
        fba_indicators = ['return-date', 'order-id', 'sku', 'asin', 'reason', 'customer-comments']
        fba_score = sum(1 for ind in fba_indicators if ind in columns_lower)
        if fba_score >= 4:
            return 'fba_returns'
        
        # Check for general return data patterns
        return_indicators = ['return', 'reason', 'refund', 'rma', 'defect', 'booking']
        return_score = sum(1 for ind in return_indicators 
                         if any(ind in col for col in columns_lower))
        
        # Check for review data patterns
        review_indicators = ['rating', 'review', 'title', 'body', 'comment', 'feedback', 'stars']
        review_score = sum(1 for ind in review_indicators 
                         if any(ind in col for col in columns_lower))
        
        # Check specific patterns
        if UniversalFileDetector._is_helium10_reviews(df):
            return 'helium10_reviews'
        elif return_score > review_score and return_score > 0:
            return 'returns'
        elif review_score > return_score and review_score > 0:
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
        if not target_asin or df.empty:
            return df
        
        # Find ASIN column (case-insensitive)
        asin_columns = [col for col in df.columns 
                       if 'asin' in col.lower()]
        
        if not asin_columns:
            return df
        
        # Filter by ASIN
        asin_col = asin_columns[0]
        
        # Handle different data types
        try:
            # Convert both to string for comparison
            df_filtered = df[df[asin_col].astype(str).str.upper() == target_asin.upper()]
            
            if df_filtered.empty:
                # Try without converting to uppercase
                df_filtered = df[df[asin_col].astype(str) == target_asin]
            
            return df_filtered
        except Exception as e:
            logger.warning(f"Error filtering by ASIN: {e}")
            return df
    
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
        date_cols = [col for col in df.columns 
                    if 'date' in col.lower()]
        if date_cols and not date_column:
            date_column = date_cols[0]
            
        if date_column and date_column in df.columns:
            try:
                df['parsed_date'] = pd.to_datetime(df[date_column], errors='coerce')
                df_with_dates = df.dropna(subset=['parsed_date'])
                if not df_with_dates.empty:
                    df_with_dates['month'] = df_with_dates['parsed_date'].dt.to_period('M')
                    metrics['by_date'] = df_with_dates.groupby('month').size().to_dict()
            except Exception as e:
                logger.warning(f"Date parsing error: {e}")
        
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
    
    @staticmethod
    def validate_file(file: ProcessedFile) -> Tuple[bool, List[str]]:
        """Validate processed file and return validation status and messages"""
        messages = []
        is_valid = True
        
        # Check if file was processed successfully
        if file.content_category == 'error':
            is_valid = False
            messages.append("File processing failed")
            return is_valid, messages
        
        # Check if data was extracted
        if file.data is None and file.raw_content is None:
            messages.append("No data extracted from file")
        
        # Check for empty dataframes
        if file.data is not None and file.data.empty:
            messages.append("File contains no data rows")
        
        # Validate based on content type
        if file.content_category == 'fba_returns' and file.data is not None:
            required_cols = ['return-date', 'order-id', 'asin', 'reason']
            missing_cols = [col for col in required_cols if col not in file.data.columns]
            if missing_cols:
                messages.append(f"Missing required columns: {', '.join(missing_cols)}")
        
        return is_valid, messages

# Export
__all__ = ['UniversalFileDetector', 'ProcessedFile']
