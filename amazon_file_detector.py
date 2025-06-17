"""
Universal File Detection and Processing Module
Version: 4.0 - Enhanced for Column K Export and Medical Device Returns
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

# Medical Device Return Keywords - Based on Amazon Return Categories
RETURN_KEYWORDS = {
    'size_fit': ['too large', 'too big', 'too loose', 'too small', 'too tight', 'too thin', 
                 'wrong size', 'doesn\'t fit', 'bad fit', 'didn\'t fit', 'doesn\'t fit well', 
                 'too tall', 'too short', 'too wide', 'sizing issues'],
    'comfort': ['uncomfortable', 'hurts customer', 'too firm', 'too hard', 'too stiff', 
                'too soft', 'not enough padding', 'not enough cushion'],
    'defects': ['defective', 'does not work properly', 'poor quality', 'ripped', 'torn', 
                'bad velcro', 'velcro doesn\'t stick', 'defective handles', 'defective suction cups', 
                'won\'t inflate', 'inflation issues', 'not working'],
    'performance': ['ineffective', 'not as expected', 'does not meet expectations', 
                    'not enough support', 'poor support', 'not enough compression', 
                    'not cold enough', 'not hot enough', 'inaccurate'],
    'stability': ['doesn\'t stay in place', 'doesn\'t stay fastened', 'slides around', 
                  'slides off', 'slides up', 'slippery', 'unstable', 'flattens'],
    'compatibility': ['doesn\'t fit walker', 'doesn\'t fit bariatric walker', 'doesn\'t fit knee walker',
                      'doesn\'t fit wheelchair', 'doesn\'t fit toilet', 'doesn\'t fit shower', 
                      'doesn\'t fit bed', 'doesn\'t fit machine', 'doesn\'t fit handle', 
                      'doesn\'t fit finger', 'not compatible', 'does not work with compression stockings'],
    'design': ['too bulky', 'too thick', 'too heavy', 'too thin', 'flimsy', 'small pulley', 
               'grip too small', 'fingers too long', 'fingers too short'],
    'wrong_product': ['wrong item', 'wrong color', 'not as advertised', 'different from website description',
                      'thought it was something else', 'thought it was scooter', 'thought it was crutches',
                      'thought pump was included', 'brace for wrong hand', 'immobilizer for wrong hand',
                      'style not as expected'],
    'missing': ['missing pieces', 'missing parts', 'missing accessories', 'no instructions', 
                'thought pump was included'],
    'customer_error': ['ordered wrong item', 'bought by mistake', 'changed mind', 'no longer needed',
                       'unauthorized purchase', 'no issue', 'customer error'],
    'shipping': ['arrived too late', 'received used item', 'received damaged item', 'item never arrived'],
    'assembly': ['difficult to use', 'difficult to adjust', 'difficult to assemble', 
                 'difficult to open valve', 'installation issues'],
    'medical': ['doctor did not approve', 'allergic reaction', 'bad smell', 'bad odor'],
    'price': ['better price available', 'found better price']
}

# Flatten all keywords for quick checking
ALL_RETURN_KEYWORDS = []
for category_keywords in RETURN_KEYWORDS.values():
    ALL_RETURN_KEYWORDS.extend(category_keywords)

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
    column_mapping: Dict[str, str] = field(default_factory=dict)  # Track important columns
    
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
            'warnings': self.warnings,
            'column_mapping': self.column_mapping
        }

class UniversalFileDetector:
    """Enhanced file detector for all Amazon marketplace formats with Column K support"""
    
    # Amazon return report patterns - expanded
    RETURN_PATTERNS = {
        'fba_returns': {
            'columns': ['return-date', 'order-id', 'sku', 'asin', 'reason', 'customer-comments', 'quantity'],
            'required': ['return-date', 'order-id', 'asin', 'reason'],
            'confidence': 0.95
        },
        'seller_central_returns': {
            'columns': ['order id', 'return reason', 'asin', 'refund amount', 'return date'],
            'indicators': ['return request', 'refund', 'rma'],
            'confidence': 0.85
        },
        'manage_returns_pdf': {
            'indicators': ['booking id', 'return request', 'defect', 'arrival time', 'manage returns'],
            'confidence': 0.9
        },
        'vive_health_returns': {
            'columns': ['categorizing/investigator complaint', 'product identifier tag'],
            'indicators': ['complaint', 'investigation', 'category'],
            'confidence': 0.9
        }
    }
    
    # Review file patterns
    REVIEW_PATTERNS = {
        'helium10': {
            'columns': ['Title', 'Body', 'Rating', 'Date', 'Verified', 'Variation'],
            'required': ['Title', 'Body', 'Rating'],
            'confidence': 0.95
        },
        'amazon_reviews': {
            'indicators': ['customer review', 'star rating', 'verified purchase', 'helpful'],
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
        """Process PDF files - placeholder for AI processing"""
        return ProcessedFile(
            file_type='pdf',
            format='pdf',
            content_category='pending_ai_analysis',
            raw_content=content,
            metadata={
                'filename': filename,
                'size': len(content),
                'message': 'PDF processing requires AI analysis'
            },
            extraction_method='ai_required'
        )
    
    @staticmethod
    def _process_image(content: bytes, filename: str) -> ProcessedFile:
        """Process image files - placeholder for vision AI"""
        return ProcessedFile(
            file_type='image',
            format='image',
            content_category='pending_vision_analysis',
            raw_content=content,
            metadata={
                'filename': filename,
                'size': len(content),
                'message': 'Image processing requires vision AI'
            },
            extraction_method='vision_ai_required'
        )
    
    @staticmethod
    def _process_text_file(content: bytes, filename: str, 
                          format_type: str, target_asin: str = None) -> ProcessedFile:
        """Process text-based files (CSV, TSV, TXT) with enhanced column detection"""
        
        # Detect encoding
        encoding = UniversalFileDetector._detect_encoding(content)
        
        try:
            text = content.decode(encoding)
            
            # Determine delimiter
            if format_type == 'tsv' or '\t' in text[:1000]:
                delimiter = '\t'
            elif format_type == 'csv':
                delimiter = ','
            else:
                delimiter = UniversalFileDetector._detect_delimiter(text)
            
            # Parse as DataFrame
            try:
                df = pd.read_csv(StringIO(text), delimiter=delimiter)
                
                # Clean column names
                df.columns = [col.strip().replace('\ufeff', '') for col in df.columns]
                
                # Detect content type and map columns
                content_category, column_mapping = UniversalFileDetector._detect_and_map_columns(df)
                
                # Filter by ASIN if provided
                if target_asin and 'asin' in column_mapping:
                    asin_col = column_mapping['asin']
                    if asin_col in df.columns:
                        df_filtered = df[df[asin_col].astype(str).str.upper() == target_asin.upper()]
                        if not df_filtered.empty:
                            df = df_filtered
                        else:
                            logger.warning(f"No data found for ASIN {target_asin}")
                
                # Ensure DataFrame has enough columns for Column K export
                if content_category in ['returns', 'fba_returns']:
                    while len(df.columns) < 11:
                        df[f'Column_{len(df.columns)}'] = ''
                
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
                        'row_count': len(df),
                        'target_asin': target_asin
                    },
                    column_mapping=column_mapping,
                    confidence=0.9,
                    extraction_method='structured_parsing'
                )
                
            except Exception as parse_error:
                # If parsing fails, return as unstructured
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
                warnings=[f'Processing error: {str(e)}']
            )
    
    @staticmethod
    def _process_excel(content: bytes, filename: str, target_asin: str = None) -> ProcessedFile:
        """Process Excel files with multi-sheet support"""
        try:
            # Try to read Excel file
            excel_file = pd.ExcelFile(BytesIO(content))
            
            # Process based on number of sheets
            if len(excel_file.sheet_names) == 1:
                # Single sheet
                df = pd.read_excel(BytesIO(content))
                df.columns = [col.strip() for col in df.columns]
                
                content_category, column_mapping = UniversalFileDetector._detect_and_map_columns(df)
                
                # Filter by ASIN
                if target_asin and 'asin' in column_mapping:
                    asin_col = column_mapping['asin']
                    if asin_col in df.columns:
                        df_filtered = df[df[asin_col].astype(str).str.upper() == target_asin.upper()]
                        if not df_filtered.empty:
                            df = df_filtered
                
                # Ensure enough columns for Column K
                if content_category in ['returns', 'fba_returns']:
                    while len(df.columns) < 11:
                        df[f'Column_{len(df.columns)}'] = ''
                
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
                    column_mapping=column_mapping,
                    confidence=0.9,
                    extraction_method='excel_parsing'
                )
            else:
                # Multiple sheets - combine if similar
                all_data = []
                sheet_info = []
                
                for sheet_name in excel_file.sheet_names:
                    try:
                        df = pd.read_excel(BytesIO(content), sheet_name=sheet_name)
                        
                        # Clean column names
                        df.columns = [col.strip() for col in df.columns]
                        
                        content_type, col_map = UniversalFileDetector._detect_and_map_columns(df)
                        
                        if target_asin and 'asin' in col_map:
                            asin_col = col_map['asin']
                            if asin_col in df.columns:
                                df_filtered = df[df[asin_col].astype(str).str.upper() == target_asin.upper()]
                                if not df_filtered.empty:
                                    df = df_filtered
                        
                        if len(df) > 0:
                            all_data.append(df)
                            sheet_info.append({
                                'name': sheet_name,
                                'type': content_type,
                                'rows': len(df)
                            })
                    except Exception as e:
                        logger.warning(f"Error processing sheet {sheet_name}: {str(e)}")
                
                # Combine data
                if all_data:
                    combined_df = pd.concat(all_data, ignore_index=True)
                    content_category, column_mapping = UniversalFileDetector._detect_and_map_columns(combined_df)
                    
                    # Ensure enough columns
                    if content_category in ['returns', 'fba_returns']:
                        while len(combined_df.columns) < 11:
                            combined_df[f'Column_{len(combined_df.columns)}'] = ''
                    
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
                        column_mapping=column_mapping,
                        extraction_method='excel_multi_sheet'
                    )
                else:
                    return ProcessedFile(
                        file_type='empty',
                        format='excel',
                        content_category='empty',
                        metadata={'filename': filename},
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
        """Detect file encoding with fallbacks"""
        sample_size = min(len(content), 10000)
        detection = chardet.detect(content[:sample_size])
        
        encoding = detection.get('encoding', 'utf-8')
        confidence = detection.get('confidence', 0)
        
        # If low confidence, try common encodings
        if confidence < 0.7:
            for enc in ['utf-8', 'cp1252', 'latin-1', 'iso-8859-1', 'utf-16']:
                try:
                    content.decode(enc)
                    return enc
                except:
                    continue
        
        return encoding or 'utf-8'
    
    @staticmethod
    def _detect_delimiter(text: str) -> str:
        """Detect delimiter in text"""
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
    def _detect_and_map_columns(df: pd.DataFrame) -> Tuple[str, Dict[str, str]]:
        """Detect content type and create column mapping"""
        if df.empty:
            return 'empty', {}
        
        try:
            columns_lower = [str(col).lower().strip() for col in df.columns]
            column_mapping = {}
            
            # Check for FBA return report
            fba_score = 0
            fba_cols = {
                'return-date': 'return_date',
                'order-id': 'order_id',
                'sku': 'sku',
                'asin': 'asin',
                'reason': 'reason',
                'customer-comments': 'customer_comments',
                'quantity': 'quantity'
            }
            
            for fba_col, map_name in fba_cols.items():
                if fba_col in columns_lower:
                    idx = columns_lower.index(fba_col)
                    column_mapping[map_name] = df.columns[idx]
                    fba_score += 1
            
            if fba_score >= 4:
                return 'fba_returns', column_mapping
            
            # Check for custom return format (Column I for complaints)
            # Look for specific column positions
            if len(df.columns) >= 9:  # At least up to column I
                # Check if column I (index 8) contains complaint-like text
                col_i = df.columns[8]
                sample_values = df[col_i].dropna().head(10).astype(str)
                
                # Check if it looks like complaints using return keywords
                if any(any(kw in str(val).lower() for kw in ALL_RETURN_KEYWORDS) for val in sample_values):
                    column_mapping['complaint'] = col_i
                    
                    # Map other columns if available
                    if len(df.columns) > 1:
                        column_mapping['sku'] = df.columns[1]  # Column B
                    if len(df.columns) > 10:
                        column_mapping['category'] = df.columns[10]  # Column K
                    
                    return 'returns', column_mapping
            
            # Check for review data
            review_indicators = ['rating', 'review', 'title', 'body', 'stars', 'verified']
            review_score = sum(1 for ind in review_indicators 
                             if any(ind in col for col in columns_lower))
            
            if review_score >= 2:
                # Map review columns
                for col_idx, col in enumerate(df.columns):
                    col_lower = str(col).lower()
                    if 'rating' in col_lower:
                        column_mapping['rating'] = col
                    elif 'title' in col_lower:
                        column_mapping['title'] = col
                    elif 'body' in col_lower or 'review' in col_lower:
                        column_mapping['body'] = col
                    elif 'asin' in col_lower:
                        column_mapping['asin'] = col
                
                return 'reviews', column_mapping
            
            # Check for general return indicators
            return_indicators = ['return', 'reason', 'complaint', 'refund', 'rma']
            return_score = sum(1 for ind in return_indicators 
                             if any(ind in col for col in columns_lower))
            
            if return_score >= 2:
                # Map what we can find
                for col_idx, col in enumerate(df.columns):
                    col_lower = str(col).lower()
                    if 'reason' in col_lower or 'complaint' in col_lower:
                        column_mapping['complaint'] = col
                    elif 'asin' in col_lower:
                        column_mapping['asin'] = col
                    elif 'sku' in col_lower:
                        column_mapping['sku'] = col
                    elif 'order' in col_lower and 'id' in col_lower:
                        column_mapping['order_id'] = col
                
                return 'returns', column_mapping
            
            # Default to data
            return 'data', column_mapping
            
        except Exception as e:
            logger.error(f"Error detecting columns: {e}")
            return 'data', {}
    
    @staticmethod
    def prepare_for_column_k_export(df: pd.DataFrame) -> pd.DataFrame:
        """Prepare DataFrame for Column K export"""
        # Ensure we have at least 11 columns (A through K)
        while len(df.columns) < 11:
            df[f'Column_{len(df.columns)}'] = ''
        
        # Ensure Column K (index 10) exists and is empty if not already populated
        if pd.isna(df.iloc[:, 10]).all():
            df.iloc[:, 10] = ''
        
        return df
    
    @staticmethod
    def validate_file(file: ProcessedFile) -> Tuple[bool, List[str]]:
        """Validate processed file"""
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
            required_cols = ['reason', 'asin']
            missing_cols = []
            for req_col in required_cols:
                if req_col not in file.column_mapping:
                    missing_cols.append(req_col)
            
            if missing_cols:
                messages.append(f"Missing required columns: {', '.join(missing_cols)}")
        
        # Check if ready for Column K export
        if file.content_category in ['returns', 'fba_returns'] and file.data is not None:
            if len(file.data.columns) < 11:
                messages.append("DataFrame needs column expansion for Column K export")
        
        return is_valid, messages
    
    @staticmethod
    def extract_return_metrics(df: pd.DataFrame, 
                             column_mapping: Dict[str, str]) -> Dict[str, Any]:
        """Extract return metrics using column mapping"""
        metrics = {
            'total_returns': len(df),
            'by_reason': {},
            'by_asin': {},
            'by_sku': {}
        }
        
        # Reason analysis
        if 'reason' in column_mapping:
            reason_col = column_mapping['reason']
            if reason_col in df.columns:
                metrics['by_reason'] = df[reason_col].value_counts().to_dict()
        elif 'complaint' in column_mapping:
            complaint_col = column_mapping['complaint']
            if complaint_col in df.columns:
                # For complaint text, we might want to categorize first
                metrics['complaints_found'] = len(df[df[complaint_col].notna()])
        
        # ASIN analysis
        if 'asin' in column_mapping:
            asin_col = column_mapping['asin']
            if asin_col in df.columns:
                metrics['by_asin'] = df[asin_col].value_counts().head(10).to_dict()
        
        # SKU analysis
        if 'sku' in column_mapping:
            sku_col = column_mapping['sku']
            if sku_col in df.columns:
                metrics['by_sku'] = df[sku_col].value_counts().head(10).to_dict()
        
        return metrics

# Export
__all__ = ['UniversalFileDetector', 'ProcessedFile']
