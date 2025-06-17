"""
Universal File Detection and Processing Module
Version: 4.0 - Enhanced for Medical Device Quality Analysis
Handles PDFs, FBA Returns, Reviews with injury/quality detection
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
import numpy as np

logger = logging.getLogger(__name__)

# Import medical device categories from AI module
try:
    from enhanced_ai_universal import MEDICAL_DEVICE_CATEGORIES, CRITICAL_KEYWORDS
except ImportError:
    # Fallback definitions if import fails
    MEDICAL_DEVICE_CATEGORIES = {}
    CRITICAL_KEYWORDS = {}

@dataclass
class ProcessedFile:
    """Structured file processing result with quality focus"""
    file_type: str
    format: str  # pdf, csv, image, etc.
    content_category: str  # returns, reviews, mixed, unknown
    data: pd.DataFrame = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    raw_content: Any = None
    confidence: float = 0.0
    extraction_method: str = ""
    warnings: List[str] = field(default_factory=list)
    critical_issues: List[Dict[str, Any]] = field(default_factory=list)
    quality_metrics: Dict[str, Any] = field(default_factory=dict)
    
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
            'critical_issues': self.critical_issues,
            'quality_metrics': self.quality_metrics
        }

class UniversalFileDetector:
    """Enhanced file detector for medical device quality analysis"""
    
    # Amazon return report patterns - expanded
    RETURN_PATTERNS = {
        'manage_returns_pdf': {
            'indicators': ['manage returns', 'return request', 'defect', 'arrival time', 
                          'order id', 'asin', 'customer comments', 'return reason'],
            'confidence': 0.9,
            'critical_check': True
        },
        'fba_returns': {
            'columns': ['return-date', 'order-id', 'sku', 'asin', 'reason', 'quantity',
                       'customer-comments', 'product-name', 'fulfillment-center-id'],
            'confidence': 0.95,
            'critical_check': True
        },
        'seller_central_returns': {
            'indicators': ['order id', 'return reason', 'asin', 'refund amount', 
                          'return status', 'buyer comments'],
            'confidence': 0.85,
            'critical_check': True
        },
        'return_merchandise_auth': {
            'indicators': ['rma', 'return merchandise', 'authorization', 'defect description'],
            'confidence': 0.8,
            'critical_check': True
        }
    }
    
    # Review file patterns
    REVIEW_PATTERNS = {
        'helium10': {
            'columns': ['Title', 'Body', 'Rating', 'Date', 'Verified', 'Profile Name'],
            'confidence': 0.95,
            'review_focused': True
        },
        'amazon_reviews': {
            'indicators': ['customer review', 'star rating', 'verified purchase', 'helpful'],
            'confidence': 0.85,
            'review_focused': True
        },
        'seller_central_reviews': {
            'columns': ['asin', 'rating', 'title', 'content', 'date', 'verified'],
            'confidence': 0.9,
            'review_focused': True
        }
    }
    
    # Quality issue keywords for quick detection
    QUALITY_KEYWORDS = {
        'injury': ['injured', 'injury', 'hurt', 'wound', 'bleeding', 'bruise', 'fracture'],
        'defect': ['defect', 'broken', 'damaged', 'malfunction', 'failed', 'cracked'],
        'safety': ['dangerous', 'unsafe', 'hazard', 'risk', 'accident'],
        'medical': ['hospital', 'doctor', 'emergency', 'medical attention', 'allergic']
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
        
        detected_format = format_map.get(ext, 'unknown')
        
        # Additional content-based detection for .txt files
        if detected_format == 'txt' and content:
            try:
                sample = content[:1000].decode('utf-8', errors='ignore')
                # Check if it's actually a TSV
                if sample.count('\t') > sample.count(',') * 2:
                    return 'tsv'
                # Check for FBA return report signature
                if 'return-date' in sample and 'order-id' in sample:
                    return 'fba_txt'
            except:
                pass
        
        return detected_format
    
    @staticmethod
    def process_file(file_content: bytes, filename: str, 
                    target_asin: str = None) -> ProcessedFile:
        """Main entry point for file processing with quality focus"""
        
        format_type = UniversalFileDetector.detect_file_format(filename, file_content)
        
        try:
            if format_type == 'pdf':
                return UniversalFileDetector._process_pdf(file_content, filename)
            elif format_type == 'image':
                return UniversalFileDetector._process_image(file_content, filename)
            elif format_type in ['csv', 'tsv', 'txt', 'fba_txt']:
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
                    warnings=['Unsupported file format'],
                    metadata={'filename': filename}
                )
                
        except Exception as e:
            logger.error(f"File processing error: {e}", exc_info=True)
            return ProcessedFile(
                file_type='error',
                format=format_type,
                content_category='error',
                warnings=[str(e)],
                metadata={'filename': filename, 'error': str(e)}
            )
    
    @staticmethod
    def _process_pdf(content: bytes, filename: str) -> ProcessedFile:
        """Process PDF files - especially from Amazon Seller Central"""
        # PDF processing will be handled by AI analyzer
        # Pre-check for critical keywords in metadata if possible
        
        return ProcessedFile(
            file_type='pdf',
            format='pdf',
            content_category='pending_ai_analysis',
            raw_content=content,
            metadata={
                'filename': filename, 
                'size': len(content),
                'source': 'amazon_seller_central'  # Assume Amazon source
            },
            extraction_method='ai_required',
            quality_metrics={'requires_ai_analysis': True}
        )
    
    @staticmethod
    def _process_image(content: bytes, filename: str) -> ProcessedFile:
        """Process image files - screenshots from Seller Central"""
        return ProcessedFile(
            file_type='image',
            format='image',
            content_category='pending_vision_analysis',
            raw_content=content,
            metadata={
                'filename': filename, 
                'size': len(content),
                'source': 'screenshot'
            },
            extraction_method='vision_ai_required'
        )
    
    @staticmethod
    def _process_text_file(content: bytes, filename: str, 
                          format_type: str, target_asin: str = None) -> ProcessedFile:
        """Process text-based files with enhanced FBA return support"""
        
        # Detect encoding
        encoding = UniversalFileDetector._detect_encoding(content)
        
        try:
            text = content.decode(encoding)
            
            # Special handling for FBA return reports
            if format_type == 'fba_txt' or UniversalFileDetector._is_fba_return_content(text):
                return UniversalFileDetector._process_fba_return_file(
                    text, filename, encoding, target_asin
                )
            
            # Determine delimiter for other text files
            if format_type == 'tsv' or '\t' in text[:1000]:
                delimiter = '\t'
            elif format_type == 'csv':
                delimiter = ','
            else:
                delimiter = UniversalFileDetector._detect_delimiter(text)
            
            # Try to parse as structured data
            try:
                df = pd.read_csv(StringIO(text), delimiter=delimiter)
                
                # Clean column names
                df.columns = [col.strip().replace('\ufeff', '') for col in df.columns]
                
                # Detect content type
                content_category = UniversalFileDetector._detect_content_type(df)
                
                # Check for critical issues
                critical_issues = UniversalFileDetector._scan_for_critical_issues(df)
                
                # Calculate quality metrics
                quality_metrics = UniversalFileDetector._calculate_quality_metrics(df)
                
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
                        'row_count': len(df),
                        'target_asin': target_asin
                    },
                    confidence=0.9,
                    extraction_method='structured_parsing',
                    critical_issues=critical_issues,
                    quality_metrics=quality_metrics
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
                warnings=[f'Decoding error: {str(e)}'],
                metadata={'filename': filename}
            )
    
    @staticmethod
    def _process_fba_return_file(text: str, filename: str, encoding: str, 
                                target_asin: str = None) -> ProcessedFile:
        """Special processing for FBA return reports"""
        try:
            # FBA returns are tab-delimited
            df = pd.read_csv(StringIO(text), delimiter='\t')
            
            # Standardize column names (handle variations)
            column_mapping = {
                'return-date': 'return_date',
                'order-id': 'order_id',
                'sku': 'sku',
                'asin': 'asin',
                'fnsku': 'fnsku',
                'product-name': 'product_name',
                'quantity': 'quantity',
                'fulfillment-center-id': 'fulfillment_center_id',
                'detailed-disposition': 'detailed_disposition',
                'reason': 'reason',
                'status': 'status',
                'license-plate-number': 'license_plate_number',
                'customer-comments': 'customer_comments'
            }
            
            # Rename columns for consistency
            df.rename(columns=column_mapping, inplace=True)
            
            # Ensure critical columns exist
            for col in ['reason', 'customer_comments']:
                if col not in df.columns:
                    df[col] = ''
            
            # Clean data
            df['customer_comments'] = df['customer_comments'].fillna('')
            df['reason'] = df['reason'].fillna('')
            
            # Convert dates
            if 'return_date' in df.columns:
                df['return_date'] = pd.to_datetime(df['return_date'], errors='coerce')
            
            # Filter by ASIN if provided
            original_count = len(df)
            if target_asin:
                df = df[df['asin'] == target_asin]
                logger.info(f"Filtered FBA returns: {original_count} -> {len(df)} for ASIN {target_asin}")
            
            # Scan for critical issues
            critical_issues = UniversalFileDetector._scan_fba_for_critical_issues(df)
            
            # Calculate quality metrics specific to FBA returns
            quality_metrics = UniversalFileDetector._calculate_fba_quality_metrics(df)
            
            return ProcessedFile(
                file_type='fba_returns',
                format='fba_txt',
                content_category='fba_returns',
                data=df,
                metadata={
                    'filename': filename,
                    'encoding': encoding,
                    'original_count': original_count,
                    'filtered_count': len(df),
                    'target_asin': target_asin,
                    'date_range': UniversalFileDetector._get_date_range(df),
                    'unique_asins': df['asin'].nunique() if 'asin' in df.columns else 0,
                    'unique_skus': df['sku'].nunique() if 'sku' in df.columns else 0
                },
                confidence=0.95,
                extraction_method='fba_parser',
                critical_issues=critical_issues,
                quality_metrics=quality_metrics,
                warnings=[] if not critical_issues else [f"Found {len(critical_issues)} critical issues"]
            )
            
        except Exception as e:
            logger.error(f"FBA return processing error: {e}")
            return ProcessedFile(
                file_type='error',
                format='fba_txt',
                content_category='error',
                warnings=[f'FBA processing error: {str(e)}'],
                metadata={'filename': filename}
            )
    
    @staticmethod
    def _scan_fba_for_critical_issues(df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Scan FBA returns for critical quality and safety issues"""
        critical_issues = []
        
        if df.empty:
            return critical_issues
        
        # Check each row for critical keywords
        for idx, row in df.iterrows():
            reason = str(row.get('reason', '')).lower()
            comment = str(row.get('customer_comments', '')).lower()
            full_text = f"{reason} {comment}"
            
            # Check for critical keywords
            found_keywords = []
            for category, keywords in QUALITY_KEYWORDS.items():
                if any(keyword in full_text for keyword in keywords):
                    found_keywords.append(category)
            
            if found_keywords:
                critical_issues.append({
                    'order_id': row.get('order_id', 'Unknown'),
                    'asin': row.get('asin', 'Unknown'),
                    'sku': row.get('sku', 'Unknown'),
                    'return_date': row.get('return_date', 'Unknown'),
                    'reason': row.get('reason', ''),
                    'customer_comment': row.get('customer_comments', ''),
                    'critical_categories': found_keywords,
                    'severity': 'critical' if 'injury' in found_keywords or 'medical' in found_keywords else 'high'
                })
        
        return critical_issues
    
    @staticmethod
    def _calculate_fba_quality_metrics(df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate quality-focused metrics for FBA returns"""
        metrics = {
            'total_returns': len(df),
            'has_customer_comments': df['customer_comments'].notna().sum() if 'customer_comments' in df.columns else 0,
            'comment_rate': 0.0,
            'reason_distribution': {},
            'disposition_distribution': {},
            'top_returned_products': [],
            'quality_indicators': {}
        }
        
        if df.empty:
            return metrics
        
        # Comment rate
        if 'customer_comments' in df.columns:
            metrics['comment_rate'] = (df['customer_comments'].notna().sum() / len(df) * 100)
        
        # Reason distribution
        if 'reason' in df.columns:
            metrics['reason_distribution'] = df['reason'].value_counts().to_dict()
            
            # Quality-related reasons
            quality_reasons = ['DEFECTIVE', 'QUALITY_NOT_ADEQUATE', 'DAMAGED_BY_FC', 
                             'DAMAGED_BY_CARRIER', 'EXPIRED_ITEM']
            quality_count = df[df['reason'].isin(quality_reasons)].shape[0]
            metrics['quality_indicators']['quality_return_rate'] = (quality_count / len(df) * 100) if len(df) > 0 else 0
        
        # Disposition distribution
        if 'detailed_disposition' in df.columns:
            metrics['disposition_distribution'] = df['detailed_disposition'].value_counts().to_dict()
        
        # Top returned products
        if 'product_name' in df.columns:
            top_products = df.groupby(['asin', 'sku', 'product_name']).size().sort_values(ascending=False).head(10)
            metrics['top_returned_products'] = [
                {
                    'asin': asin,
                    'sku': sku,
                    'product_name': name[:50] + '...' if len(name) > 50 else name,
                    'return_count': count
                }
                for (asin, sku, name), count in top_products.items()
            ]
        
        return metrics
    
    @staticmethod
    def _process_excel(content: bytes, filename: str, target_asin: str = None) -> ProcessedFile:
        """Process Excel files with enhanced quality analysis"""
        try:
            # Read all sheets
            excel_file = pd.ExcelFile(BytesIO(content))
            
            # Process each sheet
            all_data = []
            sheet_info = []
            critical_issues = []
            
            for sheet_name in excel_file.sheet_names:
                try:
                    df = pd.read_excel(BytesIO(content), sheet_name=sheet_name)
                    
                    # Clean column names
                    df.columns = [str(col).strip() for col in df.columns]
                    
                    # Detect content type
                    content_type = UniversalFileDetector._detect_content_type(df)
                    
                    # Scan for critical issues
                    sheet_critical = UniversalFileDetector._scan_for_critical_issues(df)
                    critical_issues.extend(sheet_critical)
                    
                    # Filter by ASIN if provided
                    if target_asin:
                        df = UniversalFileDetector.filter_by_asin(df, target_asin)
                    
                    if len(df) > 0:
                        all_data.append(df)
                        sheet_info.append({
                            'name': sheet_name,
                            'type': content_type,
                            'rows': len(df),
                            'critical_issues': len(sheet_critical)
                        })
                        
                except Exception as e:
                    logger.warning(f"Error processing sheet {sheet_name}: {e}")
            
            # Combine similar sheets if multiple
            if len(all_data) == 1:
                # Single sheet
                combined_df = all_data[0]
                content_category = sheet_info[0]['type']
            elif len(all_data) > 1:
                # Multiple sheets - combine if similar types
                combined_df = pd.concat(all_data, ignore_index=True)
                content_category = UniversalFileDetector._detect_content_type(combined_df)
            else:
                # No data
                return ProcessedFile(
                    file_type='excel',
                    format='excel',
                    content_category='empty',
                    metadata={'filename': filename, 'sheets': excel_file.sheet_names},
                    warnings=['No data found in Excel file']
                )
            
            # Calculate quality metrics
            quality_metrics = UniversalFileDetector._calculate_quality_metrics(combined_df)
            
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
                    'row_count': len(combined_df),
                    'target_asin': target_asin
                },
                confidence=0.9,
                extraction_method='excel_parser',
                critical_issues=critical_issues,
                quality_metrics=quality_metrics,
                warnings=[f'Combined {len(all_data)} sheets'] if len(all_data) > 1 else []
            )
                
        except Exception as e:
            logger.error(f"Excel processing error: {e}")
            return ProcessedFile(
                file_type='error',
                format='excel',
                content_category='error',
                warnings=[f'Excel processing error: {str(e)}'],
                metadata={'filename': filename}
            )
    
    @staticmethod
    def _detect_encoding(content: bytes) -> str:
        """Detect file encoding with improved accuracy"""
        # Sample for detection
        sample_size = min(len(content), 10000)
        detection = chardet.detect(content[:sample_size])
        
        encoding = detection.get('encoding', 'utf-8')
        confidence = detection.get('confidence', 0)
        
        # Common encoding fixes
        encoding_fixes = {
            'ascii': 'utf-8',
            'ISO-8859-1': 'cp1252',  # Windows Latin-1
        }
        
        encoding = encoding_fixes.get(encoding, encoding)
        
        # Fallback for low confidence
        if confidence < 0.7:
            # Try common encodings
            for enc in ['utf-8', 'cp1252', 'latin-1', 'utf-16']:
                try:
                    content.decode(enc)
                    return enc
                except:
                    continue
        
        return encoding or 'utf-8'
    
    @staticmethod
    def _detect_delimiter(text: str) -> str:
        """Detect delimiter in text with improved logic"""
        # Sample first few lines
        lines = text.split('\n')[:10]
        sample = '\n'.join(lines)
        
        # Count occurrences
        delimiters = {
            '\t': sample.count('\t'),
            ',': sample.count(','),
            '|': sample.count('|'),
            ';': sample.count(';')
        }
        
        # Check consistency across lines
        delimiter_consistency = {}
        for delim, count in delimiters.items():
            if count > 0:
                counts_per_line = [line.count(delim) for line in lines if line.strip()]
                if counts_per_line:
                    # Calculate variance
                    variance = np.var(counts_per_line)
                    delimiter_consistency[delim] = variance
        
        # Prefer delimiter with lowest variance (most consistent)
        if delimiter_consistency:
            best_delimiter = min(delimiter_consistency.items(), key=lambda x: x[1])[0]
            return best_delimiter
        
        # Fallback to most common
        return max(delimiters.items(), key=lambda x: x[1])[0] if delimiters else ','
    
    @staticmethod
    def _detect_content_type(df: pd.DataFrame) -> str:
        """Enhanced content type detection with quality focus"""
        if df.empty:
            return 'empty'
            
        columns_lower = [str(col).lower() for col in df.columns]
        
        # FBA return report detection - highest priority
        fba_required = ['return-date', 'order-id', 'sku', 'asin', 'reason']
        fba_score = sum(1 for col in fba_required if any(col in c for c in columns_lower))
        if fba_score >= 3:
            return 'fba_returns'
        
        # General return detection
        return_indicators = ['return', 'reason', 'refund', 'rma', 'defect', 'complaint',
                           'issue', 'problem', 'damaged', 'broken']
        return_score = sum(1 for ind in return_indicators 
                         if any(ind in col for col in columns_lower))
        
        # Review detection
        review_indicators = ['rating', 'review', 'title', 'body', 'comment', 'feedback', 
                           'stars', 'verified', 'helpful']
        review_score = sum(1 for ind in review_indicators 
                         if any(ind in col for col in columns_lower))
        
        # Quality report detection
        quality_indicators = ['defect', 'quality', 'inspection', 'failure', 'test', 'audit']
        quality_score = sum(1 for ind in quality_indicators
                          if any(ind in col for col in columns_lower))
        
        # Determine type based on scores
        scores = {
            'returns': return_score,
            'reviews': review_score,
            'quality_report': quality_score
        }
        
        if max(scores.values()) > 0:
            return max(scores.items(), key=lambda x: x[1])[0]
        
        # Check specific patterns
        if UniversalFileDetector._is_helium10_reviews(df):
            return 'helium10_reviews'
        
        return 'data'
    
    @staticmethod
    def _is_fba_return_content(text: str) -> bool:
        """Check if text content is FBA return report"""
        # Check first few lines for FBA signature
        lines = text.split('\n')[:5]
        header = '\n'.join(lines).lower()
        
        fba_indicators = ['return-date', 'order-id', 'fnsku', 'customer-comments', 
                         'detailed-disposition', 'fulfillment-center-id']
        
        matches = sum(1 for ind in fba_indicators if ind in header)
        return matches >= 3
    
    @staticmethod
    def _is_helium10_reviews(df: pd.DataFrame) -> bool:
        """Check if DataFrame is Helium 10 review export"""
        expected = ['Title', 'Body', 'Rating', 'Date', 'Verified']
        matches = sum(1 for exp in expected if exp in df.columns)
        return matches >= 3
    
    @staticmethod
    def _scan_for_critical_issues(df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Scan any dataframe for critical quality issues"""
        critical_issues = []
        
        if df.empty:
            return critical_issues
        
        # Columns to check for critical content
        text_columns = []
        for col in df.columns:
            if df[col].dtype == 'object':  # Text columns
                text_columns.append(col)
        
        # Check each row
        for idx, row in df.iterrows():
            combined_text = ' '.join(str(row[col]) for col in text_columns).lower()
            
            # Check for critical keywords
            found_keywords = []
            for category, keywords in QUALITY_KEYWORDS.items():
                if any(keyword in combined_text for keyword in keywords):
                    found_keywords.append(category)
            
            if found_keywords:
                issue = {
                    'row_index': idx,
                    'critical_categories': found_keywords,
                    'severity': 'critical' if any(cat in ['injury', 'medical'] for cat in found_keywords) else 'high'
                }
                
                # Add identifying information
                if 'order_id' in df.columns:
                    issue['order_id'] = row.get('order_id')
                if 'asin' in df.columns:
                    issue['asin'] = row.get('asin')
                if 'sku' in df.columns:
                    issue['sku'] = row.get('sku')
                    
                critical_issues.append(issue)
        
        return critical_issues
    
    @staticmethod
    def _calculate_quality_metrics(df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate general quality metrics for any dataframe"""
        metrics = {
            'total_rows': len(df),
            'columns': df.columns.tolist(),
            'data_quality': {}
        }
        
        if df.empty:
            return metrics
        
        # Data completeness
        metrics['data_quality']['completeness'] = (df.notna().sum().sum() / 
                                                   (len(df) * len(df.columns)) * 100)
        
        # Check for date columns and get date range
        date_columns = [col for col in df.columns if 'date' in col.lower()]
        if date_columns:
            for col in date_columns:
                try:
                    dates = pd.to_datetime(df[col], errors='coerce')
                    valid_dates = dates.dropna()
                    if not valid_dates.empty:
                        metrics[f'{col}_range'] = {
                            'start': valid_dates.min().strftime('%Y-%m-%d'),
                            'end': valid_dates.max().strftime('%Y-%m-%d')
                        }
                except:
                    pass
        
        return metrics
    
    @staticmethod
    def _get_date_range(df: pd.DataFrame) -> Dict[str, str]:
        """Get date range from return data"""
        if 'return_date' in df.columns:
            try:
                dates = pd.to_datetime(df['return_date'], errors='coerce').dropna()
                if not dates.empty:
                    return {
                        'start': dates.min().strftime('%Y-%m-%d'),
                        'end': dates.max().strftime('%Y-%m-%d'),
                        'days': (dates.max() - dates.min()).days
                    }
            except:
                pass
        return {'start': 'Unknown', 'end': 'Unknown', 'days': 0}
    
    @staticmethod
    def filter_by_asin(df: pd.DataFrame, target_asin: str) -> pd.DataFrame:
        """Enhanced ASIN filtering with better matching"""
        if not target_asin or df.empty:
            return df
        
        # Find ASIN columns (case-insensitive)
        asin_columns = []
        for col in df.columns:
            if 'asin' in str(col).lower():
                asin_columns.append(col)
        
        if not asin_columns:
            logger.warning("No ASIN column found for filtering")
            return df
        
        # Filter by ASIN
        filtered_dfs = []
        for asin_col in asin_columns:
            try:
                # Clean and standardize ASIN values
                df['_temp_asin'] = df[asin_col].astype(str).str.strip().str.upper()
                target_clean = target_asin.strip().upper()
                
                # Exact match
                mask = df['_temp_asin'] == target_clean
                filtered = df[mask]
                
                if not filtered.empty:
                    filtered_dfs.append(filtered)
                    
                df.drop('_temp_asin', axis=1, inplace=True)
                
            except Exception as e:
                logger.warning(f"Error filtering by ASIN column {asin_col}: {e}")
        
        if filtered_dfs:
            result = pd.concat(filtered_dfs, ignore_index=True).drop_duplicates()
            logger.info(f"Filtered {len(df)} rows to {len(result)} for ASIN {target_asin}")
            return result
        else:
            logger.warning(f"No matches found for ASIN {target_asin}")
            return pd.DataFrame()  # Return empty dataframe if no matches
    
    @staticmethod
    def merge_related_files(files: List[ProcessedFile], target_asin: str = None) -> Dict[str, Any]:
        """Merge related files (returns, reviews) for comprehensive analysis"""
        merged_data = {
            'returns': [],
            'reviews': [],
            'quality_reports': [],
            'other': [],
            'combined_metrics': {},
            'critical_issues': [],
            'target_asin': target_asin
        }
        
        # Categorize files
        for file in files:
            if file.content_category in ['returns', 'fba_returns']:
                if file.data is not None:
                    merged_data['returns'].append(file)
            elif file.content_category in ['reviews', 'helium10_reviews']:
                if file.data is not None:
                    merged_data['reviews'].append(file)
            elif file.content_category == 'quality_report':
                merged_data['quality_reports'].append(file)
            else:
                merged_data['other'].append(file)
            
            # Collect all critical issues
            if file.critical_issues:
                merged_data['critical_issues'].extend(file.critical_issues)
        
        # Combine returns data
        if merged_data['returns']:
            returns_dfs = [f.data for f in merged_data['returns']]
            merged_data['combined_returns'] = pd.concat(returns_dfs, ignore_index=True)
            
            # If ASIN specified, filter combined data
            if target_asin:
                merged_data['combined_returns'] = UniversalFileDetector.filter_by_asin(
                    merged_data['combined_returns'], target_asin
                )
        
        # Combine reviews data
        if merged_data['reviews']:
            reviews_dfs = [f.data for f in merged_data['reviews']]
            merged_data['combined_reviews'] = pd.concat(reviews_dfs, ignore_index=True)
            
            if target_asin:
                merged_data['combined_reviews'] = UniversalFileDetector.filter_by_asin(
                    merged_data['combined_reviews'], target_asin
                )
        
        # Calculate combined metrics
        merged_data['combined_metrics'] = UniversalFileDetector._calculate_combined_metrics(merged_data)
        
        return merged_data
    
    @staticmethod
    def _calculate_combined_metrics(merged_data: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate metrics across all data sources"""
        metrics = {
            'total_returns': 0,
            'total_reviews': 0,
            'critical_issue_count': len(merged_data.get('critical_issues', [])),
            'data_sources': len([1 for key in ['returns', 'reviews', 'quality_reports'] 
                               if merged_data.get(key)]),
            'quality_indicators': {}
        }
        
        # Returns metrics
        if 'combined_returns' in merged_data:
            returns_df = merged_data['combined_returns']
            metrics['total_returns'] = len(returns_df)
            
            # Quality-related returns
            if 'reason' in returns_df.columns:
                quality_reasons = ['DEFECTIVE', 'QUALITY_NOT_ADEQUATE', 'DAMAGED']
                quality_returns = returns_df[returns_df['reason'].str.upper().isin(quality_reasons)]
                metrics['quality_indicators']['quality_return_rate'] = (
                    len(quality_returns) / len(returns_df) * 100 if len(returns_df) > 0 else 0
                )
        
        # Reviews metrics
        if 'combined_reviews' in merged_data:
            reviews_df = merged_data['combined_reviews']
            metrics['total_reviews'] = len(reviews_df)
            
            if 'Rating' in reviews_df.columns:
                metrics['average_rating'] = reviews_df['Rating'].mean()
                metrics['negative_reviews'] = len(reviews_df[reviews_df['Rating'] <= 2])
                metrics['quality_indicators']['negative_review_rate'] = (
                    metrics['negative_reviews'] / len(reviews_df) * 100 if len(reviews_df) > 0 else 0
                )
        
        return metrics
    
    @staticmethod
    def validate_file(file: ProcessedFile) -> Tuple[bool, List[str]]:
        """Enhanced validation with quality checks"""
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
            is_valid = False
        
        # Check for critical issues
        if file.critical_issues:
            messages.append(f"⚠️ Found {len(file.critical_issues)} critical issues requiring attention")
        
        # Validate based on content type
        if file.content_category == 'fba_returns' and file.data is not None:
            required_cols = ['return_date', 'order_id', 'asin', 'reason']
            missing_cols = [col for col in required_cols 
                          if col not in file.data.columns and 
                          col.replace('_', '-') not in file.data.columns]
            if missing_cols:
                messages.append(f"Missing required columns: {', '.join(missing_cols)}")
                is_valid = False
        
        # Quality metrics validation
        if file.quality_metrics:
            if file.quality_metrics.get('quality_return_rate', 0) > 20:
                messages.append("⚠️ High quality return rate detected (>20%)")
            
            completeness = file.quality_metrics.get('data_quality', {}).get('completeness', 100)
            if completeness < 80:
                messages.append(f"Data completeness is low: {completeness:.1f}%")
        
        return is_valid, messages

# Export
__all__ = ['UniversalFileDetector', 'ProcessedFile']
