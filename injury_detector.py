"""
Enhanced Injury Detection Module - Identify Potential Injury Cases and Quality Issues
Critical for medical device safety, liability prevention, and quality management
"""

import logging
import re
from typing import Dict, List, Any, Optional, Tuple
from collections import Counter, defaultdict
from datetime import datetime

logger = logging.getLogger(__name__)

# Injury-related keywords by severity
INJURY_KEYWORDS = {
    'critical': [
        'hospital', 'emergency', 'emergency room', 'er visit', 'urgent care',
        'ambulance', 'died', 'death', 'fatal', 'life threatening',
        'severe injury', 'serious injury', 'surgery', 'operation',
        'permanent damage', 'disability', 'paralyzed', 'unconscious',
        'bleeding profusely', 'severe bleeding', 'hemorrhage',
        'anaphylactic', 'seizure', 'cardiac', 'heart attack'
    ],
    'high': [
        'injured', 'hurt badly', 'hurt seriously', 'broken bone', 'fracture',
        'bleeding', 'blood', 'wound', 'laceration', 'cut deep', 'stitches',
        'concussion', 'head injury', 'knocked out', 'passed out', 'fainted',
        'burn', 'burned', 'severe pain', 'excruciating', 'unbearable pain',
        'infection', 'infected', 'swollen badly', 'allergic reaction',
        'can\'t walk', 'can\'t move', 'immobilized', 'nerve damage',
        'hospitalized', 'medical attention', 'doctor visit'
    ],
    'medium': [
        'hurt', 'pain', 'painful', 'ache', 'sore', 'bruise', 'bruised',
        'swelling', 'swollen', 'inflammation', 'rash', 'irritation',
        'cut', 'scrape', 'scratch', 'minor bleeding', 'discomfort',
        'sprain', 'strain', 'pulled muscle', 'dizzy', 'nausea',
        'fell', 'fall', 'dropped', 'slipped', 'tripped', 'stumbled',
        'pinched', 'squeezed', 'pressure', 'numbness', 'tingling'
    ],
    'low': [
        'uncomfortable', 'slight pain', 'minor discomfort', 'tender',
        'small cut', 'minor scratch', 'slight swelling', 'red mark',
        'minor rash', 'slight irritation', 'soreness', 'stiffness'
    ]
}

# Medical device specific risk patterns
DEVICE_RISK_PATTERNS = {
    'mobility_aids': {
        'keywords': ['walker', 'wheelchair', 'cane', 'crutch', 'scooter', 'rollator', 'mobility'],
        'risks': ['fall', 'fell', 'collapsed', 'tipped', 'unstable', 'broke while using', 'gave way']
    },
    'support_devices': {
        'keywords': ['brace', 'support', 'compression', 'immobilizer', 'sling', 'splint', 'orthotic'],
        'risks': ['circulation', 'numbness', 'nerve damage', 'pressure sore', 'cut off blood', 'too tight']
    },
    'bathroom_safety': {
        'keywords': ['toilet', 'shower', 'bath', 'commode', 'grab bar', 'rail', 'seat'],
        'risks': ['slipped', 'fell in bathroom', 'broke loose', 'came off wall', 'not secure']
    },
    'beds_mattresses': {
        'keywords': ['bed', 'mattress', 'rail', 'hospital bed', 'pressure pad', 'overlay'],
        'risks': ['bed sore', 'pressure ulcer', 'fell out', 'trapped', 'entrapment', 'suffocation']
    },
    'respiratory': {
        'keywords': ['cpap', 'oxygen', 'nebulizer', 'inhaler', 'respirator', 'breathing'],
        'risks': ['breathing difficulty', 'suffocation', 'oxygen cut off', 'respiratory distress']
    },
    'monitoring': {
        'keywords': ['monitor', 'sensor', 'alarm', 'alert', 'glucose', 'blood pressure'],
        'risks': ['failed to alert', 'wrong reading', 'missed emergency', 'false alarm']
    }
}

# Quality issue patterns (non-injury)
QUALITY_PATTERNS = {
    'defective': ['defect', 'broken', 'damaged', 'doesn\'t work', 'not working', 'malfunction', 'failed'],
    'missing_parts': ['missing', 'incomplete', 'not included', 'parts missing', 'no instructions'],
    'wrong_item': ['wrong', 'different', 'not what ordered', 'incorrect'],
    'size_issues': ['too small', 'too large', 'doesn\'t fit', 'wrong size', 'sizing'],
    'performance': ['ineffective', 'doesn\'t help', 'not strong enough', 'weak', 'poor quality']
}

# Severity scoring weights
SEVERITY_WEIGHTS = {
    'critical': 1.0,
    'high': 0.8,
    'medium': 0.5,
    'low': 0.3
}

SEVERITY_LEVELS = ['critical', 'high', 'medium', 'low']

class InjuryDetector:
    """Detect and analyze potential injury cases and quality issues in return data"""
    
    def __init__(self):
        # Compile regex patterns for efficiency
        self.injury_patterns = {}
        for severity, keywords in INJURY_KEYWORDS.items():
            # Create pattern that matches whole words
            pattern = r'\b(' + '|'.join(re.escape(kw) for kw in keywords) + r')\b'
            self.injury_patterns[severity] = re.compile(pattern, re.IGNORECASE)
        
        # Compile device patterns
        self.device_patterns = {}
        for device_type, info in DEVICE_RISK_PATTERNS.items():
            device_pattern = r'\b(' + '|'.join(re.escape(kw) for kw in info['keywords']) + r')\b'
            self.device_patterns[device_type] = {
                'device': re.compile(device_pattern, re.IGNORECASE),
                'risks': info['risks']
            }
        
        # Compile quality patterns
        self.quality_patterns = {}
        for issue_type, keywords in QUALITY_PATTERNS.items():
            pattern = r'\b(' + '|'.join(re.escape(kw) for kw in keywords) + r')\b'
            self.quality_patterns[issue_type] = re.compile(pattern, re.IGNORECASE)
    
    def analyze_returns_for_injuries(self, returns: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze returns for potential injury cases and quality issues"""
        
        injury_cases = []
        quality_issues = []
        all_categorized = []
        
        severity_counts = Counter()
        injury_type_counts = Counter()
        quality_type_counts = Counter()
        
        for return_data in returns:
            # Get customer comment
            comment = self._get_comment_text(return_data)
            
            if not comment:
                # Still categorize even without comment
                categorized = self._categorize_return(return_data, '')
                all_categorized.append(categorized)
                continue
            
            # Analyze for injuries
            injury_info = self._detect_injuries(comment)
            
            # Analyze for quality issues
            quality_info = self._detect_quality_issues(comment)
            
            # Combine analysis
            return_analysis = {
                **return_data,
                'has_injury': injury_info['has_injury'],
                'has_quality_issue': quality_info['has_issue'],
                'analysis_timestamp': datetime.now().isoformat()
            }
            
            if injury_info['has_injury']:
                # Add injury information
                return_analysis.update({
                    'injury_severity': injury_info['severity'],
                    'injury_keywords': injury_info['keywords_found'],
                    'injury_risk_score': injury_info['risk_score'],
                    'device_risk': injury_info.get('device_risk', None)
                })
                
                injury_cases.append(return_analysis)
                severity_counts[injury_info['severity']] += 1
                
                # Count injury types
                for keyword in injury_info['keywords_found']:
                    injury_type_counts[keyword] += 1
            
            if quality_info['has_issue']:
                # Add quality information
                return_analysis.update({
                    'quality_issue_type': quality_info['issue_type'],
                    'quality_keywords': quality_info['keywords_found']
                })
                
                quality_issues.append(return_analysis)
                quality_type_counts[quality_info['issue_type']] += 1
            
            # Categorize for general analysis
            categorized = self._categorize_return(return_analysis, comment)
            all_categorized.append(categorized)
        
        # Separate critical cases
        critical_cases = [case for case in injury_cases if case.get('injury_severity') == 'critical']
        
        # Calculate risk assessment
        risk_assessment = self._calculate_risk_assessment(
            len(injury_cases), 
            len(returns),
            severity_counts
        )
        
        # Generate regulatory compliance check
        regulatory_info = self.check_regulatory_reporting({
            'injury_cases': injury_cases,
            'severity_breakdown': dict(severity_counts),
            'total_injuries': len(injury_cases)
        })
        
        return {
            'total_returns': len(returns),
            'total_injuries': len(injury_cases),
            'injury_cases': injury_cases,
            'critical_cases': critical_cases,
            'quality_issues': quality_issues,
            'all_categorized': all_categorized,
            'severity_breakdown': {
                'critical': severity_counts.get('critical', 0),
                'high': severity_counts.get('high', 0),
                'medium': severity_counts.get('medium', 0),
                'low': severity_counts.get('low', 0)
            },
            'injury_types': injury_type_counts,
            'quality_types': quality_type_counts,
            'risk_assessment': risk_assessment,
            'injury_rate': (len(injury_cases) / len(returns) * 100) if returns else 0,
            'quality_rate': (len(quality_issues) / len(returns) * 100) if returns else 0,
            'regulatory_info': regulatory_info
        }
    
    def _get_comment_text(self, return_data: Dict[str, Any]) -> str:
        """Extract comment text from various possible fields"""
        # Try multiple field names
        comment_fields = [
            'customer_comment', 'customer_comments', 'return_reason', 
            'reason', 'comment', 'feedback', 'issue', 'raw_content'
        ]
        
        for field in comment_fields:
            if field in return_data and return_data[field]:
                return str(return_data[field])
        
        return ''
    
    def _detect_injuries(self, text: str) -> Dict[str, Any]:
        """Detect injuries in text and determine severity"""
        
        if not text:
            return {
                'has_injury': False,
                'severity': None,
                'keywords_found': [],
                'risk_score': 0
            }
        
        keywords_found = []
        severity_scores = defaultdict(float)
        
        # Check for injury keywords by severity
        for severity, pattern in self.injury_patterns.items():
            matches = pattern.findall(text)
            if matches:
                # Store unique keywords found
                unique_matches = list(set(match.lower() for match in matches))
                keywords_found.extend(unique_matches)
                
                # Calculate severity score based on number and weight
                severity_scores[severity] += len(matches) * SEVERITY_WEIGHTS[severity]
        
        if not keywords_found:
            return {
                'has_injury': False,
                'severity': None,
                'keywords_found': [],
                'risk_score': 0
            }
        
        # Determine overall severity (highest level found)
        final_severity = None
        for severity in SEVERITY_LEVELS:
            if severity_scores[severity] > 0:
                final_severity = severity
                break
        
        # Check for device-specific risks
        device_risk = self._check_device_risks(text)
        
        # Calculate risk score (0-1)
        risk_score = min(sum(severity_scores.values()), 1.0)
        
        # Boost risk score for certain combinations
        if device_risk and final_severity in ['critical', 'high']:
            risk_score = min(risk_score * 1.2, 1.0)
        
        # Check for multiple injury indicators
        if len(keywords_found) > 3:
            risk_score = min(risk_score * 1.1, 1.0)
        
        return {
            'has_injury': True,
            'severity': final_severity,
            'keywords_found': keywords_found,
            'risk_score': risk_score,
            'device_risk': device_risk
        }
    
    def _detect_quality_issues(self, text: str) -> Dict[str, Any]:
        """Detect non-injury quality issues"""
        
        if not text:
            return {
                'has_issue': False,
                'issue_type': None,
                'keywords_found': []
            }
        
        issues_found = {}
        all_keywords = []
        
        # Check for quality issue patterns
        for issue_type, pattern in self.quality_patterns.items():
            matches = pattern.findall(text)
            if matches:
                issues_found[issue_type] = len(matches)
                all_keywords.extend([match.lower() for match in matches])
        
        if not issues_found:
            return {
                'has_issue': False,
                'issue_type': None,
                'keywords_found': []
            }
        
        # Determine primary issue type
        primary_issue = max(issues_found.items(), key=lambda x: x[1])[0]
        
        return {
            'has_issue': True,
            'issue_type': primary_issue,
            'keywords_found': list(set(all_keywords)),
            'all_issues': issues_found
        }
    
    def _check_device_risks(self, text: str) -> Optional[Dict[str, Any]]:
        """Check for device-specific injury risks"""
        
        for device_type, patterns in self.device_patterns.items():
            if patterns['device'].search(text):
                # Check if any risk patterns are present
                risks_found = []
                for risk_keyword in patterns['risks']:
                    if risk_keyword.lower() in text.lower():
                        risks_found.append(risk_keyword)
                
                if risks_found:
                    return {
                        'device_type': device_type,
                        'risks_found': risks_found,
                        'risk_level': 'high' if len(risks_found) > 1 else 'medium'
                    }
        
        return None
    
    def _categorize_return(self, return_data: Dict[str, Any], comment: str) -> Dict[str, Any]:
        """Categorize return into standard categories"""
        
        # Start with return data
        categorized = return_data.copy()
        
        # Determine category based on analysis
        if return_data.get('has_injury'):
            if return_data.get('injury_severity') in ['critical', 'high']:
                categorized['category'] = 'Medical/Health Concerns'
            else:
                categorized['category'] = 'Product Defects/Quality'
        elif return_data.get('has_quality_issue'):
            issue_type = return_data.get('quality_issue_type')
            category_map = {
                'defective': 'Product Defects/Quality',
                'missing_parts': 'Missing Components',
                'wrong_item': 'Wrong Product/Misunderstanding',
                'size_issues': 'Size/Fit Issues',
                'performance': 'Performance/Effectiveness'
            }
            categorized['category'] = category_map.get(issue_type, 'Other/Miscellaneous')
        else:
            # Use FBA reason if available
            fba_reason = return_data.get('fba_reason', '')
            if fba_reason:
                from enhanced_ai_analysis import FBA_REASON_MAP
                categorized['category'] = FBA_REASON_MAP.get(fba_reason, 'Other/Miscellaneous')
            else:
                categorized['category'] = 'Other/Miscellaneous'
        
        return categorized
    
    def _calculate_risk_assessment(self, injury_count: int, total_returns: int, 
                                 severity_counts: Counter) -> str:
        """Calculate overall risk assessment"""
        
        if total_returns == 0:
            return "No data"
        
        injury_rate = (injury_count / total_returns) * 100
        critical_count = severity_counts.get('critical', 0)
        high_count = severity_counts.get('high', 0)
        
        # Risk assessment logic
        if critical_count > 0:
            return "CRITICAL - Immediate action required"
        elif injury_rate > 5 or high_count > 3:
            return "HIGH - Urgent review needed"
        elif injury_rate > 2 or high_count > 1:
            return "MEDIUM - Close monitoring required"
        elif injury_count > 0:
            return "LOW - Continue standard monitoring"
        else:
            return "MINIMAL - No injuries detected"
    
    def generate_injury_report(self, analysis: Dict[str, Any]) -> str:
        """Generate a formatted injury report"""
        
        report = []
        report.append("INJURY & QUALITY ANALYSIS REPORT")
        report.append("=" * 50)
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        # Summary
        report.append("SUMMARY")
        report.append("-" * 30)
        report.append(f"Total Returns Analyzed: {analysis.get('total_returns', 0)}")
        report.append(f"Injury Cases Found: {analysis['total_injuries']} ({analysis.get('injury_rate', 0):.1f}%)")
        report.append(f"Quality Issues Found: {len(analysis.get('quality_issues', []))} ({analysis.get('quality_rate', 0):.1f}%)")
        report.append(f"Risk Assessment: {analysis['risk_assessment']}")
        report.append("")
        
        # Severity breakdown
        if analysis['total_injuries'] > 0:
            report.append("INJURY SEVERITY BREAKDOWN")
            report.append("-" * 30)
            for severity in SEVERITY_LEVELS:
                count = analysis['severity_breakdown'][severity]
                if count > 0:
                    report.append(f"{severity.upper()}: {count}")
            report.append("")
            
            # Top injury types
            report.append("TOP INJURY TYPES")
            report.append("-" * 30)
            for injury_type, count in analysis.get('injury_types', Counter()).most_common(10):
                report.append(f"{injury_type}: {count}")
            report.append("")
        
        # Quality issues
        if analysis.get('quality_types'):
            report.append("QUALITY ISSUE BREAKDOWN")
            report.append("-" * 30)
            for issue_type, count in analysis['quality_types'].most_common():
                report.append(f"{issue_type}: {count}")
            report.append("")
        
        # Critical cases
        if analysis.get('critical_cases'):
            report.append("CRITICAL CASES REQUIRING IMMEDIATE ATTENTION")
            report.append("-" * 30)
            for case in analysis['critical_cases'][:10]:  # Top 10
                report.append(f"Order: {case.get('order_id', 'Unknown')}")
                report.append(f"ASIN: {case.get('asin', 'Unknown')}")
                report.append(f"SKU: {case.get('sku', 'Unknown')}")
                report.append(f"Date: {case.get('return_date', 'Unknown')}")
                report.append(f"Injury Keywords: {', '.join(case.get('injury_keywords', []))}")
                report.append(f"Comment: {case.get('customer_comment', 'No comment')[:200]}...")
                report.append("-" * 20)
            report.append("")
        
        # Regulatory compliance
        if analysis.get('regulatory_info'):
            report.append("REGULATORY COMPLIANCE")
            report.append("-" * 30)
            reg_info = analysis['regulatory_info']
            report.append(f"Reporting Required: {'YES' if reg_info['reporting_required'] else 'NO'}")
            if reg_info['reporting_required']:
                report.append("Reasons:")
                for reason in reg_info['reasons']:
                    report.append(f"  - {reason}")
                report.append(f"Recommendation: {reg_info['recommendation']}")
        
        return '\n'.join(report)
    
    def check_regulatory_reporting(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Check if injuries require regulatory reporting"""
        
        reporting_required = False
        reporting_reasons = []
        reporting_type = []
        
        # FDA reporting criteria for medical devices
        if analysis['severity_breakdown'].get('critical', 0) > 0:
            reporting_required = True
            reporting_reasons.append("Critical injuries detected")
            reporting_type.append("MDR (Medical Device Report)")
        
        if analysis['severity_breakdown'].get('high', 0) >= 3:
            reporting_required = True
            reporting_reasons.append("Multiple serious injuries")
            reporting_type.append("MDR (Medical Device Report)")
        
        # Check for specific reportable events
        reportable_keywords = ['death', 'permanent', 'surgery', 'hospitalization', 'life threatening']
        for case in analysis.get('injury_cases', []):
            keywords = case.get('injury_keywords', [])
            if any(kw in ' '.join(keywords).lower() for kw in reportable_keywords):
                reporting_required = True
                reporting_reasons.append("Reportable adverse event detected")
                reporting_type.append("MDR (Medical Device Report)")
                break
        
        # EU MDR requirements (if applicable)
        if analysis.get('total_injuries', 0) >= 5:
            reporting_reasons.append("Multiple injuries - trend monitoring required")
            reporting_type.append("Trend Report")
        
        return {
            'reporting_required': reporting_required,
            'reasons': reporting_reasons,
            'reporting_types': list(set(reporting_type)),
            'recommendation': self._get_regulatory_recommendation(reporting_required, reporting_type),
            'timeline': self._get_reporting_timeline(reporting_type)
        }
    
    def _get_regulatory_recommendation(self, required: bool, types: List[str]) -> str:
        """Get specific regulatory recommendation"""
        
        if not required:
            return "Continue routine monitoring. Document all cases for trending."
        
        if "MDR (Medical Device Report)" in types:
            return "IMMEDIATE ACTION: Contact regulatory affairs. Prepare MDR within 30 days (5 days if death/serious injury)."
        elif "Trend Report" in types:
            return "Prepare trend analysis report for next regulatory submission."
        else:
            return "Review with regulatory affairs team for reporting requirements."
    
    def _get_reporting_timeline(self, types: List[str]) -> Dict[str, str]:
        """Get reporting timeline requirements"""
        
        timeline = {}
        
        if "MDR (Medical Device Report)" in types:
            timeline['MDR'] = {
                'death_serious_injury': '5 calendar days',
                'malfunction': '30 calendar days',
                'supplemental': '30 calendar days of becoming aware'
            }
        
        if "Trend Report" in types:
            timeline['trend'] = 'With periodic regulatory updates'
        
        return timeline
    
    def export_for_regulatory(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Export data formatted for regulatory reporting"""
        
        regulatory_export = {
            'report_date': datetime.now().isoformat(),
            'reporting_entity': 'Quality Management System',
            'summary': {
                'total_complaints': analysis.get('total_returns', 0),
                'injury_complaints': analysis.get('total_injuries', 0),
                'critical_events': analysis['severity_breakdown'].get('critical', 0),
                'serious_injuries': analysis['severity_breakdown'].get('high', 0)
            },
            'events': []
        }
        
        # Add critical and high severity cases
        for case in analysis.get('injury_cases', []):
            if case.get('injury_severity') in ['critical', 'high']:
                regulatory_export['events'].append({
                    'event_date': case.get('return_date', 'Unknown'),
                    'report_date': datetime.now().isoformat(),
                    'product_identifier': {
                        'asin': case.get('asin', ''),
                        'sku': case.get('sku', ''),
                        'order_id': case.get('order_id', '')
                    },
                    'event_description': case.get('customer_comment', ''),
                    'injury_keywords': case.get('injury_keywords', []),
                    'severity': case.get('injury_severity', ''),
                    'device_involvement': case.get('device_risk', {})
                })
        
        return regulatory_export

# Export classes and constants
__all__ = [
    'InjuryDetector', 
    'INJURY_KEYWORDS', 
    'SEVERITY_LEVELS',
    'DEVICE_RISK_PATTERNS',
    'QUALITY_PATTERNS'
]
