"""
Injury Detection Module - Identify Potential Injury Cases in Returns
Critical for medical device safety and liability prevention
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
        'bleeding profusely', 'severe bleeding', 'hemorrhage'
    ],
    'high': [
        'injured', 'hurt badly', 'hurt seriously', 'broken bone', 'fracture',
        'bleeding', 'blood', 'wound', 'laceration', 'cut deep', 'stitches',
        'concussion', 'head injury', 'knocked out', 'passed out',
        'burn', 'burned', 'severe pain', 'excruciating', 'unbearable pain',
        'infection', 'infected', 'swollen badly', 'allergic reaction',
        'can\'t walk', 'can\'t move', 'immobilized'
    ],
    'medium': [
        'hurt', 'pain', 'painful', 'ache', 'sore', 'bruise', 'bruised',
        'swelling', 'swollen', 'inflammation', 'rash', 'irritation',
        'cut', 'scrape', 'scratch', 'minor bleeding', 'discomfort',
        'sprain', 'strain', 'pulled muscle', 'dizzy', 'nausea',
        'fell', 'fall', 'dropped', 'slipped', 'tripped'
    ],
    'low': [
        'uncomfortable', 'slight pain', 'minor discomfort', 'tender',
        'small cut', 'minor scratch', 'slight swelling', 'red mark',
        'minor rash', 'slight irritation'
    ]
}

# Medical device specific risk patterns
DEVICE_RISK_PATTERNS = {
    'mobility_aids': {
        'keywords': ['walker', 'wheelchair', 'cane', 'crutch', 'scooter', 'rollator'],
        'risks': ['fall', 'fell', 'collapsed', 'tipped', 'unstable', 'broke while using']
    },
    'support_devices': {
        'keywords': ['brace', 'support', 'compression', 'immobilizer', 'sling'],
        'risks': ['circulation', 'numbness', 'nerve damage', 'pressure sore', 'cut off blood']
    },
    'bathroom_safety': {
        'keywords': ['toilet', 'shower', 'bath', 'commode', 'grab bar'],
        'risks': ['slipped', 'fell in bathroom', 'broke loose', 'came off wall']
    },
    'beds_mattresses': {
        'keywords': ['bed', 'mattress', 'rail', 'hospital bed'],
        'risks': ['bed sore', 'pressure ulcer', 'fell out', 'trapped', 'entrapment']
    }
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
    """Detect and analyze potential injury cases in return data"""
    
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
    
    def analyze_returns_for_injuries(self, returns: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze returns for potential injury cases"""
        
        injury_cases = []
        severity_counts = Counter()
        injury_type_counts = Counter()
        
        for return_data in returns:
            # Get customer comment
            comment = return_data.get('customer_comment', '')
            if not comment:
                # Check other possible fields
                comment = return_data.get('return_reason', '') or return_data.get('raw_content', '')
            
            if not comment:
                continue
            
            # Analyze for injuries
            injury_info = self._detect_injuries(comment)
            
            if injury_info['has_injury']:
                # Add injury information to return data
                injury_case = {
                    **return_data,
                    'severity': injury_info['severity'],
                    'injury_keywords': injury_info['keywords_found'],
                    'risk_score': injury_info['risk_score'],
                    'device_risk': injury_info.get('device_risk', None)
                }
                
                injury_cases.append(injury_case)
                severity_counts[injury_info['severity']] += 1
                
                # Count injury types
                for keyword in injury_info['keywords_found']:
                    injury_type_counts[keyword] += 1
        
        # Separate critical cases
        critical_cases = [case for case in injury_cases if case['severity'] == 'critical']
        
        # Calculate risk assessment
        risk_assessment = self._calculate_risk_assessment(
            len(injury_cases), 
            len(returns),
            severity_counts
        )
        
        return {
            'total_injuries': len(injury_cases),
            'injury_cases': injury_cases,
            'critical_cases': critical_cases,
            'severity_breakdown': {
                'critical': severity_counts.get('critical', 0),
                'high': severity_counts.get('high', 0),
                'medium': severity_counts.get('medium', 0),
                'low': severity_counts.get('low', 0)
            },
            'injury_types': injury_type_counts,
            'risk_assessment': risk_assessment,
            'injury_rate': (len(injury_cases) / len(returns) * 100) if returns else 0
        }
    
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
        
        return {
            'has_injury': True,
            'severity': final_severity,
            'keywords_found': keywords_found,
            'risk_score': risk_score,
            'device_risk': device_risk
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
                        'risks_found': risks_found
                    }
        
        return None
    
    def _calculate_risk_assessment(self, injury_count: int, total_returns: int, 
                                 severity_counts: Counter) -> str:
        """Calculate overall risk assessment"""
        
        if total_returns == 0:
            return "No data"
        
        injury_rate = (injury_count / total_returns) * 100
        critical_count = severity_counts.get('critical', 0)
        high_count = severity_counts.get('high', 0)
        
        # Risk assessment logic
        if critical_count > 0 or injury_rate > 5:
            return "CRITICAL - Immediate action required"
        elif high_count > 2 or injury_rate > 2:
            return "HIGH - Urgent review needed"
        elif injury_rate > 1:
            return "MEDIUM - Close monitoring required"
        else:
            return "LOW - Continue standard monitoring"
    
    def generate_injury_report(self, analysis: Dict[str, Any]) -> str:
        """Generate a formatted injury report"""
        
        report = []
        report.append("INJURY ANALYSIS REPORT")
        report.append("=" * 50)
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        # Summary
        report.append("SUMMARY")
        report.append("-" * 30)
        report.append(f"Total Injuries Found: {analysis['total_injuries']}")
        report.append(f"Injury Rate: {analysis['injury_rate']:.1f}%")
        report.append(f"Risk Assessment: {analysis['risk_assessment']}")
        report.append("")
        
        # Severity breakdown
        report.append("SEVERITY BREAKDOWN")
        report.append("-" * 30)
        for severity in SEVERITY_LEVELS:
            count = analysis['severity_breakdown'][severity]
            report.append(f"{severity.upper()}: {count}")
        report.append("")
        
        # Top injury types
        report.append("TOP INJURY TYPES")
        report.append("-" * 30)
        for injury_type, count in analysis['injury_types'].most_common(10):
            report.append(f"{injury_type}: {count}")
        report.append("")
        
        # Critical cases
        if analysis['critical_cases']:
            report.append("CRITICAL CASES")
            report.append("-" * 30)
            for case in analysis['critical_cases']:
                report.append(f"Order: {case.get('order_id', 'Unknown')}")
                report.append(f"ASIN: {case.get('asin', 'Unknown')}")
                report.append(f"Comment: {case.get('customer_comment', 'No comment')[:200]}...")
                report.append("")
        
        return '\n'.join(report)
    
    def check_regulatory_reporting(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Check if injuries require regulatory reporting"""
        
        reporting_required = False
        reporting_reasons = []
        
        # FDA reporting criteria for medical devices
        if analysis['severity_breakdown']['critical'] > 0:
            reporting_required = True
            reporting_reasons.append("Critical injuries detected")
        
        if analysis['severity_breakdown']['high'] >= 3:
            reporting_required = True
            reporting_reasons.append("Multiple serious injuries")
        
        # Check for specific reportable events
        reportable_keywords = ['death', 'permanent', 'surgery', 'hospitalization']
        for case in analysis['injury_cases']:
            if any(kw in ' '.join(case['injury_keywords']).lower() for kw in reportable_keywords):
                reporting_required = True
                reporting_reasons.append("Reportable adverse event detected")
                break
        
        return {
            'reporting_required': reporting_required,
            'reasons': reporting_reasons,
            'recommendation': "Contact regulatory affairs immediately" if reporting_required else "Continue monitoring"
        }

# Export
__all__ = ['InjuryDetector', 'INJURY_KEYWORDS', 'SEVERITY_LEVELS']
