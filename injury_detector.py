"""
Injury Detector Module - Critical Safety Analysis for Medical Device Returns
Identifies potential injuries and safety issues in return comments
"""

import re
import logging
from typing import Dict, List, Any, Optional, Tuple
from collections import Counter
from datetime import datetime

logger = logging.getLogger(__name__)

class InjuryDetector:
    """Detect and analyze potential injuries in medical device returns"""
    
    def __init__(self, ai_analyzer=None):
        """Initialize injury detector with optional AI enhancement"""
        self.ai_analyzer = ai_analyzer
        
        # Critical injury keywords by severity
        self.INJURY_KEYWORDS = {
            'critical': {
                'keywords': [
                    'death', 'died', 'fatal', 'emergency room', 'ER visit', 'hospitalized',
                    'ambulance', 'surgery', 'surgical', 'severe injury', 'serious injury',
                    'broken bone', 'fracture', 'concussion', 'unconscious', 'coma'
                ],
                'patterns': [
                    r'went to (the )?hospital',
                    r'rushed to (the )?ER',
                    r'called 911',
                    r'permanent (damage|injury)',
                    r'life[- ]threatening'
                ]
            },
            'high': {
                'keywords': [
                    'injured', 'hurt', 'wound', 'bleeding', 'blood', 'cut', 'laceration',
                    'burn', 'burned', 'bruise', 'bruised', 'swelling', 'swollen',
                    'infection', 'infected', 'rash', 'allergic reaction', 'pain',
                    'painful', 'doctor', 'medical attention', 'clinic'
                ],
                'patterns': [
                    r'(severe|extreme|unbearable) pain',
                    r'couldn\'t (walk|move|sleep)',
                    r'had to see (a|my) doctor',
                    r'went to urgent care',
                    r'caused an? (injury|wound)',
                    r'drew blood'
                ]
            },
            'medium': {
                'keywords': [
                    'discomfort', 'uncomfortable', 'sore', 'ache', 'aching',
                    'irritation', 'irritated', 'red', 'redness', 'mark', 'marks'
                ],
                'patterns': [
                    r'left (a )?mark',
                    r'caused discomfort',
                    r'minor (injury|pain)',
                    r'slight(ly)? (hurt|injured)'
                ]
            }
        }
        
        # Body parts often mentioned in injury reports
        self.BODY_PARTS = [
            'head', 'face', 'eye', 'ear', 'nose', 'mouth', 'teeth', 'neck',
            'shoulder', 'arm', 'elbow', 'wrist', 'hand', 'finger', 'chest',
            'back', 'spine', 'hip', 'leg', 'knee', 'ankle', 'foot', 'toe',
            'skin', 'bone', 'muscle'
        ]
        
        # Medical device failure modes that could cause injury
        self.FAILURE_MODES = {
            'mechanical': ['broke', 'snapped', 'collapsed', 'fell apart', 'shattered', 'cracked'],
            'stability': ['tipped over', 'fell', 'unstable', 'wobbled', 'gave way'],
            'sharp': ['sharp edge', 'pointed', 'cut myself', 'sliced'],
            'pinch': ['pinched', 'caught', 'trapped', 'crushed'],
            'chemical': ['burned', 'reaction', 'irritation', 'rash']
        }
    
    def analyze_returns_for_injuries(self, returns: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analyze returns for potential injuries
        
        Args:
            returns: List of return dictionaries
            
        Returns:
            Dictionary with injury analysis results
        """
        injury_cases = []
        severity_counts = Counter()
        injury_types = Counter()
        affected_products = Counter()
        
        for return_item in returns:
            # Combine all text fields for analysis
            text_to_analyze = ' '.join([
                str(return_item.get('buyer_comment', '')),
                str(return_item.get('return_reason', '')),
                str(return_item.get('customer_comment', ''))
            ]).lower()
            
            if not text_to_analyze.strip():
                continue
            
            # Detect injury
            injury_result = self._detect_injury(text_to_analyze, return_item)
            
            if injury_result['has_injury']:
                injury_case = {
                    'order_id': return_item.get('order_id', 'Unknown'),
                    'asin': return_item.get('asin', 'Unknown'),
                    'sku': return_item.get('sku', 'Unknown'),
                    'severity': injury_result['severity'],
                    'injury_type': injury_result['injury_type'],
                    'description': injury_result['description'],
                    'keywords_found': injury_result['keywords_found'],
                    'body_parts': injury_result['body_parts'],
                    'failure_mode': injury_result['failure_mode'],
                    'original_comment': return_item.get('buyer_comment', ''),
                    'return_date': return_item.get('return_date', ''),
                    'confidence': injury_result['confidence']
                }
                
                injury_cases.append(injury_case)
                severity_counts[injury_result['severity']] += 1
                injury_types[injury_result['injury_type']] += 1
                
                # Track by product
                product_key = f"{injury_case['asin']}_{injury_case['sku']}"
                affected_products[product_key] += 1
        
        # Calculate risk metrics
        total_returns = len(returns)
        total_injuries = len(injury_cases)
        injury_rate = (total_injuries / total_returns * 100) if total_returns > 0 else 0
        
        # Sort injury cases by severity
        injury_cases.sort(key=lambda x: {'critical': 0, 'high': 1, 'medium': 2, 'low': 3}.get(x['severity'], 4))
        
        # Identify high-risk products
        high_risk_products = [
            {
                'product': prod.split('_')[0],  # ASIN
                'sku': prod.split('_')[1] if '_' in prod else '',
                'injury_count': count
            }
            for prod, count in affected_products.most_common(10)
            if count >= 2  # Multiple injuries from same product
        ]
        
        return {
            'total_returns': total_returns,
            'total_injuries': total_injuries,
            'injury_rate': injury_rate,
            'severity_breakdown': dict(severity_counts),
            'injury_types': dict(injury_types),
            'high_risk_products': high_risk_products,
            'injury_cases': injury_cases,
            'critical_cases': [c for c in injury_cases if c['severity'] == 'critical'],
            'risk_assessment': self._calculate_risk_assessment(injury_cases, total_returns)
        }
    
    def _detect_injury(self, text: str, return_item: Dict[str, Any]) -> Dict[str, Any]:
        """Detect if text contains injury indicators"""
        result = {
            'has_injury': False,
            'severity': 'none',
            'injury_type': 'none',
            'description': '',
            'keywords_found': [],
            'body_parts': [],
            'failure_mode': '',
            'confidence': 0.0
        }
        
        # Check for injury keywords by severity
        for severity, data in self.INJURY_KEYWORDS.items():
            keywords_found = []
            
            # Check keywords
            for keyword in data['keywords']:
                if keyword in text:
                    keywords_found.append(keyword)
            
            # Check patterns
            for pattern in data.get('patterns', []):
                if re.search(pattern, text):
                    keywords_found.append(f"pattern: {pattern}")
            
            if keywords_found:
                result['has_injury'] = True
                result['severity'] = severity
                result['keywords_found'] = keywords_found
                break
        
        if not result['has_injury']:
            return result
        
        # Identify body parts mentioned
        body_parts = []
        for part in self.BODY_PARTS:
            if part in text:
                body_parts.append(part)
        result['body_parts'] = body_parts
        
        # Identify failure mode
        for mode, indicators in self.FAILURE_MODES.items():
            if any(indicator in text for indicator in indicators):
                result['failure_mode'] = mode
                break
        
        # Classify injury type
        result['injury_type'] = self._classify_injury_type(text, result['keywords_found'])
        
        # Extract description
        result['description'] = self._extract_injury_description(text, result['keywords_found'])
        
        # Calculate confidence
        result['confidence'] = self._calculate_confidence(result)
        
        # Use AI for enhanced analysis if available
        if self.ai_analyzer and result['confidence'] < 0.7:
            ai_result = self._enhance_with_ai(text, return_item)
            if ai_result:
                result.update(ai_result)
        
        return result
    
    def _classify_injury_type(self, text: str, keywords: List[str]) -> str:
        """Classify the type of injury"""
        # Check for specific injury types
        if any(word in text for word in ['cut', 'laceration', 'slice', 'sharp']):
            return 'laceration'
        elif any(word in text for word in ['burn', 'burned', 'burning']):
            return 'burn'
        elif any(word in text for word in ['bruise', 'bruised', 'bruising']):
            return 'contusion'
        elif any(word in text for word in ['fracture', 'broken', 'break']):
            return 'fracture'
        elif any(word in text for word in ['sprain', 'strain', 'pulled']):
            return 'sprain/strain'
        elif any(word in text for word in ['allergic', 'reaction', 'rash', 'hives']):
            return 'allergic_reaction'
        elif any(word in text for word in ['infection', 'infected']):
            return 'infection'
        elif any(word in text for word in ['fall', 'fell', 'dropped']):
            return 'fall_injury'
        else:
            return 'unspecified_injury'
    
    def _extract_injury_description(self, text: str, keywords: List[str]) -> str:
        """Extract relevant injury description from text"""
        # Find sentences containing injury keywords
        sentences = re.split(r'[.!?]', text)
        relevant_sentences = []
        
        for sentence in sentences:
            if any(keyword in sentence.lower() for keyword in keywords if not keyword.startswith('pattern:')):
                relevant_sentences.append(sentence.strip())
        
        # Return first 2 most relevant sentences
        description = '. '.join(relevant_sentences[:2])
        return description[:300] if description else "Injury mentioned but details unclear"
    
    def _calculate_confidence(self, result: Dict[str, Any]) -> float:
        """Calculate confidence score for injury detection"""
        confidence = 0.0
        
        # Base confidence on severity
        severity_scores = {'critical': 0.9, 'high': 0.7, 'medium': 0.5}
        confidence = severity_scores.get(result['severity'], 0.3)
        
        # Adjust based on keywords found
        keyword_count = len(result['keywords_found'])
        if keyword_count > 3:
            confidence += 0.2
        elif keyword_count > 1:
            confidence += 0.1
        
        # Adjust based on body parts mentioned
        if result['body_parts']:
            confidence += 0.1
        
        # Adjust based on failure mode
        if result['failure_mode']:
            confidence += 0.1
        
        return min(confidence, 1.0)
    
    def _enhance_with_ai(self, text: str, return_item: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Use AI to enhance injury detection"""
        if not self.ai_analyzer:
            return None
        
        try:
            # This would call the AI analyzer for injury detection
            # For now, return None as placeholder
            return None
        except Exception as e:
            logger.error(f"AI injury enhancement failed: {e}")
            return None
    
    def _calculate_risk_assessment(self, injury_cases: List[Dict[str, Any]], 
                                  total_returns: int) -> Dict[str, Any]:
        """Calculate overall risk assessment"""
        if total_returns == 0:
            return {
                'risk_level': 'UNKNOWN',
                'description': 'No return data to analyze',
                'recommendations': []
            }
        
        injury_rate = (len(injury_cases) / total_returns) * 100
        critical_count = sum(1 for case in injury_cases if case['severity'] == 'critical')
        high_count = sum(1 for case in injury_cases if case['severity'] == 'high')
        
        # Determine risk level
        if critical_count > 0:
            risk_level = 'CRITICAL'
            description = f"IMMEDIATE ACTION REQUIRED: {critical_count} critical injury case(s) detected"
        elif injury_rate > 5 or high_count > 3:
            risk_level = 'HIGH'
            description = f"High injury rate ({injury_rate:.1f}%) or multiple serious injuries"
        elif injury_rate > 2 or high_count > 1:
            risk_level = 'MEDIUM'
            description = f"Moderate injury rate ({injury_rate:.1f}%) requiring attention"
        elif len(injury_cases) > 0:
            risk_level = 'LOW'
            description = f"Low injury rate ({injury_rate:.1f}%) but monitoring needed"
        else:
            risk_level = 'MINIMAL'
            description = "No injuries detected"
        
        # Generate recommendations
        recommendations = self._generate_recommendations(
            risk_level, injury_cases, injury_rate
        )
        
        return {
            'risk_level': risk_level,
            'description': description,
            'injury_rate': injury_rate,
            'recommendations': recommendations
        }
    
    def _generate_recommendations(self, risk_level: str, injury_cases: List[Dict[str, Any]], 
                                injury_rate: float) -> List[str]:
        """Generate actionable recommendations based on injury analysis"""
        recommendations = []
        
        if risk_level == 'CRITICAL':
            recommendations.extend([
                "🚨 IMMEDIATE: Report critical injuries to regulatory authorities",
                "🚨 IMMEDIATE: Consider product recall or safety notice",
                "🚨 IMMEDIATE: Contact affected customers directly",
                "📋 Document all injury cases for regulatory compliance",
                "🔍 Conduct immediate root cause analysis"
            ])
        elif risk_level == 'HIGH':
            recommendations.extend([
                "⚠️ URGENT: Review product design and safety features",
                "⚠️ URGENT: Implement additional quality controls",
                "📊 Analyze injury patterns to identify common failure modes",
                "📝 Update product warnings and instructions",
                "🏥 Consider proactive customer safety communications"
            ])
        elif risk_level == 'MEDIUM':
            recommendations.extend([
                "🔍 Monitor injury trends closely",
                "📋 Review and update safety documentation",
                "🛠️ Evaluate design improvements for next revision",
                "📊 Implement enhanced quality testing for affected products"
            ])
        elif risk_level == 'LOW':
            recommendations.extend([
                "✓ Continue standard safety monitoring",
                "📊 Track injury trends over time",
                "📝 Document cases for future reference"
            ])
        
        # Add specific recommendations based on injury types
        injury_types = Counter(case['injury_type'] for case in injury_cases)
        
        if injury_types.get('laceration', 0) > 0:
            recommendations.append("🔧 Review product for sharp edges or points")
        
        if injury_types.get('fall_injury', 0) > 0:
            recommendations.append("⚖️ Evaluate product stability and anti-tip features")
        
        if injury_types.get('allergic_reaction', 0) > 0:
            recommendations.append("🧪 Review materials and allergen warnings")
        
        return recommendations
    
    def generate_injury_report(self, analysis: Dict[str, Any]) -> str:
        """Generate a formatted injury report for quality managers"""
        report = []
        report.append("=" * 60)
        report.append("MEDICAL DEVICE INJURY ANALYSIS REPORT")
        report.append("=" * 60)
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"Analysis Period: Return data analysis")
        report.append("")
        
        # Executive Summary
        report.append("EXECUTIVE SUMMARY")
        report.append("-" * 40)
        report.append(f"Risk Level: {analysis['risk_assessment']['risk_level']}")
        report.append(f"Total Returns Analyzed: {analysis['total_returns']}")
        report.append(f"Injury Cases Found: {analysis['total_injuries']}")
        report.append(f"Injury Rate: {analysis['injury_rate']:.2f}%")
        report.append(f"Critical Cases: {len(analysis['critical_cases'])}")
        report.append("")
        
        # Risk Assessment
        report.append("RISK ASSESSMENT")
        report.append("-" * 40)
        report.append(analysis['risk_assessment']['description'])
        report.append("")
        
        # Severity Breakdown
        if analysis['severity_breakdown']:
            report.append("SEVERITY BREAKDOWN")
            report.append("-" * 40)
            for severity, count in analysis['severity_breakdown'].items():
                report.append(f"{severity.upper()}: {count} cases")
            report.append("")
        
        # High Risk Products
        if analysis['high_risk_products']:
            report.append("HIGH RISK PRODUCTS")
            report.append("-" * 40)
            for product in analysis['high_risk_products']:
                report.append(f"ASIN: {product['product']}, "
                            f"SKU: {product['sku']}, "
                            f"Injuries: {product['injury_count']}")
            report.append("")
        
        # Critical Cases Detail
        if analysis['critical_cases']:
            report.append("CRITICAL CASES - IMMEDIATE ATTENTION REQUIRED")
            report.append("-" * 40)
            for case in analysis['critical_cases']:
                report.append(f"Order: {case['order_id']}")
                report.append(f"Product: {case['asin']} / {case['sku']}")
                report.append(f"Description: {case['description']}")
                report.append(f"Keywords: {', '.join(case['keywords_found'])}")
                report.append("-" * 20)
            report.append("")
        
        # Recommendations
        report.append("RECOMMENDATIONS")
        report.append("-" * 40)
        for rec in analysis['risk_assessment']['recommendations']:
            report.append(rec)
        report.append("")
        
        # Footer
        report.append("=" * 60)
        report.append("END OF REPORT")
        
        return '\n'.join(report)

# Export class
__all__ = ['InjuryDetector']
