"""
심리 분석 서비스
패턴 인식 및 위험도 평가
"""
import re
from typing import Dict, List, Tuple
import logging

logger = logging.getLogger(__name__)


class PsychologicalAnalyzer:
    """심리상담 전문 패턴 분석기"""
    
    def __init__(self):
        # 심리학적 위험 패턴
        self.risk_patterns = {
            'critical': [
                r'죽고?\s*싶', r'자살', r'사라지고?\s*싶', r'끝내고?\s*싶',
                r'소용없', r'의미없', r'가치없', r'살기?\s*싫어?'
            ],
            'high': [
                r'우울해?', r'절망', r'포기', r'한계', r'견딜?\s*수?\s*없',
                r'도움.*없', r'혼자.*감당', r'아무도.*모르'
            ],
            'medium': [
                r'불안해?', r'걱정', r'두려워?', r'무서워?', r'스트레스',
                r'화가?\s*나', r'짜증', r'분노'
            ],
            'low': [
                r'힘들', r'어렵', r'고민', r'문제', r'신경쓰'
            ]
        }
        
        # 심리상담 전문 키워드
        self.counseling_keywords = {
            'trauma': [r'트라우마', r'악몽', r'플래시백', r'기억이?\s*자꾸'],
            'depression': [r'우울', r'무기력', r'슬프', r'절망', r'힘들'],
            'anxiety': [r'불안', r'걱정', r'긴장', r'초조', r'두려'],
            'anger': [r'화', r'분노', r'짜증', r'열받', r'억울'],
            'self_harm': [r'자해', r'상처.*내', r'아프게.*하고?\s*싶'],
            'help_seeking': [r'도와', r'상담', r'치료', r'병원', r'의사']
        }
        
        # 위험도별 권장사항
        self.recommendations = {
            'critical': "즉시 전문가 상담이 필요합니다. 긴급 상담 서비스를 이용해주세요.",
            'high': "전문 상담사와의 상담을 권장합니다. 혼자 해결하려 하지 마세요.",
            'medium': "스트레스 관리가 필요합니다. 가까운 상담 센터를 방문해보세요.",
            'low': "일상적인 스트레스입니다. 휴식과 자기관리를 권장합니다.",
            'safe': "건강한 정서 상태입니다. 현재 상태를 유지하세요."
        }
    
    def analyze_risk_level(self, text: str) -> Dict:
        """위험도 평가"""
        risk_score = 0.0
        detected_keywords = []
        detected_level = 'safe'
        
        # 위험 패턴 검사
        for level, patterns in self.risk_patterns.items():
            for pattern in patterns:
                if re.search(pattern, text, re.IGNORECASE):
                    detected_keywords.append(pattern.replace(r'\s*', '').replace('?', ''))
                    
                    if level == 'critical':
                        risk_score = max(risk_score, 1.0)
                        detected_level = 'critical'
                    elif level == 'high':
                        risk_score = max(risk_score, 0.8)
                        if detected_level != 'critical':
                            detected_level = 'high'
                    elif level == 'medium':
                        risk_score = max(risk_score, 0.5)
                        if detected_level not in ['critical', 'high']:
                            detected_level = 'medium'
                    elif level == 'low':
                        risk_score = max(risk_score, 0.3)
                        if detected_level == 'safe':
                            detected_level = 'low'
        
        return {
            'level': detected_level,
            'score': risk_score,
            'keywords': list(set(detected_keywords)),
            'recommendation': self.recommendations[detected_level]
        }
    
    def detect_psychological_patterns(self, text: str) -> Dict[str, List[str]]:
        """심리 패턴 감지"""
        detected_patterns = {}
        
        for category, patterns in self.counseling_keywords.items():
            matches = []
            for pattern in patterns:
                if re.search(pattern, text, re.IGNORECASE):
                    matches.append(pattern.replace(r'\s*', '').replace('?', ''))
            
            if matches:
                detected_patterns[category] = matches
        
        return detected_patterns
    
    def generate_counseling_suggestions(
        self,
        emotion: str,
        risk_level: str,
        patterns: Dict[str, List[str]]
    ) -> List[str]:
        """상담 제안 생성"""
        suggestions = []
        
        # 감정별 기본 제안
        emotion_suggestions = {
            'joy': ["긍정적인 감정을 유지하세요", "감사 일기를 써보세요"],
            'sad': ["슬픔을 표현하는 것은 자연스러운 일입니다", "주변 사람들과 대화해보세요"],
            'anxiety': ["심호흡을 통해 긴장을 풀어보세요", "명상이나 요가를 시도해보세요"],
            'anger': ["분노를 건강하게 표현하는 방법을 찾아보세요", "운동으로 스트레스를 해소해보세요"],
            'neutral': ["현재 상태를 유지하세요", "자기관리를 계속하세요"]
        }
        
        suggestions.extend(emotion_suggestions.get(emotion, []))
        
        # 위험도별 추가 제안
        if risk_level in ['critical', 'high']:
            suggestions.append("전문가 상담을 강력히 권장합니다")
            suggestions.append("긴급 상담 핫라인: 1393 (24시간)")
        
        # 패턴별 제안
        if 'depression' in patterns:
            suggestions.append("우울감이 2주 이상 지속되면 전문가를 찾아가세요")
        if 'anxiety' in patterns:
            suggestions.append("불안 증상이 일상을 방해한다면 상담이 필요합니다")
        if 'help_seeking' in patterns:
            suggestions.append("도움을 구하는 것은 용기있는 행동입니다")
        
        return suggestions[:5]  # 최대 5개
    
    def comprehensive_analysis(self, text: str, emotion_result: Dict) -> Dict:
        """종합 심리 분석"""
        risk_assessment = self.analyze_risk_level(text)
        patterns = self.detect_psychological_patterns(text)
        suggestions = self.generate_counseling_suggestions(
            emotion_result['emotion'],
            risk_assessment['level'],
            patterns
        )
        
        return {
            'emotion': emotion_result['emotion'],
            'confidence': emotion_result['confidence'],
            'probabilities': emotion_result['probabilities'],
            'risk_assessment': risk_assessment,
            'psychological_patterns': patterns,
            'counseling_suggestions': suggestions
        }
