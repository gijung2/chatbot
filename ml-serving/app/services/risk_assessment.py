"""
Risk assessment service
"""
import re
from typing import Tuple

# 심리 위험도 패턴
RISK_PATTERNS = {
    'critical': [
        r'죽고?\s*싶', r'사라지고?\s*싶', r'끝내고?\s*싶', r'자살',
        r'소용없', r'의미없', r'가치없'
    ],
    'high': [
        r'우울해?', r'슬프', r'힘들어?', r'절망', r'포기', r'무기력',
        r'악몽', r'플래시백', r'떠올라'
    ],
    'medium': [
        r'불안해?', r'걱정', r'두려워?', r'긴장', r'떨려',
        r'화가?\s*나', r'분노', r'짜증', r'열받아?'
    ]
}

def assess_risk_level(text: str, emotion: str) -> Tuple[str, str]:
    """
    심리 위험도 평가
    
    Args:
        text: 입력 텍스트
        emotion: 감정
        
    Returns:
        (risk_level, risk_message)
    """
    text_lower = text.lower()
    
    # Critical 패턴 체크
    for pattern in RISK_PATTERNS['critical']:
        if re.search(pattern, text_lower):
            return 'critical', '⚠️ 긴급 상황이 감지되었습니다. 즉시 전문가의 도움을 받으세요.\n자살예방상담전화: 109 (24시간)'
    
    # High 패턴 체크
    for pattern in RISK_PATTERNS['high']:
        if re.search(pattern, text_lower):
            return 'high', '💛 심각한 우울감이 느껴집니다. 전문 상담사와 이야기하는 것을 권장합니다.\n정신건강상담전화: 1577-0199'
    
    # Medium 패턴 체크
    for pattern in RISK_PATTERNS['medium']:
        if re.search(pattern, text_lower):
            return 'medium', '💙 힘든 감정을 느끼고 계시네요. 충분히 휴식하고 자신을 돌보세요.'
    
    return 'low', '💚 안정적인 상태입니다. 긍정적인 마음을 유지하세요.'
