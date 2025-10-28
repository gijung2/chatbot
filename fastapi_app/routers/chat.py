"""
채팅 API 라우터
"""
from fastapi import APIRouter, HTTPException, Depends
from typing import List
import logging

from ..models.schemas import ChatMessage, ChatResponse
from ..models.emotion_model import EmotionClassifier
from ..services.psychological_service import PsychologicalAnalyzer

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/chat", tags=["Chat"])


# 상담 응답 템플릿
RESPONSE_TEMPLATES = {
    'joy': [
        "기쁜 마음이 느껴지네요! 긍정적인 에너지를 계속 유지하세요.",
        "행복한 순간을 공유해주셔서 감사합니다. 이런 감정을 잘 간직하세요."
    ],
    'sad': [
        "슬픔을 느끼고 계시는군요. 이런 감정도 자연스러운 과정입니다.",
        "힘든 시간을 보내고 계시네요. 혼자가 아니라는 것을 기억하세요."
    ],
    'anxiety': [
        "불안한 마음이 느껴집니다. 천천히 심호흡을 해보세요.",
        "걱정이 많으시군요. 지금 이 순간에 집중해보는 것은 어떨까요?"
    ],
    'anger': [
        "화가 나는 상황이시군요. 감정을 표현하는 것은 중요합니다.",
        "분노를 느끼고 계시네요. 잠시 시간을 가지고 진정해보세요."
    ],
    'neutral': [
        "현재 안정적인 상태시네요. 계속 균형을 유지하세요.",
        "편안한 상태입니다. 이 상태를 유지하는 것이 좋겠습니다."
    ]
}


@router.post("/message", response_model=ChatResponse)
async def chat_message(
    message: ChatMessage,
    model: EmotionClassifier = Depends(lambda: _get_model()),
    analyzer: PsychologicalAnalyzer = Depends(lambda: _get_analyzer())
):
    """
    채팅 메시지 처리
    
    - 감정 분석
    - 심리 상담 응답 생성
    - 제안 사항 제공
    """
    try:
        # 감정 분석
        emotion_result = model.predict_emotion(message.message)
        
        # 종합 분석
        comprehensive = analyzer.comprehensive_analysis(message.message, emotion_result)
        
        # 응답 선택
        emotion = comprehensive['emotion']
        responses = RESPONSE_TEMPLATES.get(emotion, RESPONSE_TEMPLATES['neutral'])
        response_text = responses[0]  # 첫 번째 템플릿 사용
        
        # 위험도에 따른 추가 응답
        risk_level = comprehensive['risk_assessment']['level']
        if risk_level in ['critical', 'high']:
            response_text += f"\n\n⚠️ {comprehensive['risk_assessment']['recommendation']}"
        
        return ChatResponse(
            response=response_text,
            emotion=emotion,
            confidence=comprehensive['confidence'],
            avatar_url=None,  # 향후 구현
            suggestions=comprehensive['counseling_suggestions'][:3]
        )
    
    except Exception as e:
        logger.error(f"채팅 처리 오류: {e}")
        raise HTTPException(status_code=500, detail=f"처리 실패: {str(e)}")


@router.get("/suggestions/{emotion}")
async def get_suggestions(emotion: str) -> List[str]:
    """
    감정별 상담 제안 조회
    """
    if emotion not in RESPONSE_TEMPLATES:
        raise HTTPException(status_code=404, detail="지원하지 않는 감정입니다")
    
    return RESPONSE_TEMPLATES[emotion]


def _get_model():
    """모델 가져오기"""
    from .emotion import get_model
    return get_model()


def _get_analyzer():
    """분석기 가져오기"""
    from .emotion import get_analyzer
    return get_analyzer()
