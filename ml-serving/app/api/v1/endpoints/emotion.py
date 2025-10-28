"""
Emotion analysis endpoints
"""
from fastapi import APIRouter, HTTPException
from app.schemas.emotion import EmotionAnalyzeRequest, EmotionAnalyzeResponse
from app.models.emotion_classifier import emotion_model_service
from app.services.risk_assessment import assess_risk_level
from app.services.avatar_service import get_emotion_message

router = APIRouter()

@router.post("/analyze", response_model=EmotionAnalyzeResponse)
async def analyze_emotion(request: EmotionAnalyzeRequest):
    """
    텍스트 감정 분석
    
    - **text**: 분석할 텍스트 (필수)
    - **session_id**: 세션 ID (선택)
    """
    try:
        if not emotion_model_service.is_loaded:
            raise HTTPException(status_code=503, detail="모델이 로드되지 않았습니다")
        
        # 감정 예측
        result, inference_time_ms = emotion_model_service.predict(request.text)
        
        # 위험도 평가
        risk_level, risk_message = assess_risk_level(request.text, result['emotion'])
        
        # 감정 메시지
        emotion_message = get_emotion_message(result['emotion'])
        
        return EmotionAnalyzeResponse(
            text=request.text,
            emotion=result['emotion'],
            emotion_kr=result['emotion_kr'],
            confidence=result['confidence'],
            probabilities=result['probabilities'],
            risk_level=risk_level,
            risk_message=risk_message,
            emotion_message=emotion_message,
            method=result['method'],
            inference_time_ms=inference_time_ms
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"서버 에러: {str(e)}")
