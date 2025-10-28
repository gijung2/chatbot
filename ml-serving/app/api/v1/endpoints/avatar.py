"""
Avatar generation endpoints
"""
from fastapi import APIRouter, HTTPException
from app.schemas.emotion import AvatarGenerateRequest, AvatarGenerateResponse
from app.models.emotion_classifier import emotion_model_service
from app.services.risk_assessment import assess_risk_level
from app.services.avatar_service import generate_avatar_image, get_emotion_message

router = APIRouter()

@router.post("/generate-avatar", response_model=AvatarGenerateResponse)
async def generate_avatar(request: AvatarGenerateRequest):
    """
    감정 분석 + 아바타 이미지 생성
    
    - **text**: 분석할 텍스트 (필수)
    - **emotion**: 강제 감정 설정 (선택, 없으면 자동 분석)
    - **style**: 아바타 스타일 (기본값: gradient)
    """
    try:
        # 감정이 지정되지 않았으면 자동 분석
        if not request.emotion:
            if not emotion_model_service.is_loaded:
                raise HTTPException(status_code=503, detail="모델이 로드되지 않았습니다")
            
            result, _ = emotion_model_service.predict(request.text)
            emotion = result['emotion']
            emotion_kr = result['emotion_kr']
            confidence = result['confidence']
            probabilities = result['probabilities']
        else:
            # 강제 감정 사용
            emotion = request.emotion
            from app.models.emotion_classifier import EmotionModelService
            emotion_kr = EmotionModelService.EMOTION_KR.get(emotion, emotion)
            confidence = 1.0
            probabilities = {e: (1.0 if e == emotion else 0.0) 
                           for e in EmotionModelService.EMOTION_LABELS}
        
        # 위험도 평가
        risk_level, risk_message = assess_risk_level(request.text, emotion)
        
        # 감정 메시지
        emotion_message = get_emotion_message(emotion)
        
        # 아바타 이미지 생성
        avatar_image, generation_time_ms = generate_avatar_image(emotion, request.style)
        
        return AvatarGenerateResponse(
            text=request.text,
            emotion=emotion,
            emotion_kr=emotion_kr,
            confidence=confidence,
            avatar_image=avatar_image,
            risk_level=risk_level,
            risk_message=risk_message,
            emotion_message=emotion_message,
            probabilities=probabilities,
            success=bool(avatar_image),
            generation_time_ms=generation_time_ms
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"서버 에러: {str(e)}")
