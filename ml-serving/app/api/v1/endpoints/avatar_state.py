"""
아바타 상태 매핑 엔드포인트
"""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import Optional
from app.services.avatar_mapping import avatar_mapper

router = APIRouter()


class EmotionInput(BaseModel):
    """감정 입력 스키마"""
    emotion: str = Field(..., description="감정 레이블 (joy, sad, anxiety, anger, neutral)")
    confidence: float = Field(..., ge=0.0, le=1.0, description="신뢰도 (0.0 ~ 1.0)")
    risk_level: Optional[str] = Field("low", description="위험도 레벨")


class AvatarStateResponse(BaseModel):
    """아바타 상태 응답 스키마"""
    emotion: str
    confidence: float
    risk_level: str
    expression: str
    parameters: dict
    animation: str
    transition_duration: float
    color: str
    emoji: str
    timestamp: float
    special_gesture: Optional[str] = None
    alert_level: Optional[str] = None


@router.post("/map-emotion", response_model=AvatarStateResponse)
async def map_emotion_to_avatar(data: EmotionInput):
    """
    감정을 아바타 상태로 매핑
    
    - **emotion**: 감정 레이블 (joy, sad, anxiety, anger, neutral)
    - **confidence**: 예측 신뢰도 (0.0 ~ 1.0)
    - **risk_level**: 위험도 레벨 (low, medium, high, critical)
    
    Returns:
        Live2D 파라미터 및 애니메이션 정보
    """
    valid_emotions = ["joy", "sad", "anxiety", "anger", "neutral"]
    if data.emotion not in valid_emotions:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid emotion. Must be one of {valid_emotions}"
        )
    
    avatar_state = avatar_mapper.map_emotion_to_avatar_state(
        emotion=data.emotion,
        confidence=data.confidence,
        risk_level=data.risk_level
    )
    
    return avatar_state


@router.get("/idle-state", response_model=AvatarStateResponse)
async def get_idle_state():
    """
    대기 상태 아바타 파라미터 반환
    
    Returns:
        중립 상태의 아바타 파라미터
    """
    return avatar_mapper.get_idle_state()


@router.post("/interpolate")
async def interpolate_avatar_states(
    from_emotion: str,
    to_emotion: str,
    progress: float = Field(..., ge=0.0, le=1.0)
):
    """
    두 감정 상태 사이를 보간
    
    - **from_emotion**: 시작 감정
    - **to_emotion**: 목표 감정
    - **progress**: 진행도 (0.0 ~ 1.0)
    
    Returns:
        보간된 아바타 상태
    """
    from_state = avatar_mapper.map_emotion_to_avatar_state(from_emotion, 1.0)
    to_state = avatar_mapper.map_emotion_to_avatar_state(to_emotion, 1.0)
    
    interpolated = avatar_mapper.interpolate_states(from_state, to_state, progress)
    
    return interpolated
