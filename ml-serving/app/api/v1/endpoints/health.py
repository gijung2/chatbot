"""
Health check endpoints
"""
from fastapi import APIRouter
from app.schemas.emotion import HealthResponse
from app.models.emotion_classifier import emotion_model_service
from app.config import settings

router = APIRouter()

@router.get("/health", response_model=HealthResponse)
async def health_check():
    """
    서비스 헬스체크
    
    - 모델 로드 상태 확인
    - 디바이스 정보
    - 버전 정보
    """
    return HealthResponse(
        status="healthy" if emotion_model_service.is_loaded else "degraded",
        model_loaded=emotion_model_service.is_loaded,
        device=str(emotion_model_service.device) if emotion_model_service.device else "unknown",
        model_path=settings.MODEL_PATH,
        version=settings.VERSION
    )
