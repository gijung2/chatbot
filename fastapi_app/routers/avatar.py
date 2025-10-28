"""
아바타 생성 API 라우터
"""
from fastapi import APIRouter, HTTPException, Depends
from fastapi.responses import Response
import base64
import logging

from ..models.schemas import (
    AvatarRequest,
    AvatarResponse,
    DetailedEmotionAnalysis,
    EmotionProbabilities,
    RiskAssessment
)
from ..models.emotion_model import EmotionClassifier
from ..services.psychological_service import PsychologicalAnalyzer
from ..services.avatar_service import AvatarGenerator

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/avatar", tags=["Avatar Generation"])

# 전역 인스턴스
_generator: AvatarGenerator = None


def get_generator() -> AvatarGenerator:
    """아바타 생성기 가져오기"""
    global _generator
    if _generator is None:
        _generator = AvatarGenerator()
    return _generator


@router.post("/generate", response_model=AvatarResponse)
async def generate_avatar(
    request: AvatarRequest,
    model: EmotionClassifier = Depends(lambda: _get_model_from_emotion_router()),
    analyzer: PsychologicalAnalyzer = Depends(lambda: _get_analyzer_from_emotion_router()),
    generator: AvatarGenerator = Depends(get_generator)
):
    """
    감정 기반 아바타 생성
    
    - **text**: 분석할 텍스트
    - **size**: 아바타 크기 (100-1000 픽셀)
    - **format**: 이미지 포맷 (png/jpeg)
    """
    try:
        # 감정 분석
        emotion_result = model.predict_emotion(request.text)
        
        # 종합 분석
        comprehensive = analyzer.comprehensive_analysis(request.text, emotion_result)
        
        # 아바타 생성
        avatar_data = generator.generate_detailed_avatar(comprehensive, request.size)
        
        # 응답 구성
        return AvatarResponse(
            image_base64=avatar_data['image_base64'],
            emotion=comprehensive['emotion'],
            confidence=comprehensive['confidence'],
            analysis=DetailedEmotionAnalysis(
                emotion=comprehensive['emotion'],
                confidence=comprehensive['confidence'],
                probabilities=EmotionProbabilities(**comprehensive['probabilities']),
                risk_assessment=RiskAssessment(**comprehensive['risk_assessment']),
                psychological_patterns=comprehensive['psychological_patterns'],
                counseling_suggestions=comprehensive['counseling_suggestions']
            ),
            metadata={
                'size': request.size,
                'format': request.format,
                'color': avatar_data['color']
            }
        )
    
    except Exception as e:
        logger.error(f"아바타 생성 오류: {e}")
        raise HTTPException(status_code=500, detail=f"생성 실패: {str(e)}")


@router.post("/generate/image")
async def generate_avatar_image(
    request: AvatarRequest,
    model: EmotionClassifier = Depends(lambda: _get_model_from_emotion_router()),
    generator: AvatarGenerator = Depends(get_generator)
):
    """
    아바타 이미지 직접 반환 (PNG)
    
    - Content-Type: image/png
    """
    try:
        # 감정 분석
        emotion_result = model.predict_emotion(request.text)
        
        # 아바타 생성
        avatar_base64 = generator.generate_avatar(
            emotion_result['emotion'],
            emotion_result['confidence'],
            request.size
        )
        
        # Base64 디코딩
        image_bytes = base64.b64decode(avatar_base64)
        
        return Response(content=image_bytes, media_type="image/png")
    
    except Exception as e:
        logger.error(f"아바타 이미지 생성 오류: {e}")
        raise HTTPException(status_code=500, detail=f"생성 실패: {str(e)}")


def _get_model_from_emotion_router():
    """emotion 라우터에서 모델 가져오기"""
    from .emotion import get_model
    return get_model()


def _get_analyzer_from_emotion_router():
    """emotion 라우터에서 분석기 가져오기"""
    from .emotion import get_analyzer
    return get_analyzer()
