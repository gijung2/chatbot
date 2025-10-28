"""
감정 분석 API 라우터
"""
from fastapi import APIRouter, HTTPException, Depends
from typing import Dict
import logging

from ..models.schemas import (
    EmotionRequest,
    EmotionAnalysis,
    DetailedEmotionAnalysis,
    EmotionProbabilities,
    RiskAssessment
)
from ..models.emotion_model import EmotionClassifier
from ..services.psychological_service import PsychologicalAnalyzer

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/emotion", tags=["Emotion Analysis"])

# 전역 모델 (싱글톤)
_model: EmotionClassifier = None
_analyzer: PsychologicalAnalyzer = None


def get_model() -> EmotionClassifier:
    """모델 인스턴스 가져오기"""
    global _model
    if _model is None:
        raise HTTPException(status_code=503, detail="모델이 로드되지 않았습니다")
    return _model


def get_analyzer() -> PsychologicalAnalyzer:
    """분석기 인스턴스 가져오기"""
    global _analyzer
    if _analyzer is None:
        _analyzer = PsychologicalAnalyzer()
    return _analyzer


def initialize_model(device: str = 'cpu'):
    """모델 초기화"""
    global _model
    _model = EmotionClassifier(device=device)
    logger.info("✅ 감정 분류 모델 초기화 완료")


@router.post("/analyze", response_model=EmotionAnalysis)
async def analyze_emotion(
    request: EmotionRequest,
    model: EmotionClassifier = Depends(get_model)
):
    """
    기본 감정 분석
    
    - **text**: 분석할 텍스트 (1-1000자)
    - **include_details**: 상세 분석 포함 여부
    """
    try:
        # 감정 예측
        result = model.predict_emotion(request.text)
        
        return EmotionAnalysis(
            emotion=result['emotion'],
            confidence=result['confidence'],
            probabilities=EmotionProbabilities(**result['probabilities'])
        )
    
    except Exception as e:
        logger.error(f"감정 분석 오류: {e}")
        raise HTTPException(status_code=500, detail=f"분석 실패: {str(e)}")


@router.post("/analyze/detailed", response_model=DetailedEmotionAnalysis)
async def analyze_emotion_detailed(
    request: EmotionRequest,
    model: EmotionClassifier = Depends(get_model),
    analyzer: PsychologicalAnalyzer = Depends(get_analyzer)
):
    """
    상세 감정 분석 (위험도 평가 포함)
    
    - **text**: 분석할 텍스트
    - 심리적 위험도 평가
    - 상담 제안 포함
    """
    try:
        # 감정 예측
        emotion_result = model.predict_emotion(request.text)
        
        # 종합 분석
        comprehensive = analyzer.comprehensive_analysis(request.text, emotion_result)
        
        return DetailedEmotionAnalysis(
            emotion=comprehensive['emotion'],
            confidence=comprehensive['confidence'],
            probabilities=EmotionProbabilities(**comprehensive['probabilities']),
            risk_assessment=RiskAssessment(**comprehensive['risk_assessment']),
            psychological_patterns=comprehensive['psychological_patterns'],
            counseling_suggestions=comprehensive['counseling_suggestions']
        )
    
    except Exception as e:
        logger.error(f"상세 분석 오류: {e}")
        raise HTTPException(status_code=500, detail=f"분석 실패: {str(e)}")


@router.get("/model-info")
async def get_model_info(model: EmotionClassifier = Depends(get_model)):
    """
    모델 정보 조회
    """
    return model.get_model_info()
