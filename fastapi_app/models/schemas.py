"""
Pydantic 스키마 정의
요청/응답 데이터 모델
"""
from pydantic import BaseModel, Field
from typing import Dict, List, Optional
from datetime import datetime


class EmotionRequest(BaseModel):
    """감정 분석 요청"""
    text: str = Field(..., description="분석할 텍스트", min_length=1, max_length=1000)
    include_details: bool = Field(False, description="상세 분석 포함 여부")
    
    class Config:
        json_schema_extra = {
            "example": {
                "text": "오늘 너무 우울하고 힘들어요",
                "include_details": True
            }
        }


class EmotionProbabilities(BaseModel):
    """감정별 확률"""
    joy: float
    sad: float
    anxiety: float
    anger: float
    neutral: float


class EmotionAnalysis(BaseModel):
    """기본 감정 분석 결과"""
    emotion: str = Field(..., description="예측된 감정")
    confidence: float = Field(..., description="신뢰도 (0~1)")
    probabilities: EmotionProbabilities = Field(..., description="각 감정별 확률")


class RiskAssessment(BaseModel):
    """심리적 위험도 평가"""
    level: str = Field(..., description="위험도 레벨 (safe/low/medium/high/critical)")
    score: float = Field(..., description="위험도 점수 (0~1)")
    keywords: List[str] = Field(default_factory=list, description="감지된 위험 키워드")
    recommendation: str = Field(..., description="권장 조치")


class DetailedEmotionAnalysis(EmotionAnalysis):
    """상세 감정 분석 결과"""
    risk_assessment: RiskAssessment
    psychological_patterns: Dict[str, List[str]] = Field(default_factory=dict, description="심리 패턴")
    counseling_suggestions: List[str] = Field(default_factory=list, description="상담 제안")


class AvatarRequest(BaseModel):
    """아바타 생성 요청"""
    text: str = Field(..., description="감정 분석할 텍스트", min_length=1, max_length=1000)
    size: int = Field(400, description="아바타 크기 (픽셀)", ge=100, le=1000)
    format: str = Field("png", description="이미지 포맷 (png/jpeg)")
    
    class Config:
        json_schema_extra = {
            "example": {
                "text": "오늘 너무 기쁘고 행복해요!",
                "size": 400,
                "format": "png"
            }
        }


class AvatarResponse(BaseModel):
    """아바타 생성 응답"""
    image_base64: str = Field(..., description="Base64 인코딩된 이미지")
    emotion: str = Field(..., description="감지된 감정")
    confidence: float = Field(..., description="신뢰도")
    analysis: DetailedEmotionAnalysis = Field(..., description="상세 분석")
    metadata: Dict = Field(default_factory=dict, description="메타데이터")


class ChatMessage(BaseModel):
    """채팅 메시지"""
    message: str = Field(..., description="채팅 메시지", min_length=1)
    session_id: Optional[str] = Field(None, description="세션 ID")


class ChatResponse(BaseModel):
    """채팅 응답"""
    response: str = Field(..., description="상담 응답")
    emotion: str = Field(..., description="감지된 감정")
    confidence: float = Field(..., description="신뢰도")
    avatar_url: Optional[str] = Field(None, description="아바타 이미지 URL")
    suggestions: List[str] = Field(default_factory=list, description="추가 제안")


class HealthResponse(BaseModel):
    """헬스 체크 응답"""
    status: str
    model_loaded: bool
    timestamp: datetime
    version: str


class ModelInfo(BaseModel):
    """모델 정보"""
    model_name: str
    num_classes: int
    emotion_labels: List[str]
    max_length: int
    device: str
    parameters: Optional[int] = None


class EmergencyContact(BaseModel):
    """긴급 연락처"""
    name: str
    phone: str
    description: str
    available: str


class ErrorResponse(BaseModel):
    """에러 응답"""
    error: str
    detail: Optional[str] = None
    timestamp: datetime
