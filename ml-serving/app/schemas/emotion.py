"""
Pydantic schemas for emotion analysis
"""
from pydantic import BaseModel, Field
from typing import Dict, Optional, List

class EmotionAnalyzeRequest(BaseModel):
    """감정 분석 요청"""
    text: str = Field(..., min_length=1, max_length=1000, description="분석할 텍스트")
    session_id: Optional[str] = Field(None, description="세션 ID (선택)")
    
    class Config:
        json_schema_extra = {
            "example": {
                "text": "오늘 정말 기분이 좋아요!",
                "session_id": "abc-123"
            }
        }

class EmotionAnalyzeResponse(BaseModel):
    """감정 분석 응답"""
    text: str = Field(..., description="입력 텍스트")
    emotion: str = Field(..., description="감정 (영문)")
    emotion_kr: str = Field(..., description="감정 (한글)")
    confidence: float = Field(..., ge=0.0, le=1.0, description="신뢰도")
    probabilities: Dict[str, float] = Field(..., description="각 감정별 확률")
    risk_level: str = Field(..., description="위험도 (low/medium/high/critical)")
    risk_message: str = Field(..., description="위험도 메시지")
    emotion_message: str = Field(..., description="감정 메시지")
    method: str = Field(..., description="분석 방법")
    inference_time_ms: Optional[float] = Field(None, description="추론 시간 (밀리초)")
    
    class Config:
        json_schema_extra = {
            "example": {
                "text": "오늘 정말 기분이 좋아요!",
                "emotion": "joy",
                "emotion_kr": "기쁨",
                "confidence": 0.92,
                "probabilities": {
                    "joy": 0.92,
                    "sad": 0.02,
                    "anxiety": 0.03,
                    "anger": 0.01,
                    "neutral": 0.02
                },
                "risk_level": "low",
                "risk_message": "💚 안정적인 상태입니다.",
                "emotion_message": "긍정적인 에너지가 느껴져요!",
                "method": "klue-bert-kfold",
                "inference_time_ms": 45.2
            }
        }

class AvatarGenerateRequest(BaseModel):
    """아바타 생성 요청"""
    text: str = Field(..., min_length=1, max_length=1000, description="분석할 텍스트")
    emotion: Optional[str] = Field(None, description="강제 감정 (선택)")
    style: Optional[str] = Field("gradient", description="아바타 스타일")
    
    class Config:
        json_schema_extra = {
            "example": {
                "text": "오늘 너무 행복해요!",
                "style": "gradient"
            }
        }

class AvatarGenerateResponse(BaseModel):
    """아바타 생성 응답"""
    text: str
    emotion: str
    emotion_kr: str
    confidence: float
    avatar_image: str = Field(..., description="Base64 인코딩된 이미지")
    risk_level: str
    risk_message: str
    emotion_message: str
    probabilities: Dict[str, float]
    success: bool = True
    generation_time_ms: Optional[float] = None
    
    class Config:
        json_schema_extra = {
            "example": {
                "text": "오늘 너무 행복해요!",
                "emotion": "joy",
                "emotion_kr": "기쁨",
                "confidence": 0.92,
                "avatar_image": "data:image/png;base64,...",
                "risk_level": "low",
                "risk_message": "💚 안정적인 상태입니다.",
                "emotion_message": "긍정적인 에너지가 느껴져요!",
                "probabilities": {},
                "success": True,
                "generation_time_ms": 120.5
            }
        }

class HealthResponse(BaseModel):
    """헬스체크 응답"""
    status: str = Field(..., description="상태 (healthy/unhealthy)")
    model_loaded: bool = Field(..., description="모델 로드 여부")
    device: str = Field(..., description="디바이스 (cuda/cpu)")
    model_path: str = Field(..., description="모델 경로")
    version: str = Field(..., description="버전")
