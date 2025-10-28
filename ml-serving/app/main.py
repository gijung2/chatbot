"""
FastAPI main application
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from app.config import settings
from app.models.emotion_classifier import emotion_model_service
from app.api.v1.endpoints import emotion, avatar, health, avatar_state

@asynccontextmanager
async def lifespan(app: FastAPI):
    """애플리케이션 라이프사이클 관리"""
    # Startup
    print("\n" + "="*60)
    print(f"🚀 {settings.APP_NAME} v{settings.VERSION}")
    print("="*60)
    
    # 모델 로드
    success = emotion_model_service.load_model()
    
    if not success:
        print("\n⚠️ 경고: 모델을 로드할 수 없습니다")
        print("💡 규칙 기반 분석으로 대체됩니다")
    
    print("="*60)
    print("✅ 서버 준비 완료!")
    print(f"📍 http://{settings.HOST}:{settings.PORT}")
    print(f"📚 문서: http://{settings.HOST}:{settings.PORT}/docs")
    print("="*60 + "\n")
    
    yield
    
    # Shutdown
    print("\n👋 서비스 종료")

# FastAPI 앱 생성
app = FastAPI(
    title=settings.APP_NAME,
    version=settings.VERSION,
    description="KLUE-BERT 기반 한국어 감정 분석 및 아바타 생성 API",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 프로덕션: 특정 도메인만
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 라우터 등록
app.include_router(emotion.router, prefix="/api/v1", tags=["Emotion Analysis"])
app.include_router(avatar.router, prefix="/api/v1", tags=["Avatar Generation"])
app.include_router(avatar_state.router, prefix="/api/v1", tags=["Avatar State Mapping"])
app.include_router(health.router, prefix="/api/v1", tags=["Health"])

@app.get("/", tags=["Root"])
async def root():
    """루트 엔드포인트"""
    return {
        "service": settings.APP_NAME,
        "version": settings.VERSION,
        "status": "running",
        "model_loaded": emotion_model_service.is_loaded,
        "endpoints": {
            "docs": "/docs",
            "health": "/api/v1/health",
            "analyze": "/api/v1/analyze",
            "generate_avatar": "/api/v1/generate-avatar"
        }
    }

if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "app.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
        log_level=settings.LOG_LEVEL.lower()
    )
