"""
FastAPI 메인 애플리케이션
KoBERT 기반 심리상담 챗봇 API
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from datetime import datetime
import logging
import sys
from pathlib import Path

# 프로젝트 루트를 Python 경로에 추가
sys.path.insert(0, str(Path(__file__).parent.parent))

from fastapi_app.routers import emotion, avatar, chat
from fastapi_app.models.schemas import HealthResponse, EmergencyContact

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# FastAPI 앱 생성
app = FastAPI(
    title="심리상담 챗봇 API",
    description="KoBERT 기반 감정 분석 및 심리상담 챗봇 API",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 프로덕션에서는 특정 도메인으로 제한
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def startup_event():
    """애플리케이션 시작 시 초기화"""
    logger.info("=" * 80)
    logger.info("🚀 FastAPI 심리상담 챗봇 API 시작")
    logger.info("=" * 80)
    
    try:
        # 모델 초기화
        emotion.initialize_model(device='cpu')
        logger.info("✅ 모든 서비스 초기화 완료")
    except Exception as e:
        logger.error(f"❌ 초기화 실패: {e}")
        raise


@app.on_event("shutdown")
async def shutdown_event():
    """애플리케이션 종료 시 정리"""
    logger.info("👋 FastAPI 애플리케이션 종료")


# 라우터 등록
app.include_router(emotion.router)
app.include_router(avatar.router)
app.include_router(chat.router)


@app.get("/", response_class=HTMLResponse)
async def root():
    """루트 페이지 - API 문서 링크"""
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>심리상담 챗봇 API</title>
        <meta charset="utf-8">
        <style>
            body {
                font-family: 'Segoe UI', Arial, sans-serif;
                max-width: 800px;
                margin: 50px auto;
                padding: 20px;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
            }
            .container {
                background: rgba(255, 255, 255, 0.1);
                border-radius: 15px;
                padding: 40px;
                backdrop-filter: blur(10px);
            }
            h1 {
                font-size: 2.5em;
                margin-bottom: 10px;
            }
            .subtitle {
                font-size: 1.2em;
                margin-bottom: 30px;
                opacity: 0.9;
            }
            .links {
                margin-top: 30px;
            }
            a {
                display: inline-block;
                margin: 10px 10px 10px 0;
                padding: 12px 24px;
                background: white;
                color: #667eea;
                text-decoration: none;
                border-radius: 8px;
                font-weight: bold;
                transition: transform 0.2s;
            }
            a:hover {
                transform: translateY(-2px);
                box-shadow: 0 5px 15px rgba(0,0,0,0.3);
            }
            .feature {
                margin: 20px 0;
                padding: 15px;
                background: rgba(255, 255, 255, 0.1);
                border-radius: 8px;
            }
            .emoji {
                font-size: 1.5em;
                margin-right: 10px;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🤖 심리상담 챗봇 API</h1>
            <p class="subtitle">KoBERT 기반 감정 분석 및 심리상담 서비스</p>
            
            <div class="feature">
                <span class="emoji">🎭</span>
                <strong>감정 분석:</strong> 5가지 감정 (기쁨, 슬픔, 불안, 분노, 중립) 분류
            </div>
            
            <div class="feature">
                <span class="emoji">⚕️</span>
                <strong>심리 평가:</strong> 위험도 평가 및 상담 제안
            </div>
            
            <div class="feature">
                <span class="emoji">🎨</span>
                <strong>아바타 생성:</strong> 감정 기반 실시간 아바타 생성
            </div>
            
            <div class="feature">
                <span class="emoji">💬</span>
                <strong>채팅 상담:</strong> AI 기반 심리상담 채팅
            </div>
            
            <div class="links">
                <a href="/docs">📚 API 문서 (Swagger)</a>
                <a href="/redoc">📖 API 문서 (ReDoc)</a>
                <a href="/health">❤️ 헬스 체크</a>
            </div>
        </div>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """헬스 체크"""
    try:
        model = emotion.get_model()
        model_loaded = True
    except:
        model_loaded = False
    
    return HealthResponse(
        status="healthy" if model_loaded else "degraded",
        model_loaded=model_loaded,
        timestamp=datetime.now(),
        version="2.0.0"
    )


@app.get("/emergency-contacts", response_model=list[EmergencyContact])
async def get_emergency_contacts():
    """긴급 상담 연락처"""
    return [
        EmergencyContact(
            name="생명의전화",
            phone="1393",
            description="24시간 자살 예방 상담",
            available="24시간"
        ),
        EmergencyContact(
            name="정신건강위기상담전화",
            phone="1577-0199",
            description="정신건강 위기 상담",
            available="24시간"
        ),
        EmergencyContact(
            name="청소년 상담 전화",
            phone="1388",
            description="청소년 및 학부모 상담",
            available="24시간"
        ),
        EmergencyContact(
            name="한국자살예방협회",
            phone="02-413-0892",
            description="자살 예방 및 위기 개입",
            available="평일 9-18시"
        )
    ]


@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """전역 예외 처리"""
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal Server Error",
            "detail": str(exc),
            "timestamp": datetime.now().isoformat()
        }
    )


if __name__ == "__main__":
    import uvicorn
    
    logger.info("=" * 80)
    logger.info("🚀 FastAPI 서버 시작")
    logger.info("=" * 80)
    logger.info("📍 URL: http://localhost:8000")
    logger.info("📚 API 문서: http://localhost:8000/docs")
    logger.info("=" * 80)
    
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
