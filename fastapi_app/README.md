# FastAPI 심리상담 챗봇 API

KoBERT 기반 감정 분석 및 심리상담 챗봇 API (FastAPI 리팩토링 버전)

## 🚀 주요 기능

### 1. **감정 분석 API**
- 5가지 감정 분류 (기쁨, 슬픔, 불안, 분노, 중립)
- KoBERT/KLUE-BERT 기반 고정밀 예측
- 실시간 감정 확률 분석

### 2. **심리 상담 분석**
- 심리적 위험도 평가 (5단계)
- 전문 심리 패턴 인식
- 맞춤형 상담 제안

### 3. **아바타 생성**
- 감정 기반 실시간 아바타 생성
- Base64 이미지 인코딩
- 커스터마이징 가능한 크기/포맷

### 4. **채팅 상담**
- AI 기반 심리상담 응답
- 세션 기반 대화 관리
- 긴급 상황 감지 및 대응

## 📁 프로젝트 구조

```
fastapi_app/
├── main.py                      # FastAPI 메인 앱
├── requirements.txt             # 의존성
├── models/
│   ├── schemas.py              # Pydantic 스키마
│   └── emotion_model.py        # 감정 분류 모델
├── routers/
│   ├── emotion.py              # 감정 분석 API
│   ├── avatar.py               # 아바타 생성 API
│   └── chat.py                 # 채팅 API
└── services/
    ├── psychological_service.py # 심리 분석 서비스
    └── avatar_service.py        # 아바타 생성 서비스
```

## 🔧 설치 방법

### 1. 의존성 설치

```bash
# FastAPI 앱 디렉토리로 이동
cd fastapi_app

# 의존성 설치
pip install -r requirements.txt
```

### 2. 서버 실행

```bash
# 개발 모드 (자동 재시작)
python main.py

# 또는 uvicorn 직접 실행
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

## 📚 API 사용법

### 기본 URL
```
http://localhost:8000
```

### 자동 문서
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

### 주요 엔드포인트

#### 1. 감정 분석
```bash
POST /emotion/analyze
Content-Type: application/json

{
  "text": "오늘 너무 우울하고 힘들어요",
  "include_details": false
}
```

**응답:**
```json
{
  "emotion": "sad",
  "confidence": 0.89,
  "probabilities": {
    "joy": 0.02,
    "sad": 0.89,
    "anxiety": 0.05,
    "anger": 0.02,
    "neutral": 0.02
  }
}
```

#### 2. 상세 감정 분석 (위험도 평가 포함)
```bash
POST /emotion/analyze/detailed
Content-Type: application/json

{
  "text": "살기 싫고 모든 게 의미 없어"
}
```

**응답:**
```json
{
  "emotion": "sad",
  "confidence": 0.92,
  "probabilities": {...},
  "risk_assessment": {
    "level": "critical",
    "score": 1.0,
    "keywords": ["살기싫", "의미없"],
    "recommendation": "즉시 전문가 상담이 필요합니다..."
  },
  "psychological_patterns": {
    "depression": ["우울", "절망"],
    "help_seeking": []
  },
  "counseling_suggestions": [
    "전문가 상담을 강력히 권장합니다",
    "긴급 상담 핫라인: 1393 (24시간)"
  ]
}
```

#### 3. 아바타 생성
```bash
POST /avatar/generate
Content-Type: application/json

{
  "text": "오늘 너무 기쁘고 행복해요!",
  "size": 400,
  "format": "png"
}
```

**응답:**
```json
{
  "image_base64": "iVBORw0KGgoAAAANSUhEUg...",
  "emotion": "joy",
  "confidence": 0.95,
  "analysis": {...},
  "metadata": {
    "size": 400,
    "format": "png",
    "color": [255, 223, 0]
  }
}
```

#### 4. 채팅 메시지
```bash
POST /chat/message
Content-Type: application/json

{
  "message": "요즘 너무 스트레스 받아요",
  "session_id": "user123"
}
```

**응답:**
```json
{
  "response": "불안한 마음이 느껴집니다. 천천히 심호흡을 해보세요.",
  "emotion": "anxiety",
  "confidence": 0.87,
  "avatar_url": null,
  "suggestions": [
    "심호흡을 통해 긴장을 풀어보세요",
    "명상이나 요가를 시도해보세요"
  ]
}
```

#### 5. 헬스 체크
```bash
GET /health
```

**응답:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "timestamp": "2025-10-28T10:30:00",
  "version": "2.0.0"
}
```

#### 6. 긴급 연락처
```bash
GET /emergency-contacts
```

## 🎯 Flask vs FastAPI 비교

| 항목 | Flask (기존) | FastAPI (개선) |
|------|-------------|----------------|
| 성능 | 동기 처리 | 비동기 처리 (2-3배 빠름) |
| 문서화 | 수동 작성 | 자동 생성 (/docs) |
| 타입 검증 | 수동 검증 | Pydantic 자동 검증 |
| WebSocket | 별도 라이브러리 | 내장 지원 |
| 에러 처리 | 수동 처리 | 자동 HTTP 예외 |
| 코드 구조 | 단일 파일 | 모듈화된 구조 |

## 🔐 보안 고려사항

1. **CORS**: 프로덕션에서는 `allow_origins` 제한 필요
2. **Rate Limiting**: 향후 추가 예정
3. **API Key**: 인증 시스템 구현 권장
4. **입력 검증**: Pydantic으로 자동 검증

## 📊 성능 최적화

- 모델 싱글톤 패턴으로 메모리 효율화
- 비동기 처리로 동시 요청 처리 능력 향상
- Dependency Injection으로 코드 재사용

## 🚧 향후 개선 사항

- [ ] WebSocket 실시간 채팅
- [ ] 데이터베이스 연동 (세션 관리)
- [ ] Redis 캐싱
- [ ] 배치 감정 분석 API
- [ ] 모델 버전 관리
- [ ] 프로메테우스 메트릭

## 📞 긴급 상담 연락처

- **생명의전화**: 1393 (24시간)
- **정신건강위기상담전화**: 1577-0199 (24시간)
- **청소년 상담 전화**: 1388 (24시간)

## 📝 라이선스

MIT License

## 👥 기여

이슈 및 PR을 환영합니다!
