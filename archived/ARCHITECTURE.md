# 🏗️ 심리상담 챗봇 시스템 아키텍처

## 📊 전체 시스템 구성

```
┌─────────────────────────────────────────────────────────────────────┐
│                         사용자 (Browser)                              │
└────────────────────────────┬────────────────────────────────────────┘
                             │ HTTP/WebSocket
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    프론트엔드 (Next.js 15)                            │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │  App Router (RSC)                     Port: 3000             │   │
│  │  - 페이지: Home, Chat, Analytics                             │   │
│  │  - 컴포넌트: ChatInterface, EmotionDisplay, AvatarView      │   │
│  │  - 상태관리: Zustand / React Query                           │   │
│  │  - 스타일: Tailwind CSS + shadcn/ui                         │   │
│  │  - 타입: TypeScript (strict mode)                           │   │
│  └──────────────────────────────────────────────────────────────┘   │
└────────────────────────────┬────────────────────────────────────────┘
                             │ REST API / WebSocket
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│                 백엔드 API (NestJS + TypeScript)                      │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │  API Gateway & Business Logic        Port: 3001             │   │
│  │                                                               │   │
│  │  모듈 구조:                                                    │   │
│  │  ├─ ChatModule         - 채팅 세션 관리                       │   │
│  │  ├─ EmotionModule      - 감정 분석 요청 프록시               │   │
│  │  ├─ UserModule         - 사용자 관리 (선택)                  │   │
│  │  ├─ AnalyticsModule    - 대화 분석 및 통계                   │   │
│  │  └─ WebSocketGateway   - 실시간 통신                         │   │
│  │                                                               │   │
│  │  데이터베이스: PostgreSQL (TypeORM)                           │   │
│  │  캐싱: Redis (채팅 히스토리, 세션)                            │   │
│  │  인증: JWT (선택적)                                           │   │
│  └──────────────────────────────────────────────────────────────┘   │
└────────────────────────────┬────────────────────────────────────────┘
                             │ HTTP (Internal)
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│              AI 모델 서빙 (FastAPI + Python)                          │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │  ML Model Serving                     Port: 8000            │   │
│  │                                                               │   │
│  │  엔드포인트:                                                   │   │
│  │  POST /api/v1/analyze        - 감정 분석                     │   │
│  │  POST /api/v1/generate-avatar - 아바타 생성                 │   │
│  │  POST /api/v1/assess-risk    - 위험도 평가                  │   │
│  │  GET  /api/v1/health         - 헬스체크                     │   │
│  │  GET  /metrics               - Prometheus 메트릭            │   │
│  │                                                               │   │
│  │  모델: KLUE-BERT (checkpoints_kfold/)                        │   │
│  │  캐싱: 인메모리 LRU 캐시                                      │   │
│  │  모니터링: Prometheus + Grafana                              │   │
│  └──────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 🎯 각 레이어 상세 설계

### 1️⃣ **프론트엔드 (Next.js 15 + App Router)**

#### 📁 디렉토리 구조
```
frontend-next/
├── app/
│   ├── layout.tsx                 # 루트 레이아웃
│   ├── page.tsx                   # 홈페이지
│   ├── chat/
│   │   ├── page.tsx               # 채팅 페이지
│   │   └── [sessionId]/
│   │       └── page.tsx           # 세션별 채팅
│   ├── analytics/
│   │   └── page.tsx               # 감정 분석 통계
│   └── api/                       # Route Handlers
│       └── socket/
│           └── route.ts           # WebSocket 프록시
│
├── components/
│   ├── chat/
│   │   ├── ChatInterface.tsx      # 메인 채팅 UI
│   │   ├── MessageBubble.tsx      # 메시지 버블
│   │   ├── InputBox.tsx           # 입력창
│   │   └── TypingIndicator.tsx   # 타이핑 표시
│   ├── emotion/
│   │   ├── EmotionBadge.tsx       # 감정 배지
│   │   ├── EmotionChart.tsx       # 감정 차트
│   │   └── RiskAlert.tsx          # 위험도 경고
│   ├── avatar/
│   │   ├── AvatarDisplay.tsx      # 아바타 표시
│   │   └── AvatarAnimation.tsx    # 아바타 애니메이션
│   └── ui/                        # shadcn/ui 컴포넌트
│
├── lib/
│   ├── api/
│   │   ├── chat.ts                # 채팅 API 클라이언트
│   │   ├── emotion.ts             # 감정 API 클라이언트
│   │   └── websocket.ts           # WebSocket 클라이언트
│   ├── hooks/
│   │   ├── useChat.ts             # 채팅 훅
│   │   ├── useEmotion.ts          # 감정 분석 훅
│   │   └── useAvatar.ts           # 아바타 훅
│   └── store/
│       ├── chatStore.ts           # 채팅 상태 (Zustand)
│       └── emotionStore.ts        # 감정 상태
│
├── types/
│   ├── chat.ts                    # 채팅 타입
│   ├── emotion.ts                 # 감정 타입
│   └── api.ts                     # API 응답 타입
│
└── package.json
```

#### 🔑 핵심 기술 스택
- **Framework**: Next.js 15 (App Router, RSC)
- **Language**: TypeScript 5.3+
- **Styling**: Tailwind CSS 3.4 + shadcn/ui
- **State**: Zustand + React Query (TanStack Query)
- **Real-time**: Socket.io-client
- **Forms**: React Hook Form + Zod
- **Charts**: Recharts / Chart.js
- **Animation**: Framer Motion

---

### 2️⃣ **백엔드 API (NestJS + TypeScript)**

#### 📁 디렉토리 구조
```
backend-nest/
├── src/
│   ├── main.ts                    # 애플리케이션 엔트리
│   ├── app.module.ts              # 루트 모듈
│   │
│   ├── chat/
│   │   ├── chat.module.ts
│   │   ├── chat.controller.ts     # REST 엔드포인트
│   │   ├── chat.service.ts        # 비즈니스 로직
│   │   ├── chat.gateway.ts        # WebSocket 게이트웨이
│   │   ├── entities/
│   │   │   ├── chat-session.entity.ts
│   │   │   └── message.entity.ts
│   │   └── dto/
│   │       ├── create-message.dto.ts
│   │       └── session.dto.ts
│   │
│   ├── emotion/
│   │   ├── emotion.module.ts
│   │   ├── emotion.controller.ts  # 프록시 컨트롤러
│   │   ├── emotion.service.ts     # FastAPI 호출
│   │   └── dto/
│   │       ├── analyze-emotion.dto.ts
│   │       └── emotion-result.dto.ts
│   │
│   ├── analytics/
│   │   ├── analytics.module.ts
│   │   ├── analytics.controller.ts
│   │   ├── analytics.service.ts   # 통계 계산
│   │   └── dto/
│   │       └── emotion-stats.dto.ts
│   │
│   ├── user/ (선택적)
│   │   ├── user.module.ts
│   │   ├── user.controller.ts
│   │   ├── user.service.ts
│   │   └── entities/
│   │       └── user.entity.ts
│   │
│   ├── common/
│   │   ├── filters/               # Exception filters
│   │   ├── guards/                # Auth guards
│   │   ├── interceptors/          # Response interceptors
│   │   ├── pipes/                 # Validation pipes
│   │   └── decorators/            # Custom decorators
│   │
│   └── config/
│       ├── database.config.ts     # DB 설정
│       ├── redis.config.ts        # Redis 설정
│       └── app.config.ts          # 앱 설정
│
├── test/                          # E2E 테스트
├── prisma/ (또는 TypeORM migrations)
└── package.json
```

#### 🔑 핵심 기술 스택
- **Framework**: NestJS 10+
- **Language**: TypeScript 5.3+
- **ORM**: TypeORM / Prisma
- **Database**: PostgreSQL 15+
- **Cache**: Redis 7+
- **WebSocket**: @nestjs/websockets + socket.io
- **Validation**: class-validator + class-transformer
- **HTTP Client**: Axios
- **Testing**: Jest + Supertest
- **Documentation**: Swagger (@nestjs/swagger)

---

### 3️⃣ **AI 모델 서빙 (FastAPI + Python)**

#### 📁 디렉토리 구조
```
ml-serving/
├── app/
│   ├── main.py                    # FastAPI 앱
│   ├── config.py                  # 설정
│   │
│   ├── api/
│   │   └── v1/
│   │       ├── router.py          # API 라우터
│   │       └── endpoints/
│   │           ├── emotion.py     # 감정 분석 엔드포인트
│   │           ├── avatar.py      # 아바타 생성
│   │           └── risk.py        # 위험도 평가
│   │
│   ├── models/
│   │   ├── emotion_classifier.py # 모델 래퍼
│   │   ├── avatar_generator.py   # 아바타 생성기
│   │   └── risk_assessor.py      # 위험도 평가기
│   │
│   ├── schemas/
│   │   ├── emotion.py             # Pydantic 스키마
│   │   ├── avatar.py
│   │   └── common.py
│   │
│   ├── services/
│   │   ├── inference.py           # 추론 서비스
│   │   ├── preprocessing.py      # 전처리
│   │   └── cache.py              # 캐싱 서비스
│   │
│   ├── core/
│   │   ├── model_loader.py       # 모델 로더
│   │   ├── metrics.py            # Prometheus 메트릭
│   │   └── logging.py            # 로깅 설정
│   │
│   └── utils/
│       ├── image.py              # 이미지 처리
│       └── text.py               # 텍스트 처리
│
├── checkpoints/                   # 학습된 모델 (심볼릭 링크)
├── tests/
├── requirements.txt
└── Dockerfile
```

#### 🔑 핵심 기술 스택
- **Framework**: FastAPI 0.115+
- **ML**: PyTorch 2.5+, Transformers 4.44+
- **Image**: Pillow 12+
- **Server**: Uvicorn (ASGI)
- **Validation**: Pydantic 2.12+
- **Monitoring**: prometheus-client
- **Caching**: functools.lru_cache / Redis
- **Testing**: pytest + httpx

---

## 🔄 데이터 플로우

### 채팅 + 감정 분석 플로우
```
1. 사용자 메시지 입력 (Next.js)
   ↓
2. WebSocket으로 NestJS 전송
   ↓
3. NestJS: 메시지 저장 (PostgreSQL)
   ↓
4. NestJS → FastAPI: 감정 분석 요청
   POST http://ml-serving:8000/api/v1/analyze
   {
     "text": "오늘 너무 우울해요",
     "session_id": "uuid"
   }
   ↓
5. FastAPI: 
   - KLUE-BERT 모델 추론
   - 위험도 평가
   - 아바타 이미지 생성
   ↓
6. FastAPI → NestJS: 분석 결과 반환
   {
     "emotion": "sad",
     "confidence": 0.87,
     "risk_level": "medium",
     "avatar_url": "data:image/png;base64,...",
     "probabilities": {...}
   }
   ↓
7. NestJS: 분석 결과 저장 + Redis 캐싱
   ↓
8. NestJS → Next.js: WebSocket으로 실시간 전송
   ↓
9. Next.js: UI 업데이트
   - 감정 배지 표시
   - 아바타 이미지 표시
   - 위험도 알림 (필요시)
```

---

## 🐳 Docker Compose 구성

```yaml
version: '3.8'

services:
  # 프론트엔드
  frontend:
    build: ./frontend-next
    ports:
      - "3000:3000"
    environment:
      - NEXT_PUBLIC_API_URL=http://backend:3001
      - NEXT_PUBLIC_WS_URL=ws://backend:3001
    depends_on:
      - backend

  # 백엔드 API
  backend:
    build: ./backend-nest
    ports:
      - "3001:3001"
    environment:
      - DATABASE_URL=postgresql://user:pass@postgres:5432/chatbot
      - REDIS_URL=redis://redis:6379
      - ML_SERVICE_URL=http://ml-serving:8000
    depends_on:
      - postgres
      - redis
      - ml-serving

  # ML 서빙
  ml-serving:
    build: ./ml-serving
    ports:
      - "8000:8000"
    volumes:
      - ./checkpoints_kfold:/app/checkpoints
    environment:
      - MODEL_PATH=/app/checkpoints/fold1_model_20251028_113127.pt
      - DEVICE=cuda  # 또는 cpu
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]

  # PostgreSQL
  postgres:
    image: postgres:15-alpine
    ports:
      - "5432:5432"
    environment:
      - POSTGRES_USER=chatbot_user
      - POSTGRES_PASSWORD=chatbot_pass
      - POSTGRES_DB=chatbot
    volumes:
      - postgres_data:/var/lib/postgresql/data

  # Redis
  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data

  # Nginx (리버스 프록시)
  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx/nginx.conf:/etc/nginx/nginx.conf
    depends_on:
      - frontend
      - backend

volumes:
  postgres_data:
  redis_data:
```

---

## 📡 API 설계

### NestJS Backend API

#### Chat Endpoints
```typescript
// 채팅 세션 생성
POST /api/chat/sessions
Response: { sessionId: string, createdAt: Date }

// 메시지 전송
POST /api/chat/sessions/:sessionId/messages
Body: { text: string, userId?: string }
Response: { messageId: string, emotion: EmotionResult }

// 세션 조회
GET /api/chat/sessions/:sessionId
Response: { session: Session, messages: Message[] }

// WebSocket
WS /chat
Events: 
  - message (client → server)
  - emotion-result (server → client)
  - typing (bidirectional)
```

#### Emotion Endpoints (Proxy)
```typescript
// 감정 분석
POST /api/emotion/analyze
Body: { text: string }
Response: { emotion: string, confidence: number, ... }

// 아바타 생성
POST /api/emotion/generate-avatar
Body: { text: string, emotion?: string }
Response: { avatarUrl: string, emotion: string }
```

#### Analytics Endpoints
```typescript
// 세션 통계
GET /api/analytics/sessions/:sessionId/stats
Response: { emotionDistribution: {...}, timeline: [...] }

// 전체 통계
GET /api/analytics/overview
Response: { totalSessions: number, emotionTrends: {...} }
```

### FastAPI ML Serving API

```python
# 감정 분석
POST /api/v1/analyze
{
  "text": "오늘 너무 행복해요!",
  "session_id": "optional-uuid"
}
Response: {
  "emotion": "joy",
  "emotion_kr": "기쁨",
  "confidence": 0.92,
  "probabilities": {...},
  "inference_time_ms": 45
}

# 아바타 생성
POST /api/v1/generate-avatar
{
  "emotion": "joy",
  "style": "gradient"  # optional
}
Response: {
  "avatar_url": "data:image/png;base64,...",
  "emotion": "joy",
  "generation_time_ms": 120
}

# 위험도 평가
POST /api/v1/assess-risk
{
  "text": "...",
  "emotion": "sad"
}
Response: {
  "risk_level": "high",
  "risk_score": 0.78,
  "recommendations": [...]
}

# 헬스체크
GET /api/v1/health
Response: {
  "status": "healthy",
  "model_loaded": true,
  "gpu_available": true
}
```

---

## 🚀 개발 로드맵

### Phase 1: 기반 구축 (1-2일)
- [ ] 프로젝트 구조 생성
- [ ] Docker Compose 설정
- [ ] 기본 환경 설정 (ENV, Config)

### Phase 2: ML 서빙 (1일)
- [ ] FastAPI 기본 구조
- [ ] 기존 모델 통합
- [ ] 감정 분석 API 구현
- [ ] 아바타 생성 API 구현

### Phase 3: 백엔드 API (2일)
- [ ] NestJS 프로젝트 초기화
- [ ] Database 스키마 설계
- [ ] Chat 모듈 구현
- [ ] WebSocket 게이트웨이
- [ ] FastAPI 연동

### Phase 4: 프론트엔드 (2-3일)
- [ ] Next.js 15 프로젝트 설정
- [ ] UI 컴포넌트 라이브러리 (shadcn/ui)
- [ ] 채팅 인터페이스
- [ ] 감정 시각화
- [ ] WebSocket 연동

### Phase 5: 통합 및 테스트 (1-2일)
- [ ] End-to-end 테스트
- [ ] 성능 최적화
- [ ] 에러 핸들링
- [ ] 모니터링 설정

---

## 💰 비용 및 리소스

### 개발 환경
- **로컬**: Docker Desktop + WSL2
- **비용**: 무료

### 프로덕션 (예상)
- **Frontend (Vercel)**: 무료 ~ $20/월
- **Backend (Railway/Render)**: $5-20/월
- **Database (Supabase)**: 무료 ~ $25/월
- **ML Serving (RunPod/Modal)**: $0.15-0.5/hour (GPU)
- **총 예상**: $20-100/월

---

## 🎯 장점

1. **확장성**: 각 서비스 독립적으로 스케일링
2. **유지보수**: 명확한 책임 분리
3. **개발 속도**: TypeScript로 타입 안정성
4. **성능**: Next.js RSC + Redis 캐싱
5. **전문성**: 산업 표준 기술 스택
6. **배포**: Docker로 환경 일관성
7. **모니터링**: Prometheus + Grafana 통합 가능

---

## 📚 참고 문서

- **Next.js**: https://nextjs.org/docs
- **NestJS**: https://docs.nestjs.com
- **FastAPI**: https://fastapi.tiangolo.com
- **TypeORM**: https://typeorm.io
- **Prisma**: https://www.prisma.io/docs
- **shadcn/ui**: https://ui.shadcn.com

---

**작성일**: 2025-10-28  
**버전**: 1.0.0  
**상태**: 설계 완료, 구현 대기
