# 🚀 리팩토링 실행 계획

## ✅ 현재 상황
- 학습된 KLUE-BERT 모델 보유 (59.74% 정확도)
- Flask 기반 프로토타입 존재
- 감정 분석 + 아바타 생성 로직 완성

## 🎯 목표
**모던 풀스택 마이크로서비스 아키텍처로 전환**
- Frontend: Next.js 15 (App Router)
- Backend: NestJS 10
- ML Serving: FastAPI

---

## 📋 실행 단계

### 🔴 Phase 1: FastAPI ML 서빙 구축 (우선순위 1)
> **이유**: 기존 모델을 즉시 활용 가능, 다른 서비스의 기반

#### Step 1.1: 프로젝트 초기화
```bash
cd chatbot
mkdir ml-serving
cd ml-serving

# Python 가상환경
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 의존성 설치
pip install fastapi uvicorn torch transformers pillow pydantic python-multipart
```

#### Step 1.2: 기본 구조 생성
```
ml-serving/
├── app/
│   ├── __init__.py
│   ├── main.py              # FastAPI 앱 엔트리
│   ├── config.py            # 설정 (MODEL_PATH 등)
│   ├── api/
│   │   └── v1/
│   │       ├── __init__.py
│   │       └── endpoints/
│   │           ├── emotion.py    # POST /analyze
│   │           ├── avatar.py     # POST /generate-avatar
│   │           └── health.py     # GET /health
│   ├── models/
│   │   ├── __init__.py
│   │   └── emotion_classifier.py  # 기존 코드 이전
│   └── schemas/
│       ├── __init__.py
│       ├── emotion.py       # Pydantic 모델
│       └── common.py
├── checkpoints/ -> ../checkpoints_kfold/  # 심볼릭 링크
├── requirements.txt
└── Dockerfile
```

#### Step 1.3: 핵심 코드 작성
**`app/main.py`**
```python
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.v1.endpoints import emotion, avatar, health

app = FastAPI(
    title="Emotion Analysis ML Service",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 프로덕션: 특정 도메인만
    allow_methods=["*"],
    allow_headers=["*"],
)

# 라우터 등록
app.include_router(emotion.router, prefix="/api/v1", tags=["emotion"])
app.include_router(avatar.router, prefix="/api/v1", tags=["avatar"])
app.include_router(health.router, prefix="/api/v1", tags=["health"])

@app.get("/")
def root():
    return {"service": "ML Serving", "status": "running"}
```

#### Step 1.4: 테스트 및 검증
```bash
# 실행
uvicorn app.main:app --reload --port 8000

# 테스트
curl -X POST http://localhost:8000/api/v1/analyze \
  -H "Content-Type: application/json" \
  -d '{"text": "오늘 정말 기분이 좋아!"}'
```

**예상 소요 시간**: 2-3시간

---

### 🟡 Phase 2: NestJS 백엔드 구축 (우선순위 2)

#### Step 2.1: 프로젝트 초기화
```bash
cd chatbot
npm i -g @nestjs/cli
nest new backend-nest

cd backend-nest
npm install @nestjs/websockets @nestjs/platform-socket.io
npm install @nestjs/typeorm typeorm pg redis
npm install class-validator class-transformer
npm install @nestjs/config axios
```

#### Step 2.2: 모듈 생성
```bash
nest g module chat
nest g module emotion
nest g module analytics

nest g controller chat
nest g service chat
nest g gateway chat

nest g controller emotion
nest g service emotion
```

#### Step 2.3: 핵심 구조
```typescript
// src/chat/chat.gateway.ts
@WebSocketGateway({ cors: true })
export class ChatGateway {
  @WebSocketServer()
  server: Server;

  @SubscribeMessage('message')
  async handleMessage(
    @MessageBody() data: { text: string; sessionId: string },
  ): Promise<void> {
    // 1. 메시지 저장
    const message = await this.chatService.saveMessage(data);
    
    // 2. 감정 분석 요청 (FastAPI)
    const emotion = await this.emotionService.analyze(data.text);
    
    // 3. 결과 전송
    this.server.emit('emotion-result', {
      messageId: message.id,
      emotion,
    });
  }
}
```

#### Step 2.4: FastAPI 연동
```typescript
// src/emotion/emotion.service.ts
@Injectable()
export class EmotionService {
  constructor(private readonly httpService: HttpService) {}

  async analyze(text: string): Promise<EmotionResult> {
    const response = await firstValueFrom(
      this.httpService.post('http://ml-serving:8000/api/v1/analyze', {
        text,
      }),
    );
    return response.data;
  }
}
```

**예상 소요 시간**: 4-5시간

---

### 🟢 Phase 3: Next.js 프론트엔드 (우선순위 3)

#### Step 3.1: 프로젝트 초기화
```bash
cd chatbot
npx create-next-app@latest frontend-next \
  --typescript \
  --tailwind \
  --app \
  --src-dir \
  --import-alias "@/*"

cd frontend-next
npm install socket.io-client zustand @tanstack/react-query
npx shadcn-ui@latest init
npx shadcn-ui@latest add button input card badge
```

#### Step 3.2: 기본 구조
```typescript
// src/app/chat/page.tsx
'use client';

import { useChat } from '@/lib/hooks/useChat';
import { ChatInterface } from '@/components/chat/ChatInterface';

export default function ChatPage() {
  const { messages, sendMessage, isConnected } = useChat();

  return (
    <div className="container mx-auto p-4">
      <ChatInterface 
        messages={messages}
        onSendMessage={sendMessage}
        isConnected={isConnected}
      />
    </div>
  );
}
```

#### Step 3.3: WebSocket 연동
```typescript
// src/lib/hooks/useChat.ts
import { useEffect, useState } from 'react';
import { io, Socket } from 'socket.io-client';

export function useChat() {
  const [socket, setSocket] = useState<Socket | null>(null);
  const [messages, setMessages] = useState<Message[]>([]);

  useEffect(() => {
    const socketInstance = io('http://localhost:3001/chat');
    
    socketInstance.on('emotion-result', (data) => {
      setMessages(prev => [...prev, { ...data, type: 'emotion' }]);
    });

    setSocket(socketInstance);

    return () => {
      socketInstance.disconnect();
    };
  }, []);

  const sendMessage = (text: string) => {
    socket?.emit('message', { text, sessionId: 'demo' });
    setMessages(prev => [...prev, { text, type: 'user' }]);
  };

  return { messages, sendMessage, isConnected: socket?.connected };
}
```

**예상 소요 시간**: 5-6시간

---

### 🔵 Phase 4: Docker 통합 (우선순위 4)

#### Step 4.1: 각 서비스 Dockerfile 작성

**ML Serving Dockerfile**
```dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app/ ./app/
COPY checkpoints/ ./checkpoints/

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

**Backend Dockerfile**
```dockerfile
FROM node:20-alpine AS builder

WORKDIR /app
COPY package*.json ./
RUN npm ci

COPY . .
RUN npm run build

FROM node:20-alpine

WORKDIR /app
COPY --from=builder /app/dist ./dist
COPY --from=builder /app/node_modules ./node_modules
COPY package*.json ./

EXPOSE 3001

CMD ["node", "dist/main"]
```

**Frontend Dockerfile**
```dockerfile
FROM node:20-alpine AS builder

WORKDIR /app
COPY package*.json ./
RUN npm ci

COPY . .
RUN npm run build

FROM node:20-alpine

WORKDIR /app
COPY --from=builder /app/.next ./.next
COPY --from=builder /app/node_modules ./node_modules
COPY --from=builder /app/package.json ./
COPY --from=builder /app/public ./public

EXPOSE 3000

CMD ["npm", "start"]
```

#### Step 4.2: Docker Compose
이미 ARCHITECTURE.md에 작성된 docker-compose.yml 사용

#### Step 4.3: 실행 및 테스트
```bash
# 전체 시스템 빌드 및 실행
docker-compose up --build

# 개별 서비스 재시작
docker-compose restart ml-serving

# 로그 확인
docker-compose logs -f backend
```

**예상 소요 시간**: 2-3시간

---

## 🎯 단계별 체크리스트

### Phase 1: ML Serving ✅
- [ ] FastAPI 프로젝트 구조 생성
- [ ] 기존 모델 로딩 코드 이전
- [ ] `/api/v1/analyze` 엔드포인트 구현
- [ ] `/api/v1/generate-avatar` 엔드포인트 구현
- [ ] Pydantic 스키마 정의
- [ ] 에러 핸들링
- [ ] 헬스체크 엔드포인트
- [ ] 로컬 테스트 완료

### Phase 2: NestJS Backend ✅
- [ ] NestJS 프로젝트 초기화
- [ ] PostgreSQL 연동 (TypeORM)
- [ ] Redis 연동
- [ ] Chat 모듈 (CRUD)
- [ ] WebSocket 게이트웨이
- [ ] Emotion 프록시 서비스
- [ ] FastAPI 클라이언트 구현
- [ ] Swagger 문서화
- [ ] 통합 테스트

### Phase 3: Next.js Frontend ✅
- [ ] Next.js 15 프로젝트 생성
- [ ] shadcn/ui 설정
- [ ] 채팅 UI 컴포넌트
- [ ] WebSocket 클라이언트
- [ ] 상태 관리 (Zustand)
- [ ] 감정 시각화 컴포넌트
- [ ] 아바타 표시 컴포넌트
- [ ] 반응형 디자인
- [ ] 에러 바운더리

### Phase 4: Integration ✅
- [ ] 전체 서비스 Docker 이미지 빌드
- [ ] docker-compose.yml 작성
- [ ] 환경 변수 설정
- [ ] 네트워크 통신 테스트
- [ ] End-to-end 플로우 테스트
- [ ] 성능 모니터링 설정

---

## ⚡ 빠른 시작 (Quick Start)

### Option 1: 순차적 개발 (권장)
```bash
# Day 1: ML Serving
cd ml-serving && python -m venv venv && pip install -r requirements.txt
uvicorn app.main:app --reload

# Day 2-3: Backend
cd backend-nest && npm install && npm run start:dev

# Day 4-5: Frontend
cd frontend-next && npm install && npm run dev

# Day 6: Docker 통합
docker-compose up --build
```

### Option 2: 병렬 개발 (빠른 프로토타입)
3명이 각 레이어를 동시에 개발
- Person A: ML Serving
- Person B: Backend
- Person C: Frontend

**API 계약 먼저 정의** → Swagger/OpenAPI로 Mock

---

## 📊 예상 일정

| Phase | 작업 | 소요 시간 | 인력 |
|-------|------|-----------|------|
| Phase 1 | ML Serving | 2-3시간 | 1명 |
| Phase 2 | Backend API | 4-5시간 | 1명 |
| Phase 3 | Frontend | 5-6시간 | 1명 |
| Phase 4 | Docker 통합 | 2-3시간 | 1명 |
| **총계** | | **13-17시간** | **1명 기준** |

**병렬 개발 시**: 2일 (하루 8시간 기준)

---

## 🎬 시작하시겠습니까?

다음 중 선택해주세요:

1. **Phase 1 시작**: ML Serving부터 단계별로 구축
2. **전체 스캐폴딩**: 모든 프로젝트 구조를 한 번에 생성
3. **특정 기능 먼저**: 원하는 기능부터 구현
4. **기존 코드 마이그레이션**: 현재 코드를 새 구조로 이동

어떤 방식으로 진행하시겠습니까?
