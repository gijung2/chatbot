# Next.js Frontend for Emotion Analysis Chatbot

## 📋 개요

감정 분석 챗봇의 Next.js 15 기반 프론트엔드 서비스입니다.

## 🚀 시작하기

### 설치

```bash
npm install
```

### 환경 변수 설정

```bash
cp .env.example .env.local
# .env.local 파일을 열어 필요한 값을 설정하세요
```

### 개발 모드 실행

```bash
npm run dev
```

브라우저에서 http://localhost:3000 을 열어주세요.

### 프로덕션 빌드

```bash
npm run build
npm run start
```

## 🏗️ 프로젝트 구조

```
src/
├── app/
│   ├── layout.tsx         # 루트 레이아웃
│   ├── page.tsx           # 홈페이지
│   ├── providers.tsx      # React Query Provider
│   ├── globals.css        # 전역 스타일
│   ├── chat/
│   │   └── page.tsx       # 채팅 페이지
│   └── analytics/
│       └── page.tsx       # 분석 대시보드
├── components/
│   ├── ChatMessage.tsx    # 채팅 메시지 컴포넌트
│   └── ChatInput.tsx      # 채팅 입력 컴포넌트
└── hooks/
    └── useSocket.ts       # WebSocket Hook
```

## 🔧 주요 기능

- ✅ Next.js 15 (App Router)
- ✅ React Server Components
- ✅ TypeScript
- ✅ TailwindCSS
- ✅ Socket.io Client
- ✅ React Query
- ✅ 실시간 채팅
- ✅ 감정 분석 UI
- ✅ 아바타 표시

## 📦 Docker

```bash
# 이미지 빌드
docker build -t chatbot-frontend .

# 컨테이너 실행
docker run -p 3000:3000 chatbot-frontend
```

## 🔗 관련 서비스

- **Backend API**: http://localhost:3001
- **ML Serving**: http://localhost:8000
