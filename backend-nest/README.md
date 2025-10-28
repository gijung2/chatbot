# NestJS Backend for Emotion Analysis Chatbot

## 📋 개요

감정 분석 챗봇의 NestJS 기반 백엔드 서비스입니다.

## 🚀 시작하기

### 설치

```bash
npm install
```

### 환경 변수 설정

```bash
cp .env.example .env
# .env 파일을 열어 필요한 값을 설정하세요
```

### 데이터베이스 설정

```bash
# PostgreSQL 실행 (Docker)
docker run --name chatbot-postgres \
  -e POSTGRES_USER=chatbot \
  -e POSTGRES_PASSWORD=chatbot123 \
  -e POSTGRES_DB=chatbot_db \
  -p 5432:5432 \
  -d postgres:15-alpine

# Redis 실행 (Docker)
docker run --name chatbot-redis \
  -p 6379:6379 \
  -d redis:7-alpine
```

### 개발 모드 실행

```bash
npm run start:dev
```

### 프로덕션 빌드

```bash
npm run build
npm run start:prod
```

## 📚 API 문서

서버 실행 후 Swagger 문서를 확인할 수 있습니다:

- **Swagger UI**: http://localhost:3001/api/docs

## 🏗️ 프로젝트 구조

```
src/
├── main.ts                 # 애플리케이션 엔트리포인트
├── app.module.ts           # 루트 모듈
├── app.controller.ts       # 루트 컨트롤러
├── app.service.ts          # 루트 서비스
├── modules/
│   ├── chat/              # 채팅 모듈
│   ├── emotion/           # 감정 분석 모듈
│   ├── analytics/         # 분석 모듈
│   └── user/              # 사용자 모듈
├── common/                # 공통 유틸리티
│   ├── decorators/
│   ├── filters/
│   ├── guards/
│   ├── interceptors/
│   └── pipes/
└── config/                # 설정 파일
```

## 🔧 주요 기능

- ✅ RESTful API
- ✅ WebSocket (Socket.io)
- ✅ PostgreSQL 연동
- ✅ Redis 캐싱
- ✅ Swagger API 문서
- ✅ TypeORM
- ✅ 유효성 검사 (class-validator)
- ✅ ML 서비스 연동

## 🧪 테스트

```bash
# 단위 테스트
npm run test

# e2e 테스트
npm run test:e2e

# 테스트 커버리지
npm run test:cov
```

## 📦 Docker

```bash
# 이미지 빌드
docker build -t chatbot-backend .

# 컨테이너 실행
docker run -p 3001:3001 chatbot-backend
```

## 🔗 관련 서비스

- **ML Serving**: http://localhost:8000
- **Frontend**: http://localhost:3000
- **PostgreSQL**: localhost:5432
- **Redis**: localhost:6379
