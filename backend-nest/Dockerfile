# NestJS Backend Dockerfile
FROM node:20-alpine AS builder

# 작업 디렉토리
WORKDIR /app

# 의존성 파일 복사
COPY package*.json ./
COPY tsconfig*.json ./
COPY nest-cli.json ./

# 의존성 설치
RUN npm ci

# 소스 코드 복사
COPY src ./src

# 빌드
RUN npm run build

# Production 이미지
FROM node:20-alpine

WORKDIR /app

# 프로덕션 의존성만 설치
COPY package*.json ./
RUN npm ci --only=production && npm cache clean --force

# 빌드된 파일 복사
COPY --from=builder /app/dist ./dist

# 환경 변수
ENV NODE_ENV=production
ENV PORT=3001

# 포트 노출
EXPOSE 3001

# 헬스체크
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD node -e "require('http').get('http://localhost:3001/health', (r) => r.statusCode === 200 ? process.exit(0) : process.exit(1))"

# 서버 실행
CMD ["node", "dist/main"]
