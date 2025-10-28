# 심리상담 아바타 시스템 - 경량버전 Docker 이미지
FROM python:3.11-slim

# 작업 디렉토리 설정
WORKDIR /app

# 시스템 패키지 업데이트 및 필수 도구만 설치
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Python 의존성 파일 복사 (경량버전 사용)
COPY requirements_minimal.txt requirements.txt

# Python 패키지 설치
RUN pip install --no-cache-dir -r requirements.txt

# 애플리케이션 코드 복사 (핵심 파일들만)
COPY lightweight_psychological_api.py .
COPY demo_server.py .
COPY simple_chat_demo.html .
COPY data/ ./data/

# 필요한 디렉토리 생성
RUN mkdir -p logs

# 환경 변수 설정
ENV FLASK_APP=lightweight_psychological_api.py
ENV FLASK_ENV=production
ENV PYTHONPATH=/app
ENV PORT=8003

# 비root 사용자 생성 및 권한 설정
RUN groupadd -r avatar && useradd -r -g avatar avatar
RUN chown -R avatar:avatar /app
USER avatar

# 포트 노출
EXPOSE 8003 8080

# 헬스체크 추가
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8003/health || exit 1

# 애플리케이션 실행 (개발모드로 간단하게)
CMD ["python", "lightweight_psychological_api.py"]
