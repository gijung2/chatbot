# ML Serving Dockerfile
FROM python:3.11-slim

# 작업 디렉토리 설정
WORKDIR /app

# 시스템 의존성 설치
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Python 의존성 복사 및 설치
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 애플리케이션 코드 복사
COPY app/ ./app/
COPY start_server.py .

# 모델 디렉토리 (외부 볼륨으로 마운트 예정)
RUN mkdir -p /app/checkpoints_kfold

# 환경 변수 설정
ENV PYTHONUNBUFFERED=1
ENV MODEL_PATH=/app/checkpoints_kfold/fold1_model_20251028_113127.pt
ENV DEVICE=cpu
ENV PORT=8000

# 포트 노출
EXPOSE 8000

# 헬스체크
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8000/api/v1/health')"

# 서버 실행
CMD ["python", "start_server.py"]
