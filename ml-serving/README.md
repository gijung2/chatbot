# 🚀 ML Serving - 감정 분석 API

KLUE-BERT 기반 한국어 감정 분석 및 아바타 생성 API 서버

## ⚡ 빠른 시작

### 1. 의존성 설치
```bash
cd ml-serving
pip install -r requirements.txt
```

### 2. 환경 변수 설정
```bash
cp .env.example .env
# .env 파일 편집 (필요시)
```

### 3. 서버 실행
```bash
# 개발 모드 (자동 재시작)
python -m app.main

# 또는 uvicorn 직접 실행
uvicorn app.main:app --reload --port 8000
```

### 4. API 문서 확인
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## 📡 API 엔드포인트

### 감정 분석
```bash
POST /api/v1/analyze
{
  "text": "오늘 정말 기분이 좋아요!"
}
```

### 아바타 생성
```bash
POST /api/v1/generate-avatar
{
  "text": "오늘 너무 행복해요!",
  "style": "gradient"
}
```

### 헬스체크
```bash
GET /api/v1/health
```

## 🏗️ 프로젝트 구조

```
ml-serving/
├── app/
│   ├── main.py              # FastAPI 앱
│   ├── config.py            # 설정
│   ├── api/
│   │   └── v1/
│   │       └── endpoints/   # API 엔드포인트
│   ├── models/              # ML 모델
│   ├── schemas/             # Pydantic 스키마
│   └── services/            # 비즈니스 로직
├── requirements.txt
└── .env
```

## 🔧 설정

`.env` 파일에서 다음 항목을 설정할 수 있습니다:

- `MODEL_PATH`: 학습된 모델 경로
- `DEVICE`: 디바이스 (auto, cuda, cpu)
- `PORT`: 서버 포트
- `DEBUG`: 디버그 모드

## 📊 응답 예시

```json
{
  "text": "오늘 정말 기분이 좋아요!",
  "emotion": "joy",
  "emotion_kr": "기쁨",
  "confidence": 0.92,
  "probabilities": {
    "joy": 0.92,
    "sad": 0.02,
    "anxiety": 0.03,
    "anger": 0.01,
    "neutral": 0.02
  },
  "risk_level": "low",
  "risk_message": "💚 안정적인 상태입니다.",
  "emotion_message": "긍정적인 에너지가 느껴져요!",
  "method": "klue-bert-kfold",
  "inference_time_ms": 45.2
}
```

## 🐳 Docker 실행

```bash
docker build -t ml-serving .
docker run -p 8000:8000 ml-serving
```

## 📝 개발

### 테스트
```bash
pytest tests/
```

### 코드 품질
```bash
black app/
flake8 app/
mypy app/
```
