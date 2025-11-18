# 🚀 빠른 시작 가이드

감정 분석 챗봇 프로그램을 실행하는 방법을 단계별로 설명합니다.

## 📋 목차

1. [환경 요구사항](#환경-요구사항)
2. [설치](#설치)
3. [프로그램 실행](#프로그램-실행)
4. [사용 방법](#사용-방법)
5. [문제 해결](#문제-해결)

---

## 환경 요구사항

- **Python**: 3.8 이상 (권장: 3.11)
- **운영체제**: Windows, macOS, Linux
- **메모리**: 최소 4GB RAM (권장: 8GB 이상)
- **디스크**: 약 2GB 여유 공간

---

## 설치

### 1. 프로젝트 클론

```bash
git clone https://github.com/gijung2/chatbot.git
cd chatbot
```

### 2. 가상환경 생성 및 활성화

**Windows (PowerShell):**
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**macOS/Linux:**
```bash
python -m venv .venv
source .venv/bin/activate
```

### 3. 의존성 패키지 설치

```bash
pip install -r requirements.txt
```

> ⏱️ 설치에 약 5-10분 정도 소요될 수 있습니다.

---

## 프로그램 실행

### 방법 1: FastAPI 서버 + 심플 챗 데모 (권장)

이 방법은 웹 브라우저에서 실시간 채팅 형식으로 감정 분석을 테스트할 수 있습니다.

#### 1단계: FastAPI 서버 실행

터미널에서 다음 명령어를 실행합니다:

```bash
python fastapi_app/main.py
```

**실행 결과:**
```
INFO:     Uvicorn running on http://127.0.0.1:8000 (Press CTRL+C to quit)
INFO:     Started reloader process
```

> 🟢 서버가 실행 중일 때는 터미널을 종료하지 마세요!

#### 2단계: 심플 챗 데모 열기

1. **브라우저에서 직접 파일 열기:**
   - `simple_chat_demo.html` 파일을 더블 클릭하거나
   - 브라우저 주소창에 파일 경로를 입력합니다
   
   ```
   file:///C:/Users/[사용자명]/Desktop/chatbot/simple_chat_demo.html
   ```

2. **또는 VS Code에서 Live Server 사용:**
   - VS Code에서 `simple_chat_demo.html` 우클릭
   - "Open with Live Server" 선택

#### 3단계: 채팅 시작

웹 페이지가 열리면 바로 채팅을 시작할 수 있습니다!

**예시 문장:**
```
오늘 정말 기분이 좋아! 😊
시험에 떨어져서 너무 슬퍼...
내일 발표인데 불안해
```

각 메시지에 대해 감정(기쁨, 슬픔, 불안, 분노, 중립)이 분석되어 표시됩니다.

---

### 방법 2: 감정 테스트 페이지

더 상세한 분석 결과를 확인하고 싶다면 이 방법을 사용하세요.

#### 1단계: FastAPI 서버 실행 (동일)

```bash
python fastapi_app/main.py
```

#### 2단계: 감정 테스트 페이지 열기

`emotion_test.html` 파일을 브라우저에서 엽니다.

#### 기능:

- **📊 기본 분석**: 간단한 감정 분류 및 확률 분포
- **🔍 상세 분석**: 위험도 평가 및 상담 제안 포함

**위험도 평가 예시:**
- **낮음**: 일반적인 감정 표현
- **중간**: 약간의 부정적 감정
- **높음**: 자살/자해 관련 표현 감지 → 전문 상담 권장

---

### 방법 3: Python 스크립트로 직접 테스트

코드 레벨에서 모델을 테스트하고 싶다면:

```bash
python test_new_model.py
```

**출력 예시:**
```
🧪 Colab 학습 모델 테스트
================================================================================

📦 모델 로드 중...
✅ 모델 로드 완료

📊 모델 정보:
   - model_type: skt/kobert-base-v1
   - total_parameters: 92,190,725
   - device: cpu

🎯 감정 분석 테스트
================================================================================

[1] 오늘 정말 기분이 좋아! 너무 행복해!
   🎭 감정: joy (신뢰도: 99.74%)
   📊 확률 분포:
      - joy     : 99.74% ████████████████████
      - neutral :  0.12% 
      - anxiety :  0.08% 
```

---

## 사용 방법

### 심플 챗 데모 사용법

1. **메시지 입력**: 하단 입력창에 문장을 입력합니다.
2. **전송**: Enter 키를 누르거나 전송 버튼을 클릭합니다.
3. **결과 확인**: 
   - 왼쪽(사용자 메시지): 입력한 문장
   - 오른쪽(봇 응답): 감정 분석 결과
     - 감정 태그 (예: 🎉 기쁨)
     - 신뢰도 (예: 99.74%)
     - 감정별 확률 분포

### API 엔드포인트

FastAPI 서버가 제공하는 API:

#### 1. 기본 감정 분석
```http
POST http://localhost:8000/emotion/analyze
Content-Type: application/json

{
  "text": "오늘 기분이 좋아!"
}
```

**응답:**
```json
{
  "emotion": "joy",
  "confidence": 0.9974,
  "probabilities": {
    "joy": 0.9974,
    "neutral": 0.0012,
    "anxiety": 0.0008,
    "sad": 0.0004,
    "anger": 0.0002
  }
}
```

#### 2. 상세 분석 (위험도 평가 포함)
```http
POST http://localhost:8000/emotion/analyze/detailed
Content-Type: application/json

{
  "text": "너무 우울해서 아무것도 하기 싫어"
}
```

**응답:**
```json
{
  "emotion": "sad",
  "confidence": 0.9234,
  "probabilities": { ... },
  "risk_assessment": {
    "level": "medium",
    "score": 6,
    "recommendation": "전문가 상담 권장",
    "risk_factors": ["우울감 표현"]
  },
  "counseling_suggestions": [
    "우울한 감정을 느끼고 계시는군요. 괜찮으신가요?",
    "전문 상담사와 대화해보는 것을 추천드립니다."
  ]
}
```

#### 3. 헬스 체크
```http
GET http://localhost:8000/health
```

#### 4. 모델 정보
```http
GET http://localhost:8000/emotion/model-info
```

---

## 문제 해결

### ❌ "ModuleNotFoundError" 오류

**원인**: 필요한 패키지가 설치되지 않음

**해결:**
```bash
pip install -r requirements.txt
```

### ❌ "Connection refused" 오류 (심플 챗 페이지)

**원인**: FastAPI 서버가 실행되지 않음

**해결:**
1. 서버가 실행 중인지 확인: `http://localhost:8000/health` 접속
2. 서버 재시작: `python fastapi_app/main.py`

### ❌ "trust_remote_code" 오류

**원인**: KoBERT 토크나이저 신뢰 설정 필요

**해결**: 이미 코드에 적용되어 있습니다. 최신 코드를 사용하세요.

### ❌ 모델 파일이 없다는 오류

**원인**: `best_emotion_model` 폴더가 누락됨

**해결:**
1. Colab에서 학습한 모델을 다운로드
2. `best_emotion_model/` 폴더에 배치
3. 폴더 구조 확인:
   ```
   best_emotion_model/
   ├── training_results.json
   └── best_emotion_model/
       ├── config.json
       ├── model.safetensors
       ├── tokenizer.json
       └── ...
   ```

### 🐢 모델 로딩이 너무 느림

**원인**: CPU 모드로 실행 중

**해결 (GPU 사용):**
```python
# fastapi_app/main.py 수정
model = EmotionClassifierHF(device='cuda')  # GPU 사용
```

> ⚠️ CUDA 지원 PyTorch가 설치되어 있어야 합니다.

### 💬 심플 챗이 작동하지 않음

**체크리스트:**
1. ✅ FastAPI 서버가 실행 중인가? → `http://localhost:8000/health` 확인
2. ✅ 브라우저 콘솔에 CORS 오류가 있는가? → F12로 콘솔 확인
3. ✅ 포트가 맞는가? → `simple_chat_demo.html`의 API_BASE 확인

---

## 📚 추가 문서

- **모델 학습**: [KOTE_TRAINING_GUIDE.md](KOTE_TRAINING_GUIDE.md)
- **Colab 사용**: [COLAB_QUICK_START.md](COLAB_QUICK_START.md)
- **성능 개선**: [PERFORMANCE_IMPROVEMENT_GUIDE.md](PERFORMANCE_IMPROVEMENT_GUIDE.md)
- **Unity 연동**: [UNITY_INTEGRATION_GUIDE.md](UNITY_INTEGRATION_GUIDE.md)
- **배포 가이드**: [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md)

---

## 🎯 다음 단계

### 1. 모델 성능 확인
```bash
python test_new_model.py
```

### 2. API 통합 테스트
```bash
python test_api_integration.py
```

### 3. 성능 벤치마크
```bash
python test_model_performance.py --detailed
```

### 4. 커스터마이징

원하는 감정 클래스나 응답을 수정하려면:
- 모델: `fastapi_app/models/emotion_model_hf.py`
- 라우터: `fastapi_app/routers/emotion.py`
- 서비스: `fastapi_app/services/emotion_service.py`

---

## 📞 지원

문제가 발생하면:
1. [Issues](https://github.com/gijung2/chatbot/issues)에 등록
2. 에러 로그 및 환경 정보 포함
3. 재현 가능한 최소 예제 제공

---

**즐거운 감정 분석 되세요! 🎭✨**
