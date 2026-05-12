# 🧠 심리상담 감정 분석 챗봇

**KR-BERT 기반 한국어 감정 분석 및 심리상담 챗봇 시스템**

[![Python](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104-green.svg)](https://fastapi.tiangolo.com/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)

<img width="801" height="770" alt="image" src="https://github.com/user-attachments/assets/140e8d7b-608e-45ea-b465-06202e50e100" />


## 🎯 주요 기능

### � **실시간 감정 분석**
- 감정분석 모델 기반 5가지 감정 분류 (joy, sad, anxiety, anger, neutral)
- 클래스 가중치 적용으로 불균형 데이터 보정
- 신뢰도 점수 제공

### � **Live2D 아바타 채팅**
- 감정별 아바타 표정 변화
- 실시간 감정 동기화
- 다크/라이트 모드 지원

### 🤖 **심리상담 응답**
- 감정별 맞춤 상담 메시지

## 📂 프로젝트 구조

```
chatbot/
├── fastapi_app/              # FastAPI 백엔드 서버
│   ├── main.py              # 메인 애플리케이션
│   ├── routers/             # API 라우터 (chat, emotion, avatar)
│   ├── models/              # 감정 분류 모델
│   └── services/            # 심리 분석 서비스
│
├── simple_chat_demo.html     # 채팅 데모 페이지
├── colab_training.ipynb      # Google Colab 학습 노트북
│
├── training/                 # 로컬 학습 스크립트
│   ├── train_krbert_hf.py   # KR-BERT 학습 (클래스 가중치)
│   ├── data_loader.py       # 데이터 로더
│   └── visualize.py         # 학습 결과 시각화
│
├── data/                     # 감정 데이터셋
│   ├── processed/           # 전처리된 데이터
│   └── raw/                 # 원본 데이터
│
├── checkpoints_kfold/        # 학습된 모델 체크포인트
├── docs/                     # 가이드 문서들
└── requirements.txt          # Python 패키지 의존성
```

## 🛠️ 설치 및 실행

### **1. 설치**

```bash
# 저장소 클론
git clone https://github.com/gijung2/chatbot.git
cd chatbot

# 가상환경 생성 (선택)
python -m venv .venv
.venv\Scripts\activate  # Windows
source .venv/bin/activate  # Mac/Linux

# 의존성 설치
pip install -r requirements.txt
```

### **2. FastAPI 서버 실행**

```bash
cd chatbot
python fastapi_app/main.py
```

서버가 `http://localhost:8000` 에서 실행됩니다.

### **3. 채팅 데모 실행**

```powershell
# HTML 서버 시작 (새 터미널)
python -m http.server 8080

# 브라우저로 접속
# http://localhost:8080/simple_chat_demo.html
```

---

## 🧪 API 사용법

### **감정 분석**

```python
import requests

response = requests.post('http://localhost:8000/emotion/analyze', 
    json={'text': '오늘 너무 행복해요!'})

result = response.json()
print(f"감정: {result['emotion']}")
print(f"신뢰도: {result['confidence']}")
```

### **채팅 메시지**

```python
response = requests.post('http://localhost:8000/chat/message', 
    json={
        'message': '걱정이 너무 많아요',
        'session_id': 'user-123'
    })

result = response.json()
print(f"응답: {result['response']}")
print(f"감정: {result['emotion']}")
print(f"제안: {result['suggestions']}")
```

### **API 문서**

- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

---

## 🎓 모델 학습

### **Google Colab에서 학습 (권장 ⭐)**

1. `colab_training.ipynb` 를 Google Colab에 업로드
2. **런타임** → **런타임 유형 변경** → **T4 GPU** 선택
3. `data/processed/emotion_corpus_merged.csv` (131K samples) 업로드
4. 셀 순서대로 실행 (90-120분 소요)

**학습 설정:**
- 모델: `ko-bert`
- 데이터 분할: 80/20 (104K train / 26K test)
- 클래스 가중치: [3.01, 1.50, 1.18, 1.14, 0.48]
- Epochs: 10
- Batch Size: 16


### **로컬에서 학습 (CPU/GPU)**

```bash
cd training
python train_krbert_hf.py \
  --data_path ../data/processed/emotion_corpus_merged.csv \
  --epochs 10 \
  --batch_size 16
```

### **학습된 모델 통합**

1. Colab에서 모델 다운로드 (`best_model_*.zip`)
2. `checkpoints_kfold/` 에 압축 해제
3. `fastapi_app/routers/emotion.py` 수정:
   ```python
   from ..models.emotion_model_hf import EmotionClassifierHF
   ```
4. 서버 재시작

자세한 가이드: [docs/MODEL_INTEGRATION_GUIDE.md](docs/MODEL_INTEGRATION_GUIDE.md)

---

## 🐳 Docker 배포

```bash
# 빌드
docker-compose build

# 실행
docker-compose up -d

# 학습용 (GPU 필요)
docker-compose -f docker-compose.training.yml up
```

---

## 📁 주요 파일 설명

| 파일/폴더 | 설명 |
|-----------|------|
| `fastapi_app/main.py` | FastAPI 메인 애플리케이션 |
| `fastapi_app/models/emotion_model.py` | KLUE-BERT 감정 분류 모델 (기본) |
| `fastapi_app/models/emotion_model_hf.py` | KR-BERT 감정 분류 모델 (학습 후) |
| `simple_chat_demo.html` | Live2D 채팅 데모 페이지 |
| `colab_training.ipynb` | Google Colab 학습 노트북 (단일 분할) |
| `training/train_krbert_hf.py` | 로컬 학습 스크립트 |
| `test_model_integration.py` | 모델 통합 테스트 |

---

## 📚 문서

- [Colab 학습 가이드](docs/COLAB_GUIDE.md)
- [모델 통합 가이드](docs/MODEL_INTEGRATION_GUIDE.md)
- [Colab 다운로드 가이드](docs/COLAB_DOWNLOAD_GUIDE.md)
- [Docker GPU 가이드](docs/DOCKER_GPU_GUIDE.md)
- [배포 가이드](docs/DEPLOYMENT.md)

---

## 🛠️ 기술 스택

### **Backend**
- FastAPI 0.104
- Python 3.11
- PyTorch 2.0+

### **Frontend**
- HTML5 + JavaScript (simple_chat_demo.html)
- Live2D SDK

### **AI/ML**
- KR-BERT (snunlp/KR-Medium)
- KLUE-BERT (klue/bert-base)
- scikit-learn
- Hugging Face Datasets
- Transformers 4.35+
---

## 🧪 테스트 예시

### **기쁨** 😊
- "오늘 정말 행복한 일이 생겼어요!"
- "시험에 합격해서 너무 기뻐요!"

### **슬픔** 😢
- "너무 힘들어서 눈물이 나요"
- "요즘 우울해서 잠을 못 자요"

### **불안** 😰
- "시험이 다가와서 너무 불안해요"
- "계속 걱정이 되어서 잠이 안 와요"

### **분노** 😠
- "정말 화가 나서 참을 수 없어요"
- "그 사람이 나를 무시해서 짜증나요"

### **중립** 😐
- "그냥 평범한 하루였어요"
- "특별한 일은 없었어요"

---



##  다음 할 일

- [ ] Colab에서 KR-BERT 학습 (90-120분)
- [ ] 학습된 모델 다운로드
- [ ] 로컬 챗봇에 통합
- [ ] 성능 비교 (기존 vs 새 모델)
- [ ] 프로덕션 배포


