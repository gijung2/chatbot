# 📊 데이터 전처리 및 학습 가이드

## 🗂️ 데이터셋 구조

### 통합 데이터셋 (emotion_corpus_merged.csv) - **권장**
- **총 샘플**: 131,091개
- **출처**: 3개 데이터셋 통합
  1. 감성대화말뭉치 (AI Hub): 41,387 samples
  2. 한국어_단발성_대화_데이터셋: 38,594 samples
  3. 한국어_연속적_대화_데이터셋: 55,600 samples

### 감정 분포
| 감정 | 개수 | 비율 |
|------|------|------|
| joy (기쁨) | 9,037 | 6.9% |
| sad (슬픔) | 18,074 | 13.8% |
| anxiety (불안) | 23,090 | 17.6% |
| anger (분노) | 23,854 | 18.2% |
| neutral (중립) | 57,036 | 43.5% |

---

## 🚀 1단계: 데이터 전처리

### 기존 데이터만 사용 (41K samples)
```bash
cd data
python preprocess_emotion_corpus.py
```
출력: `processed/emotion_corpus_full.csv`

### 새 데이터 추가 + 통합 (131K samples) ⭐ **권장**
```bash
cd data
python preprocess_new_datasets.py
```
출력: `processed/emotion_corpus_merged.csv`

---

## 🎓 2단계: 모델 학습

### 옵션 A: Google Colab (GPU 무료, 2-3시간) ⭐ **가장 빠름**

1. **Colab 접속**: https://colab.research.google.com/
2. **노트북 업로드**: `colab_training.ipynb`
3. **GPU 활성화**: 런타임 > 런타임 유형 변경 > GPU
4. **데이터 업로드**: `emotion_corpus_merged.csv` (또는 `emotion_corpus_full.csv`)
5. **실행**: 런타임 > 모두 실행

**예상 시간**: 2-3시간 (GPU T4)  
**예상 성능**: Accuracy 87-92%, F1 0.86-0.91

---

### 옵션 B: 로컬 CPU 학습 (6-10시간)

#### KR-BERT 학습 (Hugging Face Trainer)
```bash
# 통합 데이터 (131K samples, 권장)
python training/train_krbert_hf.py \
    --data_path data/processed/emotion_corpus_merged.csv \
    --model_name snunlp/KR-Medium \
    --epochs 12 \
    --batch_size 32 \
    --k_folds 2

# 기존 데이터만 (41K samples)
python training/train_krbert_hf.py \
    --data_path data/processed/emotion_corpus_full.csv \
    --model_name snunlp/KR-Medium \
    --epochs 12 \
    --batch_size 32 \
    --k_folds 2
```

#### KLUE BERT 학습 (커스텀 Trainer)
```bash
python training/main_kfold.py \
    --data_path data/processed/emotion_corpus_merged.csv \
    --model_name klue/bert-base \
    --epochs 12 \
    --batch_size 16 \
    --k_folds 2
```

**예상 시간**: 
- 통합 데이터 (131K): 10-12시간
- 기존 데이터 (41K): 6-8시간

---

## 📦 3단계: 모델 저장 위치

### Colab 학습 후
```
다운로드 파일:
├── best_model_fold1_20251102_XXXXXX.zip  (모델 압축 파일)
└── kfold_summary.json                     (학습 결과 요약)
```

**압축 해제 위치**: `checkpoints_kfold/fold1_best_model_20251102_XXXXXX/`

### 로컬 학습 후
```
checkpoints_krbert/
├── fold1_best_model_20251102_XXXXXX/
│   ├── config.json
│   ├── pytorch_model.bin
│   ├── tokenizer_config.json
│   └── vocab.txt
├── fold2_best_model_20251102_XXXXXX/
└── kfold_summary_20251102_XXXXXX.json
```

---

## 🧪 4단계: 모델 테스트

### 모델 로드 및 추론
```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# 모델 로드
model_path = "checkpoints_krbert/fold1_best_model_20251102_XXXXXX"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)

# 감정 매핑
emotion_labels = ['joy', 'sad', 'anxiety', 'anger', 'neutral']

# 추론
def predict_emotion(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=128)
    
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        predicted_class = torch.argmax(logits, dim=1).item()
    
    emotion = emotion_labels[predicted_class]
    confidence = torch.softmax(logits, dim=1)[0][predicted_class].item()
    
    return emotion, confidence

# 테스트
text = "오늘 정말 기분이 좋아요!"
emotion, confidence = predict_emotion(text)
print(f"입력: {text}")
print(f"예측 감정: {emotion} (신뢰도: {confidence:.2%})")
```

---

## 📊 성능 비교

| 데이터셋 | 샘플 수 | 균형도 | 예상 Accuracy | 예상 F1 | 학습 시간 (Colab) |
|---------|---------|--------|--------------|---------|------------------|
| 기존 (emotion_corpus_full) | 41,387 | 불균형 | 85-90% | 0.83-0.88 | 1-2시간 |
| **통합 (emotion_corpus_merged)** | **131,091** | **균형 개선** | **87-92%** | **0.86-0.91** | **2-3시간** |

---

## 💡 추가 개선 방안

### 1. 더 큰 모델 사용
```bash
python training/train_krbert_hf.py \
    --model_name klue/roberta-large \
    --batch_size 16 \
    --epochs 15
```

### 2. 클래스 가중치 조정
소수 클래스(joy)에 더 높은 가중치 부여하여 성능 향상

### 3. 데이터 증강
- 역번역 (Back-translation)
- 동의어 치환 (Synonym replacement)

### 4. 앙상블 모델
- KR-BERT + KLUE BERT + RoBERTa 앙상블

---

## 🔧 문제 해결

### RTX 5070 GPU 사용 불가
**원인**: SM 12.0 (Compute Capability 12.0) 미지원  
**해결**: Google Colab 또는 CPU 학습 사용

### Out of Memory (OOM)
**해결**: 
```bash
# 배치 크기 줄이기
--batch_size 16  # 기본 32에서 16으로

# Gradient Accumulation 사용
--gradient_accumulation_steps 2
```

### 데이터 파일 없음
```bash
# 전처리 스크립트 재실행
cd data
python preprocess_new_datasets.py
```

---

## 📚 참고 자료

- **KR-BERT**: https://github.com/snunlp/KR-BERT
- **KLUE**: https://github.com/KLUE-benchmark/KLUE
- **Hugging Face Transformers**: https://huggingface.co/docs/transformers/
- **Google Colab**: https://colab.research.google.com/

---

## 📞 문의

문제가 발생하면 다음을 확인하세요:
1. Python 3.11 사용 중인지
2. 필요한 패키지 모두 설치되었는지 (`pip install -r requirements.txt`)
3. 데이터 파일이 올바른 경로에 있는지
4. GPU 메모리가 충분한지 (Colab 사용 권장)
