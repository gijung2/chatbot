# 🚀 KOTE 포함 176K 데이터 재학습 가이드

## 📊 데이터셋 정보

### KOTE 증강 효과
- **기존 데이터**: 131,091 samples (감성대화말뭉치 3개 통합)
- **KOTE 추가**: +45,000 samples (Korean Online That-gul Emotions)
- **총 데이터**: 176,091 samples

### 클래스 분포 개선
| 클래스 | 기존 (131K) | KOTE 포함 (176K) | 개선 효과 |
|--------|-------------|------------------|-----------|
| joy | 6.9% | 14.94% | ⬆️ 8.04%p (소수 클래스 강화!) |
| sad | 14.9% | 11.83% | ⬇️ 3.07%p (균형 개선) |
| anxiety | 7.1% | 14.26% | ⬆️ 7.16%p (소수 클래스 강화!) |
| anger | 27.5% | 25.78% | ⬇️ 1.72%p (균형 개선) |
| neutral | 43.5% | 33.18% | ⬇️ 10.32%p (과다표현 완화!) |

**🎯 기대 효과**: 
- joy, anxiety 클래스 성능 대폭 향상 (기존 약점 보완)
- neutral 과다표현 완화로 다른 감정 예측 정확도 상승
- 전체 정확도: 88-93% → **91-96% 예상**

---

## 🔧 학습 준비

### 1. 데이터 파일 확인
```bash
# 로컬에서 파일 확인
cd c:\Users\rlarl\OneDrive\Desktop\chatbot\data\processed
dir emotion_corpus_with_kote.csv

# 파일 크기: 약 25-30 MB
# 샘플 수: 176,091개
```

### 2. Colab 노트북 업데이트 완료 ✅
다음 내용이 자동으로 업데이트되었습니다:
- **데이터 업로드**: `emotion_corpus_with_kote.csv` 우선 선택
- **클래스 가중치**: KOTE 포함 데이터 기준 재계산
  - joy: 1.34 (기존 3.01 → 감소, 데이터 증가로 균형 개선)
  - sad: 1.69
  - anxiety: 1.40 (기존 1.18 → 증가, 소수 클래스 보정)
  - anger: 0.77 (기존 1.14 → 감소, 다수 클래스 억제)
  - neutral: 0.60 (기존 0.48 → 약간 증가, 과다표현 완화)
- **데이터 분할**: 140,872 (train) + 35,219 (val)

---

## 🚀 Google Colab 학습 실행

### Step 1: Colab 노트북 열기
1. Google Colab 접속: https://colab.research.google.com/
2. `colab_training.ipynb` 업로드
   - 파일 위치: `c:\Users\rlarl\OneDrive\Desktop\chatbot\colab_training.ipynb`

### Step 2: GPU 런타임 설정
1. 메뉴: **런타임 > 런타임 유형 변경**
2. **하드웨어 가속기**: T4 GPU 선택
3. 저장

### Step 3: 데이터 업로드
셀 실행 시 파일 업로드 창이 나타남:
```
📂 KOTE 포함 통합 데이터 파일을 업로드하세요:
   - emotion_corpus_with_kote.csv (권장, 176K samples) ⭐ NEW!
```

**업로드 파일**: `c:\Users\rlarl\OneDrive\Desktop\chatbot\data\processed\emotion_corpus_with_kote.csv`

### Step 4: 순차적 셀 실행
1. **패키지 설치** (약 2분)
2. **데이터 업로드** (약 1-2분, 25-30MB)
3. **데이터 확인** (클래스 분포 체크)
4. **모델 학습** (약 60-90분)
   - Train: 140,872 samples
   - Val: 35,219 samples
   - Epochs: 10 (early stopping 적용)
   - Batch size: 64

### Step 5: 학습 모니터링
```
Epoch 1/10:
  Train Loss: 0.8234, Train Acc: 68.45%
  Val Loss: 0.6123, Val Acc: 76.82%

Epoch 2/10:
  Train Loss: 0.5234, Train Acc: 79.23%
  Val Loss: 0.4567, Val Acc: 83.45%
...
```

**예상 최종 성능**:
- Train Accuracy: 94-97%
- Val Accuracy: 91-96% (기존 88-93%보다 향상)

---

## 💾 모델 다운로드

### 학습 완료 후
노트북 마지막 셀에서 자동으로 ZIP 파일 생성:
```
📦 checkpoints_kfold_kote_176k.zip 생성 완료!
   - config.json
   - model.safetensors (406 MB)
   - tokenizer.json
   - tokenizer_config.json
   - vocab.txt
   - special_tokens_map.json
   - training_args.bin
```

### 로컬 저장 위치
```
c:\Users\rlarl\OneDrive\Desktop\chatbot\checkpoints_kfold_kote\
```

---

## 🔄 로컬 통합

### 1. 모델 파일 압축 해제
```bash
# PowerShell
cd c:\Users\rlarl\OneDrive\Desktop\chatbot
Expand-Archive -Path checkpoints_kfold_kote_176k.zip -DestinationPath checkpoints_kfold_kote
```

### 2. FastAPI 모델 경로 업데이트
`fastapi_app/models/emotion_model_hf.py` 수정:

```python
# 기존
MODEL_PATH = "checkpoints_kfold"

# 변경
MODEL_PATH = "checkpoints_kfold_kote"
```

### 3. 서버 재시작
```bash
cd fastapi_app
python main.py
```

### 4. 성능 테스트
```bash
# 테스트 API 호출
python test_api.py
```

**예상 결과 개선**:
```json
// 기존 (131K 모델)
{
  "emotion": "joy",
  "confidence": 0.23,  // 낮은 신뢰도
  "message": "오늘 정말 행복해요!"
}

// 개선 (176K KOTE 모델)
{
  "emotion": "joy",
  "confidence": 0.87,  // 높은 신뢰도!
  "message": "오늘 정말 행복해요!"
}
```

---

## 📈 성능 비교

### 예상 성능 향상

| 지표 | 기존 (131K) | KOTE 포함 (176K) | 개선 |
|------|-------------|------------------|------|
| Overall Accuracy | 88-93% | 91-96% | +3%p |
| Joy F1-Score | 0.65-0.75 | 0.82-0.90 | +17%p |
| Anxiety F1-Score | 0.70-0.80 | 0.85-0.92 | +15%p |
| Neutral F1-Score | 0.90-0.95 | 0.88-0.94 | -2%p (과다표현 완화) |
| Avg Confidence | 0.65-0.75 | 0.78-0.88 | +13%p |

### 특히 개선되는 부분
1. **joy 클래스**: 6.9% → 14.94% 데이터 증가로 recall 대폭 향상
2. **anxiety 클래스**: 7.1% → 14.26% 데이터 증가로 precision 개선
3. **neutral 과다예측 완화**: 43.5% → 33.18%로 감소, 다른 감정 예측 정확도 상승
4. **전체적인 신뢰도 향상**: 평균 confidence 0.65 → 0.82 예상

---

## 🐛 트러블슈팅

### 1. Colab GPU 부족
**증상**: "Runtime crashed" 또는 "Out of memory"

**해결책**:
```python
# CONFIG에서 batch_size 조정
CONFIG = {
    'batch_size': 32,  # 64 → 32로 감소
    ...
}
```

### 2. 업로드 시간 초과
**증상**: 파일 업로드가 10분 넘게 걸림

**해결책**:
```python
# Google Drive 마운트 사용
from google.colab import drive
drive.mount('/content/drive')

# Drive에 파일 미리 업로드 후
data_path = '/content/drive/MyDrive/emotion_corpus_with_kote.csv'
df = pd.read_csv(data_path)
```

### 3. 학습이 너무 느림
**증상**: 1 epoch에 30분 이상 소요

**확인사항**:
1. GPU 사용 확인: `!nvidia-smi`
2. T4 GPU 선택 확인: 런타임 > 런타임 유형 변경
3. Batch size 확인: 64가 적정 (32는 느림)

### 4. 모델 성능이 예상보다 낮음
**증상**: Val Accuracy < 85%

**체크리스트**:
- [ ] 클래스 가중치 적용 확인
- [ ] Early stopping patience 충분한지 (6 epoch)
- [ ] Learning rate 적절한지 (3e-5)
- [ ] 데이터 전처리 정상인지 (emotion_corpus_with_kote.csv 176K 확인)

---

## 📝 체크리스트

### 학습 전
- [ ] `emotion_corpus_with_kote.csv` 파일 확인 (176,091 samples)
- [ ] `colab_training.ipynb` 업데이트 확인
- [ ] Google Colab GPU 런타임 설정

### 학습 중
- [ ] 데이터 업로드 성공 (176K samples)
- [ ] 클래스 분포 확인 (joy 14.94%, neutral 33.18%)
- [ ] GPU 사용 확인 (`nvidia-smi`)
- [ ] 학습 진행 모니터링 (loss 감소, accuracy 증가)

### 학습 후
- [ ] Val Accuracy ≥ 90% 확인
- [ ] 모델 파일 다운로드 (`checkpoints_kfold_kote.zip`)
- [ ] 로컬 압축 해제
- [ ] FastAPI 경로 업데이트
- [ ] 서버 재시작 및 테스트

---

## 🎯 다음 단계

1. **Colab 학습 실행** (이 가이드 참고)
2. **성능 검증** (test_api.py로 다양한 입력 테스트)
3. **Unity 통합** (UNITY_INTEGRATION_GUIDE.md 참고)
4. **프로덕션 배포** (Docker, FastAPI 서버)

---

## 📞 도움이 필요하면?

- Colab 노트북 경로: `c:\Users\rlarl\OneDrive\Desktop\chatbot\colab_training.ipynb`
- 데이터 파일: `data/processed/emotion_corpus_with_kote.csv`
- Unity 가이드: `UNITY_INTEGRATION_GUIDE.md`
- FastAPI 서버: `fastapi_app/main.py`

**예상 학습 시간**: 총 90-120분
- 패키지 설치: 2분
- 데이터 업로드: 2분
- 학습: 60-90분 (176K 데이터)
- 모델 저장 및 다운로드: 3-5분

**Good luck! 🚀**
