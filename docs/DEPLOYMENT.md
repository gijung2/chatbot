# 🚀 모델 배포 가이드

## 📦 학습된 모델 정보

### K-Fold Cross Validation 결과 (2024-10-28)

**학습 설정:**
- 데이터셋: 감성대화말뭉치 (41,387 samples)
- 모델: klue/bert-base
- K-Fold: 2-fold (테스트)
- Epoch: 1 (빠른 검증)
- Batch Size: 16
- Learning Rate: 2e-5

**Fold 1 성능:**
- Validation Accuracy: **59.74%**
- Validation F1 Score: **59.19%** (weighted)
- Validation Loss: **0.9398**

**클래스별 성능:**
| 감정 | Precision | Recall | F1-Score |
|------|-----------|--------|----------|
| joy | 0.478 | 0.734 | 0.579 |
| sad | 0.519 | 0.556 | 0.537 |
| anxiety | 0.632 | 0.715 | **0.671** ⭐ |
| anger | 0.700 | 0.446 | 0.545 |
| neutral | 0.616 | 0.301 | 0.405 |

**모델 파일 위치:**
```
checkpoints_kfold/fold1_model_20251028_113127.pt
```

---

## 🔧 모델 로드 및 사용 방법

### 1. 모델 로드 스크립트

```python
import torch
from transformers import AutoTokenizer
from training.model import create_model

# 디바이스 설정
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 토크나이저 로드
tokenizer = AutoTokenizer.from_pretrained('klue/bert-base')

# 모델 생성
model = create_model(
    model_name='klue/bert-base',
    num_labels=5,
    dropout_rate=0.3,
    freeze_bert=False,
    device=device
)

# 체크포인트 로드
checkpoint = torch.load('checkpoints_kfold/fold1_model_20251028_113127.pt', 
                       map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

print("✅ 모델 로드 완료!")
print(f"📊 모델 설정: {checkpoint['model_config']}")
```

### 2. 감정 예측 함수

```python
def predict_emotion(text: str, model, tokenizer, device):
    """
    입력 텍스트의 감정을 예측
    
    Args:
        text: 예측할 텍스트
        model: 학습된 모델
        tokenizer: 토크나이저
        device: 디바이스
    
    Returns:
        predicted_label: 예측된 감정 (0-4)
        probabilities: 각 클래스별 확률
        emotion_name: 감정 이름
    """
    # 텍스트 토큰화
    encoding = tokenizer(
        text,
        max_length=128,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )
    
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)
    
    # 예측
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs['logits']
        probabilities = torch.softmax(logits, dim=-1)
        predicted_label = torch.argmax(probabilities, dim=-1).item()
    
    # 감정 매핑
    emotion_map = {
        0: 'joy',      # 기쁨
        1: 'sad',      # 슬픔
        2: 'anxiety',  # 불안
        3: 'anger',    # 분노
        4: 'neutral'   # 중립
    }
    
    return predicted_label, probabilities[0].cpu().numpy(), emotion_map[predicted_label]

# 사용 예시
text = "오늘 정말 기분이 좋아!"
label, probs, emotion = predict_emotion(text, model, tokenizer, device)
print(f"텍스트: {text}")
print(f"예측 감정: {emotion} (라벨: {label})")
print(f"확률: {probs}")
```

---

## 🌐 API 서버에 통합

### FastAPI 통합 예시

```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
from transformers import AutoTokenizer
from training.model import create_model

app = FastAPI(title="감정 분류 API")

# 전역 변수로 모델 로드 (서버 시작 시 1회)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
tokenizer = AutoTokenizer.from_pretrained('klue/bert-base')

model = create_model(
    model_name='klue/bert-base',
    num_labels=5,
    device=device
)

checkpoint = torch.load('checkpoints_kfold/fold1_model_20251028_113127.pt', 
                       map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

class EmotionRequest(BaseModel):
    text: str

class EmotionResponse(BaseModel):
    text: str
    emotion: str
    label: int
    probabilities: dict

@app.post("/predict", response_model=EmotionResponse)
async def predict_emotion_api(request: EmotionRequest):
    """감정 예측 API"""
    label, probs, emotion = predict_emotion(
        request.text, model, tokenizer, device
    )
    
    return EmotionResponse(
        text=request.text,
        emotion=emotion,
        label=label,
        probabilities={
            'joy': float(probs[0]),
            'sad': float(probs[1]),
            'anxiety': float(probs[2]),
            'anger': float(probs[3]),
            'neutral': float(probs[4])
        }
    )

# 실행: uvicorn api:app --reload
```

---

## 📊 모델 성능 개선 방안

### 1. **더 많은 Epoch 학습**
현재 1 epoch만 학습되었습니다. 권장: 10-20 epochs

```powershell
python training\main_kfold.py --k_folds 5 --epochs 10 --batch_size 16
```

### 2. **하이퍼파라미터 튜닝**
- Learning Rate: 1e-5 ~ 5e-5 실험
- Batch Size: 32 (GPU 메모리 충분 시)
- Dropout: 0.1 ~ 0.5 범위 테스트

### 3. **데이터 증강**
- Back-translation
- Synonym replacement
- Random deletion/insertion

### 4. **앙상블**
5-fold 모든 모델의 예측을 평균내어 사용

---

## 🔒 모델 파일 관리

### .gitignore 설정 (이미 적용됨)
```
checkpoints/
checkpoints_kfold/
*.pt
*.pth
```

### 모델 파일 크기
- `fold1_model_20251028_113127.pt`: ~420MB

### 대용량 파일 관리 옵션

**Option 1: Git LFS (추천)**
```bash
git lfs install
git lfs track "*.pt"
git add .gitattributes
git add checkpoints_kfold/*.pt
git commit -m "Add trained models with LFS"
```

**Option 2: 외부 스토리지**
- Google Drive / Dropbox
- AWS S3 / Azure Blob Storage
- Hugging Face Model Hub

**Option 3: README에 다운로드 링크**
모델 파일을 별도로 공유하고 README에 다운로드 링크 제공

---

## 📝 배포 체크리스트

- [x] 모델 학습 완료
- [x] 모델 성능 검증
- [x] .gitignore 설정
- [x] 배포 가이드 문서화
- [ ] API 서버 통합 테스트
- [ ] 프론트엔드 연동
- [ ] 성능 모니터링 설정
- [ ] 에러 핸들링 구현

---

## 🎯 다음 단계

1. **전체 K-Fold 학습 (권장)**
   ```powershell
   python training\main_kfold.py --k_folds 5 --epochs 10
   ```

2. **API 서버에 통합**
   - `fastapi_app/models/emotion_model.py` 업데이트
   - 체크포인트 경로 설정

3. **프론트엔드 연동**
   - 감정 분석 결과를 UI에 표시
   - 실시간 아바타 표정 변화

4. **모니터링 및 로깅**
   - 예측 정확도 추적
   - 응답 시간 모니터링

---

## 📞 문의 및 지원

문제가 발생하거나 질문이 있으시면 이슈를 남겨주세요.

**업데이트 날짜:** 2024-10-28  
**버전:** v1.0.0 (Initial K-Fold Test)
