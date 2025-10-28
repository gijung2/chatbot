# 🚀 Google Colab에서 학습하기

## 📋 준비물
- Google 계정
- GitHub 저장소 (이미 푸시됨)
- 15-20시간 (여러 세션)

---

## 🎯 Step 1: Colab 노트북 열기

1. **Google Colab 접속**: https://colab.research.google.com/
2. **파일 업로드**:
   - `File` → `Upload notebook`
   - `colab_training.ipynb` 업로드

또는

3. **GitHub에서 직접 열기**:
   - https://colab.research.google.com/github/gijung2/chatbot/blob/main/colab_training.ipynb

---

## ⚙️ Step 2: GPU 활성화

1. `Runtime` → `Change runtime type`
2. **Hardware accelerator**: `T4 GPU` 선택
3. `Save` 클릭

---

## 🏃 Step 3: 순서대로 실행

### 1) 환경 설정 (5분)
```python
# GPU 확인
!nvidia-smi

# 저장소 클론
!git clone https://github.com/gijung2/chatbot.git
%cd chatbot

# 패키지 설치
!pip install -q transformers torch pandas scikit-learn matplotlib seaborn tqdm
```

### 2) 데이터 확인 (1분)
```python
import pandas as pd
df = pd.read_csv('data/processed/emotion_corpus_full.csv')
print(f"전체 데이터: {len(df):,}개")
```

### 3) 테스트 학습 - 10 Epochs (2-3시간)
```bash
!python training/main_kfold.py \
  --data_path data/processed/emotion_corpus_full.csv \
  --model_name klue/bert-base \
  --k_folds 5 \
  --epochs 10 \
  --batch_size 32 \
  --output_dir checkpoints_kfold
```

**예상 결과**: 70-75% 정확도

### 4) 본격 학습 - 100 Epochs (20-30시간)
```bash
!python training/main_kfold.py \
  --data_path data/processed/emotion_corpus_full.csv \
  --model_name klue/bert-base \
  --k_folds 5 \
  --epochs 100 \
  --batch_size 32 \
  --output_dir checkpoints_kfold
```

**목표**: 78%+ 정확도

---

## 💾 Step 4: 모델 저장 및 다운로드

### 옵션 1: Google Drive에 저장 (권장)
```python
from google.colab import drive
drive.mount('/content/drive')

# 압축
!zip -r checkpoints_kfold.zip checkpoints_kfold/

# Drive에 복사
!cp checkpoints_kfold.zip /content/drive/MyDrive/
```

### 옵션 2: 직접 다운로드
```python
from google.colab import files
files.download('checkpoints_kfold.zip')
```

---

## ⚠️ 주의사항

### Colab 무료 버전 제한
- **최대 12시간** 연속 실행
- 90분 비활성화 시 연결 끊김
- GPU 사용 시간 제한

### 해결 방법

#### 1) 세션 유지 스크립트
브라우저 콘솔(`F12`)에서 실행:
```javascript
function KeepClicking(){
  console.log("Clicking");
  document.querySelector("colab-connect-button").click();
}
setInterval(KeepClicking, 60000); // 1분마다 클릭
```

#### 2) 중간 저장 활용
- 각 Fold 완료 시 자동 저장됨
- 연결 끊겨도 체크포인트에서 재개 가능

#### 3) 여러 세션으로 나누기
- 10 Epochs씩 나눠서 학습
- 매번 저장 → Drive 업로드

#### 4) Colab Pro 고려 ($9.99/월)
- 24시간 연속 실행
- 더 좋은 GPU (V100, A100)
- 우선 순위 접근

---

## 📊 예상 학습 시간 (T4 GPU 기준)

| Epochs | 소요 시간 | 예상 정확도 | 세션 수 |
|--------|----------|------------|---------|
| 10     | 2-3시간  | 70-75%     | 1개     |
| 50     | 10-15시간| 76-78%     | 2-3개   |
| 100    | 20-30시간| **78-80%** | 3-4개   |

---

## 🎯 학습 완료 후

### 1. 로컬로 모델 가져오기
```powershell
# Drive에서 다운로드 또는
# Colab에서 직접 다운로드한 zip 파일 압축 해제
```

### 2. 모델 테스트
```powershell
cd c:\Users\rlarl\OneDrive\Desktop\chatbot
python training/load_model.py `
  --model_path checkpoints_kfold/fold1_model_최신.pt `
  --interactive
```

### 3. API 서버에 통합
```python
# backend/emotion_server.py에서 사용
model_path = "checkpoints_kfold/fold1_model_최신.pt"
```

---

## 💡 팁

1. **10 Epochs로 먼저 테스트**
   - 제대로 작동하는지 확인
   - 2-3시간이면 완료

2. **batch_size 조정**
   - GPU 메모리 부족 시: `--batch_size 16`
   - 여유 있으면: `--batch_size 64`

3. **Early Stopping 활용**
   - `--early_stopping_patience 10`
   - 성능 향상 없으면 자동 중단

4. **학습 로그 저장**
   - Colab 출력 복사
   - 또는 로그 파일로 저장

---

## 🆘 문제 해결

### GPU 할당 안됨
```
Runtime → Disconnect and delete runtime
Runtime → Change runtime type → GPU
```

### 메모리 부족
```bash
--batch_size 16  # 줄이기
--max_length 64  # 줄이기
```

### 패키지 에러
```bash
!pip install --upgrade transformers torch
```

---

## 📞 도움말

문제가 생기면:
1. Colab 노트북의 에러 메시지 확인
2. GitHub Issues에 질문
3. 학습 로그 공유

**Happy Training! 🚀**
