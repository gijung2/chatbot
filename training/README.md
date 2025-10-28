# 감정 분류 모델 학습 가이드

## 📋 개요
이 디렉토리는 감정 분류 모델(KLUE/KoBERT)을 VSCode에서 학습하기 위한 모듈화된 코드를 포함합니다.

## 🗂️ 파일 구조
```
training/
├── main.py                      # 메인 실행 스크립트 (CLI)
├── data_loader.py               # 데이터 로드 및 전처리
├── model.py                     # 모델 정의 (KLUE/KoBERT)
├── train.py                     # 학습 및 검증 로직
├── visualize.py                 # 학습 결과 시각화
├── requirements_training.txt    # 학습용 패키지 요구사항
└── README.md                    # 이 파일
```

## 🚀 빠른 시작

### 1. 필요 패키지 설치
```powershell
pip install -r training\requirements_training.txt
```

### 2. 기본 학습 실행
```powershell
python training\main.py --mode train --batch_size 16 --epochs 10
```

### 3. 배치 크기와 에폭 조정
```powershell
# 배치 32, 에폭 5로 학습
python training\main.py --mode train --batch_size 32 --epochs 5

# GPU 메모리가 부족하면 배치를 줄이세요
python training\main.py --mode train --batch_size 8 --epochs 10
```

## 🎛️ 주요 커맨드 라인 옵션

### 학습 모드
```powershell
python training\main.py --mode train \
    --batch_size 16 \
    --epochs 10 \
    --learning_rate 2e-5 \
    --model_name klue/bert-base \
    --output_dir checkpoints \
    --save_history
```

### 평가 모드
```powershell
python training\main.py --mode evaluate \
    --model_path checkpoints/best_model_20250228_123456.pt \
    --batch_size 32
```

## 📊 하이퍼파라미터

| 파라미터 | 기본값 | 설명 |
|---------|--------|------|
| `--batch_size` | 16 | 배치 크기 (GPU 메모리에 따라 조정) |
| `--epochs` | 10 | 학습 에폭 수 |
| `--learning_rate` | 2e-5 | 학습률 |
| `--max_length` | 128 | 최대 시퀀스 길이 |
| `--dropout_rate` | 0.3 | Dropout 비율 |
| `--early_stopping_patience` | 3 | Early stopping 인내심 |

## 💾 출력 파일

학습 완료 후 `checkpoints/` 디렉토리에 생성되는 파일:
- `best_model_YYYYMMDD_HHMMSS.pt` - 최고 성능 모델 체크포인트
- `history_YYYYMMDD_HHMMSS.json` - 학습 히스토리 (--save_history 옵션)
- `training_history_YYYYMMDD_HHMMSS.png` - 학습 결과 그래프

## 🖥️ GPU/CPU 사용

- **GPU 사용**: CUDA가 설치되어 있으면 자동으로 GPU를 사용합니다.
- **CPU만 사용**: GPU가 없어도 정상 작동하지만 느립니다.
- **GPU 확인**:
  ```powershell
  python -c "import torch; print('CUDA:', torch.cuda.is_available())"
  ```

## 📈 학습 모니터링

학습 중에는 다음 정보가 실시간으로 출력됩니다:
- Train Loss (에폭별)
- Validation Loss
- Validation Accuracy
- Validation F1 Score (weighted)
- 클래스별 Precision, Recall, F1

학습 완료 후 그래프가 자동으로 생성되고 표시됩니다.

## 🔧 문제 해결

### GPU 메모리 부족
```powershell
# 배치 크기를 줄이세요
python training\main.py --batch_size 8 --epochs 10
```

### 학습이 너무 느림
```powershell
# 시퀀스 길이를 줄이세요
python training\main.py --max_length 64 --batch_size 32
```

### KoBERT 사용
```powershell
# kobert-tokenizer 설치 후
python training\main.py --model_name skt/kobert-base-v1
```

## 📝 예제 명령어

### 빠른 테스트 (작은 에폭)
```powershell
python training\main.py --mode train --batch_size 32 --epochs 3
```

### 완전 학습 (큰 배치, 긴 에폭)
```powershell
python training\main.py --mode train --batch_size 32 --epochs 20 --early_stopping_patience 5 --save_history
```

### BERT 파라미터 동결 (분류 헤드만 학습)
```powershell
python training\main.py --mode train --freeze_bert --epochs 5
```

## 🎯 다음 단계

1. 학습 완료 후 `checkpoints/best_model_*.pt` 파일을 `kobert_psychological_api.py`에 통합
2. 평가 모드로 테스트 데이터 성능 확인
3. 프론트엔드와 연결하여 실시간 감정 분석 데모 구축

## 📚 참고

- [Hugging Face Transformers 문서](https://huggingface.co/docs/transformers)
- [KLUE 벤치마크](https://klue-benchmark.com/)
- [PyTorch 튜토리얼](https://pytorch.org/tutorials/)
