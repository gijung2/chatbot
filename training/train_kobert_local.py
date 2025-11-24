"""
KoBERT 감정 분석 모델 로컬 학습 스크립트 (최적화 버전)
- GPU 자동 감지 및 사용
- K-Fold 교차 검증
- 데이터 불균형 해소
- Early Stopping
- Label Smoothing + Class Weights
"""

import os
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from typing import Dict, List, Tuple, Optional
import warnings
from datetime import datetime
warnings.filterwarnings('ignore')

# 머신러닝
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix
from imblearn.over_sampling import RandomOverSampler

# 딥러닝
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback
)
from datasets import Dataset as HFDataset

# 진행상황 표시
from tqdm import tqdm
tqdm.pandas()

# ========================================
# 설정 (Configuration)
# ========================================

# 모델 설정
MODEL_NAME = "monologg/kobert"  # KoBERT 모델

# 데이터 설정
DATA_PATH = "data/processed/emotion_corpus_with_kote.csv"  # 로컬 경로
MAX_LENGTH = 128  # 최대 토큰 길이

# 학습 설정 (최적화됨 - 정확도 향상)
LEARNING_RATE = 2e-5  # 파인튜닝용 낮은 학습률
BATCH_SIZE = 32  # RTX 3080 Ti는 32 가능 (메모리 부족 시 16으로 줄이기)
NUM_EPOCHS = 15  # 에포크 수 (5 → 15로 증가)
WEIGHT_DECAY = 0.01  # 정규화
WARMUP_RATIO = 0.1  # 워밍업 비율 (전체 스텝의 10%)
GRADIENT_ACCUMULATION_STEPS = 2  # Gradient Accumulation (효과적인 배치 크기 64)

# K-Fold 설정
N_FOLDS = 5  # K-Fold 교차 검증 폴드 수
RANDOM_SEED = 42  # 재현성을 위한 시드

# 고급 학습 기법
USE_LABEL_SMOOTHING = True  # Label Smoothing (과적합 방지)
LABEL_SMOOTHING_FACTOR = 0.1  # Label Smoothing 강도
USE_CLASS_WEIGHTS = True  # 클래스 가중치 (불균형 데이터 대응)
EARLY_STOPPING_PATIENCE = 4  # Early Stopping 인내심 (2 → 4로 증가)

# 데이터 불균형 해소 설정
USE_SAMPLING = True  # 샘플링 사용 여부
SAMPLING_STRATEGY = "balanced"  # 'balanced', 'minority', 또는 비율 지정

# 감정 클래스 매핑
LABEL2ID = {'joy': 0, 'sad': 1, 'anxiety': 2, 'anger': 3, 'neutral': 4}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}
NUM_LABELS = len(LABEL2ID)

# 출력 경로
OUTPUT_DIR = "checkpoints_kobert_local"
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("="*60)
print("[KoBERT 감정 분석 모델 로컬 학습 - 최적화 버전]")
print("="*60)
print(f"모델: {MODEL_NAME}")
print(f"최대 길이: {MAX_LENGTH}")
print(f"학습률: {LEARNING_RATE}")
print(f"배치 크기: {BATCH_SIZE} (효과적: {BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS})")
print(f"에포크: {NUM_EPOCHS}")
print(f"K-Fold: {N_FOLDS}")
print(f"Label Smoothing: {LABEL_SMOOTHING_FACTOR if USE_LABEL_SMOOTHING else 'Off'}")
print(f"클래스 가중치: {'On' if USE_CLASS_WEIGHTS else 'Off'}")
print(f"Early Stopping: {EARLY_STOPPING_PATIENCE} epochs")
print(f"출력 경로: {OUTPUT_DIR}")
print("="*60)

# GPU 확인
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\nPyTorch 버전: {torch.__version__}")
print(f"CUDA 사용 가능: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU 메모리: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB")
print("="*60)


# ========================================
# 데이터 전처리 함수
# ========================================

def clean_text(text: str) -> str:
    """텍스트 정제 함수"""
    if pd.isna(text) or text == "":
        return ""
    
    text = str(text)
    
    # HTML 태그 제거
    text = re.sub(r'<[^>]+>', '', text)
    
    # URL 제거
    text = re.sub(r'http\S+|www\.\S+', '', text)
    
    # 이메일 제거
    text = re.sub(r'\S+@\S+', '', text)
    
    # 사용자 아이디 제거
    text = re.sub(r'@\w+', '', text)
    
    # 연속된 공백 제거
    text = re.sub(r'\s+', ' ', text)
    
    # 앞뒤 공백 제거
    text = text.strip()
    
    return text


def normalize_emoticons(text: str) -> str:
    """이모티콘 정규화"""
    # 긍정 이모티콘
    text = re.sub(r'ㅋ{3,}', '[강한긍정]', text)
    text = re.sub(r'ㅎ{3,}', '[강한긍정]', text)
    text = re.sub(r'ㅋ{1,2}', '[긍정]', text)
    text = re.sub(r'ㅎ{1,2}', '[긍정]', text)
    
    # 슬픔 이모티콘
    text = re.sub(r'ㅠ{2,}', '[슬픔]', text)
    text = re.sub(r'ㅜ{2,}', '[슬픔]', text)
    
    return text


def load_and_preprocess_data(file_path: str) -> pd.DataFrame:
    """데이터 로드 및 전처리"""
    print("\n[데이터 로딩 중...]")
    df = pd.read_csv(file_path, low_memory=False)
    print(f"  원본 데이터: {len(df):,}개 샘플")
    
    # 필수 컬럼 확인
    required_cols = ['text', 'emotion', 'label_id']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"필수 컬럼 누락: {missing_cols}")
    
    # 텍스트 정제
    print("\n[텍스트 정제 중...]")
    df['text'] = df['text'].apply(clean_text)
    df['text'] = df['text'].apply(normalize_emoticons)
    
    # 빈 텍스트 제거
    before_len = len(df)
    df = df[df['text'].str.len() > 0].copy()
    after_len = len(df)
    print(f"  빈 텍스트 제거: {before_len - after_len}개 샘플 제거")
    
    # label_id 처리
    if df['label_id'].dtype == 'object':
        df['label_id'] = df['emotion'].map(LABEL2ID)
    
    # NaN 제거
    df = df.dropna(subset=['text', 'emotion', 'label_id']).copy()
    
    # label_id를 정수형으로 변환
    df['label_id'] = df['label_id'].astype(int)
    
    print(f"\n최종 데이터: {len(df):,}개 샘플")
    
    return df


def balance_dataset(df: pd.DataFrame, strategy: str = "balanced") -> pd.DataFrame:
    """데이터 불균형 해소"""
    print("\n[데이터 불균형 해소 중...]")
    
    # 현재 분포
    print("\n샘플링 전 분포:")
    before_counts = df['label_id'].value_counts().sort_index()
    for label_id, count in before_counts.items():
        emotion = ID2LABEL[label_id]
        print(f"  {emotion:10s} (ID {label_id}): {count:6,}개")
    
    min_samples = before_counts.min()
    max_samples = before_counts.max()
    mean_samples = int(before_counts.mean())
    
    print(f"\n  최소: {min_samples:,}개, 최대: {max_samples:,}개, 평균: {mean_samples:,}개")
    print(f"  불균형 비율: {max_samples/min_samples:.2f}:1")
    
    # 샘플링 전략 설정
    if strategy == "balanced":
        sampling_strategy = {}
        for label_id, count in before_counts.items():
            if count < mean_samples:
                sampling_strategy[label_id] = mean_samples
    else:
        sampling_strategy = strategy
    
    if not sampling_strategy:
        print("\n  샘플링할 클래스가 없습니다.")
        return df
    
    print(f"\n  샘플링 대상: {len(sampling_strategy)}개 클래스")
    for label_id, target_count in sampling_strategy.items():
        current_count = before_counts[label_id]
        emotion = ID2LABEL[label_id]
        print(f"    {emotion:10s} (ID {label_id}): {current_count:6,}개 -> {target_count:6,}개")
    
    # Oversampling
    oversampler = RandomOverSampler(
        sampling_strategy=sampling_strategy,
        random_state=RANDOM_SEED
    )
    
    X = df[['text']].values
    y = df['label_id'].values
    
    X_resampled, y_resampled = oversampler.fit_resample(X, y)
    
    # 데이터프레임 재구성
    df_balanced = pd.DataFrame({
        'text': X_resampled.flatten(),
        'label_id': y_resampled
    })
    
    df_balanced['emotion'] = df_balanced['label_id'].map(ID2LABEL)
    
    # 샘플링 후 분포
    print("\n샘플링 후 분포:")
    after_counts = df_balanced['label_id'].value_counts().sort_index()
    for label_id, count in after_counts.items():
        emotion = ID2LABEL[label_id]
        print(f"  {emotion:10s} (ID {label_id}): {count:6,}개")
    
    print(f"\n  총 샘플 수: {len(df_balanced):,}개 (이전: {len(df):,}개)")
    
    return df_balanced


def compute_class_weights(labels: List[int]) -> torch.Tensor:
    """클래스 가중치 계산 (불균형 데이터 대응)"""
    from sklearn.utils.class_weight import compute_class_weight
    
    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(labels),
        y=labels
    )
    
    return torch.tensor(class_weights, dtype=torch.float32)


def compute_metrics(eval_pred):
    """평가 메트릭 계산"""
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    
    accuracy = accuracy_score(labels, predictions)
    f1 = f1_score(labels, predictions, average='weighted')
    precision = precision_score(labels, predictions, average='weighted')
    recall = recall_score(labels, predictions, average='weighted')
    
    return {
        'accuracy': accuracy,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }


class WeightedLossTrainer(Trainer):
    """클래스 가중치 + Label Smoothing을 적용한 커스텀 Trainer"""
    
    def __init__(self, *args, class_weights=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights
        
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        
        # 클래스 가중치 + Label Smoothing 적용
        if self.class_weights is not None:
            loss_fct = nn.CrossEntropyLoss(
                weight=self.class_weights.to(model.device),
                label_smoothing=LABEL_SMOOTHING_FACTOR if USE_LABEL_SMOOTHING else 0.0
            )
        else:
            loss_fct = nn.CrossEntropyLoss(
                label_smoothing=LABEL_SMOOTHING_FACTOR if USE_LABEL_SMOOTHING else 0.0
            )
        
        loss = loss_fct(logits, labels)
        
        return (loss, outputs) if return_outputs else loss


# ========================================
# 메인 학습 루프
# ========================================

def main():
    # 데이터 로드 및 전처리
    df = load_and_preprocess_data(DATA_PATH)
    
    # 감정 분포 확인
    print("\n" + "="*60)
    print("[감정 분포]")
    print("="*60)
    emotion_counts = df['emotion'].value_counts()
    for emotion, count in emotion_counts.items():
        percentage = count / len(df) * 100
        print(f"{emotion:10s}: {count:6,}개 ({percentage:5.2f}%)")
    
    # 불균형 해소
    if USE_SAMPLING:
        df = balance_dataset(df, strategy=SAMPLING_STRATEGY)
    
    # 토크나이저 로드
    print("\n" + "="*60)
    print(f"[토크나이저 로딩: {MODEL_NAME}]")
    print("="*60)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    print("토크나이저 로드 완료")
    
    # 데이터 준비
    texts = df['text'].tolist()
    labels = df['label_id'].tolist()
    
    # K-Fold 분할
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_SEED)
    
    # 결과 저장
    fold_results = []
    all_predictions = []
    all_labels = []
    
    print(f"\n{'='*60}")
    print(f"[K-Fold 교차 검증 시작 ({N_FOLDS} folds)]")
    print(f"{'='*60}")
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(texts, labels), 1):
        print(f"\n{'='*60}")
        print(f"[Fold {fold}/{N_FOLDS}]")
        print(f"{'='*60}")
        
        # 학습/검증 데이터 분할
        train_texts = [texts[i] for i in train_idx]
        train_labels = [labels[i] for i in train_idx]
        val_texts = [texts[i] for i in val_idx]
        val_labels = [labels[i] for i in val_idx]
        
        print(f"\n데이터 분할:")
        print(f"  학습: {len(train_texts):,}개")
        print(f"  검증: {len(val_texts):,}개")
        
        # 클래스 가중치 계산
        class_weights = None
        if USE_CLASS_WEIGHTS:
            class_weights = compute_class_weights(train_labels)
            print(f"\n클래스 가중치:")
            for label_id, weight in enumerate(class_weights):
                emotion = ID2LABEL[label_id]
                print(f"  {emotion:10s}: {weight:.4f}")
            print(f"  (높을수록 소수 클래스, 낮을수록 다수 클래스)")
        
        # HuggingFace Dataset으로 변환
        train_dataset = HFDataset.from_dict({
            'text': train_texts,
            'label': train_labels
        })
        
        val_dataset = HFDataset.from_dict({
            'text': val_texts,
            'label': val_labels
        })
        
        # 토큰화
        def tokenize_function(examples):
            return tokenizer(
                examples['text'],
                truncation=True,
                padding='max_length',
                max_length=MAX_LENGTH
            )
        
        print("\n토크나이징 중...")
        train_dataset = train_dataset.map(tokenize_function, batched=True)
        val_dataset = val_dataset.map(tokenize_function, batched=True)
        
        # 라벨 설정
        train_dataset = train_dataset.rename_column('label', 'labels')
        val_dataset = val_dataset.rename_column('label', 'labels')
        
        train_dataset.set_format('torch', columns=['input_ids', 'attention_mask', 'labels'])
        val_dataset.set_format('torch', columns=['input_ids', 'attention_mask', 'labels'])
        
        # 모델 초기화
        print(f"\n모델 로딩: {MODEL_NAME}")
        model = AutoModelForSequenceClassification.from_pretrained(
            MODEL_NAME,
            num_labels=NUM_LABELS,
            id2label=ID2LABEL,
            label2id=LABEL2ID,
            trust_remote_code=True
        )
        
        # 학습 인자 설정
        fold_output_dir = os.path.join(OUTPUT_DIR, f'fold_{fold}')
        
        training_args = TrainingArguments(
            output_dir=fold_output_dir,
            num_train_epochs=NUM_EPOCHS,
            per_device_train_batch_size=BATCH_SIZE,
            per_device_eval_batch_size=BATCH_SIZE * 2,
            learning_rate=LEARNING_RATE,
            weight_decay=WEIGHT_DECAY,
            warmup_ratio=WARMUP_RATIO,
            gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
            logging_dir=os.path.join(fold_output_dir, 'logs'),
            logging_steps=50,
            eval_strategy='epoch',
            save_strategy='epoch',
            load_best_model_at_end=True,
            metric_for_best_model='f1',
            greater_is_better=True,
            save_total_limit=2,
            seed=RANDOM_SEED,
            fp16=torch.cuda.is_available(),
            dataloader_num_workers=4,
            report_to='none',
        )
        
        # Trainer 생성 (클래스 가중치 적용)
        if USE_CLASS_WEIGHTS and class_weights is not None:
            trainer = WeightedLossTrainer(
                model=model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=val_dataset,
                compute_metrics=compute_metrics,
                class_weights=class_weights,
                callbacks=[EarlyStoppingCallback(early_stopping_patience=EARLY_STOPPING_PATIENCE)]
            )
        else:
            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=val_dataset,
                compute_metrics=compute_metrics,
                callbacks=[EarlyStoppingCallback(early_stopping_patience=EARLY_STOPPING_PATIENCE)]
            )
        
        # 학습
        print(f"\n[학습 시작...]")
        train_result = trainer.train()
        
        # 평가
        print(f"\n[평가 중...]")
        eval_result = trainer.evaluate()
        
        # 예측
        predictions = trainer.predict(val_dataset)
        y_pred = np.argmax(predictions.predictions, axis=1)
        
        # 결과 저장
        fold_results.append({
            'fold': fold,
            'accuracy': eval_result['eval_accuracy'],
            'f1': eval_result['eval_f1'],
            'precision': eval_result['eval_precision'],
            'recall': eval_result['eval_recall']
        })
        
        all_predictions.extend(y_pred.tolist())
        all_labels.extend(val_labels)
        
        print(f"\n[Fold {fold} 완료!]")
        print(f"  Accuracy: {eval_result['eval_accuracy']:.4f}")
        print(f"  F1 Score: {eval_result['eval_f1']:.4f}")
        print(f"  Precision: {eval_result['eval_precision']:.4f}")
        print(f"  Recall: {eval_result['eval_recall']:.4f}")
        
        # 최고 성능 모델 저장
        best_model_path = os.path.join(OUTPUT_DIR, f'best_model_fold_{fold}')
        trainer.save_model(best_model_path)
        tokenizer.save_pretrained(best_model_path)
        print(f"모델 저장: {best_model_path}")
        
        # 메모리 정리
        del model, trainer, train_dataset, val_dataset
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    # 최종 결과 분석
    results_df = pd.DataFrame(fold_results)
    
    print(f"\n{'='*60}")
    print("[K-Fold 교차 검증 완료!]")
    print(f"{'='*60}")
    print("\n[K-Fold 결과 요약]")
    print("="*60)
    print(results_df.to_string(index=False))
    
    print("\n" + "="*60)
    print("[평균 성능]")
    print("="*60)
    print(f"Accuracy : {results_df['accuracy'].mean():.4f} (+/-{results_df['accuracy'].std():.4f})")
    print(f"F1 Score : {results_df['f1'].mean():.4f} (+/-{results_df['f1'].std():.4f})")
    print(f"Precision: {results_df['precision'].mean():.4f} (+/-{results_df['precision'].std():.4f})")
    print(f"Recall   : {results_df['recall'].mean():.4f} (+/-{results_df['recall'].std():.4f})")
    
    # 최고 성능 Fold 찾기
    best_fold = results_df.loc[results_df['f1'].idxmax(), 'fold']
    best_f1 = results_df.loc[results_df['f1'].idxmax(), 'f1']
    
    print(f"\n{'='*60}")
    print(f"[최고 성능 모델: Fold {int(best_fold)} (F1: {best_f1:.4f})]")
    print(f"모델 경로: {OUTPUT_DIR}/best_model_fold_{int(best_fold)}")
    print(f"{'='*60}")
    
    # 결과 저장
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = os.path.join(OUTPUT_DIR, f'training_results_{timestamp}.json')
    
    import json
    results = {
        'timestamp': timestamp,
        'model_name': MODEL_NAME,
        'num_folds': N_FOLDS,
        'num_epochs': NUM_EPOCHS,
        'batch_size': BATCH_SIZE,
        'learning_rate': LEARNING_RATE,
        'fold_results': fold_results,
        'average_metrics': {
            'accuracy': float(results_df['accuracy'].mean()),
            'f1': float(results_df['f1'].mean()),
            'precision': float(results_df['precision'].mean()),
            'recall': float(results_df['recall'].mean())
        },
        'best_fold': int(best_fold),
        'best_f1': float(best_f1),
        'best_model_path': f"{OUTPUT_DIR}/best_model_fold_{int(best_fold)}"
    }
    
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\n결과 저장: {results_file}")
    print("\n[학습 완료!]")


if __name__ == "__main__":
    main()
