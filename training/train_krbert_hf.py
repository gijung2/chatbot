"""
KR-BERT 기반 감정 분류 학습 (Hugging Face Trainer 사용)
제공된 코드 기반으로 현재 프로젝트에 최적화

통합 데이터셋 사용 (131K samples):
- 기존 감성대화말뭉치 (41K)
- 한국어_단발성_대화_데이터셋 (38K)
- 한국어_연속적_대화_데이터셋 (55K)

사용법:
    # 통합 데이터 (권장)
    python training/train_krbert_hf.py --data_path data/processed/emotion_corpus_merged.csv --epochs 12 --batch_size 64 --k_folds 2
    
    # 기존 데이터만
    python training/train_krbert_hf.py --data_path data/processed/emotion_corpus_full.csv --epochs 12 --batch_size 64
"""
import argparse
import os
import numpy as np
import pandas as pd
import torch
from datetime import datetime

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from sklearn.preprocessing import LabelEncoder

from torch.utils.data import Dataset

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AutoConfig,
    BertForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback
)

import torch.nn as nn
import logging
import json
import gc

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# 클래스 가중치 설정 (불균형 데이터 보정)
# 가중치 = 전체 샘플 수 / (클래스 수 * 각 클래스 샘플 수)
# [joy, sad, anxiety, anger, neutral]
CLASS_WEIGHTS = torch.tensor([3.01, 1.50, 1.18, 1.14, 0.48], dtype=torch.float32)


class WeightedLossBert(BertForSequenceClassification):
    """클래스 가중치를 적용한 CrossEntropyLoss를 사용하는 BERT 모델"""
    def __init__(self, config):
        super().__init__(config)
        
        # Loss 함수 정의 시 class_weights 적용
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.loss_fct = nn.CrossEntropyLoss(weight=CLASS_WEIGHTS.to(device))
        logger.info(f"✅ 클래스 가중치 Loss 함수 초기화 완료 (device: {device})")
        logger.info(f"   - 가중치: {CLASS_WEIGHTS.tolist()}")

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, labels=None, **kwargs):
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        sequence_output = outputs[0]
        
        logits = self.classifier(sequence_output[:, 0, :])
        
        loss = None
        if labels is not None:
            # 정의된 가중치 Loss 함수로 Loss 계산
            loss = self.loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            
        return (loss, logits) if loss is not None else (logits,)


class EmotionDataset(Dataset):
    """Hugging Face Trainer용 데이터셋"""
    def __init__(self, encodings, labels=None):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        if self.labels is not None:
            item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.encodings["input_ids"])


def compute_metrics(p):
    """평가 메트릭 계산"""
    pred, labels = p
    pred = np.argmax(pred, axis=1)

    accuracy = accuracy_score(y_true=labels, y_pred=pred)
    recall_micro = recall_score(y_true=labels, y_pred=pred, average="micro")
    recall_macro = recall_score(y_true=labels, y_pred=pred, average="macro")
    precision_micro = precision_score(y_true=labels, y_pred=pred, average="micro")
    precision_macro = precision_score(y_true=labels, y_pred=pred, average="macro")
    f1_macro = f1_score(y_true=labels, y_pred=pred, average="macro")

    return {
        "accuracy": accuracy,
        "recall_micro": recall_micro,
        "recall_macro": recall_macro,
        "precision_micro": precision_micro,
        "precision_macro": precision_macro,
        "f1_macro": f1_macro
    }


def preprocess_data(file_path, text_column, label_column, test_size, tokenizer, max_length, seed):
    """데이터 로드 및 전처리"""
    logger.info(f"\n📂 데이터 로드: {file_path}")
    
    df = pd.read_csv(file_path)
    logger.info(f"✅ 총 샘플 수: {len(df):,}")
    
    # 텍스트와 레이블 추출
    X = list(df[text_column])
    
    # label_id가 이미 있으면 사용, 없으면 emotion에서 생성
    if label_column in df.columns:
        y = list(df[label_column])
    elif 'emotion' in df.columns:
        lbe = LabelEncoder()
        y = list(lbe.fit_transform(df['emotion']))
        logger.info(f"   - 라벨 인코딩: {dict(zip(lbe.classes_, lbe.transform(lbe.classes_)))}")
    else:
        raise ValueError(f"'{label_column}' 또는 'emotion' 컬럼을 찾을 수 없습니다.")
    
    # 클래스 분포 출력
    unique, counts = np.unique(y, return_counts=True)
    logger.info(f"\n📊 클래스 분포:")
    for label_id, count in zip(unique, counts):
        percentage = count / len(y) * 100
        logger.info(f"   - Class {label_id}: {count:,} ({percentage:.1f}%)")
    
    # Train/Val 분할
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=test_size, shuffle=True, stratify=y, random_state=seed
    )
    
    logger.info(f"\n✅ 데이터 분할 완료:")
    logger.info(f"   - Train: {len(X_train):,} samples")
    logger.info(f"   - Val: {len(X_val):,} samples")
    
    # 토크나이징
    logger.info(f"\n🔤 토크나이징 중... (max_length={max_length})")
    X_train_tokenized = tokenizer(
        X_train, padding=True, truncation=True, max_length=max_length
    )
    X_val_tokenized = tokenizer(
        X_val, padding=True, truncation=True, max_length=max_length
    )
    logger.info(f"✅ 토크나이징 완료")
    
    return X_train_tokenized, X_val_tokenized, y_train, y_val


def train_kfold(args):
    """K-Fold Cross Validation 학습"""
    logger.info("\n" + "=" * 80)
    logger.info(f"🔀 {args.k_folds}-Fold Cross Validation 시작")
    logger.info("=" * 80)
    
    # 데이터 로드
    df = pd.read_csv(args.data_path)
    X = list(df[args.text_column])
    
    if args.label_column in df.columns:
        y = np.array(df[args.label_column])
    elif 'emotion' in df.columns:
        lbe = LabelEncoder()
        y = lbe.fit_transform(df['emotion'])
    else:
        raise ValueError(f"'{args.label_column}' 또는 'emotion' 컬럼을 찾을 수 없습니다.")
    
    # 토크나이저 로드
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    logger.info(f"✅ 토크나이저 로드: {args.model_name}")
    
    # K-Fold 분할
    skf = StratifiedKFold(n_splits=args.k_folds, shuffle=True, random_state=args.seed)
    
    all_fold_results = []
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        logger.info("\n" + "=" * 80)
        logger.info(f"📊 Fold {fold_idx + 1}/{args.k_folds} 학습 시작")
        logger.info("=" * 80)
        
        # Fold 데이터 준비
        X_train = [X[i] for i in train_idx]
        X_val = [X[i] for i in val_idx]
        y_train = y[train_idx].tolist()
        y_val = y[val_idx].tolist()
        
        logger.info(f"   - Train: {len(X_train):,} samples")
        logger.info(f"   - Val: {len(X_val):,} samples")
        
        # 토크나이징
        X_train_tokenized = tokenizer(
            X_train, padding=True, truncation=True, max_length=args.max_length
        )
        X_val_tokenized = tokenizer(
            X_val, padding=True, truncation=True, max_length=args.max_length
        )
        
        # 데이터셋 생성
        train_dataset = EmotionDataset(X_train_tokenized, y_train)
        val_dataset = EmotionDataset(X_val_tokenized, y_val)
        
        # 클래스 가중치 적용 모델 로드 (각 fold마다 새로 초기화)
        logger.info(f"\n🤖 가중치 적용 모델 로드: {args.model_name}")
        config = AutoConfig.from_pretrained(args.model_name, num_labels=args.num_labels)
        model = WeightedLossBert.from_pretrained(args.model_name, config=config)
        logger.info(f"✅ 클래스 가중치 적용 모델 로드 완료")
        
        # Training Arguments (transformers 최신 버전 호환)
        output_dir = os.path.join(args.output_dir, f"fold{fold_idx+1}_{timestamp}")
        training_args = TrainingArguments(
            output_dir=output_dir,
            eval_strategy="steps",  # evaluation_strategy에서 변경
            eval_steps=args.eval_steps,
            per_device_train_batch_size=args.batch_size,
            per_device_eval_batch_size=args.batch_size,
            num_train_epochs=args.epochs,
            seed=args.seed,
            load_best_model_at_end=True,
            learning_rate=args.learning_rate,
            save_total_limit=1,
            logging_steps=100,
            save_strategy="steps",
            save_steps=args.eval_steps,
            metric_for_best_model="f1_macro",
            greater_is_better=True,
            warmup_steps=args.warmup_steps,
            weight_decay=0.01,
            fp16=False,  # CPU 모드에서는 False
        )
        
        # Trainer 생성
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=args.early_stopping_patience)]
            if args.early_stopping_patience > 0 else None,
        )
        
        # 학습
        logger.info(f"\n🚀 Fold {fold_idx + 1} 학습 시작...")
        train_result = trainer.train()
        
        # 평가
        eval_result = trainer.evaluate()
        
        logger.info(f"\n📊 Fold {fold_idx + 1} 결과:")
        logger.info(f"   - Accuracy: {eval_result['eval_accuracy']:.4f}")
        logger.info(f"   - F1 Macro: {eval_result['eval_f1_macro']:.4f}")
        logger.info(f"   - Precision Macro: {eval_result['eval_precision_macro']:.4f}")
        logger.info(f"   - Recall Macro: {eval_result['eval_recall_macro']:.4f}")
        
        # 모델 저장
        model_path = os.path.join(args.output_dir, f"fold{fold_idx+1}_best_model_{timestamp}")
        trainer.save_model(model_path)
        logger.info(f"💾 모델 저장: {model_path}")
        
        # 결과 기록
        fold_results = {
            'fold': fold_idx + 1,
            'accuracy': eval_result['eval_accuracy'],
            'f1_macro': eval_result['eval_f1_macro'],
            'precision_macro': eval_result['eval_precision_macro'],
            'recall_macro': eval_result['eval_recall_macro'],
            'loss': eval_result['eval_loss'],
            'model_path': model_path
        }
        all_fold_results.append(fold_results)
        
        # 메모리 정리
        del model, trainer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    # 전체 결과 요약
    logger.info("\n" + "=" * 80)
    logger.info(f"📊 {args.k_folds}-Fold Cross Validation 최종 결과")
    logger.info("=" * 80)
    
    avg_acc = np.mean([r['accuracy'] for r in all_fold_results])
    std_acc = np.std([r['accuracy'] for r in all_fold_results])
    avg_f1 = np.mean([r['f1_macro'] for r in all_fold_results])
    std_f1 = np.std([r['f1_macro'] for r in all_fold_results])
    
    logger.info(f"\n📈 평균 성능:")
    logger.info(f"   - Accuracy: {avg_acc:.4f} ± {std_acc:.4f}")
    logger.info(f"   - F1 Macro: {avg_f1:.4f} ± {std_f1:.4f}")
    
    # 최고 성능 fold
    best_fold_idx = np.argmax([r['f1_macro'] for r in all_fold_results])
    best_fold = all_fold_results[best_fold_idx]
    
    logger.info(f"\n🏆 최고 성능 Fold: {best_fold['fold']}")
    logger.info(f"   - Accuracy: {best_fold['accuracy']:.4f}")
    logger.info(f"   - F1 Macro: {best_fold['f1_macro']:.4f}")
    logger.info(f"   - 모델 경로: {best_fold['model_path']}")
    
    # 결과 저장
    results_summary = {
        'timestamp': timestamp,
        'k_folds': args.k_folds,
        'avg_accuracy': float(avg_acc),
        'std_accuracy': float(std_acc),
        'avg_f1_macro': float(avg_f1),
        'std_f1_macro': float(std_f1),
        'best_fold': int(best_fold['fold']),
        'fold_results': all_fold_results
    }
    
    summary_path = os.path.join(args.output_dir, f'kfold_summary_{timestamp}.json')
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(results_summary, f, indent=2, ensure_ascii=False)
    
    logger.info(f"\n💾 결과 요약 저장: {summary_path}")
    
    return all_fold_results


def train_single(args):
    """단일 Train/Val split 학습"""
    logger.info("\n" + "=" * 80)
    logger.info("🚀 단일 Train/Val 학습 시작")
    logger.info("=" * 80)
    
    # 토크나이저 로드
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    logger.info(f"✅ 토크나이저 로드: {args.model_name}")
    
    # 데이터 전처리
    X_train_tokenized, X_val_tokenized, y_train, y_val = preprocess_data(
        file_path=args.data_path,
        text_column=args.text_column,
        label_column=args.label_column,
        test_size=args.test_size,
        tokenizer=tokenizer,
        max_length=args.max_length,
        seed=args.seed
    )
    
    # 데이터셋 생성
    train_dataset = EmotionDataset(X_train_tokenized, y_train)
    val_dataset = EmotionDataset(X_val_tokenized, y_val)
    
    # 클래스 가중치 적용 모델 로드
    logger.info(f"\n🤖 가중치 적용 모델 로드: {args.model_name}")
    config = AutoConfig.from_pretrained(args.model_name, num_labels=args.num_labels)
    model = WeightedLossBert.from_pretrained(args.model_name, config=config)
    logger.info(f"✅ 클래스 가중치 적용 모델 로드 완료 (num_labels={args.num_labels})")
    
    # Training Arguments (transformers 최신 버전 호환)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(args.output_dir, f"single_{timestamp}")
    
    training_args = TrainingArguments(
        output_dir=output_dir,
        eval_strategy="steps",  # evaluation_strategy에서 변경
        eval_steps=args.eval_steps,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        seed=args.seed,
        load_best_model_at_end=True,
        learning_rate=args.learning_rate,
        save_total_limit=1,
        logging_steps=100,
        save_strategy="steps",
        save_steps=args.eval_steps,
        metric_for_best_model="f1_macro",
        greater_is_better=True,
        warmup_steps=args.warmup_steps,
        weight_decay=0.01,
        fp16=False,  # CPU 모드
    )
    
    # Trainer 생성
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=args.early_stopping_patience)]
        if args.early_stopping_patience > 0 else None,
    )
    
    # 학습
    logger.info("\n🚀 학습 시작...")
    trainer.train()
    
    # 최종 평가
    eval_result = trainer.evaluate()
    
    logger.info("\n" + "=" * 80)
    logger.info("📊 최종 평가 결과")
    logger.info("=" * 80)
    logger.info(f"   - Accuracy: {eval_result['eval_accuracy']:.4f}")
    logger.info(f"   - F1 Macro: {eval_result['eval_f1_macro']:.4f}")
    logger.info(f"   - Precision Macro: {eval_result['eval_precision_macro']:.4f}")
    logger.info(f"   - Recall Macro: {eval_result['eval_recall_macro']:.4f}")
    logger.info(f"   - Loss: {eval_result['eval_loss']:.4f}")
    
    # 모델 저장
    model_path = os.path.join(args.output_dir, f"best_model_{timestamp}")
    trainer.save_model(model_path)
    tokenizer.save_pretrained(model_path)
    logger.info(f"\n💾 모델 및 토크나이저 저장: {model_path}")
    
    # 결과 저장
    results = {
        'timestamp': timestamp,
        'model_name': args.model_name,
        'accuracy': eval_result['eval_accuracy'],
        'f1_macro': eval_result['eval_f1_macro'],
        'precision_macro': eval_result['eval_precision_macro'],
        'recall_macro': eval_result['eval_recall_macro'],
        'loss': eval_result['eval_loss'],
        'model_path': model_path,
        'hyperparameters': vars(args)
    }
    
    results_path = os.path.join(args.output_dir, f'results_{timestamp}.json')
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    logger.info(f"💾 결과 저장: {results_path}")
    logger.info("\n✅ 학습 완료!")


def parse_args():
    """커맨드 라인 인자 파싱"""
    parser = argparse.ArgumentParser(description='KR-BERT 감정 분류 학습')
    
    # 데이터
    parser.add_argument('--data_path', type=str,
                        default='data/processed/emotion_corpus_merged.csv',
                        help='데이터 경로 (기본: 통합 데이터 131K samples)')
    parser.add_argument('--text_column', type=str, default='text',
                        help='텍스트 컬럼명')
    parser.add_argument('--label_column', type=str, default='label_id',
                        help='라벨 컬럼명')
    parser.add_argument('--test_size', type=float, default=0.05,
                        help='Validation 비율 (기본: 0.05)')
    
    # K-Fold 옵션
    parser.add_argument('--k_folds', type=int, default=0,
                        help='K-Fold 수 (0이면 단일 train/val split)')
    
    # 모델
    parser.add_argument('--model_name', type=str,
                        default='snunlp/KR-Medium',
                        help='Hugging Face 모델 이름 (기본: snunlp/KR-Medium)')
    parser.add_argument('--num_labels', type=int, default=5,
                        help='감정 클래스 수 (기본: 5)')
    
    # 학습 하이퍼파라미터
    parser.add_argument('--batch_size', type=int, default=64,
                        help='배치 크기 (기본: 64)')
    parser.add_argument('--epochs', type=int, default=7,
                        help='에폭 수 (기본: 7)')
    parser.add_argument('--learning_rate', type=float, default=5e-5,
                        help='학습률 (기본: 5e-5)')
    parser.add_argument('--max_length', type=int, default=128,
                        help='최대 시퀀스 길이 (기본: 128)')
    parser.add_argument('--warmup_steps', type=int, default=0,
                        help='Warmup 스텝 수')
    parser.add_argument('--eval_steps', type=int, default=500,
                        help='평가 주기 (기본: 500)')
    parser.add_argument('--early_stopping_patience', type=int, default=3,
                        help='Early stopping 인내심 (0이면 비활성화)')
    
    # 저장
    parser.add_argument('--output_dir', type=str, default='checkpoints_krbert',
                        help='모델 저장 디렉토리')
    
    # 기타
    parser.add_argument('--seed', type=int, default=42,
                        help='랜덤 시드')
    
    return parser.parse_args()


def main():
    """메인 함수"""
    args = parse_args()
    
    # 설정 출력
    logger.info("\n" + "=" * 80)
    logger.info("⚙️ KR-BERT 감정 분류 학습 설정")
    logger.info("=" * 80)
    for arg, value in vars(args).items():
        logger.info(f"   - {arg}: {value}")
    
    # 출력 디렉토리 생성
    os.makedirs(args.output_dir, exist_ok=True)
    
    # K-Fold 또는 단일 학습
    if args.k_folds > 1:
        train_kfold(args)
    else:
        train_single(args)


if __name__ == "__main__":
    main()
