"""
K-Fold Cross Validation 메인 실행 스크립트
감정 분류 모델 K-fold 교차검증 학습

사용법:
    python training/main_kfold.py --data_path data/processed/emotion_corpus_full.csv --k_folds 5 --epochs 10
"""
import argparse
import os
import torch
from transformers import AutoTokenizer
import logging
import json
from datetime import datetime
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score

from data_loader import EmotionDataset
from model import create_model
from train import Trainer
from visualize import plot_training_history

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_args():
    """커맨드 라인 인자 파싱"""
    parser = argparse.ArgumentParser(description='K-Fold 감정 분류 모델 학습')
    
    # 데이터
    parser.add_argument('--data_path', type=str,
                        default='data/processed/emotion_corpus_full.csv',
                        help='전체 데이터 경로')
    parser.add_argument('--text_column', type=str, default='text',
                        help='텍스트 컬럼명')
    parser.add_argument('--label_column', type=str, default='label_id',
                        help='라벨 컬럼명')
    parser.add_argument('--k_folds', type=int, default=5,
                        help='K-Fold 수 (기본: 5)')
    
    # 모델
    parser.add_argument('--model_name', type=str,
                        default='klue/bert-base',
                        help='Hugging Face 모델 이름')
    parser.add_argument('--num_labels', type=int, default=5,
                        help='감정 클래스 수')
    parser.add_argument('--dropout_rate', type=float, default=0.3,
                        help='Dropout 비율')
    parser.add_argument('--freeze_bert', action='store_true',
                        help='BERT 파라미터 동결')
    
    # 학습 하이퍼파라미터
    parser.add_argument('--batch_size', type=int, default=16,
                        help='배치 크기')
    parser.add_argument('--epochs', type=int, default=10,
                        help='에폭 수')
    parser.add_argument('--learning_rate', type=float, default=2e-5,
                        help='학습률')
    parser.add_argument('--max_length', type=int, default=128,
                        help='최대 시퀀스 길이')
    parser.add_argument('--warmup_steps', type=int, default=0,
                        help='Warmup 스텝 수')
    parser.add_argument('--max_grad_norm', type=float, default=1.0,
                        help='Gradient clipping 임계값')
    parser.add_argument('--early_stopping_patience', type=int, default=3,
                        help='Early stopping 인내심')
    
    # 저장
    parser.add_argument('--output_dir', type=str, default='checkpoints_kfold',
                        help='모델 저장 디렉토리')
    parser.add_argument('--save_all_folds', action='store_true',
                        help='모든 fold 모델 저장 (기본: 최고 fold만)')
    
    # 기타
    parser.add_argument('--num_workers', type=int, default=0,
                        help='데이터 로더 워커 수')
    parser.add_argument('--seed', type=int, default=42,
                        help='랜덤 시드')
    
    return parser.parse_args()


def set_seed(seed: int):
    """재현성을 위한 시드 고정"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    logger.info(f"🌱 시드 설정: {seed}")


def create_kfold_splits(df: pd.DataFrame, k_folds: int, label_column: str, seed: int):
    """Stratified K-Fold 분할 생성"""
    skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=seed)
    splits = []
    
    X = df.index.values
    y = df[label_column].values
    
    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        splits.append((train_idx, val_idx))
        logger.info(f"   Fold {fold_idx+1}: Train={len(train_idx):,}, Val={len(val_idx):,}")
    
    return splits


def train_single_fold(
    fold_idx: int,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    args,
    tokenizer,
    device,
    timestamp: str
):
    """단일 Fold 학습"""
    logger.info("\n" + "=" * 80)
    logger.info(f"📊 Fold {fold_idx + 1}/{args.k_folds} 학습 시작")
    logger.info("=" * 80)
    
    # DataLoader 생성
    from torch.utils.data import DataLoader
    
    train_dataset = EmotionDataset(
        texts=train_df[args.text_column].values,
        labels=train_df[args.label_column].values,
        tokenizer=tokenizer,
        max_length=args.max_length
    )
    
    val_dataset = EmotionDataset(
        texts=val_df[args.text_column].values,
        labels=val_df[args.label_column].values,
        tokenizer=tokenizer,
        max_length=args.max_length
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers
    )
    
    logger.info(f"✅ Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
    
    # 모델 생성 (각 fold마다 새로 초기화)
    model = create_model(
        model_name=args.model_name,
        num_labels=args.num_labels,
        dropout_rate=args.dropout_rate,
        freeze_bert=args.freeze_bert,
        device=device
    )
    
    # Trainer 생성
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        max_grad_norm=args.max_grad_norm
    )
    
    # 학습 실행
    save_path = os.path.join(
        args.output_dir, 
        f'fold{fold_idx+1}_model_{timestamp}.pt'
    )
    
    history = trainer.train(
        num_epochs=args.epochs,
        save_path=save_path,
        early_stopping_patience=args.early_stopping_patience
    )
    
    # 최고 성능 기록
    best_epoch = np.argmax(history['val_f1'])
    fold_results = {
        'fold': fold_idx + 1,
        'best_val_acc': history['val_acc'][best_epoch],
        'best_val_f1': history['val_f1'][best_epoch],
        'best_val_loss': history['val_loss'][best_epoch],
        'best_epoch': best_epoch + 1,
        'model_path': save_path,
        'history': history
    }
    
    logger.info(f"\n📊 Fold {fold_idx + 1} 결과:")
    logger.info(f"   - Best Epoch: {fold_results['best_epoch']}")
    logger.info(f"   - Best Val Acc: {fold_results['best_val_acc']:.4f}")
    logger.info(f"   - Best Val F1: {fold_results['best_val_f1']:.4f}")
    logger.info(f"   - Best Val Loss: {fold_results['best_val_loss']:.4f}")
    
    return fold_results


def main():
    """메인 함수"""
    args = parse_args()
    
    # 시드 고정
    set_seed(args.seed)
    
    # 설정 출력
    logger.info("\n" + "=" * 80)
    logger.info("⚙️ K-Fold Cross Validation 설정")
    logger.info("=" * 80)
    for arg, value in vars(args).items():
        logger.info(f"   - {arg}: {value}")
    
    # 디바이스 설정
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"\n🖥️ 디바이스: {device}")
    if torch.cuda.is_available():
        logger.info(f"   - GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"   - CUDA 버전: {torch.version.cuda}")
    
    # 출력 디렉토리 생성
    os.makedirs(args.output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 데이터 로드
    logger.info("\n" + "=" * 80)
    logger.info("📂 전체 데이터 로드")
    logger.info("=" * 80)
    logger.info(f"   - 경로: {args.data_path}")
    
    df = pd.read_csv(args.data_path)
    logger.info(f"✅ 데이터 로드 완료: {len(df):,} samples")
    logger.info(f"   - 컬럼: {list(df.columns)}")
    
    # 클래스 분포
    logger.info(f"\n📊 클래스 분포:")
    for label_id in sorted(df[args.label_column].unique()):
        count = (df[args.label_column] == label_id).sum()
        percentage = count / len(df) * 100
        emotion = df[df[args.label_column] == label_id]['emotion'].iloc[0] if 'emotion' in df.columns else label_id
        logger.info(f"   - {emotion} (id={label_id}): {count:,} ({percentage:.1f}%)")
    
    # K-Fold 분할 생성
    logger.info("\n" + "=" * 80)
    logger.info(f"🔀 {args.k_folds}-Fold Stratified 분할 생성")
    logger.info("=" * 80)
    splits = create_kfold_splits(
        df=df,
        k_folds=args.k_folds,
        label_column=args.label_column,
        seed=args.seed
    )
    
    # 토크나이저 로드
    logger.info("\n" + "=" * 80)
    logger.info("🔤 토크나이저 로드")
    logger.info("=" * 80)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    logger.info(f"✅ 토크나이저 로드 완료: {args.model_name}")
    
    # K-Fold 학습 실행
    all_fold_results = []
    
    for fold_idx, (train_idx, val_idx) in enumerate(splits):
        train_df = df.iloc[train_idx].reset_index(drop=True)
        val_df = df.iloc[val_idx].reset_index(drop=True)
        
        fold_results = train_single_fold(
            fold_idx=fold_idx,
            train_df=train_df,
            val_df=val_df,
            args=args,
            tokenizer=tokenizer,
            device=device,
            timestamp=timestamp
        )
        
        all_fold_results.append(fold_results)
    
    # 전체 결과 요약
    logger.info("\n" + "=" * 80)
    logger.info(f"📊 {args.k_folds}-Fold Cross Validation 최종 결과")
    logger.info("=" * 80)
    
    avg_acc = np.mean([r['best_val_acc'] for r in all_fold_results])
    std_acc = np.std([r['best_val_acc'] for r in all_fold_results])
    avg_f1 = np.mean([r['best_val_f1'] for r in all_fold_results])
    std_f1 = np.std([r['best_val_f1'] for r in all_fold_results])
    avg_loss = np.mean([r['best_val_loss'] for r in all_fold_results])
    
    logger.info(f"\n📈 평균 성능:")
    logger.info(f"   - Accuracy: {avg_acc:.4f} ± {std_acc:.4f}")
    logger.info(f"   - F1 Score: {avg_f1:.4f} ± {std_f1:.4f}")
    logger.info(f"   - Loss: {avg_loss:.4f}")
    
    logger.info(f"\n📋 Fold별 상세 결과:")
    for result in all_fold_results:
        logger.info(f"   Fold {result['fold']}: "
                   f"Acc={result['best_val_acc']:.4f}, "
                   f"F1={result['best_val_f1']:.4f}, "
                   f"Loss={result['best_val_loss']:.4f}, "
                   f"Epoch={result['best_epoch']}")
    
    # 최고 성능 fold 찾기
    best_fold_idx = np.argmax([r['best_val_f1'] for r in all_fold_results])
    best_fold = all_fold_results[best_fold_idx]
    
    logger.info(f"\n🏆 최고 성능 Fold: {best_fold['fold']}")
    logger.info(f"   - Accuracy: {best_fold['best_val_acc']:.4f}")
    logger.info(f"   - F1 Score: {best_fold['best_val_f1']:.4f}")
    logger.info(f"   - 모델 경로: {best_fold['model_path']}")
    
    # 결과 저장
    results_summary = {
        'timestamp': timestamp,
        'k_folds': args.k_folds,
        'total_samples': len(df),
        'avg_accuracy': float(avg_acc),
        'std_accuracy': float(std_acc),
        'avg_f1': float(avg_f1),
        'std_f1': float(std_f1),
        'avg_loss': float(avg_loss),
        'best_fold': int(best_fold['fold']),
        'best_fold_acc': float(best_fold['best_val_acc']),
        'best_fold_f1': float(best_fold['best_val_f1']),
        'fold_results': [
            {
                'fold': r['fold'],
                'best_val_acc': float(r['best_val_acc']),
                'best_val_f1': float(r['best_val_f1']),
                'best_val_loss': float(r['best_val_loss']),
                'best_epoch': r['best_epoch'],
                'model_path': r['model_path']
            }
            for r in all_fold_results
        ]
    }
    
    summary_path = os.path.join(args.output_dir, f'kfold_summary_{timestamp}.json')
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(results_summary, f, indent=2, ensure_ascii=False)
    
    logger.info(f"\n💾 결과 요약 저장: {summary_path}")
    
    # 최고 성능 fold의 학습 곡선 시각화
    graph_path = os.path.join(args.output_dir, f'best_fold_history_{timestamp}.png')
    plot_training_history(best_fold['history'], save_path=graph_path)
    logger.info(f"📈 최고 성능 Fold 그래프 저장: {graph_path}")
    
    # 최고 성능 fold가 아닌 모델 삭제 (옵션)
    if not args.save_all_folds:
        logger.info(f"\n🗑️ 최고 성능 Fold 외 모델 삭제 중...")
        for result in all_fold_results:
            if result['fold'] != best_fold['fold']:
                if os.path.exists(result['model_path']):
                    os.remove(result['model_path'])
                    logger.info(f"   - 삭제: {result['model_path']}")
    
    logger.info("\n" + "=" * 80)
    logger.info("✅ K-Fold Cross Validation 완료!")
    logger.info("=" * 80)


if __name__ == '__main__':
    main()
