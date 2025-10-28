"""
메인 실행 스크립트
감정 분류 모델 학습 파이프라인

사용법:
    python main.py --mode train --batch_size 16 --epochs 10
    python main.py --mode evaluate --model_path checkpoints/best_model.pt
"""
import argparse
import os
import torch
from transformers import AutoTokenizer
import logging
import json
from datetime import datetime

from data_loader import load_emotion_data, create_data_loaders
from model import create_model
from train import Trainer
from visualize import plot_training_history

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_args():
    """커맨드 라인 인자 파싱"""
    parser = argparse.ArgumentParser(description='감정 분류 모델 학습')
    
    # 모드
    parser.add_argument('--mode', type=str, default='train',
                        choices=['train', 'evaluate', 'predict'],
                        help='실행 모드: train, evaluate, predict')
    
    # 데이터
    parser.add_argument('--train_data', type=str,
                        default='data/processed/train.csv',
                        help='학습 데이터 경로')
    parser.add_argument('--val_data', type=str,
                        default='data/processed/val.csv',
                        help='검증 데이터 경로')
    parser.add_argument('--test_data', type=str,
                        default='data/processed/test.csv',
                        help='테스트 데이터 경로')
    parser.add_argument('--text_column', type=str, default='text',
                        help='텍스트 컬럼명')
    parser.add_argument('--label_column', type=str, default='label',
                        help='라벨 컬럼명')
    
    # 모델
    parser.add_argument('--model_name', type=str,
                        default='klue/bert-base',
                        help='Hugging Face 모델 이름 (예: klue/bert-base, skt/kobert-base-v1)')
    parser.add_argument('--num_labels', type=int, default=5,
                        help='감정 클래스 수')
    parser.add_argument('--dropout_rate', type=float, default=0.3,
                        help='Dropout 비율')
    parser.add_argument('--freeze_bert', action='store_true',
                        help='BERT 파라미터 동결 (분류 헤드만 학습)')
    
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
    
    # 저장/로드
    parser.add_argument('--output_dir', type=str, default='checkpoints',
                        help='모델 저장 디렉토리')
    parser.add_argument('--model_path', type=str, default=None,
                        help='로드할 모델 경로 (evaluate/predict 모드용)')
    parser.add_argument('--save_history', action='store_true',
                        help='학습 히스토리 JSON으로 저장')
    
    # 기타
    parser.add_argument('--num_workers', type=int, default=0,
                        help='데이터 로더 워커 수 (Windows는 0 권장)')
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


def train_mode(args):
    """학습 모드"""
    logger.info("=" * 80)
    logger.info("🚀 감정 분류 모델 학습 시작")
    logger.info("=" * 80)
    
    # 디바이스 설정
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"🖥️ 디바이스: {device}")
    if torch.cuda.is_available():
        logger.info(f"   - GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"   - CUDA 버전: {torch.version.cuda}")
    
    # 출력 디렉토리 생성
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 1. 데이터 로드
    logger.info("\n" + "=" * 80)
    logger.info("📂 데이터 로드")
    logger.info("=" * 80)
    train_df, val_df, test_df = load_emotion_data(
        train_path=args.train_data,
        val_path=args.val_data,
        test_path=args.test_data if os.path.exists(args.test_data) else None,
        text_column=args.text_column,
        label_column=args.label_column
    )
    
    # 2. 토크나이저 로드
    logger.info("\n" + "=" * 80)
    logger.info("🔤 토크나이저 로드")
    logger.info("=" * 80)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    logger.info(f"✅ 토크나이저 로드 완료: {args.model_name}")
    
    # 3. DataLoader 생성
    logger.info("\n" + "=" * 80)
    logger.info("🔄 DataLoader 생성")
    logger.info("=" * 80)
    train_loader, val_loader, test_loader = create_data_loaders(
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        tokenizer=tokenizer,
        batch_size=args.batch_size,
        max_length=args.max_length,
        text_column=args.text_column,
        label_column=args.label_column,
        num_workers=args.num_workers
    )
    
    # 4. 모델 생성
    logger.info("\n" + "=" * 80)
    logger.info("🤖 모델 생성")
    logger.info("=" * 80)
    model = create_model(
        model_name=args.model_name,
        num_labels=args.num_labels,
        dropout_rate=args.dropout_rate,
        freeze_bert=args.freeze_bert,
        device=device
    )
    
    # 5. Trainer 생성
    logger.info("\n" + "=" * 80)
    logger.info("🏋️ Trainer 생성")
    logger.info("=" * 80)
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        max_grad_norm=args.max_grad_norm
    )
    
    # 6. 학습 실행
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = os.path.join(args.output_dir, f'best_model_{timestamp}.pt')
    
    history = trainer.train(
        num_epochs=args.epochs,
        save_path=save_path,
        early_stopping_patience=args.early_stopping_patience
    )
    
    # 7. 학습 히스토리 저장 (옵션)
    if args.save_history:
        history_path = os.path.join(args.output_dir, f'history_{timestamp}.json')
        with open(history_path, 'w', encoding='utf-8') as f:
            json.dump(history, f, indent=2, ensure_ascii=False)
        logger.info(f"💾 학습 히스토리 저장: {history_path}")
    
    # 8. 시각화
    logger.info("\n" + "=" * 80)
    logger.info("📈 학습 결과 시각화")
    logger.info("=" * 80)
    graph_path = os.path.join(args.output_dir, f'training_history_{timestamp}.png')
    plot_training_history(history, save_path=graph_path)
    
    logger.info("\n" + "=" * 80)
    logger.info("✅ 학습 완료!")
    logger.info(f"   - 모델 저장: {save_path}")
    logger.info(f"   - 그래프 저장: {graph_path}")
    logger.info("=" * 80)


def evaluate_mode(args):
    """평가 모드"""
    logger.info("=" * 80)
    logger.info("📊 모델 평가")
    logger.info("=" * 80)
    
    if not args.model_path or not os.path.exists(args.model_path):
        logger.error("❌ --model_path를 지정해주세요.")
        return
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"🖥️ 디바이스: {device}")
    
    # 데이터 로드
    _, val_df, test_df = load_emotion_data(
        train_path=args.train_data,
        val_path=args.val_data,
        test_path=args.test_data if os.path.exists(args.test_data) else None,
        text_column=args.text_column,
        label_column=args.label_column
    )
    
    # 토크나이저
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    
    # DataLoader
    _, val_loader, test_loader = create_data_loaders(
        train_df=val_df,  # 더미
        val_df=test_df if test_df is not None else val_df,
        test_df=None,
        tokenizer=tokenizer,
        batch_size=args.batch_size,
        max_length=args.max_length,
        text_column=args.text_column,
        label_column=args.label_column,
        num_workers=args.num_workers
    )
    
    # 모델 로드
    model = create_model(
        model_name=args.model_name,
        num_labels=args.num_labels,
        dropout_rate=args.dropout_rate,
        freeze_bert=False,
        device=device
    )
    
    checkpoint = torch.load(args.model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    logger.info(f"✅ 모델 로드: {args.model_path}")
    
    # Trainer로 평가
    trainer = Trainer(
        model=model,
        train_loader=val_loader,  # 더미
        val_loader=val_loader,
        device=device
    )
    
    val_loss, val_acc, val_f1, report = trainer.validate()
    
    logger.info(f"\n📊 평가 결과:")
    logger.info(f"   - Loss: {val_loss:.4f}")
    logger.info(f"   - Accuracy: {val_acc:.4f}")
    logger.info(f"   - F1 (weighted): {val_f1:.4f}")
    
    logger.info(f"\n📈 클래스별 성능:")
    for label_name in model.id2label.values():
        if label_name in report:
            metrics = report[label_name]
            logger.info(f"   - {label_name}: "
                      f"P={metrics['precision']:.3f}, "
                      f"R={metrics['recall']:.3f}, "
                      f"F1={metrics['f1-score']:.3f}, "
                      f"Support={metrics['support']}")


def main():
    """메인 함수"""
    args = parse_args()
    
    # 시드 고정
    set_seed(args.seed)
    
    # 설정 출력
    logger.info("\n" + "=" * 80)
    logger.info("⚙️ 실행 설정")
    logger.info("=" * 80)
    for arg, value in vars(args).items():
        logger.info(f"   - {arg}: {value}")
    
    # 모드별 실행
    if args.mode == 'train':
        train_mode(args)
    elif args.mode == 'evaluate':
        evaluate_mode(args)
    elif args.mode == 'predict':
        logger.info("❌ predict 모드는 아직 구현되지 않았습니다.")
    else:
        logger.error(f"❌ 알 수 없는 모드: {args.mode}")


if __name__ == '__main__':
    main()
