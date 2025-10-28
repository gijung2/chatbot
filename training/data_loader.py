"""
데이터 로더 모듈
CSV 파일에서 감정 데이터를 로드하고 전처리합니다.
"""
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
import torch
from typing import Tuple, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EmotionDataset(Dataset):
    """감정 분류 데이터셋"""
    
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }


def load_emotion_data(
    train_path: str,
    val_path: str,
    test_path: Optional[str] = None,
    text_column: str = 'text',
    label_column: str = 'label'
) -> Tuple[pd.DataFrame, pd.DataFrame, Optional[pd.DataFrame]]:
    """
    CSV 파일에서 감정 데이터 로드
    
    Args:
        train_path: 학습 데이터 경로
        val_path: 검증 데이터 경로
        test_path: 테스트 데이터 경로 (선택)
        text_column: 텍스트 컬럼명
        label_column: 라벨 컬럼명
    
    Returns:
        train_df, val_df, test_df (또는 None)
    """
    logger.info(f"📂 학습 데이터 로드: {train_path}")
    train_df = pd.read_csv(train_path)
    
    logger.info(f"📂 검증 데이터 로드: {val_path}")
    val_df = pd.read_csv(val_path)
    
    test_df = None
    if test_path:
        logger.info(f"📂 테스트 데이터 로드: {test_path}")
        test_df = pd.read_csv(test_path)
    
    # 데이터 정보 출력
    logger.info(f"✅ 학습 데이터: {len(train_df)}개")
    logger.info(f"✅ 검증 데이터: {len(val_df)}개")
    if test_df is not None:
        logger.info(f"✅ 테스트 데이터: {len(test_df)}개")
    
    # 라벨 분포 확인
    logger.info(f"📊 학습 데이터 라벨 분포:\n{train_df[label_column].value_counts()}")
    
    return train_df, val_df, test_df


def create_data_loaders(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: Optional[pd.DataFrame],
    tokenizer,
    batch_size: int = 16,
    max_length: int = 128,
    text_column: str = 'text',
    label_column: str = 'label',
    num_workers: int = 0
) -> Tuple[DataLoader, DataLoader, Optional[DataLoader]]:
    """
    PyTorch DataLoader 생성
    
    Args:
        train_df, val_df, test_df: 데이터프레임
        tokenizer: Hugging Face 토크나이저
        batch_size: 배치 크기
        max_length: 최대 시퀀스 길이
        text_column: 텍스트 컬럼명
        label_column: 라벨 컬럼명
        num_workers: 데이터 로더 워커 수
    
    Returns:
        train_loader, val_loader, test_loader
    """
    # 데이터셋 생성
    train_dataset = EmotionDataset(
        texts=train_df[text_column].values,
        labels=train_df[label_column].values,
        tokenizer=tokenizer,
        max_length=max_length
    )
    
    val_dataset = EmotionDataset(
        texts=val_df[text_column].values,
        labels=val_df[label_column].values,
        tokenizer=tokenizer,
        max_length=max_length
    )
    
    test_dataset = None
    if test_df is not None:
        test_dataset = EmotionDataset(
            texts=test_df[text_column].values,
            labels=test_df[label_column].values,
            tokenizer=tokenizer,
            max_length=max_length
        )
    
    # DataLoader 생성
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    test_loader = None
    if test_dataset is not None:
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True if torch.cuda.is_available() else False
        )
    
    logger.info(f"✅ DataLoader 생성 완료 (batch_size={batch_size})")
    logger.info(f"   - 학습 배치 수: {len(train_loader)}")
    logger.info(f"   - 검증 배치 수: {len(val_loader)}")
    if test_loader:
        logger.info(f"   - 테스트 배치 수: {len(test_loader)}")
    
    return train_loader, val_loader, test_loader
