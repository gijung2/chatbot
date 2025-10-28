"""
시각화 모듈
학습 결과 시각화
"""
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from typing import Dict, List
import logging

# 한글 폰트 설정
matplotlib.rc('font', family='Malgun Gothic')  # Windows
matplotlib.rcParams['axes.unicode_minus'] = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def plot_training_history(history: Dict, save_path: str = None):
    """
    학습 히스토리 시각화
    
    Args:
        history: train_loss, val_loss, val_accuracy, val_f1 포함한 딕셔너리
        save_path: 그래프 저장 경로 (선택)
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('학습 결과 시각화', fontsize=16, fontweight='bold')
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    # 1. Loss
    axes[0, 0].plot(epochs, history['train_loss'], 'b-o', label='Train Loss', linewidth=2)
    axes[0, 0].plot(epochs, history['val_loss'], 'r-o', label='Val Loss', linewidth=2)
    axes[0, 0].set_title('Loss', fontsize=14, fontweight='bold')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Accuracy
    axes[0, 1].plot(epochs, history['val_accuracy'], 'g-o', label='Val Accuracy', linewidth=2)
    axes[0, 1].set_title('Validation Accuracy', fontsize=14, fontweight='bold')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. F1 Score
    axes[1, 0].plot(epochs, history['val_f1'], 'm-o', label='Val F1 (weighted)', linewidth=2)
    axes[1, 0].set_title('Validation F1 Score', fontsize=14, fontweight='bold')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('F1 Score')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. 요약 테이블
    axes[1, 1].axis('off')
    summary_text = f"""
    📊 학습 요약
    
    • 총 에폭: {len(epochs)}
    • 최종 Train Loss: {history['train_loss'][-1]:.4f}
    • 최종 Val Loss: {history['val_loss'][-1]:.4f}
    • 최종 Val Accuracy: {history['val_accuracy'][-1]:.4f}
    • 최종 Val F1: {history['val_f1'][-1]:.4f}
    
    • Best Val Loss: {min(history['val_loss']):.4f} (Epoch {np.argmin(history['val_loss'])+1})
    • Best Val Accuracy: {max(history['val_accuracy']):.4f} (Epoch {np.argmax(history['val_accuracy'])+1})
    • Best Val F1: {max(history['val_f1']):.4f} (Epoch {np.argmax(history['val_f1'])+1})
    """
    axes[1, 1].text(0.1, 0.5, summary_text, fontsize=11, verticalalignment='center',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"💾 그래프 저장: {save_path}")
    
    plt.show()
    logger.info("📈 그래프 표시 완료")


def plot_loss_only(train_loss: List[float], val_loss: List[float], save_path: str = None):
    """
    Loss만 간단하게 시각화
    
    Args:
        train_loss: 학습 손실 리스트
        val_loss: 검증 손실 리스트
        save_path: 저장 경로
    """
    plt.figure(figsize=(10, 6))
    epochs = range(1, len(train_loss) + 1)
    
    plt.plot(epochs, train_loss, 'b-o', label='Training Loss', linewidth=2, markersize=8)
    plt.plot(epochs, val_loss, 'r-o', label='Validation Loss', linewidth=2, markersize=8)
    
    plt.title('Training and Validation Loss', fontsize=16, fontweight='bold')
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"💾 Loss 그래프 저장: {save_path}")
    
    plt.show()
