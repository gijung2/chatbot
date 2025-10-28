"""
학습 모듈
모델 학습 및 검증 로직
"""
import torch
import torch.nn as nn
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from tqdm import tqdm
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, classification_report
import logging
from typing import Dict, List, Tuple
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Trainer:
    """모델 학습 및 검증"""
    
    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        device,
        learning_rate: float = 2e-5,
        warmup_steps: int = 0,
        max_grad_norm: float = 1.0
    ):
        """
        Args:
            model: 학습할 모델
            train_loader: 학습 데이터 로더
            val_loader: 검증 데이터 로더
            device: 'cuda' 또는 'cpu'
            learning_rate: 학습률
            warmup_steps: Warmup 스텝 수
            max_grad_norm: Gradient clipping 임계값
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.learning_rate = learning_rate
        self.max_grad_norm = max_grad_norm
        
        # Optimizer
        self.optimizer = AdamW(
            model.parameters(),
            lr=learning_rate,
            eps=1e-8
        )
        
        # Scheduler
        total_steps = len(train_loader) * 10  # 임시 (에폭 수 * 배치 수)
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )
        
        # 기록
        self.train_loss_history = []
        self.val_loss_history = []
        self.val_acc_history = []
        self.val_f1_history = []
        
        logger.info(f"✅ Trainer 초기화 완료")
        logger.info(f"   - Learning rate: {learning_rate}")
        logger.info(f"   - Warmup steps: {warmup_steps}")
        logger.info(f"   - Max grad norm: {max_grad_norm}")
    
    def train_epoch(self) -> float:
        """1 에폭 학습"""
        self.model.train()
        total_loss = 0
        
        progress_bar = tqdm(self.train_loader, desc="Training")
        for batch in progress_bar:
            # 데이터를 디바이스로 이동
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)
            
            # Forward pass
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            loss = outputs['loss']
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.max_grad_norm
            )
            
            # Optimizer step
            self.optimizer.step()
            self.scheduler.step()
            
            total_loss += loss.item()
            
            # 진행 상황 업데이트
            progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        avg_loss = total_loss / len(self.train_loader)
        return avg_loss
    
    def validate(self) -> Tuple[float, float, float, Dict]:
        """검증"""
        self.model.eval()
        total_loss = 0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            progress_bar = tqdm(self.val_loader, desc="Validation")
            for batch in progress_bar:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                
                loss = outputs['loss']
                logits = outputs['logits']
                
                total_loss += loss.item()
                
                # 예측
                preds = torch.argmax(logits, dim=-1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        avg_loss = total_loss / len(self.val_loader)
        accuracy = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds, average='weighted')
        
        # 상세 리포트
        report = classification_report(
            all_labels,
            all_preds,
            target_names=[self.model.id2label[i] for i in range(self.model.num_labels)],
            output_dict=True
        )
        
        return avg_loss, accuracy, f1, report
    
    def train(
        self,
        num_epochs: int,
        save_path: str = None,
        early_stopping_patience: int = 3
    ) -> Dict:
        """
        전체 학습 프로세스
        
        Args:
            num_epochs: 에폭 수
            save_path: 모델 저장 경로 (선택)
            early_stopping_patience: Early stopping 인내심
        
        Returns:
            학습 히스토리
        """
        logger.info(f"🚀 학습 시작: {num_epochs} 에폭")
        
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(num_epochs):
            start_time = time.time()
            
            logger.info(f"\n{'='*60}")
            logger.info(f"Epoch {epoch + 1}/{num_epochs}")
            logger.info(f"{'='*60}")
            
            # 학습
            train_loss = self.train_epoch()
            self.train_loss_history.append(train_loss)
            
            # 검증
            val_loss, val_acc, val_f1, report = self.validate()
            self.val_loss_history.append(val_loss)
            self.val_acc_history.append(val_acc)
            self.val_f1_history.append(val_f1)
            
            # 결과 출력
            elapsed = time.time() - start_time
            logger.info(f"\n📊 Epoch {epoch + 1} 결과:")
            logger.info(f"   - Train Loss: {train_loss:.4f}")
            logger.info(f"   - Val Loss: {val_loss:.4f}")
            logger.info(f"   - Val Accuracy: {val_acc:.4f}")
            logger.info(f"   - Val F1 (weighted): {val_f1:.4f}")
            logger.info(f"   - Time: {elapsed:.2f}s")
            
            # 클래스별 성능
            logger.info(f"\n📈 클래스별 성능:")
            for label_name in self.model.id2label.values():
                if label_name in report:
                    metrics = report[label_name]
                    logger.info(f"   - {label_name}: "
                              f"P={metrics['precision']:.3f}, "
                              f"R={metrics['recall']:.3f}, "
                              f"F1={metrics['f1-score']:.3f}")
            
            # Best 모델 저장
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                
                if save_path:
                    self.save_model(save_path)
                    logger.info(f"💾 Best 모델 저장: {save_path}")
            else:
                patience_counter += 1
                logger.info(f"⏳ Early stopping counter: {patience_counter}/{early_stopping_patience}")
            
            # Early stopping
            if patience_counter >= early_stopping_patience:
                logger.info(f"⛔ Early stopping at epoch {epoch + 1}")
                break
        
        logger.info(f"\n{'='*60}")
        logger.info(f"✅ 학습 완료!")
        logger.info(f"   - Best Val Loss: {best_val_loss:.4f}")
        logger.info(f"{'='*60}\n")
        
        return {
            'train_loss': self.train_loss_history,
            'val_loss': self.val_loss_history,
            'val_accuracy': self.val_acc_history,
            'val_f1': self.val_f1_history
        }
    
    def save_model(self, save_path: str):
        """모델 저장"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'train_loss_history': self.train_loss_history,
            'val_loss_history': self.val_loss_history,
            'val_acc_history': self.val_acc_history,
            'val_f1_history': self.val_f1_history,
            'model_config': {
                'model_name': self.model.model_name,
                'num_labels': self.model.num_labels,
                'id2label': self.model.id2label,
                'label2id': self.model.label2id
            }
        }, save_path)
    
    @staticmethod
    def load_model(model, load_path: str, device: str = 'cuda'):
        """모델 로드"""
        checkpoint = torch.load(load_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        return model, checkpoint
