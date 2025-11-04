"""
감정 분류 모델 (Hugging Face Transformers 기반)
학습된 KR-BERT 모델을 직접 로드하여 사용
"""
import torch
from typing import Dict
import logging
from pathlib import Path

from transformers import AutoTokenizer, AutoModelForSequenceClassification

logger = logging.getLogger(__name__)


class EmotionClassifierHF:
    """Hugging Face Transformers 기반 감정 분류 모델"""
    
    def __init__(
        self,
        model_path: str = None,
        device: str = 'cpu',
        num_classes: int = 5
    ):
        """
        Args:
            model_path: 학습된 모델 경로 (예: checkpoints_kfold/fold1_best_model_20251104_XXXXXX)
            device: 'cpu' 또는 'cuda'
            num_classes: 감정 클래스 수 (기본: 5)
        """
        self.device = device
        self.num_classes = num_classes
        
        # 감정 라벨 (학습 시 사용한 순서와 동일하게)
        self.emotion_labels = ["joy", "sad", "anxiety", "anger", "neutral"]
        self.label_to_id = {label: i for i, label in enumerate(self.emotion_labels)}
        self.id_to_label = {i: label for i, label in enumerate(self.emotion_labels)}
        
        # 모델 및 토크나이저 로드
        self._load_model(model_path)
        
        logger.info(f"✅ 감정 분류 모델 초기화 완료")
        logger.info(f"   - 모델 경로: {model_path}")
        logger.info(f"   - Device: {device}")
        logger.info(f"   - 감정 클래스: {self.emotion_labels}")
    
    def _load_model(self, model_path: str = None):
        """모델 및 토크나이저 로드"""
        if model_path is None:
            # 기본 모델 경로 (가장 최신 학습 모델 자동 탐색)
            model_path = self._find_latest_model()
        
        if model_path is None:
            raise ValueError(
                "모델 경로를 지정하거나 checkpoints_kfold/ 폴더에 학습된 모델을 배치하세요."
            )
        
        try:
            logger.info(f"📦 모델 로드 중: {model_path}")
            
            # 토크나이저 로드
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            
            # 모델 로드
            self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
            self.model.to(self.device)
            self.model.eval()
            
            logger.info(f"✅ 모델 로드 완료")
            
        except Exception as e:
            logger.error(f"❌ 모델 로드 실패: {e}")
            raise
    
    def _find_latest_model(self) -> str:
        """최신 학습 모델 자동 탐색"""
        project_root = Path(__file__).parent.parent.parent
        checkpoints_dir = project_root / "checkpoints_kfold"
        
        if not checkpoints_dir.exists():
            return None
        
        # fold*_best_model_* 형식의 폴더 찾기
        model_dirs = list(checkpoints_dir.glob("fold*_best_model_*"))
        
        if not model_dirs:
            logger.warning(f"⚠️ {checkpoints_dir}에서 학습된 모델을 찾을 수 없습니다.")
            return None
        
        # 가장 최신 폴더 선택 (타임스탬프 기준)
        latest_model = max(model_dirs, key=lambda p: p.name)
        logger.info(f"🔍 최신 모델 발견: {latest_model.name}")
        
        return str(latest_model)
    
    def predict_emotion(
        self,
        text: str,
        max_length: int = 128
    ) -> Dict:
        """
        텍스트에서 감정 예측
        
        Args:
            text: 분석할 텍스트
            max_length: 최대 토큰 길이
            
        Returns:
            {
                'emotion': str,           # 예측된 감정 ('joy', 'sad', 'anxiety', 'anger', 'neutral')
                'confidence': float,      # 신뢰도 (0-1)
                'probabilities': dict     # 각 감정별 확률
            }
        """
        self.model.eval()
        
        # 토큰화
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        
        input_ids = encoding['input_ids'].to(self.device)
        attention_mask = encoding['attention_mask'].to(self.device)
        
        # 예측
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            probabilities = torch.softmax(logits, dim=-1)
            predicted_class = torch.argmax(probabilities, dim=-1).item()
            confidence = probabilities[0][predicted_class].item()
        
        return {
            'emotion': self.emotion_labels[predicted_class],
            'confidence': float(confidence),
            'probabilities': {
                self.emotion_labels[i]: float(prob.item())
                for i, prob in enumerate(probabilities[0])
            }
        }
    
    def predict_batch(
        self,
        texts: list[str],
        max_length: int = 128
    ) -> list[Dict]:
        """배치 예측 (여러 텍스트 동시 처리)"""
        self.model.eval()
        
        # 배치 토큰화
        encoding = self.tokenizer(
            texts,
            add_special_tokens=True,
            max_length=max_length,
            padding=True,
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        
        input_ids = encoding['input_ids'].to(self.device)
        attention_mask = encoding['attention_mask'].to(self.device)
        
        # 배치 예측
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            probabilities = torch.softmax(logits, dim=-1)
            predicted_classes = torch.argmax(probabilities, dim=-1)
        
        # 결과 변환
        results = []
        for i in range(len(texts)):
            predicted_class = predicted_classes[i].item()
            confidence = probabilities[i][predicted_class].item()
            
            results.append({
                'emotion': self.emotion_labels[predicted_class],
                'confidence': float(confidence),
                'probabilities': {
                    self.emotion_labels[j]: float(probabilities[i][j].item())
                    for j in range(self.num_classes)
                }
            })
        
        return results
    
    def get_model_info(self) -> Dict:
        """모델 정보 반환"""
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        return {
            'model_type': type(self.model).__name__,
            'num_classes': self.num_classes,
            'emotion_labels': self.emotion_labels,
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'device': str(self.device),
            'tokenizer_vocab_size': self.tokenizer.vocab_size
        }
