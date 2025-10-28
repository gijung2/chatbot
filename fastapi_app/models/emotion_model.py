"""
감정 분류 모델
KoBERT 기반 PyTorch 모델
"""
import torch
import torch.nn as nn
from typing import Dict, Optional
import logging

logger = logging.getLogger(__name__)

# Transformers 및 KoBERT 관련 import
try:
    from transformers import BertModel, AutoTokenizer, AutoModel
    from kobert_tokenizer import KoBERTTokenizer
    KOBERT_AVAILABLE = True
except ImportError:
    KOBERT_AVAILABLE = False
    logger.warning("KoBERT 라이브러리 없음. KLUE-BERT 사용")


class EmotionClassifier(nn.Module):
    """KoBERT/KLUE-BERT 기반 감정 분류 모델"""
    
    def __init__(
        self,
        num_classes: int = 5,
        model_name: str = 'skt/kobert-base-v1',
        dropout_rate: float = 0.3,
        device: str = 'cpu'
    ):
        super(EmotionClassifier, self).__init__()
        
        self.num_classes = num_classes
        self.model_name = model_name
        self.device = device
        
        # 감정 라벨
        self.emotion_labels = ["joy", "sad", "anxiety", "anger", "neutral"]
        self.label_to_id = {label: i for i, label in enumerate(self.emotion_labels)}
        self.id_to_label = {i: label for i, label in enumerate(self.emotion_labels)}
        
        # 모델 및 토크나이저 로드
        self._load_model()
        
        # 분류 헤드
        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Linear(self.bert_hidden_size, num_classes)
        
        self.to(device)
        logger.info(f"✅ 모델 초기화 완료: {model_name}, Device: {device}")
    
    def _load_model(self):
        """BERT 모델 및 토크나이저 로드"""
        if KOBERT_AVAILABLE and 'kobert' in self.model_name.lower():
            try:
                self.tokenizer = KoBERTTokenizer.from_pretrained(self.model_name)
                self.bert = BertModel.from_pretrained(self.model_name)
                logger.info("✅ KoBERT 로드 성공")
            except Exception as e:
                logger.warning(f"KoBERT 로드 실패: {e}, KLUE-BERT 사용")
                self._load_klue_bert()
        else:
            self._load_klue_bert()
        
        self.bert_hidden_size = self.bert.config.hidden_size
    
    def _load_klue_bert(self):
        """KLUE-BERT 로드 (fallback)"""
        self.tokenizer = AutoTokenizer.from_pretrained('klue/bert-base')
        self.bert = AutoModel.from_pretrained('klue/bert-base')
        self.model_name = 'klue/bert-base'
        logger.info("✅ KLUE-BERT 로드 성공")
    
    def forward(self, input_ids, attention_mask):
        """순전파"""
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        return logits
    
    def predict_emotion(
        self,
        text: str,
        max_length: int = 128
    ) -> Dict:
        """텍스트에서 감정 예측"""
        self.eval()
        
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
        
        with torch.no_grad():
            logits = self.forward(input_ids, attention_mask)
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
    
    def load_checkpoint(self, checkpoint_path: str):
        """체크포인트 로드"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.load_state_dict(checkpoint['model_state_dict'])
        logger.info(f"✅ 체크포인트 로드: {checkpoint_path}")
    
    def get_model_info(self) -> Dict:
        """모델 정보 반환"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'model_name': self.model_name,
            'num_classes': self.num_classes,
            'emotion_labels': self.emotion_labels,
            'hidden_size': self.bert_hidden_size,
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'device': str(self.device)
        }
