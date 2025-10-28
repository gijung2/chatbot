"""
Emotion Classification Model
"""
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
from typing import Dict, Tuple
import time

from app.config import settings

class EmotionClassifier(nn.Module):
    """KLUE-BERT 기반 감정 분류 모델"""
    
    def __init__(self, bert_model, num_labels: int = 5):
        super().__init__()
        self.bert = bert_model
        self.dropout = nn.Dropout(settings.DROPOUT_RATE)
        self.classifier = nn.Linear(bert_model.config.hidden_size, num_labels)
        
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        return logits

class EmotionModelService:
    """감정 분석 모델 서비스"""
    
    EMOTION_LABELS = ['joy', 'sad', 'anxiety', 'anger', 'neutral']
    EMOTION_KR = {
        'joy': '기쁨',
        'sad': '슬픔',
        'anxiety': '불안',
        'anger': '분노',
        'neutral': '중립'
    }
    
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.device = None
        self._is_loaded = False
        
    def load_model(self) -> bool:
        """모델 로드"""
        try:
            # 디바이스 설정
            if settings.DEVICE == "auto":
                self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            else:
                self.device = torch.device(settings.DEVICE)
            
            print(f"🔧 디바이스: {self.device}")
            
            # 토크나이저 로드
            self.tokenizer = AutoTokenizer.from_pretrained(settings.MODEL_NAME)
            print("✅ 토크나이저 로드 완료")
            
            # BERT 모델 로드
            bert_model = AutoModel.from_pretrained(settings.MODEL_NAME)
            self.model = EmotionClassifier(bert_model, num_labels=settings.NUM_LABELS)
            
            # 체크포인트 로드
            import os
            if not os.path.exists(settings.MODEL_PATH):
                print(f"⚠️ 모델 파일이 없습니다: {settings.MODEL_PATH}")
                return False
            
            checkpoint = torch.load(settings.MODEL_PATH, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.to(self.device)
            self.model.eval()
            
            print(f"✅ 모델 로드 완료: {settings.MODEL_PATH}")
            if 'val_acc_history' in checkpoint and len(checkpoint['val_acc_history']) > 0:
                acc = checkpoint['val_acc_history'][0]
                print(f"📊 검증 정확도: {acc:.2%}")
            
            self._is_loaded = True
            return True
            
        except Exception as e:
            print(f"❌ 모델 로드 실패: {e}")
            self._is_loaded = False
            return False
    
    def predict(self, text: str) -> Tuple[Dict, float]:
        """
        감정 예측
        
        Returns:
            (result_dict, inference_time_ms)
        """
        if not self._is_loaded:
            raise RuntimeError("모델이 로드되지 않았습니다")
        
        start_time = time.time()
        
        # 토크나이징
        encoding = self.tokenizer(
            text,
            max_length=settings.MAX_LENGTH,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        input_ids = encoding['input_ids'].to(self.device)
        attention_mask = encoding['attention_mask'].to(self.device)
        
        # 예측
        with torch.no_grad():
            logits = self.model(input_ids, attention_mask)
            probabilities = torch.nn.functional.softmax(logits, dim=1)
            confidence, predicted_class = torch.max(probabilities, dim=1)
        
        emotion = self.EMOTION_LABELS[predicted_class.item()]
        
        # 확률 딕셔너리
        probs_dict = {
            self.EMOTION_LABELS[i]: float(probabilities[0][i])
            for i in range(len(self.EMOTION_LABELS))
        }
        
        inference_time_ms = (time.time() - start_time) * 1000
        
        result = {
            'emotion': emotion,
            'emotion_kr': self.EMOTION_KR[emotion],
            'confidence': float(confidence.item()),
            'probabilities': probs_dict,
            'method': 'klue-bert-kfold'
        }
        
        return result, inference_time_ms
    
    @property
    def is_loaded(self) -> bool:
        """모델 로드 여부"""
        return self._is_loaded

# 전역 모델 인스턴스
emotion_model_service = EmotionModelService()
