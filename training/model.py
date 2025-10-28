"""
모델 정의 모듈
KLUE/KoBERT 기반 감정 분류 모델
"""
import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EmotionClassifier(nn.Module):
    """
    KLUE/KoBERT 기반 감정 분류 모델
    """
    
    def __init__(
        self,
        model_name: str = 'klue/bert-base',
        num_labels: int = 5,
        dropout_rate: float = 0.3,
        freeze_bert: bool = False
    ):
        """
        Args:
            model_name: Hugging Face 모델 이름
            num_labels: 감정 클래스 수
            dropout_rate: Dropout 비율
            freeze_bert: BERT 파라미터 동결 여부
        """
        super(EmotionClassifier, self).__init__()
        
        self.model_name = model_name
        self.num_labels = num_labels
        
        # BERT 모델 로드
        logger.info(f"🤖 모델 로드: {model_name}")
        self.bert = AutoModel.from_pretrained(model_name)
        self.config = AutoConfig.from_pretrained(model_name)
        
        # BERT 파라미터 동결 (옵션)
        if freeze_bert:
            logger.info("❄️ BERT 파라미터 동결")
            for param in self.bert.parameters():
                param.requires_grad = False
        
        # 분류 헤드
        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Linear(self.config.hidden_size, num_labels)
        
        # 감정 라벨 매핑 (기본값)
        self.id2label = {
            0: "joy",
            1: "sad",
            2: "anxiety",
            3: "anger",
            4: "neutral"
        }
        self.label2id = {v: k for k, v in self.id2label.items()}
        
        logger.info(f"✅ 모델 초기화 완료")
        logger.info(f"   - BERT hidden size: {self.config.hidden_size}")
        logger.info(f"   - 클래스 수: {num_labels}")
        logger.info(f"   - Dropout: {dropout_rate}")
    
    def forward(self, input_ids, attention_mask, labels=None):
        """
        Forward pass
        
        Args:
            input_ids: [batch_size, seq_len]
            attention_mask: [batch_size, seq_len]
            labels: [batch_size] (옵션)
        
        Returns:
            loss (라벨이 있는 경우), logits, hidden_states
        """
        # BERT 인코딩
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )
        
        # [CLS] 토큰의 hidden state 사용
        pooled_output = outputs.pooler_output  # [batch_size, hidden_size]
        
        # Dropout + 분류
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)  # [batch_size, num_labels]
        
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits, labels)
        
        return {
            'loss': loss,
            'logits': logits,
            'hidden_states': outputs.last_hidden_state
        }
    
    def predict(self, input_ids, attention_mask):
        """
        예측 (평가 모드)
        
        Returns:
            predicted_labels, probabilities
        """
        self.eval()
        with torch.no_grad():
            outputs = self.forward(input_ids, attention_mask)
            logits = outputs['logits']
            probabilities = torch.softmax(logits, dim=-1)
            predicted_labels = torch.argmax(probabilities, dim=-1)
        
        return predicted_labels, probabilities
    
    def get_trainable_parameters(self):
        """학습 가능한 파라미터 수 반환"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def get_total_parameters(self):
        """전체 파라미터 수 반환"""
        return sum(p.numel() for p in self.parameters())


def create_model(
    model_name: str = 'klue/bert-base',
    num_labels: int = 5,
    dropout_rate: float = 0.3,
    freeze_bert: bool = False,
    device: str = 'cuda'
) -> EmotionClassifier:
    """
    모델 생성 및 디바이스 이동
    
    Args:
        model_name: Hugging Face 모델 이름
        num_labels: 감정 클래스 수
        dropout_rate: Dropout 비율
        freeze_bert: BERT 파라미터 동결 여부
        device: 'cuda' 또는 'cpu'
    
    Returns:
        모델 객체
    """
    model = EmotionClassifier(
        model_name=model_name,
        num_labels=num_labels,
        dropout_rate=dropout_rate,
        freeze_bert=freeze_bert
    )
    
    # 디바이스 이동
    model = model.to(device)
    
    # 파라미터 정보 출력
    total_params = model.get_total_parameters()
    trainable_params = model.get_trainable_parameters()
    
    logger.info(f"🔢 전체 파라미터: {total_params:,}")
    logger.info(f"🔢 학습 가능 파라미터: {trainable_params:,} ({trainable_params/total_params*100:.2f}%)")
    logger.info(f"🖥️ 디바이스: {device}")
    
    return model
