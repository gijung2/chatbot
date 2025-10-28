"""
감정 분석 API 서버 (Port 5000)
현재 학습된 KLUE-BERT 모델 사용 (나중에 Colab 모델로 교체 가능)
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import torch
from transformers import AutoTokenizer, AutoModel
import uvicorn
import os
import sys
import base64
import io
import re
from PIL import Image, ImageDraw, ImageFont
from typing import Optional

# 프로젝트 루트 추가
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

app = FastAPI(title="감정 분석 API")

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 전역 변수
model = None
tokenizer = None
device = None
MODEL_PATH = "checkpoints_kfold/fold1_model_20251028_113127.pt"  # 현재 모델
emotion_labels = ['joy', 'sad', 'anxiety', 'anger', 'neutral']
emotion_kr = {
    'joy': '기쁨',
    'sad': '슬픔',
    'anxiety': '불안',
    'anger': '분노',
    'neutral': '중립'
}

# 아바타 색상 및 메시지
AVATAR_COLORS = {
    'joy': {
        'bg_start': (255, 235, 59),
        'bg_end': (255, 193, 7),
        'emoji': '😊',
        'message': '긍정적인 에너지가 느껴져요! 좋은 감정을 유지하세요 ✨'
    },
    'sad': {
        'bg_start': (100, 181, 246),
        'bg_end': (63, 81, 181),
        'emoji': '😢',
        'message': '힘든 감정이 느껴지네요. 괜찮아요, 함께 이야기해봐요 💙'
    },
    'anxiety': {
        'bg_start': (186, 104, 200),
        'bg_end': (123, 31, 162),
        'emoji': '😰',
        'message': '불안한 마음이 있으시군요. 천천히 깊게 숨을 쉬어보세요 🌸'
    },
    'anger': {
        'bg_start': (255, 138, 128),
        'bg_end': (244, 67, 54),
        'emoji': '😠',
        'message': '화가 나셨군요. 감정을 표현하는 것은 좋은 일이에요 🔥'
    },
    'neutral': {
        'bg_start': (189, 189, 189),
        'bg_end': (117, 117, 117),
        'emoji': '😐',
        'message': '평온한 상태시네요. 어떤 이야기든 편하게 나눠보세요 💬'
    }
}

# 심리 위험도 패턴
RISK_PATTERNS = {
    'critical': [
        r'죽고?\s*싶', r'사라지고?\s*싶', r'끝내고?\s*싶', r'자살',
        r'소용없', r'의미없', r'가치없'
    ],
    'high': [
        r'우울해?', r'슬프', r'힘들어?', r'절망', r'포기', r'무기력',
        r'악몽', r'플래시백', r'떠올라'
    ],
    'medium': [
        r'불안해?', r'걱정', r'두려워?', r'긴장', r'떨려',
        r'화가?\s*나', r'분노', r'짜증', r'열받아?'
    ]
}

class TextInput(BaseModel):
    text: str

class EmotionOutput(BaseModel):
    text: str
    emotion: str
    emotion_kr: str
    confidence: float
    method: str
    probabilities: dict
    risk_level: Optional[str] = 'low'
    risk_message: Optional[str] = None
    emotion_message: Optional[str] = None

class AvatarOutput(BaseModel):
    text: str
    emotion: str
    emotion_kr: str
    confidence: float
    avatar_image: str  # Base64 encoded image
    risk_level: str
    risk_message: str
    emotion_message: str
    probabilities: dict
    success: bool = True

def load_model():
    """학습된 모델 로드"""
    global model, tokenizer, device
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🔧 디바이스: {device}")
    
    # 토크나이저 로드
    tokenizer = AutoTokenizer.from_pretrained('klue/bert-base')
    print("✅ 토크나이저 로드 완료")
    
    # 모델 구조 생성
    class EmotionClassifier(torch.nn.Module):
        def __init__(self, bert_model, num_labels=5):
            super().__init__()
            self.bert = bert_model
            self.dropout = torch.nn.Dropout(0.3)
            self.classifier = torch.nn.Linear(bert_model.config.hidden_size, num_labels)
        
        def forward(self, input_ids, attention_mask):
            outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
            pooled_output = outputs.pooler_output
            pooled_output = self.dropout(pooled_output)
            logits = self.classifier(pooled_output)
            return logits
    
    # 체크포인트 확인
    if not os.path.exists(MODEL_PATH):
        print(f"⚠️ 모델 파일이 없습니다: {MODEL_PATH}")
        print("💡 Colab에서 학습 완료 후 모델을 다운로드하세요")
        return False
    
    # BERT 모델 로드
    bert_model = AutoModel.from_pretrained('klue/bert-base')
    model = EmotionClassifier(bert_model, num_labels=5)
    
    # 체크포인트 로드
    checkpoint = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    print(f"✅ 모델 로드 완료: {MODEL_PATH}")
    if 'val_acc_history' in checkpoint and len(checkpoint['val_acc_history']) > 0:
        acc = checkpoint['val_acc_history'][0]
        print(f"📊 검증 정확도: {acc:.2%}")
    
    return True

def predict_emotion(text: str) -> dict:
    """감정 예측"""
    if model is None or tokenizer is None:
        raise RuntimeError("모델이 로드되지 않았습니다")
    
    # 토크나이징
    encoding = tokenizer(
        text,
        max_length=128,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )
    
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)
    
    # 예측
    with torch.no_grad():
        logits = model(input_ids, attention_mask)
        probabilities = torch.nn.functional.softmax(logits, dim=1)
        confidence, predicted_class = torch.max(probabilities, dim=1)
    
    emotion = emotion_labels[predicted_class.item()]
    
    # 확률 딕셔너리
    probs_dict = {
        emotion_labels[i]: float(probabilities[0][i])
        for i in range(len(emotion_labels))
    }
    
    return {
        'text': text,
        'emotion': emotion,
        'emotion_kr': emotion_kr[emotion],
        'confidence': float(confidence.item()),
        'method': 'klue-bert-kfold',
        'probabilities': probs_dict
    }

def assess_risk_level(text: str, emotion: str) -> tuple:
    """심리 위험도 평가"""
    text_lower = text.lower()
    
    # Critical 패턴 체크
    for pattern in RISK_PATTERNS['critical']:
        if re.search(pattern, text_lower):
            return 'critical', '⚠️ 긴급 상황이 감지되었습니다. 즉시 전문가의 도움을 받으세요.\n자살예방상담전화: 109 (24시간)'
    
    # High 패턴 체크
    for pattern in RISK_PATTERNS['high']:
        if re.search(pattern, text_lower):
            return 'high', '💛 심각한 우울감이 느껴집니다. 전문 상담사와 이야기하는 것을 권장합니다.\n정신건강상담전화: 1577-0199'
    
    # Medium 패턴 체크
    for pattern in RISK_PATTERNS['medium']:
        if re.search(pattern, text_lower):
            return 'medium', '💙 힘든 감정을 느끼고 계시네요. 충분히 휴식하고 자신을 돌보세요.'
    
    return 'low', '💚 안정적인 상태입니다. 긍정적인 마음을 유지하세요.'

def generate_avatar_image(emotion: str) -> str:
    """감정별 아바타 이미지 생성 (Base64)"""
    try:
        width, height = 400, 400
        img = Image.new('RGB', (width, height), 'white')
        draw = ImageDraw.Draw(img)
        
        emotion_data = AVATAR_COLORS.get(emotion, AVATAR_COLORS['neutral'])
        bg_start = emotion_data['bg_start']
        bg_end = emotion_data['bg_end']
        
        # 그라데이션 배경
        for y in range(height):
            ratio = y / height
            r = int(bg_start[0] + (bg_end[0] - bg_start[0]) * ratio)
            g = int(bg_start[1] + (bg_end[1] - bg_start[1]) * ratio)
            b = int(bg_start[2] + (bg_end[2] - bg_start[2]) * ratio)
            draw.line([(0, y), (width, y)], fill=(r, g, b))
        
        # 흰색 원
        circle_radius = 120
        circle_center = (width // 2, height // 2)
        draw.ellipse(
            [circle_center[0] - circle_radius, circle_center[1] - circle_radius,
             circle_center[0] + circle_radius, circle_center[1] + circle_radius],
            fill='white', outline=(200, 200, 200), width=3
        )
        
        # 이모지/텍스트
        emoji = emotion_data['emoji']
        try:
            font = ImageFont.truetype("seguiemj.ttf", 150)
        except:
            try:
                font = ImageFont.truetype("arial.ttf", 80)
                emoji = emotion_kr[emotion]
            except:
                font = ImageFont.load_default()
                emoji = emotion_kr[emotion]
        
        bbox = draw.textbbox((0, 0), emoji, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        text_x = (width - text_width) // 2
        text_y = (height - text_height) // 2 - 20
        draw.text((text_x, text_y), emoji, fill='black', font=font)
        
        # Base64 인코딩
        buffer = io.BytesIO()
        img.save(buffer, format='PNG')
        buffer.seek(0)
        img_base64 = base64.b64encode(buffer.read()).decode('utf-8')
        
        return f"data:image/png;base64,{img_base64}"
    except Exception as e:
        print(f"⚠️ 아바타 생성 실패: {e}")
        return ""

@app.on_event("startup")
async def startup_event():
    """서버 시작 시 모델 로드"""
    print("\n" + "="*60)
    print("🚀 감정 분석 API 서버 시작")
    print("="*60)
    
    success = load_model()
    
    if not success:
        print("\n⚠️ 경고: 모델을 로드할 수 없습니다")
        print("💡 임시로 규칙 기반 분석을 사용합니다")
        print("📝 Colab에서 학습 완료 후:")
        print(f"   1. model.zip 다운로드")
        print(f"   2. 압축 해제하여 {MODEL_PATH} 위치에 저장")
        print(f"   3. 서버 재시작")
    
    print("="*60)
    print("✅ 서버 준비 완료!")
    print("📍 http://localhost:5000")
    print("📚 문서: http://localhost:5000/docs")
    print("="*60 + "\n")

@app.get("/")
async def root():
    """루트 엔드포인트"""
    return {
        "service": "감정 분석 API",
        "version": "1.0",
        "model": "KLUE-BERT K-Fold",
        "model_loaded": model is not None,
        "emotions": emotion_labels,
        "model_path": MODEL_PATH,
        "status": "ready" if model is not None else "waiting for model"
    }

@app.post("/analyze", response_model=EmotionOutput)
async def analyze(input_data: TextInput):
    """감정 분석 엔드포인트"""
    try:
        if not input_data.text.strip():
            raise HTTPException(status_code=400, detail="텍스트가 비어있습니다")
        
        # 모델이 없으면 간단한 규칙 기반 사용
        if model is None:
            result = rule_based_emotion(input_data.text)
        else:
            result = predict_emotion(input_data.text)
        
        # 위험도 평가
        risk_level, risk_message = assess_risk_level(input_data.text, result['emotion'])
        
        # 감정 메시지
        emotion_message = AVATAR_COLORS[result['emotion']]['message']
        
        return EmotionOutput(
            text=input_data.text,
            emotion=result['emotion'],
            emotion_kr=result['emotion_kr'],
            confidence=result['confidence'],
            method=result['method'],
            probabilities=result['probabilities'],
            risk_level=risk_level,
            risk_message=risk_message,
            emotion_message=emotion_message
        )
        
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=f"모델 에러: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"서버 에러: {str(e)}")

@app.post("/generate_avatar", response_model=AvatarOutput)
async def generate_avatar(input_data: TextInput):
    """감정 분석 + 아바타 생성 엔드포인트"""
    try:
        if not input_data.text.strip():
            raise HTTPException(status_code=400, detail="텍스트가 비어있습니다")
        
        # 감정 예측
        if model is None:
            result = rule_based_emotion(input_data.text)
        else:
            result = predict_emotion(input_data.text)
        
        # 위험도 평가
        risk_level, risk_message = assess_risk_level(input_data.text, result['emotion'])
        
        # 감정 메시지
        emotion_message = AVATAR_COLORS[result['emotion']]['message']
        
        # 아바타 이미지 생성
        avatar_image = generate_avatar_image(result['emotion'])
        
        return AvatarOutput(
            text=input_data.text,
            emotion=result['emotion'],
            emotion_kr=result['emotion_kr'],
            confidence=result['confidence'],
            avatar_image=avatar_image,
            risk_level=risk_level,
            risk_message=risk_message,
            emotion_message=emotion_message,
            probabilities=result['probabilities'],
            success=bool(avatar_image)
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"서버 에러: {str(e)}")

def rule_based_emotion(text: str) -> dict:
    """간단한 규칙 기반 감정 분석 (임시)"""
    text_lower = text.lower()
    
    # 키워드 기반 감정 판단
    if any(word in text_lower for word in ['기쁘', '행복', '좋아', '웃', '즐거', '최고']):
        emotion = 'joy'
    elif any(word in text_lower for word in ['슬프', '우울', '눈물', '힘들', '아프']):
        emotion = 'sad'
    elif any(word in text_lower for word in ['불안', '걱정', '두렵', '무서', '떨려']):
        emotion = 'anxiety'
    elif any(word in text_lower for word in ['화', '짜증', '분노', '싫어', '미워']):
        emotion = 'anger'
    else:
        emotion = 'neutral'
    
    return {
        'text': text,
        'emotion': emotion,
        'emotion_kr': emotion_kr[emotion],
        'confidence': 0.7,
        'method': 'rule-based (임시)',
        'probabilities': {e: (0.7 if e == emotion else 0.075) for e in emotion_labels}
    }

@app.get("/health")
async def health_check():
    """헬스 체크"""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "device": str(device) if device else "unknown"
    }

@app.post("/update_model")
async def update_model(model_path: str = MODEL_PATH):
    """모델 업데이트 (Colab 학습 완료 후 사용)"""
    global MODEL_PATH
    MODEL_PATH = model_path
    
    success = load_model()
    
    if success:
        return {"status": "success", "message": f"모델 업데이트 완료: {MODEL_PATH}"}
    else:
        raise HTTPException(status_code=500, detail="모델 로드 실패")

if __name__ == "__main__":
    print("\n💡 사용 방법:")
    print("   python emotion_api_server.py")
    print("\n📝 모델 교체 방법:")
    print("   1. Colab에서 학습 완료 후 model.zip 다운로드")
    print("   2. 압축 해제: checkpoints_kfold/ 폴더")
    print("   3. 서버 재시작")
    print()
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=5000,
        log_level="info"
    )
