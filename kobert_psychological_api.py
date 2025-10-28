"""
KoBERT 기반 고급 심리상담 아바타 API
- PyTorch + KoBERT 감정 분석
- 심리상담 전문 패턴 인식
- 고정밀 감정 예측
"""

from flask import Flask, request, jsonify, render_template_string
from flask_cors import CORS
import torch
import torch.nn as nn
import base64
import io
import json
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from datetime import datetime
import logging
import re
from typing import Dict, List, Tuple, Optional

# Transformers 및 KoBERT 관련 import
try:
    from transformers import (
        BertTokenizer, BertModel, BertConfig,
        AutoTokenizer, AutoModel
    )
    from kobert_tokenizer import KoBERTTokenizer
    KOBERT_AVAILABLE = True
    print("✅ KoBERT 토크나이저 로드 성공")
except ImportError as e:
    print(f"⚠️ KoBERT 관련 라이브러리 로드 실패: {e}")
    print("📦 다음 명령어로 설치하세요: pip install kobert-tokenizer transformers torch")
    KOBERT_AVAILABLE = False

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

class KoBERTEmotionClassifier(nn.Module):
    """KoBERT 기반 감정 분류 모델"""
    
    def __init__(self, num_classes=5, model_name='skt/kobert-base-v1', dropout_rate=0.3):
        super(KoBERTEmotionClassifier, self).__init__()
        
        self.num_classes = num_classes
        self.model_name = model_name
        
        # KoBERT 모델 로드
        if KOBERT_AVAILABLE:
            try:
                # KoBERT 토크나이저
                self.tokenizer = KoBERTTokenizer.from_pretrained(model_name)
                # BERT 모델
                self.bert = BertModel.from_pretrained(model_name)
                logger.info("✅ KoBERT 모델 로드 성공")
            except Exception as e:
                logger.warning(f"KoBERT 로드 실패: {e}, KLUE-BERT 사용")
                self.tokenizer = AutoTokenizer.from_pretrained('klue/bert-base')
                self.bert = AutoModel.from_pretrained('klue/bert-base')
        else:
            # Fallback to KLUE-BERT
            self.tokenizer = AutoTokenizer.from_pretrained('klue/bert-base')
            self.bert = AutoModel.from_pretrained('klue/bert-base')
            
        self.bert_hidden_size = self.bert.config.hidden_size
        
        # 분류 헤드
        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Linear(self.bert_hidden_size, num_classes)
        
        # 감정 라벨
        self.emotion_labels = ["joy", "sad", "anxiety", "anger", "neutral"]
        self.label_to_id = {label: i for i, label in enumerate(self.emotion_labels)}
        self.id_to_label = {i: label for i, label in enumerate(self.emotion_labels)}
    
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        return logits
    
    def predict_emotion(self, text: str, max_length: int = 128):
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
        
        input_ids = encoding['input_ids']
        attention_mask = encoding['attention_mask']
        
        with torch.no_grad():
            logits = self.forward(input_ids, attention_mask)
            probabilities = torch.softmax(logits, dim=-1)
            predicted_class = torch.argmax(probabilities, dim=-1).item()
            confidence = probabilities[0][predicted_class].item()
        
        return {
            'emotion': self.emotion_labels[predicted_class],
            'confidence': confidence,
            'probabilities': {
                self.emotion_labels[i]: prob.item() 
                for i, prob in enumerate(probabilities[0])
            }
        }

class PsychologicalPatternAnalyzer:
    """심리상담 전문 패턴 분석기"""
    
    def __init__(self):
        # 심리학적 위험 패턴 (KoBERT 보완용)
        self.risk_patterns = {
            'critical': [
                r'죽고?\s*싶', r'자살', r'사라지고?\s*싶', r'끝내고?\s*싶',
                r'소용없', r'의미없', r'가치없', r'살기?\s*싫어?'
            ],
            'high': [
                r'우울해?', r'절망', r'포기', r'한계', r'견딜?\s*수?\s*없',
                r'도움.*없', r'혼자.*감당', r'아무도.*모르'
            ],
            'medium': [
                r'불안해?', r'걱정', r'두려워?', r'무서워?', r'스트레스',
                r'화가?\s*나', r'짜증', r'분노'
            ]
        }
        
        # 심리상담 전문 키워드
        self.counseling_keywords = {
            'trauma': [r'트라우마', r'악몽', r'플래시백', r'기억이?\s*자꾸'],
            'depression': [r'우울', r'무기력', r'슬프', r'절망', r'힘들'],
            'anxiety': [r'불안', r'걱정', r'긴장', r'초조', r'두려'],
            'anger': [r'화', r'분노', r'짜증', r'열받', r'억울'],
            'self_harm': [r'자해', r'상처.*내', r'아프게.*하고?\s*싶'],
            'help_seeking': [r'도와', r'상담', r'치료', r'병원', r'의사']
        }
    
    def analyze_risk_level(self, text: str) -> Dict:
        """위험도 분석"""
        text_lower = text.lower()
        
        # 위험도별 점수 계산
        risk_scores = {'critical': 0, 'high': 0, 'medium': 0}
        detected_patterns = []
        
        for risk_level, patterns in self.risk_patterns.items():
            for pattern in patterns:
                matches = len(re.findall(pattern, text_lower))
                if matches > 0:
                    risk_scores[risk_level] += matches
                    detected_patterns.append((risk_level, pattern, matches))
        
        # 최종 위험도 결정
        if risk_scores['critical'] > 0:
            final_risk = 'high'
        elif risk_scores['high'] > 0:
            final_risk = 'medium' if risk_scores['high'] < 3 else 'high'
        elif risk_scores['medium'] > 0:
            final_risk = 'low' if risk_scores['medium'] < 2 else 'medium'
        else:
            final_risk = 'low'
        
        # 심리상담 키워드 분석
        counseling_analysis = {}
        for category, patterns in self.counseling_keywords.items():
            score = 0
            for pattern in patterns:
                score += len(re.findall(pattern, text_lower))
            counseling_analysis[category] = score
        
        return {
            'risk_level': final_risk,
            'risk_scores': risk_scores,
            'detected_patterns': detected_patterns,
            'counseling_analysis': counseling_analysis,
            'needs_immediate_help': risk_scores['critical'] > 0 or risk_scores['high'] > 2
        }

class AdvancedAvatarGenerator:
    """고급 아바타 생성기 (KoBERT + 패턴 분석)"""
    
    def __init__(self):
        # GPU 사용 가능 여부 확인
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"🖥️ 사용 중인 디바이스: {self.device}")
        
        # KoBERT 모델 초기화
        try:
            self.kobert_model = KoBERTEmotionClassifier().to(self.device)
            logger.info("✅ KoBERT 감정 분석 모델 초기화 완료")
        except Exception as e:
            logger.error(f"❌ KoBERT 모델 초기화 실패: {e}")
            self.kobert_model = None
        
        # 패턴 분석기
        self.pattern_analyzer = PsychologicalPatternAnalyzer()
        
        # 아바타 스타일 (더 세밀한 색상)
        self.avatar_styles = {
            'joy': {
                'face_color': (255, 228, 196),
                'background': (255, 245, 238),
                'accent_color': (255, 160, 122),
                'message': "기쁨을 느끼고 계시는군요! 이런 긍정적인 감정을 소중히 하세요 😊"
            },
            'sad': {
                'face_color': (221, 221, 238),
                'background': (240, 248, 255),
                'accent_color': (123, 104, 238),
                'message': "힘든 시간을 보내고 계시는군요. 이런 감정도 자연스러운 것입니다 💙"
            },
            'anxiety': {
                'face_color': (255, 239, 213),
                'background': (253, 245, 230),
                'accent_color': (205, 133, 63),
                'message': "불안하신 마음이 느껴져요. 깊게 숨을 들이쉬고 천천히 내쉬어보세요 🌼"
            },
            'anger': {
                'face_color': (255, 218, 185),
                'background': (255, 240, 245),
                'accent_color': (205, 92, 92),
                'message': "화가 나셨군요. 이 감정을 인정하고 건전하게 표현해보세요 🔥"
            },
            'neutral': {
                'face_color': (250, 240, 230),
                'background': (248, 248, 255),
                'accent_color': (169, 169, 169),
                'message': "안정된 상태이신 것 같네요. 현재의 평온함을 느껴보세요 ✨"
            }
        }
        
        # 위험도별 대응
        self.risk_responses = {
            'high': {
                'message': "⚠️ 매우 힘든 상황이신 것 같습니다. 혼자 견디지 마시고 전문가의 도움을 받으세요.",
                'emergency_contact': "자살예방상담전화: 109 (24시간)",
                'additional_message': "당신의 생명은 소중합니다. 지금의 고통은 영원하지 않습니다."
            },
            'medium': {
                'message': "💛 지금 어려운 시간을 겪고 계시는군요. 도움을 요청하는 것은 용기있는 일입니다.",
                'support_message': "정신건강상담전화: 1577-0199"
            },
            'low': {
                'message': "현재 심리적으로 안정된 상태로 보입니다 ✅"
            }
        }
    
    def analyze_comprehensive_emotion(self, text: str) -> Dict:
        """종합적 감정 분석 (KoBERT + 패턴 분석)"""
        
        # 1. KoBERT 감정 분석
        kobert_result = None
        if self.kobert_model:
            try:
                kobert_result = self.kobert_model.predict_emotion(text)
                logger.info(f"🤖 KoBERT 분석: {kobert_result['emotion']} (신뢰도: {kobert_result['confidence']:.3f})")
            except Exception as e:
                logger.error(f"KoBERT 예측 오류: {e}")
        
        # 2. 패턴 기반 위험도 분석
        risk_analysis = self.pattern_analyzer.analyze_risk_level(text)
        
        # 3. 결과 통합
        if kobert_result:
            final_emotion = kobert_result['emotion']
            confidence = kobert_result['confidence']
            emotion_probs = kobert_result['probabilities']
        else:
            # KoBERT 실패시 패턴 기반 fallback
            final_emotion = self._pattern_based_emotion(text)
            confidence = 0.7
            emotion_probs = {final_emotion: 0.7}
        
        # 위험도가 높으면 감정을 조정 (안전 우선)
        if risk_analysis['risk_level'] == 'high':
            final_emotion = 'sad'  # 위험 상황은 슬픔으로 처리하여 적절한 대응
        
        return {
            'emotion': final_emotion,
            'confidence': confidence,
            'emotion_probabilities': emotion_probs,
            'risk_level': risk_analysis['risk_level'],
            'risk_analysis': risk_analysis,
            'kobert_available': kobert_result is not None,
            'needs_immediate_help': risk_analysis['needs_immediate_help'],
            'analysis_method': 'kobert_enhanced' if kobert_result else 'pattern_based'
        }
    
    def _pattern_based_emotion(self, text: str) -> str:
        """패턴 기반 감정 분석 (fallback)"""
        text_lower = text.lower()
        
        if any(word in text_lower for word in ['죽고싶', '자살', '끝내고싶', '의미없']):
            return 'sad'
        elif any(word in text_lower for word in ['우울', '슬프', '힘들', '절망']):
            return 'sad'
        elif any(word in text_lower for word in ['불안', '걱정', '무서', '두려']):
            return 'anxiety'
        elif any(word in text_lower for word in ['화', '짜증', '분노', '열받']):
            return 'anger'
        elif any(word in text_lower for word in ['기쁘', '좋', '행복', '즐거']):
            return 'joy'
        else:
            return 'neutral'
    
    def generate_avatar_with_analysis(self, text: str) -> Dict:
        """감정 분석 + 아바타 생성"""
        
        # 종합 감정 분석
        analysis = self.analyze_comprehensive_emotion(text)
        
        # 아바타 생성
        avatar_result = self._generate_detailed_avatar(analysis)
        
        # 상담 메시지 생성
        counseling_message = self._generate_counseling_message(analysis, text)
        
        return {
            'success': True,
            'avatar_image': avatar_result['avatar_image'],
            'emotion': analysis['emotion'],
            'emotion_message': counseling_message['primary_message'],
            'risk_level': analysis['risk_level'],
            'risk_message': counseling_message['risk_message'],
            'emergency_info': counseling_message.get('emergency_info'),
            'confidence': analysis['confidence'],
            'emotion_probabilities': analysis['emotion_probabilities'],
            'analysis_method': analysis['analysis_method'],
            'kobert_available': analysis['kobert_available'],
            'needs_immediate_help': analysis['needs_immediate_help'],
            'counseling_analysis': analysis['risk_analysis']['counseling_analysis'],
            'timestamp': datetime.now().isoformat()
        }
    
    def _generate_detailed_avatar(self, analysis: Dict) -> Dict:
        """상세한 아바타 생성"""
        emotion = analysis['emotion']
        confidence = analysis['confidence']
        risk_level = analysis['risk_level']
        
        style = self.avatar_styles[emotion].copy()
        
        # 신뢰도에 따른 색상 조정
        if confidence > 0.8:
            # 고신뢰도: 색상을 더 진하게
            style['face_color'] = tuple(max(0, c - 20) for c in style['face_color'])
        elif confidence < 0.6:
            # 저신뢰도: 색상을 더 밝게
            style['face_color'] = tuple(min(255, c + 20) for c in style['face_color'])
        
        # 이미지 생성 (고해상도)
        width, height = 600, 600
        image = Image.new('RGB', (width, height), style['background'])
        draw = ImageDraw.Draw(image)
        
        # 배경 그라데이션
        self._draw_advanced_background(draw, width, height, style)
        
        # 얼굴 그리기
        face_size = int(350 * (0.9 + confidence * 0.2))
        face_x = (width - face_size) // 2
        face_y = (height - face_size) // 2 - 30
        
        # 얼굴 그림자
        shadow_offset = 8
        draw.ellipse([face_x + shadow_offset, face_y + shadow_offset, 
                     face_x + face_size + shadow_offset, face_y + face_size + shadow_offset], 
                    fill=(0, 0, 0, 40))
        
        # 메인 얼굴
        draw.ellipse([face_x, face_y, face_x + face_size, face_y + face_size], 
                    fill=style['face_color'], outline=style['accent_color'], width=5)
        
        # 감정별 세부 요소
        self._draw_emotion_details(draw, face_x, face_y, face_size, emotion, confidence)
        
        # 위험도 표시
        if risk_level == 'high':
            self._add_emergency_indicator(draw, width, height)
        
        # Base64 변환
        buffer = io.BytesIO()
        image.save(buffer, format='PNG', quality=100)
        buffer.seek(0)
        
        image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        
        return {
            'avatar_image': f"data:image/png;base64,{image_base64}"
        }
    
    def _draw_advanced_background(self, draw, width, height, style):
        """고급 배경 그리기"""
        bg_color = style['background']
        accent_color = style['accent_color']
        
        # 방사형 그라데이션 효과
        center_x, center_y = width // 2, height // 2
        max_radius = min(width, height) // 2
        
        for radius in range(max_radius, 0, -5):
            ratio = radius / max_radius
            r = int(bg_color[0] * ratio + accent_color[0] * (1 - ratio) * 0.1)
            g = int(bg_color[1] * ratio + accent_color[1] * (1 - ratio) * 0.1)
            b = int(bg_color[2] * ratio + accent_color[2] * (1 - ratio) * 0.1)
            
            draw.ellipse([center_x - radius, center_y - radius,
                         center_x + radius, center_y + radius], 
                        fill=(r, g, b))
    
    def _draw_emotion_details(self, draw, face_x, face_y, face_size, emotion, confidence):
        """감정별 세부 표현"""
        # 눈 그리기
        eye_y = face_y + face_size // 3
        left_eye_x = face_x + face_size // 3
        right_eye_x = face_x + 2 * face_size // 3
        
        eye_width = max(25, face_size // 15)
        eye_height = max(20, face_size // 18)
        
        if emotion == 'joy':
            # 웃는 눈
            draw.arc([left_eye_x - eye_width, eye_y - eye_height//2, 
                     left_eye_x + eye_width, eye_y + eye_height//2], 
                    0, 180, fill=(0, 0, 0), width=5)
            draw.arc([right_eye_x - eye_width, eye_y - eye_height//2, 
                     right_eye_x + eye_width, eye_y + eye_height//2], 
                    0, 180, fill=(0, 0, 0), width=5)
            
            # 웃는 입
            mouth_y = face_y + 2 * face_size // 3
            mouth_width = int(face_size // 3 * (1 + confidence * 0.5))
            draw.arc([face_x + face_size//2 - mouth_width, mouth_y - 20,
                     face_x + face_size//2 + mouth_width, mouth_y + 40], 
                    0, 180, fill=(200, 50, 50), width=8)
            
        elif emotion == 'sad':
            # 슬픈 눈
            draw.ellipse([left_eye_x - eye_width//2, eye_y - eye_height//2, 
                         left_eye_x + eye_width//2, eye_y + eye_height//2], 
                        fill=(50, 50, 120))
            draw.ellipse([right_eye_x - eye_width//2, eye_y - eye_height//2, 
                         right_eye_x + eye_width//2, eye_y + eye_height//2], 
                        fill=(50, 50, 120))
            
            # 눈물 (고신뢰도일 때)
            if confidence > 0.7:
                tear_x = left_eye_x + eye_width//3
                tear_y = eye_y + eye_height
                for i in range(3):
                    draw.ellipse([tear_x - 4, tear_y + i*10, tear_x + 4, tear_y + i*10 + 20], 
                                fill=(150, 200, 255))
            
            # 슬픈 입
            mouth_y = face_y + 2 * face_size // 3
            mouth_width = face_size // 4
            draw.arc([face_x + face_size//2 - mouth_width, mouth_y - 30,
                     face_x + face_size//2 + mouth_width, mouth_y + 20], 
                    180, 360, fill=(100, 100, 150), width=6)
        
        # 다른 감정들도 비슷하게 구현...
    
    def _add_emergency_indicator(self, draw, width, height):
        """응급 상황 표시"""
        # 빨간색 경고 테두리
        draw.rectangle([0, 0, width, height], outline=(255, 0, 0), width=10)
        
        # 경고 아이콘
        warning_size = 50
        warning_x = width - warning_size - 20
        warning_y = 20
        
        # 삼각형 경고 표시
        draw.polygon([
            (warning_x + warning_size//2, warning_y),
            (warning_x, warning_y + warning_size),
            (warning_x + warning_size, warning_y + warning_size)
        ], fill=(255, 0, 0))
        
        draw.ellipse([warning_x + warning_size//2 - 5, warning_y + 15,
                     warning_x + warning_size//2 + 5, warning_y + 25], fill=(255, 255, 255))
        draw.ellipse([warning_x + warning_size//2 - 3, warning_y + 35,
                     warning_x + warning_size//2 + 3, warning_y + 40], fill=(255, 255, 255))
    
    def _generate_counseling_message(self, analysis: Dict, original_text: str) -> Dict:
        """상담 메시지 생성"""
        emotion = analysis['emotion']
        risk_level = analysis['risk_level']
        
        # 기본 메시지
        primary_message = self.avatar_styles[emotion]['message']
        
        # 위험도별 메시지
        risk_message = self.risk_responses[risk_level]['message']
        
        result = {
            'primary_message': primary_message,
            'risk_message': risk_message
        }
        
        # 응급 상황 정보
        if analysis['needs_immediate_help']:
            result['emergency_info'] = {
                'contact': self.risk_responses['high']['emergency_contact'],
                'additional': self.risk_responses['high']['additional_message'],
                'urgent': True
            }
        
        return result

# 전역 아바타 생성기
avatar_generator = AdvancedAvatarGenerator()

@app.route('/health', methods=['GET'])
def health_check():
    """헬스 체크"""
    return jsonify({
        'status': 'healthy',
        'service': 'KoBERT Advanced Psychological Avatar API',
        'kobert_available': KOBERT_AVAILABLE,
        'device': str(avatar_generator.device),
        'timestamp': datetime.now().isoformat()
    })

@app.route('/analyze', methods=['POST'])
def analyze_text():
    """텍스트 심리 분석"""
    try:
        data = request.get_json()
        text = data.get('text', '')
        
        if not text:
            return jsonify({'error': 'Text is required'}), 400
        
        analysis = avatar_generator.analyze_comprehensive_emotion(text)
        
        return jsonify({
            'success': True,
            'analysis': analysis,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Analysis error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/generate_avatar', methods=['POST'])
def generate_avatar():
    """KoBERT 기반 아바타 생성"""
    try:
        data = request.get_json()
        text = data.get('text', '')
        
        if not text:
            return jsonify({'error': 'Text is required'}), 400
        
        result = avatar_generator.generate_avatar_with_analysis(text)
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Avatar generation error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/model_info', methods=['GET'])
def get_model_info():
    """모델 정보"""
    return jsonify({
        'model_type': 'KoBERT Enhanced',
        'kobert_available': KOBERT_AVAILABLE,
        'device': str(avatar_generator.device),
        'features': [
            'KoBERT 감정 분석',
            '심리상담 패턴 인식',
            '위험도 평가',
            '고해상도 아바타 생성'
        ]
    })

@app.route('/emergency_contacts', methods=['GET'])
def get_emergency_contacts():
    """응급 연락처 정보"""
    return jsonify({
        'emergency_contacts': {
            'suicide_prevention': {
                'name': '자살예방상담전화',
                'number': '109',
                'available': '24시간'
            },
            'mental_health': {
                'name': '정신건강상담전화',
                'number': '1577-0199',
                'available': '24시간'
            },
            'crisis_center': {
                'name': '생명의전화',
                'number': '1588-9191',
                'available': '24시간'
            }
        }
    })

@app.route('/', methods=['GET'])
def index():
    """메인 페이지"""
    return render_template_string("""
    <!DOCTYPE html>
    <html>
    <head>
        <title>KoBERT 심리상담 아바타 API</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 40px; background: #f0f8ff; }
            .container { max-width: 900px; margin: 0 auto; background: white; padding: 40px; border-radius: 15px; box-shadow: 0 8px 16px rgba(0,0,0,0.1); }
            .header { text-align: center; color: #2c3e50; margin-bottom: 40px; }
            .status { display: flex; justify-content: space-around; margin: 30px 0; }
            .status-item { text-align: center; padding: 20px; background: #f8f9fa; border-radius: 10px; }
            .feature { background: #e8f5e8; padding: 25px; margin: 15px 0; border-radius: 10px; border-left: 5px solid #27ae60; }
            .warning { background: #fff3cd; border: 1px solid #ffeaa7; padding: 20px; border-radius: 8px; margin: 25px 0; }
            .emergency { background: #f8d7da; border: 1px solid #f5c6cb; padding: 20px; border-radius: 8px; margin: 25px 0; }
            .model-info { background: #d1ecf1; border: 1px solid #bee5eb; padding: 20px; border-radius: 8px; margin: 25px 0; }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>🧠 KoBERT 심리상담 아바타 API</h1>
                <p>Advanced Psychological Analysis with Korean BERT</p>
            </div>
            
            <div class="status">
                <div class="status-item">
                    <h4>🤖 KoBERT</h4>
                    <p>{{ '✅ 활성화' if kobert_available else '❌ 비활성화' }}</p>
                </div>
                <div class="status-item">
                    <h4>🖥️ 디바이스</h4>
                    <p>{{ device }}</p>
                </div>
                <div class="status-item">
                    <h4>🚀 상태</h4>
                    <p>✅ 정상 작동</p>
                </div>
            </div>
            
            <div class="model-info">
                <h3>🔬 모델 특징</h3>
                <ul>
                    <li><strong>KoBERT 감정 분석</strong>: 한국어 특화 BERT 모델로 정밀한 감정 인식</li>
                    <li><strong>심리상담 패턴</strong>: 자살사고, 우울, 불안, 트라우마 전문 감지</li>
                    <li><strong>위험도 평가</strong>: 3단계 위험도 분류 및 즉시 개입</li>
                    <li><strong>고해상도 아바타</strong>: 600x600 해상도의 정교한 감정 표현</li>
                </ul>
            </div>
            
            <div class="feature">
                <h3>📡 API 엔드포인트</h3>
                <p><code>POST /generate_avatar</code> - KoBERT 기반 아바타 생성</p>
                <p><code>POST /analyze</code> - 감정 분석만 수행</p>
                <p><code>GET /model_info</code> - 모델 정보 확인</p>
                <p><code>GET /emergency_contacts</code> - 응급 연락처</p>
            </div>
            
            <div class="warning">
                <h4>⚠️ 중요 안내</h4>
                <p>이 시스템은 <strong>심리상담의 보조 도구</strong>입니다. 전문 상담사나 의료진의 진단을 대체할 수 없습니다.</p>
            </div>
            
            <div class="emergency">
                <h4>🆘 응급상황 대응</h4>
                <p><strong>자살예방상담전화: 109 (24시간)</strong></p>
                <p><strong>정신건강상담전화: 1577-0199 (24시간)</strong></p>
                <p><strong>생명의전화: 1588-9191 (24시간)</strong></p>
            </div>
        </div>
    </body>
    </html>
    """, kobert_available=KOBERT_AVAILABLE, device=avatar_generator.device)

if __name__ == '__main__':
    print("🧠 KoBERT 심리상담 아바타 API 시작...")
    print("🔗 API 주소: http://localhost:8003")
    print("📋 문서: http://localhost:8003")
    print(f"🤖 KoBERT 사용 가능: {KOBERT_AVAILABLE}")
    print(f"🖥️ 디바이스: {avatar_generator.device}")
    
    app.run(
        host='0.0.0.0',
        port=8003,
        debug=True,
        threaded=True
    )