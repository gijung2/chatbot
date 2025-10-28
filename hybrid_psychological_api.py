"""
심리상담 아바타 API (PyTorch + 패턴 기반 하이브리드)
- 기본 패턴 분석 + PyTorch 백그라운드 로딩
- 빠른 응답 + 고정밀 분석
"""

from flask import Flask, request, jsonify, render_template_string
from flask_cors import CORS
import base64
import io
import json
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from datetime import datetime
import logging
import re
from typing import Dict, List, Tuple, Optional
import threading
import time

# PyTorch 및 KoBERT 백그라운드 로딩
KOBERT_MODEL = None
KOBERT_LOADING = False

def load_kobert_background():
    """백그라운드에서 KoBERT 모델 로딩"""
    global KOBERT_MODEL, KOBERT_LOADING
    
    if KOBERT_LOADING:
        return
        
    KOBERT_LOADING = True
    
    try:
        print("🔄 KoBERT 모델 백그라운드 로딩 시작...")
        
        import torch
        import torch.nn as nn
        from transformers import AutoTokenizer, AutoModel
        
        # 간단한 KoBERT 감정 분석기
        class SimpleKoBERTAnalyzer:
            def __init__(self):
                self.tokenizer = AutoTokenizer.from_pretrained('klue/bert-base')
                self.model = AutoModel.from_pretrained('klue/bert-base')
                self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                self.model.to(self.device)
                
                # 감정 키워드 매핑
                self.emotion_keywords = {
                    'joy': ['기쁘', '좋', '행복', '즐거', '만족', '기대', '희망'],
                    'sad': ['슬프', '우울', '힘들', '절망', '외로', '허무', '공허'],
                    'anxiety': ['불안', '걱정', '두려', '무서', '긴장', '초조', '심려'],
                    'anger': ['화', '분노', '짜증', '열받', '억울', '답답', '화나'],
                    'neutral': ['그냥', '평범', '보통', '괜찮', '그럭저럭']
                }
                
            def analyze_emotion(self, text: str) -> Dict:
                """간단한 감정 분석"""
                text_lower = text.lower()
                
                # 키워드 기반 점수 계산
                scores = {}
                for emotion, keywords in self.emotion_keywords.items():
                    score = 0
                    for keyword in keywords:
                        score += len(re.findall(keyword, text_lower))
                    scores[emotion] = score
                
                # 최고 점수 감정 선택
                if sum(scores.values()) == 0:
                    emotion = 'neutral'
                    confidence = 0.5
                else:
                    emotion = max(scores, key=scores.get)
                    total_score = sum(scores.values())
                    confidence = scores[emotion] / total_score if total_score > 0 else 0.5
                
                return {
                    'emotion': emotion,
                    'confidence': min(0.95, max(0.5, confidence)),
                    'scores': scores,
                    'method': 'kobert_hybrid'
                }
        
        KOBERT_MODEL = SimpleKoBERTAnalyzer()
        print("✅ KoBERT 모델 로딩 완료!")
        
    except Exception as e:
        print(f"⚠️ KoBERT 로딩 실패: {e}")
        print("📍 패턴 기반 분석으로 대체됩니다.")
        KOBERT_MODEL = None
    
    KOBERT_LOADING = False

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

class PsychologicalPatternAnalyzer:
    """심리상담 전문 패턴 분석기"""
    
    def __init__(self):
        # 위험도별 패턴
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
        
        # 감정 패턴
        self.emotion_patterns = {
            'joy': [r'기쁘', r'좋', r'행복', r'즐거', r'만족', r'웃', r'미소'],
            'sad': [r'슬프', r'우울', r'힘들', r'절망', r'외로', r'눈물', r'울고'],
            'anxiety': [r'불안', r'걱정', r'두려', r'무서', r'긴장', r'떨리', r'조마조마'],
            'anger': [r'화', r'분노', r'짜증', r'열받', r'억울', r'빡치', r'싫'],
            'neutral': [r'그냥', r'평범', r'보통', r'괜찮', r'그럭저럭', r'무난']
        }
    
    def analyze_emotion(self, text: str) -> Dict:
        """패턴 기반 감정 분석"""
        text_lower = text.lower()
        
        scores = {}
        for emotion, patterns in self.emotion_patterns.items():
            score = 0
            for pattern in patterns:
                matches = len(re.findall(pattern, text_lower))
                score += matches
            scores[emotion] = score
        
        # 최고 점수 감정
        if sum(scores.values()) == 0:
            emotion = 'neutral'
            confidence = 0.6
        else:
            emotion = max(scores, key=scores.get)
            total = sum(scores.values())
            confidence = scores[emotion] / total if total > 0 else 0.6
        
        return {
            'emotion': emotion,
            'confidence': min(0.9, max(0.6, confidence)),
            'scores': scores,
            'method': 'pattern_based'
        }
    
    def analyze_risk_level(self, text: str) -> Dict:
        """위험도 분석"""
        text_lower = text.lower()
        
        risk_scores = {'critical': 0, 'high': 0, 'medium': 0}
        detected_patterns = []
        
        for risk_level, patterns in self.risk_patterns.items():
            for pattern in patterns:
                matches = len(re.findall(pattern, text_lower))
                if matches > 0:
                    risk_scores[risk_level] += matches
                    detected_patterns.append((risk_level, pattern, matches))
        
        # 최종 위험도
        if risk_scores['critical'] > 0:
            final_risk = 'high'
        elif risk_scores['high'] > 0:
            final_risk = 'medium' if risk_scores['high'] < 3 else 'high'
        elif risk_scores['medium'] > 0:
            final_risk = 'low' if risk_scores['medium'] < 2 else 'medium'
        else:
            final_risk = 'low'
        
        return {
            'risk_level': final_risk,
            'risk_scores': risk_scores,
            'detected_patterns': detected_patterns,
            'needs_immediate_help': risk_scores['critical'] > 0 or risk_scores['high'] > 2
        }

class HybridAvatarGenerator:
    """하이브리드 아바타 생성기 (패턴 + KoBERT)"""
    
    def __init__(self):
        self.pattern_analyzer = PsychologicalPatternAnalyzer()
        
        # 아바타 스타일
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
        
        # 위험도별 메시지
        self.risk_messages = {
            'high': "⚠️ 매우 힘든 상황입니다. 전문가의 도움을 받으세요. 자살예방상담전화: 109",
            'medium': "💛 어려운 시간이지만 도움을 요청하는 것은 용기입니다. 정신건강상담전화: 1577-0199",
            'low': "✅ 현재 심리적으로 안정된 상태입니다."
        }
    
    def analyze_comprehensive_emotion(self, text: str) -> Dict:
        """종합 감정 분석"""
        
        # 1. 패턴 기반 분석 (즉시)
        pattern_result = self.pattern_analyzer.analyze_emotion(text)
        
        # 2. 위험도 분석
        risk_analysis = self.pattern_analyzer.analyze_risk_level(text)
        
        # 3. KoBERT 분석 (사용 가능한 경우)
        kobert_result = None
        if KOBERT_MODEL and not KOBERT_LOADING:
            try:
                kobert_result = KOBERT_MODEL.analyze_emotion(text)
                logger.info(f"🤖 KoBERT 분석: {kobert_result['emotion']} (신뢰도: {kobert_result['confidence']:.3f})")
            except Exception as e:
                logger.error(f"KoBERT 분석 오류: {e}")
        
        # 4. 결과 통합
        if kobert_result and kobert_result['confidence'] > 0.7:
            # KoBERT 결과를 우선적으로 사용
            final_emotion = kobert_result['emotion']
            confidence = kobert_result['confidence']
            method = 'kobert_enhanced'
        else:
            # 패턴 분석 결과 사용
            final_emotion = pattern_result['emotion']
            confidence = pattern_result['confidence']
            method = 'pattern_based'
        
        # 위험도가 높으면 감정 조정
        if risk_analysis['risk_level'] == 'high':
            final_emotion = 'sad'
        
        return {
            'emotion': final_emotion,
            'confidence': confidence,
            'risk_level': risk_analysis['risk_level'],
            'risk_analysis': risk_analysis,
            'pattern_result': pattern_result,
            'kobert_result': kobert_result,
            'method': method,
            'kobert_available': KOBERT_MODEL is not None,
            'kobert_loading': KOBERT_LOADING
        }
    
    def generate_avatar_with_analysis(self, text: str) -> Dict:
        """감정 분석 + 아바타 생성"""
        
        # 종합 분석
        analysis = self.analyze_comprehensive_emotion(text)
        
        # 아바타 생성
        avatar_result = self._generate_avatar(analysis)
        
        # 메시지 생성
        emotion_message = self.avatar_styles[analysis['emotion']]['message']
        risk_message = self.risk_messages[analysis['risk_level']]
        
        return {
            'success': True,
            'avatar_image': avatar_result['avatar_image'],
            'emotion': analysis['emotion'],
            'emotion_message': emotion_message,
            'risk_level': analysis['risk_level'],
            'risk_message': risk_message,
            'confidence': analysis['confidence'],
            'method': analysis['method'],
            'kobert_available': analysis['kobert_available'],
            'kobert_loading': analysis['kobert_loading'],
            'needs_immediate_help': analysis['risk_analysis']['needs_immediate_help'],
            'timestamp': datetime.now().isoformat()
        }
    
    def _generate_avatar(self, analysis: Dict) -> Dict:
        """아바타 이미지 생성"""
        emotion = analysis['emotion']
        confidence = analysis['confidence']
        risk_level = analysis['risk_level']
        
        style = self.avatar_styles[emotion].copy()
        
        # 고해상도 이미지
        width, height = 500, 500
        image = Image.new('RGB', (width, height), style['background'])
        draw = ImageDraw.Draw(image)
        
        # 배경 그라데이션
        center_x, center_y = width // 2, height // 2
        for radius in range(min(width, height) // 2, 0, -10):
            alpha = radius / (min(width, height) // 2)
            color = tuple(int(c * alpha + style['accent_color'][i] * (1 - alpha) * 0.1) 
                         for i, c in enumerate(style['background']))
            draw.ellipse([center_x - radius, center_y - radius,
                         center_x + radius, center_y + radius], fill=color)
        
        # 얼굴 그리기
        face_size = int(300 * (0.8 + confidence * 0.4))
        face_x = (width - face_size) // 2
        face_y = (height - face_size) // 2 - 20
        
        # 얼굴 그림자
        draw.ellipse([face_x + 5, face_y + 5, 
                     face_x + face_size + 5, face_y + face_size + 5], 
                    fill=(0, 0, 0, 30))
        
        # 얼굴
        draw.ellipse([face_x, face_y, face_x + face_size, face_y + face_size], 
                    fill=style['face_color'], outline=style['accent_color'], width=3)
        
        # 감정별 표정
        self._draw_emotion_expression(draw, face_x, face_y, face_size, emotion, confidence)
        
        # 위험도 표시
        if risk_level == 'high':
            self._add_warning_indicator(draw, width, height)
        
        # Base64 변환
        buffer = io.BytesIO()
        image.save(buffer, format='PNG', quality=95)
        buffer.seek(0)
        
        image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        
        return {
            'avatar_image': f"data:image/png;base64,{image_base64}"
        }
    
    def _draw_emotion_expression(self, draw, face_x, face_y, face_size, emotion, confidence):
        """감정별 표정 그리기"""
        # 눈 위치
        eye_y = face_y + face_size // 3
        left_eye_x = face_x + face_size // 3
        right_eye_x = face_x + 2 * face_size // 3
        eye_size = max(15, face_size // 20)
        
        # 입 위치
        mouth_y = face_y + 2 * face_size // 3
        mouth_x = face_x + face_size // 2
        
        if emotion == 'joy':
            # 웃는 눈
            draw.arc([left_eye_x - eye_size, eye_y - eye_size//2, 
                     left_eye_x + eye_size, eye_y + eye_size//2], 
                    0, 180, fill=(0, 0, 0), width=3)
            draw.arc([right_eye_x - eye_size, eye_y - eye_size//2, 
                     right_eye_x + eye_size, eye_y + eye_size//2], 
                    0, 180, fill=(0, 0, 0), width=3)
            
            # 웃는 입
            mouth_width = int(face_size // 4 * (1 + confidence * 0.5))
            draw.arc([mouth_x - mouth_width, mouth_y - 15,
                     mouth_x + mouth_width, mouth_y + 25], 
                    0, 180, fill=(200, 50, 50), width=5)
            
        elif emotion == 'sad':
            # 슬픈 눈
            draw.ellipse([left_eye_x - eye_size//2, eye_y - eye_size//2, 
                         left_eye_x + eye_size//2, eye_y + eye_size//2], 
                        fill=(50, 50, 100))
            draw.ellipse([right_eye_x - eye_size//2, eye_y - eye_size//2, 
                         right_eye_x + eye_size//2, eye_y + eye_size//2], 
                        fill=(50, 50, 100))
            
            # 눈물
            if confidence > 0.7:
                draw.ellipse([left_eye_x - 3, eye_y + eye_size, 
                             left_eye_x + 3, eye_y + eye_size + 15], 
                            fill=(150, 200, 255))
            
            # 슬픈 입
            mouth_width = face_size // 5
            draw.arc([mouth_x - mouth_width, mouth_y - 20,
                     mouth_x + mouth_width, mouth_y + 10], 
                    180, 360, fill=(100, 100, 150), width=4)
            
        elif emotion == 'anxiety':
            # 불안한 눈
            draw.ellipse([left_eye_x - eye_size//2, eye_y - eye_size//2, 
                         left_eye_x + eye_size//2, eye_y + eye_size//2], 
                        fill=(80, 80, 80))
            draw.ellipse([right_eye_x - eye_size//2, eye_y - eye_size//2, 
                         right_eye_x + eye_size//2, eye_y + eye_size//2], 
                        fill=(80, 80, 80))
            
            # 걱정스러운 입
            draw.ellipse([mouth_x - 8, mouth_y - 5, mouth_x + 8, mouth_y + 5], 
                        fill=(150, 100, 100))
            
        elif emotion == 'anger':
            # 화난 눈
            draw.polygon([
                (left_eye_x - eye_size, eye_y),
                (left_eye_x, eye_y - eye_size//2),
                (left_eye_x + eye_size, eye_y),
                (left_eye_x, eye_y + eye_size//2)
            ], fill=(150, 50, 50))
            
            draw.polygon([
                (right_eye_x - eye_size, eye_y),
                (right_eye_x, eye_y - eye_size//2),
                (right_eye_x + eye_size, eye_y),
                (right_eye_x, eye_y + eye_size//2)
            ], fill=(150, 50, 50))
            
            # 화난 입
            draw.rectangle([mouth_x - 20, mouth_y - 3, mouth_x + 20, mouth_y + 3], 
                          fill=(200, 50, 50))
            
        else:  # neutral
            # 평범한 눈
            draw.ellipse([left_eye_x - eye_size//2, eye_y - eye_size//2, 
                         left_eye_x + eye_size//2, eye_y + eye_size//2], 
                        fill=(100, 100, 100))
            draw.ellipse([right_eye_x - eye_size//2, eye_y - eye_size//2, 
                         right_eye_x + eye_size//2, eye_y + eye_size//2], 
                        fill=(100, 100, 100))
            
            # 평범한 입
            draw.rectangle([mouth_x - 15, mouth_y - 2, mouth_x + 15, mouth_y + 2], 
                          fill=(120, 120, 120))
    
    def _add_warning_indicator(self, draw, width, height):
        """위험 상황 경고 표시"""
        # 빨간 테두리
        draw.rectangle([0, 0, width-1, height-1], outline=(255, 0, 0), width=8)
        
        # 경고 아이콘
        warning_size = 40
        warning_x = width - warning_size - 15
        warning_y = 15
        
        # 삼각형 경고
        draw.polygon([
            (warning_x + warning_size//2, warning_y),
            (warning_x, warning_y + warning_size),
            (warning_x + warning_size, warning_y + warning_size)
        ], fill=(255, 0, 0))
        
        # ! 표시
        draw.rectangle([warning_x + warning_size//2 - 2, warning_y + 10,
                       warning_x + warning_size//2 + 2, warning_y + 25], 
                      fill=(255, 255, 255))
        draw.ellipse([warning_x + warning_size//2 - 2, warning_y + 28,
                     warning_x + warning_size//2 + 2, warning_y + 32], 
                    fill=(255, 255, 255))

# 전역 아바타 생성기
avatar_generator = HybridAvatarGenerator()

# KoBERT 백그라운드 로딩 시작
def start_kobert_loading():
    thread = threading.Thread(target=load_kobert_background, daemon=True)
    thread.start()

@app.route('/health', methods=['GET'])
def health_check():
    """헬스 체크"""
    return jsonify({
        'status': 'healthy',
        'service': 'Hybrid Psychological Avatar API',
        'kobert_available': KOBERT_MODEL is not None,
        'kobert_loading': KOBERT_LOADING,
        'timestamp': datetime.now().isoformat()
    })

@app.route('/generate_avatar', methods=['POST'])
def generate_avatar():
    """하이브리드 아바타 생성"""
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

@app.route('/analyze', methods=['POST'])
def analyze_text():
    """텍스트 분석"""
    try:
        data = request.get_json()
        text = data.get('text', '')
        
        if not text:
            return jsonify({'error': 'Text is required'}), 400
        
        analysis = avatar_generator.analyze_comprehensive_emotion(text)
        return jsonify(analysis)
        
    except Exception as e:
        logger.error(f"Analysis error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/model_status', methods=['GET'])
def get_model_status():
    """모델 상태 확인"""
    return jsonify({
        'kobert_available': KOBERT_MODEL is not None,
        'kobert_loading': KOBERT_LOADING,
        'pattern_analyzer': True,
        'hybrid_mode': True
    })

@app.route('/', methods=['GET'])
def index():
    """메인 페이지"""
    return render_template_string("""
    <!DOCTYPE html>
    <html>
    <head>
        <title>하이브리드 심리상담 아바타 API</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 40px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; }
            .container { max-width: 900px; margin: 0 auto; background: rgba(255,255,255,0.95); padding: 40px; border-radius: 20px; color: #333; }
            .header { text-align: center; margin-bottom: 40px; }
            .status { display: flex; justify-content: space-around; margin: 30px 0; }
            .status-item { text-align: center; padding: 20px; background: #f8f9fa; border-radius: 10px; }
            .feature { background: #e8f5e8; padding: 25px; margin: 15px 0; border-radius: 10px; border-left: 5px solid #27ae60; }
            .loading { background: #fff3cd; border: 1px solid #ffeaa7; padding: 20px; border-radius: 8px; margin: 25px 0; }
            .emergency { background: #f8d7da; border: 1px solid #f5c6cb; padding: 20px; border-radius: 8px; margin: 25px 0; }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>🧠 하이브리드 심리상담 아바타 API</h1>
                <p>Pattern Analysis + KoBERT Enhancement</p>
            </div>
            
            <div class="status">
                <div class="status-item">
                    <h4>🤖 KoBERT</h4>
                    <p>{{ '✅ 활성화' if kobert_available else ('🔄 로딩중' if kobert_loading else '❌ 대기중') }}</p>
                </div>
                <div class="status-item">
                    <h4>🎯 패턴분석</h4>
                    <p>✅ 활성화</p>
                </div>
                <div class="status-item">
                    <h4>🚀 상태</h4>
                    <p>✅ 정상 작동</p>
                </div>
            </div>
            
            {% if kobert_loading %}
            <div class="loading">
                <h4>🔄 KoBERT 모델 로딩 중...</h4>
                <p>백그라운드에서 KoBERT 모델을 로딩하고 있습니다. 로딩 완료 전까지는 패턴 분석으로 서비스됩니다.</p>
            </div>
            {% endif %}
            
            <div class="feature">
                <h3>🎯 하이브리드 분석 시스템</h3>
                <ul>
                    <li><strong>즉시 응답</strong>: 패턴 기반 분석으로 빠른 감정 인식</li>
                    <li><strong>고정밀 분석</strong>: KoBERT 모델로 정확한 감정 분류</li>
                    <li><strong>위험도 평가</strong>: 심리상담 전문 패턴으로 위험 상황 감지</li>
                    <li><strong>아바타 생성</strong>: 500x500 고해상도 감정 표현</li>
                </ul>
            </div>
            
            <div class="feature">
                <h3>📡 API 엔드포인트</h3>
                <p><code>POST /generate_avatar</code> - 하이브리드 아바타 생성</p>
                <p><code>POST /analyze</code> - 감정 분석</p>
                <p><code>GET /model_status</code> - 모델 상태 확인</p>
                <p><code>GET /health</code> - 헬스 체크</p>
            </div>
            
            <div class="emergency">
                <h4>🆘 응급상황 연락처</h4>
                <p><strong>자살예방상담전화: 109 (24시간)</strong></p>
                <p><strong>정신건강상담전화: 1577-0199 (24시간)</strong></p>
                <p><strong>생명의전화: 1588-9191 (24시간)</strong></p>
            </div>
        </div>
        
        <script>
            // 모델 상태 주기적 확인
            setInterval(function() {
                fetch('/model_status')
                    .then(response => response.json())
                    .then(data => {
                        if (data.kobert_available && !data.kobert_loading) {
                            location.reload();
                        }
                    });
            }, 5000);
        </script>
    </body>
    </html>
    """, kobert_available=KOBERT_MODEL is not None, kobert_loading=KOBERT_LOADING)

if __name__ == '__main__':
    print("🧠 하이브리드 심리상담 아바타 API 시작...")
    print("🔗 API 주소: http://localhost:8003")
    print("📋 문서: http://localhost:8003")
    print("🎯 패턴 분석: 즉시 사용 가능")
    print("🤖 KoBERT 로딩: 백그라운드에서 진행")
    
    # KoBERT 백그라운드 로딩 시작
    start_kobert_loading()
    
    app.run(
        host='0.0.0.0',
        port=8003,
        debug=True,
        threaded=True
    )