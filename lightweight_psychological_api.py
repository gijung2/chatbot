"""
심리상담 전문 아바타 API (Lightweight Version)
- 키워드 기반 심리 상태 분석
- 심리상담 도메인 특화 패턴 인식
- PyTorch 없이 동작하는 경량 버전
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

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

class PsychologicalPatternAnalyzer:
    """심리상담 전문 패턴 분석기 (키워드 기반)"""
    
    def __init__(self):
        # 심리학적 패턴 정의
        self.psychological_patterns = {
            # 자살 사고 지표 (최고 위험도)
            'suicidal_ideation': {
                'patterns': [
                    r'죽고?\s*싶', r'사라지고?\s*싶', r'끝내고?\s*싶', r'자살',
                    r'소용없', r'의미없', r'가치없', r'살기?\s*싫어?',
                    r'세상에?\s*없었으면', r'사라져?\s*버리고?\s*싶'
                ],
                'weight': 10,
                'risk_level': 'critical'
            },
            
            # 우울 증상
            'depression_symptoms': {
                'patterns': [
                    r'우울해?', r'슬프', r'힘들어?', r'절망', r'포기', r'무기력',
                    r'잠을?\s*못\s*자', r'식욕없', r'기력없', r'집중안?\s*돼?',
                    r'아무것도\s*하기\s*싫', r'흥미없', r'재미없', r'공허해?'
                ],
                'weight': 7,
                'risk_level': 'high'
            },
            
            # 불안 증상
            'anxiety_symptoms': {
                'patterns': [
                    r'불안해?', r'걱정', r'두려워?', r'무서워?', r'긴장', r'떨려',
                    r'심장이?\s*두근', r'식은땀', r'숨쉬기\s*힘들', r'답답해?',
                    r'계속\s*생각나', r'멈춰지지\s*않아?', r'잠이?\s*안\s*와?'
                ],
                'weight': 6,
                'risk_level': 'medium'
            },
            
            # 분노 조절 문제
            'anger_issues': {
                'patterns': [
                    r'화가?\s*나', r'분노', r'짜증', r'열받아?', r'빡쳐', r'미쳐?',
                    r'욕이?\s*나와?', r'때리고?\s*싶', r'부수고?\s*싶', r'참을?\s*수?\s*없',
                    r'억울해?', r'분해?', r'약\s*올라?'
                ],
                'weight': 5,
                'risk_level': 'medium'
            },
            
            # 트라우마/PTSD
            'trauma_ptsd': {
                'patterns': [
                    r'악몽', r'플래시백', r'기억이?\s*자꾸', r'떠올라',
                    r'다시\s*보여?', r'반복적?으로', r'잊을?\s*수?\s*없',
                    r'그때가?\s*생각나?', r'무서운\s*기억'
                ],
                'weight': 8,
                'risk_level': 'high'
            },
            
            # 자존감 문제
            'self_esteem_issues': {
                'patterns': [
                    r'나는?\s*안?\s*돼?', r'못생겨?', r'바보같', r'쓸모없',
                    r'실패작', r'자신없', r'확신없', r'부족해?', r'못나',
                    r'열등감', r'비교가?\s*돼?', r'초라해?'
                ],
                'weight': 4,
                'risk_level': 'medium'
            },
            
            # 관계 문제
            'relationship_issues': {
                'patterns': [
                    r'혼자', r'외로워?', r'쓸쓸해?', r'버림받', r'떠날까?\s*봐?',
                    r'잃을까?\s*봐?', r'배신', r'속았', r'거짓말', r'무시',
                    r'따돌림', r'어울리지?\s*못해?'
                ],
                'weight': 3,
                'risk_level': 'low'
            },
            
            # 긍정적 지표
            'positive_indicators': {
                'patterns': [
                    r'기뻐?', r'행복해?', r'좋아?', r'즐거워?', r'만족',
                    r'감사해?', r'성공', r'해냈', r'이뤘', r'달성',
                    r'희망', r'기대', r'계획', r'꿈'
                ],
                'weight': -2,  # 음수로 위험도 감소
                'risk_level': 'positive'
            },
            
            # 도움 요청 신호
            'help_seeking': {
                'patterns': [
                    r'도와줘?', r'어떻게\s*해야', r'방법이?\s*있을까?', r'해결',
                    r'상담', r'치료', r'병원', r'의사', r'상담사', r'도움'
                ],
                'weight': -1,  # 도움 요청은 긍정적 신호
                'risk_level': 'positive'
            }
        }
        
        # 감정 강도 지표
        self.intensity_patterns = {
            'high_intensity': [r'너무', r'정말', r'완전', r'진짜', r'엄청', r'매우', r'극도로'],
            'medium_intensity': [r'좀', r'조금', r'약간', r'살짝', r'다소', r'어느\s*정도'],
            'temporal_frequency': [r'항상', r'계속', r'자꾸', r'매번', r'늘', r'맨날', r'하루종일']
        }
        
        # 부정어 패턴
        self.negation_patterns = [r'안', r'않', r'못', r'없', r'아니', r'말고', r'빼고', r'거부', r'싫']

    def analyze_psychological_state(self, text):
        """텍스트 심리 상태 종합 분석"""
        text_lower = text.lower()
        
        # 각 패턴별 점수 계산
        pattern_scores = {}
        total_risk_score = 0
        detected_patterns = []
        
        for category, pattern_data in self.psychological_patterns.items():
            score = 0
            matches = []
            
            for pattern in pattern_data['patterns']:
                found_matches = re.findall(pattern, text_lower)
                if found_matches:
                    matches.extend(found_matches)
                    score += len(found_matches)
            
            if score > 0:
                weighted_score = score * pattern_data['weight']
                pattern_scores[category] = {
                    'raw_score': score,
                    'weighted_score': weighted_score,
                    'matches': matches,
                    'risk_level': pattern_data['risk_level']
                }
                total_risk_score += weighted_score
                detected_patterns.append(category)
        
        # 감정 강도 계산
        intensity = self._calculate_intensity(text_lower)
        
        # 부정어 영향 계산
        negation_count = sum(len(re.findall(pattern, text_lower)) for pattern in self.negation_patterns)
        has_negation = negation_count > 0
        
        # 기본 감정 분류
        primary_emotion = self._classify_primary_emotion(pattern_scores, text_lower, has_negation)
        
        # 위험도 평가
        risk_level = self._assess_risk_level(total_risk_score, pattern_scores)
        
        # 특별 관심사항 감지
        special_concerns = self._detect_special_concerns(pattern_scores)
        
        return {
            'emotion': primary_emotion,
            'confidence': min(0.9, 0.5 + len(detected_patterns) * 0.1),
            'intensity': intensity,
            'risk_level': risk_level,
            'risk_score': total_risk_score,
            'pattern_scores': pattern_scores,
            'detected_patterns': detected_patterns,
            'special_concerns': special_concerns,
            'has_negation': has_negation,
            'analysis_method': 'psychological_pattern_analysis'
        }
    
    def _calculate_intensity(self, text_lower):
        """감정 강도 계산"""
        intensity = 0.5  # 기본값
        
        # 고강도 지표
        for pattern in self.intensity_patterns['high_intensity']:
            matches = len(re.findall(pattern, text_lower))
            intensity += matches * 0.15
        
        # 시간적 빈도 (지속성)
        for pattern in self.intensity_patterns['temporal_frequency']:
            matches = len(re.findall(pattern, text_lower))
            intensity += matches * 0.1
        
        # 느낌표 개수
        exclamation_count = text_lower.count('!')
        intensity += exclamation_count * 0.05
        
        # 텍스트 길이 (긴 텍스트는 복잡한 감정)
        if len(text_lower) > 100:
            intensity += 0.1
        
        return min(1.0, intensity)
    
    def _classify_primary_emotion(self, pattern_scores, text_lower, has_negation):
        """주요 감정 분류"""
        # 자살 사고나 심각한 우울이 감지되면 슬픔으로 분류
        if 'suicidal_ideation' in pattern_scores or \
           ('depression_symptoms' in pattern_scores and pattern_scores['depression_symptoms']['weighted_score'] > 10):
            return 'sad'
        
        # 우울 증상
        if 'depression_symptoms' in pattern_scores:
            return 'sad'
        
        # 불안 증상
        if 'anxiety_symptoms' in pattern_scores:
            return 'anxiety'
        
        # 분노 증상
        if 'anger_issues' in pattern_scores:
            return 'anger'
        
        # 긍정적 지표 (부정어가 없을 때만)
        if 'positive_indicators' in pattern_scores and not has_negation:
            return 'joy'
        
        # 기본적인 키워드 분석
        if any(word in text_lower for word in ['기쁘', '좋', '행복', '즐거']) and not has_negation:
            return 'joy'
        elif any(word in text_lower for word in ['슬프', '우울', '힘들', '아파']):
            return 'sad'
        elif any(word in text_lower for word in ['불안', '걱정', '무서', '두려']):
            return 'anxiety'
        elif any(word in text_lower for word in ['화', '짜증', '분노', '열받']):
            return 'anger'
        
        return 'neutral'
    
    def _assess_risk_level(self, total_risk_score, pattern_scores):
        """위험도 평가"""
        # 자살 사고가 감지되면 즉시 고위험
        if 'suicidal_ideation' in pattern_scores:
            return 'high'
        
        # 점수 기반 위험도 평가
        if total_risk_score >= 20:
            return 'high'
        elif total_risk_score >= 10:
            return 'medium'
        else:
            return 'low'
    
    def _detect_special_concerns(self, pattern_scores):
        """특별 관심사항 감지"""
        concerns = []
        
        for category, score_data in pattern_scores.items():
            if category == 'suicidal_ideation':
                concerns.append({
                    'type': 'suicidal_ideation',
                    'severity': 'critical',
                    'message': '자살 사고가 감지되었습니다. 즉시 전문가의 도움이 필요합니다.',
                    'emergency_number': '109'
                })
            elif category == 'depression_symptoms' and score_data['weighted_score'] > 15:
                concerns.append({
                    'type': 'severe_depression',
                    'severity': 'high',
                    'message': '심각한 우울 증상이 감지되었습니다.'
                })
            elif category == 'trauma_ptsd' and score_data['weighted_score'] > 10:
                concerns.append({
                    'type': 'trauma_ptsd',
                    'severity': 'medium',
                    'message': '트라우마 관련 증상이 감지되었습니다.'
                })
            elif category == 'help_seeking':
                concerns.append({
                    'type': 'help_seeking',
                    'severity': 'positive',
                    'message': '도움을 요청하시는 건강한 신호입니다.'
                })
        
        return concerns

class LightweightAvatarGenerator:
    """경량 아바타 생성기"""
    
    def __init__(self):
        self.analyzer = PsychologicalPatternAnalyzer()
        
        # 아바타 스타일 정의 (더 아름다운 색상)
        self.avatar_styles = {
            'joy': {
                'face_color': (255, 228, 196),  # 따뜻한 피치
                'background': (255, 245, 238),  # 부드러운 크림
                'accent_color': (255, 160, 122),  # 코랄 핑크
                'message': "기쁨과 행복을 느끼고 계시는군요! 😊"
            },
            'sad': {
                'face_color': (221, 221, 238),  # 부드러운 라벤더
                'background': (240, 248, 255),  # 앨리스 블루
                'accent_color': (123, 104, 238),  # 미디움 슬레이트 블루
                'message': "슬픔을 느끼고 계시는군요. 함께 이야기해요 💙"
            },
            'anxiety': {
                'face_color': (255, 239, 213),  # 따뜻한 피치 퍼프
                'background': (253, 245, 230),  # 올드 레이스
                'accent_color': (205, 133, 63),   # 페루
                'message': "불안하신 마음이 느껴져요. 천천히 호흡해보세요 🌼"
            },
            'anger': {
                'face_color': (255, 218, 185),  # 부드러운 복숭아
                'background': (255, 240, 245),  # 라벤더 블러시
                'accent_color': (205, 92, 92),   # 인디언 레드
                'message': "화가 나셨군요. 감정을 천천히 풀어보아요 🔥"
            },
            'neutral': {
                'face_color': (250, 240, 230),  # 리넨
                'background': (248, 248, 255),  # 고스트 화이트
                'accent_color': (169, 169, 169), # 다크 그레이
                'message': "편안한 상태이신 것 같네요 ✨"
            }
        }
        
        # 위험도별 대응
        self.risk_responses = {
            'high': {
                'message': "⚠️ 심각한 심리적 어려움이 감지됩니다. 전문가의 도움을 받으시길 권합니다.",
                'emergency_contact': "자살예방상담전화: 109 (24시간)",
                'color_overlay': (255, 0, 0, 50)
            },
            'medium': {
                'message': "💛 지금 힘드신 상황이 이해됩니다. 혼자 견디지 마세요.",
                'support_message': "정신건강상담전화: 1577-0199",
                'color_overlay': (255, 165, 0, 30)
            },
            'low': {
                'message': "현재 상태가 안정적으로 보입니다 ✅",
                'color_overlay': (0, 255, 0, 20)
            }
        }
    
    def analyze_and_generate(self, text):
        """텍스트 분석 및 아바타 생성"""
        # 심리 상태 분석
        analysis = self.analyzer.analyze_psychological_state(text)
        
        # 아바타 생성
        avatar_result = self._generate_avatar(analysis)
        
        return {
            'success': True,
            'avatar_image': avatar_result['avatar_image'],
            'emotion': analysis['emotion'],
            'emotion_message': avatar_result['emotion_message'],
            'risk_level': analysis['risk_level'],
            'risk_message': avatar_result['risk_message'],
            'intensity': analysis['intensity'],
            'confidence': analysis['confidence'],
            'special_concerns': analysis['special_concerns'],
            'analysis_method': analysis['analysis_method'],
            'detected_patterns': analysis['detected_patterns'],
            'timestamp': datetime.now().isoformat()
        }
    
    def _generate_avatar(self, analysis):
        """아바타 이미지 생성 - 고급 디자인"""
        emotion = analysis['emotion']
        intensity = analysis['intensity']
        risk_level = analysis['risk_level']
        
        style = self.avatar_styles[emotion].copy()
        
        # 강도에 따른 색상 조정
        if intensity > 0.7:
            style['face_color'] = tuple(max(0, c - 20) for c in style['face_color'])
        elif intensity < 0.3:
            style['face_color'] = tuple(min(255, c + 20) for c in style['face_color'])
        
        # 이미지 생성 (고해상도)
        width, height = 500, 500
        image = Image.new('RGB', (width, height), style['background'])
        draw = ImageDraw.Draw(image)
        
        # 배경 그라데이션 효과
        self._draw_gradient_background(draw, width, height, style['background'], style['accent_color'])
        
        # 얼굴 그리기 (더 정교한 디자인)
        face_size = int(280 * (0.8 + intensity * 0.3))
        face_x = (width - face_size) // 2
        face_y = (height - face_size) // 2 - 20  # 약간 위로
        
        # 얼굴 그림자 효과
        shadow_offset = 5
        draw.ellipse([face_x + shadow_offset, face_y + shadow_offset, 
                     face_x + face_size + shadow_offset, face_y + face_size + shadow_offset], 
                    fill=(0, 0, 0, 30))
        
        # 메인 얼굴
        draw.ellipse([face_x, face_y, face_x + face_size, face_y + face_size], 
                    fill=style['face_color'], outline=style['accent_color'], width=4)
        
        # 얼굴 하이라이트
        highlight_size = face_size // 3
        highlight_x = face_x + face_size // 4
        highlight_y = face_y + face_size // 5
        draw.ellipse([highlight_x, highlight_y, highlight_x + highlight_size, highlight_y + highlight_size//2], 
                    fill=(255, 255, 255, 80))
        
        # 눈 그리기 (감정별 디자인)
        self._draw_eyes(draw, face_x, face_y, face_size, emotion, intensity)
        
        # 입 그리기 (감정별 디자인)
        self._draw_mouth(draw, face_x, face_y, face_size, emotion, intensity)
        
        # 볼 그리기 (감정별)
        self._draw_cheeks(draw, face_x, face_y, face_size, emotion, intensity)
        
        # 머리카락 추가
        self._draw_hair(draw, face_x, face_y, face_size, emotion)
        
        # 위험도 오버레이 (더 세련된 효과)
        if risk_level in self.risk_responses:
            self._apply_risk_overlay(image, risk_level)
        
        # 감정별 특수 효과
        self._add_emotion_effects(draw, width, height, emotion, intensity)
        
        # Base64 변환
        buffer = io.BytesIO()
        image.save(buffer, format='PNG', quality=95)
        buffer.seek(0)
        
        image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        
        return {
            'avatar_image': f"data:image/png;base64,{image_base64}",
            'emotion_message': style['message'],
            'risk_message': self.risk_responses[risk_level]['message']
        }
    
    def _draw_gradient_background(self, draw, width, height, bg_color, accent_color):
        """그라데이션 배경 그리기"""
        for y in range(height):
            # 수직 그라데이션
            ratio = y / height
            r = int(bg_color[0] * (1 - ratio) + accent_color[0] * ratio * 0.3)
            g = int(bg_color[1] * (1 - ratio) + accent_color[1] * ratio * 0.3)
            b = int(bg_color[2] * (1 - ratio) + accent_color[2] * ratio * 0.3)
            draw.line([(0, y), (width, y)], fill=(r, g, b))
    
    def _draw_eyes(self, draw, face_x, face_y, face_size, emotion, intensity):
        """감정별 눈 그리기"""
        eye_y = face_y + face_size // 3
        left_eye_x = face_x + face_size // 3
        right_eye_x = face_x + 2 * face_size // 3
        
        eye_width = max(20, face_size // 12)
        eye_height = max(15, face_size // 15)
        
        if emotion == 'joy':
            # 웃는 눈 (초승달 모양)
            draw.arc([left_eye_x - eye_width, eye_y - eye_height//2, 
                     left_eye_x + eye_width, eye_y + eye_height//2], 
                    0, 180, fill=(0, 0, 0), width=4)
            draw.arc([right_eye_x - eye_width, eye_y - eye_height//2, 
                     right_eye_x + eye_width, eye_y + eye_height//2], 
                    0, 180, fill=(0, 0, 0), width=4)
        elif emotion == 'sad':
            # 슬픈 눈 (아래로 처진)
            draw.ellipse([left_eye_x - eye_width//2, eye_y - eye_height//2, 
                         left_eye_x + eye_width//2, eye_y + eye_height//2], 
                        fill=(50, 50, 100))
            draw.ellipse([right_eye_x - eye_width//2, eye_y - eye_height//2, 
                         right_eye_x + eye_width//2, eye_y + eye_height//2], 
                        fill=(50, 50, 100))
            # 눈물
            if intensity > 0.6:
                tear_x = left_eye_x + eye_width//4
                tear_y = eye_y + eye_height
                draw.ellipse([tear_x - 3, tear_y, tear_x + 3, tear_y + 15], fill=(150, 200, 255))
        elif emotion == 'anxiety':
            # 불안한 눈 (넓게 뜬)
            draw.ellipse([left_eye_x - eye_width, eye_y - eye_height, 
                         left_eye_x + eye_width, eye_y + eye_height], 
                        fill=(255, 255, 255), outline=(0, 0, 0), width=2)
            draw.ellipse([right_eye_x - eye_width, eye_y - eye_height, 
                         right_eye_x + eye_width, eye_y + eye_height], 
                        fill=(255, 255, 255), outline=(0, 0, 0), width=2)
            # 동공
            draw.ellipse([left_eye_x - eye_width//3, eye_y - eye_height//3, 
                         left_eye_x + eye_width//3, eye_y + eye_height//3], fill=(0, 0, 0))
            draw.ellipse([right_eye_x - eye_width//3, eye_y - eye_height//3, 
                         right_eye_x + eye_width//3, eye_y + eye_height//3], fill=(0, 0, 0))
        elif emotion == 'anger':
            # 화난 눈 (찡그린)
            draw.polygon([left_eye_x - eye_width, eye_y - eye_height//2,
                         left_eye_x + eye_width, eye_y,
                         left_eye_x + eye_width, eye_y + eye_height//2,
                         left_eye_x - eye_width, eye_y + eye_height//2], 
                        fill=(150, 0, 0))
            draw.polygon([right_eye_x - eye_width, eye_y,
                         right_eye_x + eye_width, eye_y - eye_height//2,
                         right_eye_x + eye_width, eye_y + eye_height//2,
                         right_eye_x - eye_width, eye_y + eye_height//2], 
                        fill=(150, 0, 0))
        else:  # neutral
            # 중립적인 눈
            draw.ellipse([left_eye_x - eye_width//2, eye_y - eye_height//2, 
                         left_eye_x + eye_width//2, eye_y + eye_height//2], 
                        fill=(80, 80, 80))
            draw.ellipse([right_eye_x - eye_width//2, eye_y - eye_height//2, 
                         right_eye_x + eye_width//2, eye_y + eye_height//2], 
                        fill=(80, 80, 80))
    
    def _draw_mouth(self, draw, face_x, face_y, face_size, emotion, intensity):
        """감정별 입 그리기"""
        mouth_y = face_y + 2 * face_size // 3
        mouth_width = int(face_size // 4 * (1 + intensity * 0.5))
        mouth_center_x = face_x + face_size // 2
        
        if emotion == 'joy':
            # 웃는 입 (더 풍부한 표현)
            draw.arc([mouth_center_x - mouth_width, mouth_y - 15,
                     mouth_center_x + mouth_width, mouth_y + 25], 
                    0, 180, fill=(200, 50, 50), width=6)
            # 치아
            if intensity > 0.5:
                draw.rectangle([mouth_center_x - mouth_width//2, mouth_y - 5,
                               mouth_center_x + mouth_width//2, mouth_y + 5], 
                              fill=(255, 255, 255))
        elif emotion == 'sad':
            # 슬픈 입 (아래로 휜)
            draw.arc([mouth_center_x - mouth_width, mouth_y - 25,
                     mouth_center_x + mouth_width, mouth_y + 15], 
                    180, 360, fill=(100, 100, 150), width=5)
        elif emotion == 'anxiety':
            # 불안한 입 (작고 떨리는)
            small_width = mouth_width // 2
            draw.ellipse([mouth_center_x - small_width//2, mouth_y - 3,
                         mouth_center_x + small_width//2, mouth_y + 3], 
                        fill=(150, 150, 150))
        elif emotion == 'anger':
            # 화난 입 (찡그린)
            draw.polygon([mouth_center_x - mouth_width, mouth_y,
                         mouth_center_x, mouth_y - 10,
                         mouth_center_x + mouth_width, mouth_y], 
                        fill=(150, 0, 0))
        else:  # neutral
            # 중립적인 입
            draw.line([mouth_center_x - mouth_width//2, mouth_y,
                      mouth_center_x + mouth_width//2, mouth_y], 
                     fill=(120, 120, 120), width=4)
    
    def _draw_cheeks(self, draw, face_x, face_y, face_size, emotion, intensity):
        """볼 그리기"""
        if emotion == 'joy' and intensity > 0.4:
            # 기쁠 때 볼 빨갛게
            cheek_size = face_size // 8
            left_cheek_x = face_x + face_size // 4
            right_cheek_x = face_x + 3 * face_size // 4
            cheek_y = face_y + face_size // 2
            
            draw.ellipse([left_cheek_x - cheek_size, cheek_y - cheek_size//2,
                         left_cheek_x + cheek_size, cheek_y + cheek_size//2], 
                        fill=(255, 150, 150))
            draw.ellipse([right_cheek_x - cheek_size, cheek_y - cheek_size//2,
                         right_cheek_x + cheek_size, cheek_y + cheek_size//2], 
                        fill=(255, 150, 150))
        elif emotion == 'anger' and intensity > 0.6:
            # 화날 때 볼 빨갛게
            cheek_size = face_size // 6
            left_cheek_x = face_x + face_size // 4
            right_cheek_x = face_x + 3 * face_size // 4
            cheek_y = face_y + face_size // 2
            
            draw.ellipse([left_cheek_x - cheek_size, cheek_y - cheek_size//2,
                         left_cheek_x + cheek_size, cheek_y + cheek_size//2], 
                        fill=(200, 50, 50))
            draw.ellipse([right_cheek_x - cheek_size, cheek_y - cheek_size//2,
                         right_cheek_x + cheek_size, cheek_y + cheek_size//2], 
                        fill=(200, 50, 50))
    
    def _draw_hair(self, draw, face_x, face_y, face_size, emotion):
        """머리카락 그리기"""
        hair_color = (101, 67, 33)  # 갈색 머리
        
        # 앞머리
        bang_y = face_y - face_size // 8
        bang_width = face_size // 3
        for i in range(5):
            x_offset = (i - 2) * bang_width // 4
            draw.ellipse([face_x + face_size//2 + x_offset - bang_width//8, bang_y,
                         face_x + face_size//2 + x_offset + bang_width//8, bang_y + face_size//4], 
                        fill=hair_color)
        
        # 옆머리
        draw.ellipse([face_x - face_size//8, face_y + face_size//8,
                     face_x + face_size//4, face_y + face_size//2], 
                    fill=hair_color)
        draw.ellipse([face_x + 3*face_size//4, face_y + face_size//8,
                     face_x + face_size + face_size//8, face_y + face_size//2], 
                    fill=hair_color)
    
    def _apply_risk_overlay(self, image, risk_level):
        """위험도 오버레이 적용"""
        overlay = Image.new('RGBA', image.size, (0, 0, 0, 0))
        overlay_draw = ImageDraw.Draw(overlay)
        
        if risk_level == 'high':
            # 빨간색 경고 테두리
            overlay_draw.rectangle([0, 0, image.width, image.height], 
                                 outline=(255, 0, 0, 100), width=8)
        elif risk_level == 'medium':
            # 주황색 주의 테두리
            overlay_draw.rectangle([0, 0, image.width, image.height], 
                                 outline=(255, 165, 0, 80), width=6)
        
        # 오버레이 합성
        image.paste(Image.alpha_composite(image.convert('RGBA'), overlay))
    
    def _add_emotion_effects(self, draw, width, height, emotion, intensity):
        """감정별 특수 효과"""
        if emotion == 'joy' and intensity > 0.7:
            # 기쁨 - 반짝이는 별
            import random
            for _ in range(8):
                x = random.randint(50, width - 50)
                y = random.randint(50, height - 50)
                star_size = random.randint(8, 15)
                draw.polygon([x, y - star_size, x + star_size//2, y - star_size//3,
                             x + star_size, y, x + star_size//2, y + star_size//3,
                             x, y + star_size, x - star_size//2, y + star_size//3,
                             x - star_size, y, x - star_size//2, y - star_size//3], 
                            fill=(255, 255, 100))
        
        elif emotion == 'sad' and intensity > 0.6:
            # 슬픔 - 빗방울 효과
            import random
            for _ in range(12):
                x = random.randint(0, width)
                y = random.randint(0, height//2)
                drop_length = random.randint(20, 40)
                draw.line([x, y, x + 5, y + drop_length], 
                         fill=(150, 200, 255), width=2)
        
        elif emotion == 'anger' and intensity > 0.7:
            # 분노 - 화염 효과
            flame_x = width // 2
            flame_y = height - 50
            draw.polygon([flame_x - 20, flame_y, flame_x - 10, flame_y - 30,
                         flame_x, flame_y - 40, flame_x + 10, flame_y - 30,
                         flame_x + 20, flame_y], 
                        fill=(255, 100, 0))

# 전역 아바타 생성기
avatar_generator = LightweightAvatarGenerator()

@app.route('/health', methods=['GET'])
def health_check():
    """헬스 체크"""
    return jsonify({
        'status': 'healthy',
        'service': 'Lightweight Psychological Avatar API',
        'version': 'v1.0',
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
        
        analysis = avatar_generator.analyzer.analyze_psychological_state(text)
        
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
    """심리 상태 기반 아바타 생성"""
    try:
        data = request.get_json()
        text = data.get('text', '')
        
        if not text:
            return jsonify({'error': 'Text is required'}), 400
        
        result = avatar_generator.analyze_and_generate(text)
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Avatar generation error: {e}")
        return jsonify({'error': str(e)}), 500

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
            'youth_counseling': {
                'name': '청소년전화',
                'number': '1388',
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
        <title>심리상담 전문 아바타 API (경량버전)</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 40px; background: #f0f8ff; }
            .container { max-width: 800px; margin: 0 auto; background: white; padding: 30px; border-radius: 10px; }
            .header { text-align: center; color: #2c3e50; margin-bottom: 30px; }
            .feature { background: #f8f9fa; padding: 20px; margin: 10px 0; border-radius: 8px; }
            .emergency { background: #f8d7da; padding: 15px; border-radius: 5px; margin: 20px 0; }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>🧠 심리상담 전문 아바타 API</h1>
                <p>Lightweight Psychological Pattern Analysis System</p>
            </div>
            
            <div class="feature">
                <h3>🚀 주요 기능</h3>
                <ul>
                    <li><strong>패턴 기반 분석</strong>: 정규표현식 기반 심리 패턴 인식</li>
                    <li><strong>위험도 평가</strong>: 자살사고, 우울, 불안 등 위험 요소 감지</li>
                    <li><strong>실시간 아바타</strong>: 심리 상태에 따른 즉시 아바타 생성</li>
                    <li><strong>경량 시스템</strong>: PyTorch 없이 빠른 처리</li>
                </ul>
            </div>
            
            <div class="emergency">
                <h4>🆘 응급상황</h4>
                <p><strong>자살예방상담전화: 109 (24시간)</strong></p>
                <p><strong>정신건강상담전화: 1577-0199</strong></p>
            </div>
        </div>
    </body>
    </html>
    """)

if __name__ == '__main__':
    print("🧠 심리상담 전문 아바타 API (경량버전) 시작...")
    print("🔗 API 주소: http://localhost:8003")
    print("📋 문서: http://localhost:8003")
    
    app.run(
        host='0.0.0.0',
        port=8003,
        debug=True,
        threaded=True
    )