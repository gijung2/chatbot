"""
아바타 생성 서비스
감정 기반 아바타 이미지 생성
"""
import base64
import io
from PIL import Image, ImageDraw, ImageFont
from typing import Dict, Tuple
import logging

logger = logging.getLogger(__name__)


class AvatarGenerator:
    """감정 기반 아바타 생성기"""
    
    def __init__(self):
        # 감정별 색상 (RGB)
        self.emotion_colors = {
            'joy': (255, 223, 0),      # 밝은 노랑
            'sad': (100, 149, 237),    # 파랑
            'anxiety': (255, 140, 0),  # 주황
            'anger': (220, 20, 60),    # 빨강
            'neutral': (169, 169, 169) # 회색
        }
        
        # 감정별 표정
        self.emotion_faces = {
            'joy': {'eyes': '^_^', 'mouth': 'smile'},
            'sad': {'eyes': 'T_T', 'mouth': 'sad'},
            'anxiety': {'eyes': '@_@', 'mouth': 'worried'},
            'anger': {'eyes': '>_<', 'mouth': 'angry'},
            'neutral': {'eyes': '-_-', 'mouth': 'neutral'}
        }
    
    def generate_avatar(
        self,
        emotion: str,
        confidence: float,
        size: int = 400
    ) -> str:
        """아바타 이미지 생성 (Base64)"""
        # 이미지 생성
        image = Image.new('RGB', (size, size), color='white')
        draw = ImageDraw.Draw(image)
        
        # 배경 원 (감정 색상)
        color = self.emotion_colors.get(emotion, (169, 169, 169))
        padding = size // 10
        draw.ellipse(
            [padding, padding, size - padding, size - padding],
            fill=color,
            outline=(50, 50, 50),
            width=3
        )
        
        # 얼굴 그리기
        self._draw_face(draw, size, emotion, confidence)
        
        # Base64 인코딩
        buffer = io.BytesIO()
        image.save(buffer, format='PNG')
        img_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        
        return img_base64
    
    def _draw_face(
        self,
        draw: ImageDraw.Draw,
        size: int,
        emotion: str,
        confidence: float
    ):
        """얼굴 표정 그리기"""
        center_x, center_y = size // 2, size // 2
        face_size = size // 3
        
        # 눈 그리기
        eye_y = center_y - face_size // 3
        eye_offset = face_size // 3
        eye_size = size // 20
        
        # 왼쪽 눈
        draw.ellipse(
            [center_x - eye_offset - eye_size, eye_y - eye_size,
             center_x - eye_offset + eye_size, eye_y + eye_size],
            fill=(0, 0, 0)
        )
        
        # 오른쪽 눈
        draw.ellipse(
            [center_x + eye_offset - eye_size, eye_y - eye_size,
             center_x + eye_offset + eye_size, eye_y + eye_size],
            fill=(0, 0, 0)
        )
        
        # 입 그리기
        mouth_y = center_y + face_size // 3
        mouth_width = face_size // 2
        
        if emotion == 'joy':
            # 웃는 입
            draw.arc(
                [center_x - mouth_width, mouth_y - 20,
                 center_x + mouth_width, mouth_y + 40],
                start=0, end=180,
                fill=(0, 0, 0), width=5
            )
        elif emotion == 'sad':
            # 슬픈 입
            draw.arc(
                [center_x - mouth_width, mouth_y - 40,
                 center_x + mouth_width, mouth_y + 20],
                start=180, end=360,
                fill=(0, 0, 0), width=5
            )
        elif emotion == 'anger':
            # 화난 입
            draw.line(
                [center_x - mouth_width, mouth_y,
                 center_x + mouth_width, mouth_y],
                fill=(0, 0, 0), width=5
            )
        elif emotion == 'anxiety':
            # 불안한 입 (물결)
            draw.arc(
                [center_x - mouth_width, mouth_y - 10,
                 center_x, mouth_y + 10],
                start=180, end=360,
                fill=(0, 0, 0), width=3
            )
            draw.arc(
                [center_x, mouth_y - 10,
                 center_x + mouth_width, mouth_y + 10],
                start=0, end=180,
                fill=(0, 0, 0), width=3
            )
        else:  # neutral
            # 평범한 입
            draw.line(
                [center_x - mouth_width // 2, mouth_y,
                 center_x + mouth_width // 2, mouth_y],
                fill=(0, 0, 0), width=3
            )
        
        # 감정 텍스트 추가
        try:
            # 한글 폰트 시도
            font = ImageFont.truetype("malgun.ttf", size // 15)
        except:
            font = ImageFont.load_default()
        
        emotion_text = f"{emotion.upper()}"
        confidence_text = f"{confidence:.1%}"
        
        # 텍스트 위치 계산
        text_y = size - size // 8
        
        # 감정 라벨
        draw.text(
            (center_x, text_y - 30),
            emotion_text,
            fill=(0, 0, 0),
            font=font,
            anchor="mm"
        )
        
        # 신뢰도
        draw.text(
            (center_x, text_y),
            confidence_text,
            fill=(50, 50, 50),
            font=font,
            anchor="mm"
        )
    
    def generate_detailed_avatar(
        self,
        analysis: Dict,
        size: int = 400
    ) -> Dict:
        """상세 분석 포함 아바타 생성"""
        emotion = analysis['emotion']
        confidence = analysis['confidence']
        
        # 기본 아바타 생성
        avatar_base64 = self.generate_avatar(emotion, confidence, size)
        
        return {
            'image_base64': avatar_base64,
            'emotion': emotion,
            'confidence': confidence,
            'color': self.emotion_colors.get(emotion),
            'size': size
        }
