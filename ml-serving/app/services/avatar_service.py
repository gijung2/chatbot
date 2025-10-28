"""
Avatar generation service
"""
import base64
import io
import time
from PIL import Image, ImageDraw, ImageFont
from typing import Tuple

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

EMOTION_KR = {
    'joy': '기쁨',
    'sad': '슬픔',
    'anxiety': '불안',
    'anger': '분노',
    'neutral': '중립'
}

def get_emotion_message(emotion: str) -> str:
    """감정 메시지 가져오기"""
    return AVATAR_COLORS.get(emotion, AVATAR_COLORS['neutral'])['message']

def generate_avatar_image(emotion: str, style: str = "gradient") -> Tuple[str, float]:
    """
    감정별 아바타 이미지 생성
    
    Args:
        emotion: 감정
        style: 스타일 (현재는 gradient만 지원)
        
    Returns:
        (base64_image, generation_time_ms)
    """
    start_time = time.time()
    
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
                emoji = EMOTION_KR[emotion]
            except:
                font = ImageFont.load_default()
                emoji = EMOTION_KR[emotion]
        
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
        
        generation_time_ms = (time.time() - start_time) * 1000
        
        return f"data:image/png;base64,{img_base64}", generation_time_ms
        
    except Exception as e:
        print(f"⚠️ 아바타 생성 실패: {e}")
        return "", 0
