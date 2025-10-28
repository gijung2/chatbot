"""
프론트엔드 웹 서버
감정 분석 + 미리 생성된 아바타 이미지 표시
"""

from flask import Flask, render_template, request, jsonify, send_from_directory
import requests
import json
import random
import os
from datetime import datetime

app = Flask(__name__)

# 설정
EMOTION_SERVER_URL = "http://localhost:5000"
AVATAR_PATH = "public/avatars"

# 감정별 사용 가능한 스타일 (Colab에서 생성된 이미지 개수)
AVAILABLE_STYLES = {
    "joy": ["style_1.png", "style_2.png", "style_3.png"],
    "sad": ["style_1.png", "style_2.png", "style_3.png"],
    "anxiety": ["style_1.png", "style_2.png", "style_3.png"],
    "anger": ["style_1.png", "style_2.png", "style_3.png"],
    "neutral": ["style_1.png", "style_2.png", "style_3.png"]
}

@app.route('/')
def index():
    """메인 페이지"""
    return render_template('index.html')

@app.route('/avatars/<emotion>/<filename>')
def serve_avatar(emotion, filename):
    """아바타 이미지 서빙"""
    avatar_dir = os.path.join(AVATAR_PATH, emotion)
    return send_from_directory(avatar_dir, filename)

@app.route('/api/analyze', methods=['POST'])
def analyze_text():
    """텍스트 감정 분석 + 아바타 이미지 반환"""
    try:
        data = request.get_json()
        text = data.get('text', '')
        
        if not text:
            return jsonify({'error': 'Text is required'}), 400
        
        # 감정 분석 API 호출
        emotion_response = requests.post(
            f"{EMOTION_SERVER_URL}/analyze",
            json={'text': text},
            timeout=10
        )
        
        if emotion_response.status_code != 200:
            return jsonify({'error': 'Emotion analysis failed'}), 500
        
        emotion_result = emotion_response.json()
        emotion = emotion_result.get('emotion', 'neutral')
        
        # 해당 감정의 랜덤 스타일 선택
        available_styles = AVAILABLE_STYLES.get(emotion, ['style_1.png'])
        selected_style = random.choice(available_styles)
        
        # 아바타 이미지 URL 생성
        avatar_url = f"/avatars/{emotion}/{selected_style}"
        
        return jsonify({
            'text': text,
            'emotion': emotion,
            'emotion_kr': emotion_result.get('emotion_kr', emotion),
            'confidence': emotion_result.get('confidence', 0.0),
            'method': emotion_result.get('method', 'unknown'),
            'avatar_url': avatar_url,
            'avatar_style': selected_style.replace('.png', ''),
            'available_styles': len(available_styles),
            'timestamp': datetime.now().isoformat()
        })
        
    except requests.exceptions.RequestException as e:
        return jsonify({'error': f'Connection error: {str(e)}'}), 500
    except Exception as e:
        return jsonify({'error': f'Server error: {str(e)}'}), 500

@app.route('/api/avatar/<emotion>')
def get_random_avatar(emotion):
    """특정 감정의 랜덤 아바타 이미지 URL 반환"""
    if emotion not in AVAILABLE_STYLES:
        return jsonify({'error': 'Invalid emotion'}), 400
    
    available_styles = AVAILABLE_STYLES[emotion]
    selected_style = random.choice(available_styles)
    avatar_url = f"/avatars/{emotion}/{selected_style}"
    
    return jsonify({
        'emotion': emotion,
        'avatar_url': avatar_url,
        'style': selected_style.replace('.png', ''),
        'total_styles': len(available_styles)
    })

@app.route('/api/emotions')
def get_emotions():
    """사용 가능한 모든 감정과 스타일 정보 반환"""
    return jsonify({
        'emotions': list(AVAILABLE_STYLES.keys()),
        'styles_info': {
            emotion: {
                'count': len(styles),
                'files': styles
            }
            for emotion, styles in AVAILABLE_STYLES.items()
        }
    })

@app.route('/gallery')
def gallery():
    """아바타 갤러리 페이지"""
    return render_template('gallery.html', emotions=AVAILABLE_STYLES)

@app.route('/test')
def test_page():
    """테스트 페이지"""
    return render_template('test.html')

if __name__ == '__main__':
    # 정적 파일 디렉토리 확인
    if not os.path.exists(AVATAR_PATH):
        print(f"⚠️  Avatar directory not found: {AVATAR_PATH}")
        print("Colab에서 생성된 이미지들을 해당 디렉토리에 저장해주세요.")
    else:
        print(f"✅ Avatar directory found: {AVATAR_PATH}")
    
    print("🌐 프론트엔드 서버 시작...")
    print("=" * 50)
    print("📝 사용 방법:")
    print("  1. 메인 페이지: http://localhost:3000")
    print("  2. 갤러리: http://localhost:3000/gallery")
    print("  3. 테스트: http://localhost:3000/test")
    print("  4. API: http://localhost:3000/api/analyze")
    print("=" * 50)
    
    app.run(
        host='0.0.0.0',
        port=3000,
        debug=True
    )
