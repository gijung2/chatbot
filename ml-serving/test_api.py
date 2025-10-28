"""
Simple test script for ML serving API
"""
import requests
import json

BASE_URL = "http://localhost:8000"

def test_health():
    """Test health endpoint"""
    print("=" * 60)
    print("Testing Health Endpoint")
    print("=" * 60)
    try:
        response = requests.get(f"{BASE_URL}/api/v1/health")
        print(f"Status Code: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2, ensure_ascii=False)}")
    except Exception as e:
        print(f"Error: {e}")

def test_emotion_analysis():
    """Test emotion analysis endpoint"""
    print("\n" + "=" * 60)
    print("Testing Emotion Analysis Endpoint")
    print("=" * 60)
    
    test_cases = [
        "오늘 정말 행복해요!",
        "너무 슬프고 우울해요",
        "걱정이 너무 많아서 불안해요",
        "정말 화가 나서 미치겠어요",
        "그냥 평범한 하루였어요"
    ]
    
    for text in test_cases:
        print(f"\n📝 입력: {text}")
        try:
            response = requests.post(
                f"{BASE_URL}/api/v1/analyze",
                json={"text": text}
            )
            print(f"Status Code: {response.status_code}")
            if response.status_code == 200:
                result = response.json()
                print(f"감정: {result['emotion']}")
                print(f"신뢰도: {result['confidence']:.2%}")
                print(f"위험도: {result['risk_level']}")
                if result.get('risk_message'):
                    print(f"메시지: {result['risk_message']}")
            else:
                print(f"Error: {response.text}")
        except Exception as e:
            print(f"Error: {e}")

def test_avatar_generation():
    """Test avatar generation endpoint"""
    print("\n" + "=" * 60)
    print("Testing Avatar Generation Endpoint")
    print("=" * 60)
    
    try:
        response = requests.post(
            f"{BASE_URL}/api/v1/generate-avatar",
            json={"text": "오늘 정말 행복해요!", "style": "gradient"}
        )
        print(f"Status Code: {response.status_code}")
        if response.status_code == 200:
            result = response.json()
            print(f"감정: {result['emotion']}")
            print(f"아바타 생성 완료 (Base64 길이: {len(result['avatar_image'])})")
            print(f"처리 시간: {result['processing_time']:.3f}초")
        else:
            print(f"Error: {response.text}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    test_health()
    test_emotion_analysis()
    test_avatar_generation()
    print("\n" + "=" * 60)
    print("✅ 모든 테스트 완료!")
    print("=" * 60)
