"""
FastAPI 테스트 클라이언트
API 엔드포인트 테스트
"""
import requests
import json

BASE_URL = "http://localhost:8000"


def test_health():
    """헬스 체크 테스트"""
    print("\n" + "=" * 80)
    print("🏥 헬스 체크")
    print("=" * 80)
    
    response = requests.get(f"{BASE_URL}/health")
    print(f"Status: {response.status_code}")
    print(json.dumps(response.json(), indent=2, ensure_ascii=False))


def test_emotion_analysis():
    """감정 분석 테스트"""
    print("\n" + "=" * 80)
    print("🎭 감정 분석")
    print("=" * 80)
    
    data = {
        "text": "오늘 너무 기쁘고 행복해요!",
        "include_details": False
    }
    
    response = requests.post(f"{BASE_URL}/emotion/analyze", json=data)
    print(f"Status: {response.status_code}")
    print(json.dumps(response.json(), indent=2, ensure_ascii=False))


def test_detailed_analysis():
    """상세 분석 테스트"""
    print("\n" + "=" * 80)
    print("📊 상세 감정 분석 (위험도 평가)")
    print("=" * 80)
    
    data = {
        "text": "요즘 너무 우울하고 힘들어서 아무것도 하고 싶지 않아요"
    }
    
    response = requests.post(f"{BASE_URL}/emotion/analyze/detailed", json=data)
    print(f"Status: {response.status_code}")
    print(json.dumps(response.json(), indent=2, ensure_ascii=False))


def test_avatar_generation():
    """아바타 생성 테스트"""
    print("\n" + "=" * 80)
    print("🎨 아바타 생성")
    print("=" * 80)
    
    data = {
        "text": "너무 화가 나요!",
        "size": 400,
        "format": "png"
    }
    
    response = requests.post(f"{BASE_URL}/avatar/generate", json=data)
    print(f"Status: {response.status_code}")
    
    if response.status_code == 200:
        result = response.json()
        print(f"Emotion: {result['emotion']}")
        print(f"Confidence: {result['confidence']:.2%}")
        print(f"Image Base64 Length: {len(result['image_base64'])}")
        print(f"Metadata: {result['metadata']}")


def test_chat():
    """채팅 테스트"""
    print("\n" + "=" * 80)
    print("💬 채팅")
    print("=" * 80)
    
    data = {
        "message": "요즘 스트레스가 너무 심해요",
        "session_id": "test_user"
    }
    
    response = requests.post(f"{BASE_URL}/chat/message", json=data)
    print(f"Status: {response.status_code}")
    print(json.dumps(response.json(), indent=2, ensure_ascii=False))


def test_emergency_contacts():
    """긴급 연락처 테스트"""
    print("\n" + "=" * 80)
    print("📞 긴급 연락처")
    print("=" * 80)
    
    response = requests.get(f"{BASE_URL}/emergency-contacts")
    print(f"Status: {response.status_code}")
    print(json.dumps(response.json(), indent=2, ensure_ascii=False))


def test_model_info():
    """모델 정보 테스트"""
    print("\n" + "=" * 80)
    print("🤖 모델 정보")
    print("=" * 80)
    
    response = requests.get(f"{BASE_URL}/emotion/model-info")
    print(f"Status: {response.status_code}")
    print(json.dumps(response.json(), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    print("=" * 80)
    print("🧪 FastAPI 테스트 시작")
    print("=" * 80)
    print(f"Base URL: {BASE_URL}")
    print(f"API 문서: {BASE_URL}/docs")
    print("=" * 80)
    
    try:
        test_health()
        test_emotion_analysis()
        test_detailed_analysis()
        test_avatar_generation()
        test_chat()
        test_emergency_contacts()
        test_model_info()
        
        print("\n" + "=" * 80)
        print("✅ 모든 테스트 완료!")
        print("=" * 80)
    
    except requests.exceptions.ConnectionError:
        print("\n❌ 서버에 연결할 수 없습니다.")
        print("서버를 먼저 실행하세요: python main.py")
    
    except Exception as e:
        print(f"\n❌ 테스트 실패: {e}")
