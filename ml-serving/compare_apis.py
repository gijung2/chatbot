"""
기존 emotion_api_server.py와 새 ml-serving API 비교
"""
import requests
import json
from typing import Dict, Any

OLD_API = "http://localhost:5000"
NEW_API = "http://localhost:8000"

def compare_emotion_analysis():
    """감정 분석 API 비교"""
    print("\n" + "="*60)
    print("감정 분석 API 비교")
    print("="*60)
    
    test_texts = [
        "오늘 정말 행복해요!",
        "너무 슬프고 우울해요",
        "걱정이 너무 많아서 불안해요"
    ]
    
    for text in test_texts:
        print(f"\n📝 테스트 텍스트: {text}")
        print("-" * 60)
        
        # 기존 API
        try:
            old_response = requests.post(
                f"{OLD_API}/analyze",
                json={"text": text},
                timeout=10
            )
            if old_response.status_code == 200:
                old_data = old_response.json()
                print(f"🔴 기존 API: {old_data.get('emotion')} ({old_data.get('confidence', 0):.2%})")
            else:
                print(f"🔴 기존 API: 오류 (서버 미실행 또는 오류)")
        except Exception as e:
            print(f"🔴 기존 API: 연결 실패 - {e}")
        
        # 새 API
        try:
            new_response = requests.post(
                f"{NEW_API}/api/v1/analyze",
                json={"text": text},
                timeout=10
            )
            if new_response.status_code == 200:
                new_data = new_response.json()
                print(f"🟢 새 API: {new_data['emotion']} ({new_data['confidence']:.2%})")
                print(f"   위험도: {new_data['risk_level']}")
            else:
                print(f"🟢 새 API: 오류 {new_response.status_code}")
        except Exception as e:
            print(f"🟢 새 API: 연결 실패 - {e}")

def compare_response_structure():
    """응답 구조 비교"""
    print("\n" + "="*60)
    print("응답 구조 비교")
    print("="*60)
    
    test_text = "오늘 정말 행복해요!"
    
    print("\n🔴 기존 API 응답 구조:")
    print("-" * 60)
    try:
        old_response = requests.post(
            f"{OLD_API}/analyze",
            json={"text": test_text},
            timeout=10
        )
        if old_response.status_code == 200:
            print(json.dumps(old_response.json(), indent=2, ensure_ascii=False))
        else:
            print("서버 미실행 또는 오류")
    except Exception as e:
        print(f"연결 실패: {e}")
    
    print("\n🟢 새 API 응답 구조:")
    print("-" * 60)
    try:
        new_response = requests.post(
            f"{NEW_API}/api/v1/analyze",
            json={"text": test_text},
            timeout=10
        )
        if new_response.status_code == 200:
            print(json.dumps(new_response.json(), indent=2, ensure_ascii=False))
    except Exception as e:
        print(f"연결 실패: {e}")

if __name__ == "__main__":
    print("\n🔍 API 비교 테스트 시작")
    print("⚠️  기존 API (emotion_api_server.py)가 포트 5000에서 실행 중이어야 합니다")
    print("⚠️  새 API (ml-serving)가 포트 8000에서 실행 중이어야 합니다")
    
    compare_emotion_analysis()
    compare_response_structure()
    
    print("\n" + "="*60)
    print("✅ 비교 완료")
    print("="*60)
