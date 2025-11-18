"""
FastAPI 감정 분석 API 테스트
"""
import requests
import json


def test_emotion_api():
    """감정 분석 API 테스트"""
    base_url = "http://localhost:8000"
    
    print("=" * 80)
    print("🧪 FastAPI 감정 분석 API 테스트")
    print("=" * 80)
    
    # 1. Health Check
    print("\n1️⃣ Health Check")
    response = requests.get(f"{base_url}/health")
    print(f"   Status: {response.status_code}")
    print(f"   Response: {json.dumps(response.json(), indent=2, ensure_ascii=False)}")
    
    # 2. 기본 감정 분석
    print("\n2️⃣ 기본 감정 분석")
    test_texts = [
        "오늘 정말 기분이 좋아! 너무 행복해!",
        "시험에 떨어져서 너무 슬퍼...",
        "내일 발표인데 너무 불안하고 떨려",
        "이건 정말 화나는 일이야!",
        "오늘 점심 뭐 먹을까?",
    ]
    
    for text in test_texts:
        response = requests.post(
            f"{base_url}/emotion/analyze",
            json={"text": text}
        )
        result = response.json()
        print(f"\n   텍스트: {text}")
        print(f"   감정: {result['emotion']} (신뢰도: {result['confidence']:.2%})")
        print(f"   확률 분포:")
        for emotion, prob in sorted(result['probabilities'].items(), key=lambda x: x[1], reverse=True)[:3]:
            print(f"      - {emotion}: {prob:.2%}")
    
    # 3. 상세 감정 분석 (위험도 평가 포함)
    print("\n3️⃣ 상세 감정 분석 (위험도 평가)")
    detailed_tests = [
        "죽고 싶어... 더 이상 살아갈 이유가 없어",
        "요즘 우울하고 아무것도 하기 싫어",
        "오늘 맛있는 거 먹어서 행복해!",
    ]
    
    for text in detailed_tests:
        response = requests.post(
            f"{base_url}/emotion/analyze/detailed",
            json={"text": text}
        )
        result = response.json()
        print(f"\n   텍스트: {text}")
        print(f"   감정: {result['emotion']} (신뢰도: {result['confidence']:.2%})")
        print(f"   위험도: {result['risk_assessment']['level']} ({result['risk_assessment']['score']}/10)")
        print(f"   위험 요인: {', '.join(result['risk_assessment']['risk_factors']) if result['risk_assessment']['risk_factors'] else '없음'}")
        if result['counseling_suggestions']:
            print(f"   상담 제안:")
            for suggestion in result['counseling_suggestions'][:2]:
                print(f"      - {suggestion}")
    
    # 4. 모델 정보
    print("\n4️⃣ 모델 정보")
    response = requests.get(f"{base_url}/emotion/model-info")
    info = response.json()
    print(f"   모델 타입: {info['model_type']}")
    print(f"   감정 클래스: {', '.join(info['emotion_labels'])}")
    print(f"   총 파라미터: {info['total_parameters']:,}")
    print(f"   디바이스: {info['device']}")
    
    print("\n" + "=" * 80)
    print("✅ 테스트 완료!")
    print("=" * 80)


if __name__ == "__main__":
    try:
        test_emotion_api()
    except requests.exceptions.ConnectionError:
        print("❌ 서버가 실행되지 않았습니다.")
        print("   먼저 'python fastapi_app/main.py'로 서버를 실행하세요.")
    except Exception as e:
        print(f"❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()
