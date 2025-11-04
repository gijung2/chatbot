"""
학습한 모델 통합 테스트 스크립트
모델이 정상적으로 로드되고 예측이 작동하는지 확인
"""
import sys
from pathlib import Path

# 프로젝트 루트를 Python 경로에 추가
sys.path.insert(0, str(Path(__file__).parent))

from fastapi_app.models.emotion_model_hf import EmotionClassifierHF


def test_model_loading():
    """모델 로드 테스트"""
    print("=" * 80)
    print("🧪 모델 로드 테스트")
    print("=" * 80)
    
    try:
        # 모델 초기화 (최신 모델 자동 탐색)
        model = EmotionClassifierHF(device='cpu')
        print("✅ 모델 로드 성공!\n")
        
        # 모델 정보 출력
        info = model.get_model_info()
        print("📊 모델 정보:")
        print(f"   - 모델 타입: {info['model_type']}")
        print(f"   - 총 파라미터: {info['total_parameters']:,}")
        print(f"   - 학습 가능 파라미터: {info['trainable_parameters']:,}")
        print(f"   - 감정 클래스: {info['emotion_labels']}")
        print(f"   - 어휘 크기: {info['tokenizer_vocab_size']:,}")
        print(f"   - Device: {info['device']}\n")
        
        return model
    
    except Exception as e:
        print(f"❌ 모델 로드 실패: {e}")
        print("\n💡 해결 방법:")
        print("   1. checkpoints_kfold/ 폴더에 학습된 모델이 있는지 확인")
        print("   2. 모델 폴더 이름이 'fold*_best_model_*' 형식인지 확인")
        print("   3. 또는 model_path를 직접 지정:")
        print("      model = EmotionClassifierHF(model_path='경로/to/모델')")
        return None


def test_single_prediction(model):
    """단일 예측 테스트"""
    print("=" * 80)
    print("🧪 단일 감정 예측 테스트")
    print("=" * 80)
    
    test_cases = [
        "오늘 정말 기쁜 일이 있었어요!",
        "너무 슬프고 우울해요...",
        "시험이 다가와서 불안하네요",
        "이런 일에 화가 나네요",
        "오늘 날씨가 좋네요"
    ]
    
    expected_emotions = ["joy", "sad", "anxiety", "anger", "neutral"]
    
    print("\n📝 테스트 케이스:")
    correct = 0
    
    for i, (text, expected) in enumerate(zip(test_cases, expected_emotions), 1):
        result = model.predict_emotion(text)
        
        is_correct = result['emotion'] == expected
        if is_correct:
            correct += 1
        
        status = "✅" if is_correct else "⚠️"
        
        print(f"\n{i}. {text}")
        print(f"   {status} 예측: {result['emotion']} (신뢰도: {result['confidence']:.2%})")
        print(f"   기대: {expected}")
        
        # 확률 분포
        print("   확률 분포:")
        for emotion, prob in sorted(result['probabilities'].items(), key=lambda x: x[1], reverse=True)[:3]:
            bar = "█" * int(prob * 20)
            print(f"      {emotion:8s}: {bar} {prob:.2%}")
    
    accuracy = correct / len(test_cases)
    print(f"\n📊 정확도: {correct}/{len(test_cases)} ({accuracy:.0%})")
    
    if accuracy >= 0.8:
        print("✅ 모델이 정상적으로 작동합니다!")
    else:
        print("⚠️ 정확도가 낮습니다. 모델을 다시 확인하세요.")


def test_batch_prediction(model):
    """배치 예측 테스트"""
    print("\n" + "=" * 80)
    print("🧪 배치 감정 예측 테스트")
    print("=" * 80)
    
    texts = [
        "정말 행복한 하루였어요",
        "너무 힘들고 지쳐요",
        "걱정이 많아요"
    ]
    
    print("\n📝 배치 예측 (3개 텍스트):")
    results = model.predict_batch(texts)
    
    for text, result in zip(texts, results):
        print(f"\n   텍스트: {text}")
        print(f"   결과: {result['emotion']} (신뢰도: {result['confidence']:.2%})")
    
    print("\n✅ 배치 예측이 정상적으로 작동합니다!")


def test_edge_cases(model):
    """엣지 케이스 테스트"""
    print("\n" + "=" * 80)
    print("🧪 엣지 케이스 테스트")
    print("=" * 80)
    
    edge_cases = [
        ("", "빈 문자열"),
        ("ㅋㅋㅋㅋㅋ", "자음만"),
        ("123456", "숫자만"),
        ("!!!!!!", "특수문자만"),
        ("very long text " * 50, "매우 긴 텍스트"),
    ]
    
    print("\n📝 엣지 케이스:")
    
    for text, description in edge_cases:
        try:
            display_text = text[:30] + "..." if len(text) > 30 else text
            result = model.predict_emotion(text)
            print(f"\n   ✅ {description}: {display_text}")
            print(f"      예측: {result['emotion']} (신뢰도: {result['confidence']:.2%})")
        except Exception as e:
            print(f"\n   ❌ {description}: {e}")


def main():
    """메인 테스트 실행"""
    print("\n")
    print("█" * 80)
    print("█" + " " * 78 + "█")
    print("█" + "  🤖 학습한 모델 통합 테스트  ".center(78) + "█")
    print("█" + " " * 78 + "█")
    print("█" * 80)
    print("\n")
    
    # 1. 모델 로드
    model = test_model_loading()
    if model is None:
        print("\n❌ 모델을 로드할 수 없어 테스트를 중단합니다.")
        return
    
    # 2. 단일 예측 테스트
    test_single_prediction(model)
    
    # 3. 배치 예측 테스트
    test_batch_prediction(model)
    
    # 4. 엣지 케이스 테스트
    test_edge_cases(model)
    
    # 최종 결과
    print("\n" + "=" * 80)
    print("🎉 모든 테스트 완료!")
    print("=" * 80)
    print("\n✅ 다음 단계:")
    print("   1. FastAPI 서버 시작: python fastapi_app/main.py")
    print("   2. API 문서 확인: http://localhost:8000/docs")
    print("   3. 테스트 API 호출")
    print("\n")


if __name__ == "__main__":
    main()
