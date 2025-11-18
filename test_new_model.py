"""
Colab에서 가져온 best_emotion_model 테스트
"""
import sys
from pathlib import Path

# 프로젝트 루트 추가
sys.path.insert(0, str(Path(__file__).parent))

from fastapi_app.models.emotion_model_hf import EmotionClassifierHF


def test_model():
    """모델 테스트"""
    print("=" * 80)
    print("🧪 Colab 학습 모델 테스트")
    print("=" * 80)
    
    try:
        # 모델 로드
        print("\n📦 모델 로드 중...")
        model = EmotionClassifierHF(device='cpu')
        
        # 모델 정보 출력
        print("\n📊 모델 정보:")
        info = model.get_model_info()
        for key, value in info.items():
            print(f"   - {key}: {value}")
        
        # 테스트 문장들
        test_texts = [
            "오늘 정말 기분이 좋아! 너무 행복해!",
            "시험에 떨어져서 너무 슬퍼... 눈물이 나",
            "내일 발표인데 너무 불안하고 떨려",
            "이건 정말 화나는 일이야! 참을 수가 없어!",
            "오늘 점심 뭐 먹을까?",
            "ㅋㅋㅋㅋ 진짜 웃겨 ㅎㅎㅎ",
            "ㅠㅠ 너무 슬프다 ㅜㅜ",
            "존맛탱! JMT!!",
        ]
        
        print("\n" + "=" * 80)
        print("🎯 감정 분석 테스트")
        print("=" * 80)
        
        for i, text in enumerate(test_texts, 1):
            result = model.predict_emotion(text)
            
            print(f"\n[{i}] {text}")
            print(f"   🎭 감정: {result['emotion']} (신뢰도: {result['confidence']:.2%})")
            print("   📊 확률 분포:")
            for emotion, prob in sorted(result['probabilities'].items(), key=lambda x: x[1], reverse=True):
                bar = "█" * int(prob * 20)
                print(f"      - {emotion:8s}: {prob:6.2%} {bar}")
        
        print("\n" + "=" * 80)
        print("✅ 테스트 완료!")
        print("=" * 80)
        
    except Exception as e:
        print(f"\n❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_model()
