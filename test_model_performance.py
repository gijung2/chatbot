"""
감정 분류 모델 성능 테스트 스크립트

사용법:
    python test_model_performance.py

옵션:
    --model_path: 모델 경로 (기본: checkpoints_kfold)
    --detailed: 상세 결과 출력
"""

import sys
import argparse
from pathlib import Path

# FastAPI 앱 경로 추가
sys.path.insert(0, str(Path(__file__).parent / 'fastapi_app'))

from fastapi_app.models.emotion_model_hf import EmotionClassifierHF


# 테스트 케이스 (다양한 감정 표현)
TEST_CASES = [
    # Joy (기쁨)
    ("오늘 정말 행복해요!", "joy"),
    ("완전 기분 좋아!", "joy"),
    ("너무 기뻐서 날아갈 것 같아요", "joy"),
    ("와! 대박! 정말 좋아요!", "joy"),
    ("행복한 하루였어요", "joy"),
    
    # Sad (슬픔)
    ("너무 슬퍼서 눈물이 나요", "sad"),
    ("우울해 죽겠어", "sad"),
    ("마음이 아파요", "sad"),
    ("슬픈 일이 있었어요", "sad"),
    ("기분이 너무 다운돼요", "sad"),
    
    # Anxiety (불안)
    ("시험이 걱정돼요", "anxiety"),
    ("떨려요 너무", "anxiety"),
    ("불안해서 잠을 못 자요", "anxiety"),
    ("무서워요", "anxiety"),
    ("걱정이 많아요", "anxiety"),
    
    # Anger (분노)
    ("화가 나서 미칠 것 같아요", "anger"),
    ("짜증나!", "anger"),
    ("정말 열받아요", "anger"),
    ("너무 화나서 말도 못하겠어요", "anger"),
    ("진짜 빡쳐요", "anger"),
    
    # Neutral (중립)
    ("그냥 그래요", "neutral"),
    ("별로 특별한 일 없어요", "neutral"),
    ("평범한 하루예요", "neutral"),
    ("그저 그래요", "neutral"),
    ("뭐 그냥 보통이에요", "neutral"),
    
    # Edge Cases (경계 케이스)
    ("기쁘면서도 슬퍼요", "neutral"),  # 혼합 감정
    ("화나는데 걱정도 돼요", "anxiety"),  # 복합 감정
    ("", "neutral"),  # 빈 문자열
    ("ㅋㅋㅋㅋ", "joy"),  # 이모티콘
    ("ㅠㅠㅠㅠ", "sad"),  # 이모티콘
]


def test_model(model_path: str = None, detailed: bool = False):
    """모델 성능 테스트"""
    
    print("=" * 80)
    print("🧪 감정 분류 모델 성능 테스트")
    print("=" * 80)
    
    # 모델 로드
    try:
        print(f"\n📦 모델 로드 중...")
        model = EmotionClassifierHF(model_path=model_path)
        print(f"✅ 모델 로드 완료: {model_path or '기본 경로'}\n")
    except Exception as e:
        print(f"❌ 모델 로드 실패: {e}")
        return
    
    # 모델 정보 출력
    if detailed:
        info = model.get_model_info()
        print("📊 모델 정보:")
        print(f"   - 모델 타입: {info['model_type']}")
        print(f"   - 파라미터 수: {info['total_parameters']:,}")
        print(f"   - Device: {info['device']}")
        print(f"   - 감정 클래스: {', '.join(info['emotion_labels'])}")
        print()
    
    # 테스트 실행
    print("=" * 80)
    print("📝 테스트 케이스 실행")
    print("=" * 80)
    
    results = {
        'total': 0,
        'correct': 0,
        'by_emotion': {emotion: {'total': 0, 'correct': 0} for emotion in model.emotion_labels}
    }
    
    confidences = []
    
    for text, expected in TEST_CASES:
        if not text:  # 빈 문자열 스킵
            continue
            
        result = model.predict_emotion(text)
        predicted = result['emotion']
        confidence = result['confidence']
        
        is_correct = predicted == expected
        results['total'] += 1
        results['by_emotion'][expected]['total'] += 1
        
        if is_correct:
            results['correct'] += 1
            results['by_emotion'][expected]['correct'] += 1
        
        confidences.append(confidence)
        
        # 결과 출력
        status = "✅" if is_correct else "❌"
        print(f"\n{status} \"{text}\"")
        print(f"   예측: {predicted} ({confidence:.2%} 신뢰도)")
        
        if not is_correct:
            print(f"   정답: {expected}")
        
        if detailed:
            print(f"   확률 분포:")
            for emotion, prob in sorted(result['probabilities'].items(), key=lambda x: -x[1]):
                print(f"      - {emotion}: {prob:.2%}")
    
    # 종합 결과
    print("\n" + "=" * 80)
    print("📊 종합 결과")
    print("=" * 80)
    
    overall_accuracy = results['correct'] / results['total'] * 100
    avg_confidence = sum(confidences) / len(confidences)
    
    print(f"\n🎯 전체 정확도: {overall_accuracy:.1f}% ({results['correct']}/{results['total']})")
    print(f"📈 평균 신뢰도: {avg_confidence:.2%}")
    
    # 감정별 정확도
    print(f"\n📋 감정별 정확도:")
    for emotion in model.emotion_labels:
        total = results['by_emotion'][emotion]['total']
        correct = results['by_emotion'][emotion]['correct']
        
        if total > 0:
            accuracy = correct / total * 100
            print(f"   - {emotion:8s}: {accuracy:5.1f}% ({correct}/{total})")
        else:
            print(f"   - {emotion:8s}: N/A (테스트 케이스 없음)")
    
    # 성능 평가
    print("\n" + "=" * 80)
    print("💡 성능 평가")
    print("=" * 80)
    
    if overall_accuracy >= 95:
        grade = "🏆 우수 (Excellent)"
        message = "모델이 매우 잘 작동합니다!"
    elif overall_accuracy >= 90:
        grade = "🥇 좋음 (Good)"
        message = "모델이 잘 작동합니다."
    elif overall_accuracy >= 80:
        grade = "🥈 보통 (Fair)"
        message = "추가 학습이 권장됩니다."
    else:
        grade = "🥉 개선 필요 (Needs Improvement)"
        message = "모델 재학습이 필요합니다."
    
    print(f"\n등급: {grade}")
    print(f"평가: {message}")
    
    if avg_confidence < 0.7:
        print(f"\n⚠️ 평균 신뢰도가 낮습니다 ({avg_confidence:.2%})")
        print("   → KOTE 데이터로 재학습 권장")
    
    # 개선 제안
    print("\n📌 개선 제안:")
    
    weak_emotions = []
    for emotion in model.emotion_labels:
        total = results['by_emotion'][emotion]['total']
        correct = results['by_emotion'][emotion]['correct']
        if total > 0 and correct / total < 0.8:
            weak_emotions.append(emotion)
    
    if weak_emotions:
        print(f"   - 약한 클래스: {', '.join(weak_emotions)}")
        print(f"   → 해당 클래스 데이터 증강 권장")
    
    if overall_accuracy < 95:
        print(f"   - KOTE 데이터로 재학습 (176K samples)")
        print(f"   - 하이퍼파라미터 튜닝 (epochs=15, lr=2e-5)")
        print(f"   - Label smoothing 적용")
    
    if avg_confidence < 0.8:
        print(f"   - Temperature scaling 적용")
        print(f"   - Threshold 조정")
    
    print("\n" + "=" * 80)
    print(f"상세 가이드: PERFORMANCE_IMPROVEMENT_GUIDE.md")
    print("=" * 80)
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="감정 분류 모델 성능 테스트")
    parser.add_argument(
        '--model_path',
        type=str,
        default=None,
        help='모델 경로 (기본: checkpoints_kfold)'
    )
    parser.add_argument(
        '--detailed',
        action='store_true',
        help='상세 결과 출력'
    )
    
    args = parser.parse_args()
    
    test_model(model_path=args.model_path, detailed=args.detailed)
