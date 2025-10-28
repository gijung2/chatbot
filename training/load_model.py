"""
학습된 모델 로드 및 사용 유틸리티

사용법:
    python training/load_model.py --model_path checkpoints_kfold/fold1_model_20251028_113127.pt
"""
import argparse
import torch
from transformers import AutoTokenizer
from model import create_model
import numpy as np


def load_trained_model(model_path: str, device: str = 'cpu'):
    """
    학습된 모델 로드
    
    Args:
        model_path: 모델 체크포인트 경로
        device: 'cuda' 또는 'cpu'
    
    Returns:
        model: 로드된 모델
        tokenizer: 토크나이저
        config: 모델 설정 정보
    """
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    
    # 체크포인트 로드
    print(f"📂 체크포인트 로드 중: {model_path}")
    checkpoint = torch.load(model_path, map_location=device)
    
    # 모델 설정 확인
    model_config = checkpoint.get('model_config', {})
    print(f"\n📊 모델 설정:")
    for key, value in model_config.items():
        print(f"   - {key}: {value}")
    
    # 토크나이저 로드
    model_name = model_config.get('model_name', 'klue/bert-base')
    print(f"\n🔤 토크나이저 로드: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # 모델 생성
    print(f"\n🤖 모델 생성 중...")
    model = create_model(
        model_name=model_name,
        num_labels=model_config.get('num_labels', 5),
        dropout_rate=0.3,
        freeze_bert=False,
        device=device
    )
    
    # 가중치 로드
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"✅ 모델 로드 완료!")
    print(f"🖥️ 디바이스: {device}")
    
    # 학습 히스토리 출력 (있는 경우)
    if 'val_acc_history' in checkpoint:
        print(f"\n📈 학습 히스토리:")
        print(f"   - Best Val Accuracy: {max(checkpoint['val_acc_history']):.4f}")
        print(f"   - Best Val F1: {max(checkpoint['val_f1_history']):.4f}")
        print(f"   - Final Val Loss: {checkpoint['val_loss_history'][-1]:.4f}")
    
    return model, tokenizer, model_config


def predict_emotion(text: str, model, tokenizer, device, max_length: int = 128):
    """
    입력 텍스트의 감정을 예측
    
    Args:
        text: 예측할 텍스트
        model: 학습된 모델
        tokenizer: 토크나이저
        device: 디바이스
        max_length: 최대 시퀀스 길이
    
    Returns:
        predicted_label: 예측된 감정 라벨 (0-4)
        probabilities: 각 클래스별 확률
        emotion_name: 감정 이름
        korean_name: 한글 감정 이름
    """
    # 감정 매핑
    emotion_map = {
        0: ('joy', '기쁨'),
        1: ('sad', '슬픔'),
        2: ('anxiety', '불안'),
        3: ('anger', '분노'),
        4: ('neutral', '중립')
    }
    
    # 텍스트 토큰화
    encoding = tokenizer(
        text,
        max_length=max_length,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )
    
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)
    
    # 예측
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs['logits']
        probabilities = torch.softmax(logits, dim=-1)
        predicted_label = torch.argmax(probabilities, dim=-1).item()
    
    probs = probabilities[0].cpu().numpy()
    emotion_eng, emotion_kor = emotion_map[predicted_label]
    
    return predicted_label, probs, emotion_eng, emotion_kor


def main():
    """메인 함수"""
    parser = argparse.ArgumentParser(description='학습된 모델 로드 및 테스트')
    parser.add_argument('--model_path', type=str, required=True,
                       help='모델 체크포인트 경로')
    parser.add_argument('--device', type=str, default='cpu',
                       choices=['cpu', 'cuda'],
                       help='디바이스 선택')
    parser.add_argument('--interactive', action='store_true',
                       help='대화형 모드로 실행')
    
    args = parser.parse_args()
    
    # 모델 로드
    model, tokenizer, config = load_trained_model(args.model_path, args.device)
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    
    # 테스트 예시
    test_texts = [
        "오늘 정말 기분이 좋아!",
        "너무 슬프고 우울해...",
        "시험 결과가 걱정돼서 잠이 안 와.",
        "정말 화가 나서 참을 수가 없어!",
        "오늘 날씨가 맑네요."
    ]
    
    if not args.interactive:
        # 테스트 예시 실행
        print("\n" + "=" * 80)
        print("🧪 테스트 예시")
        print("=" * 80)
        
        for text in test_texts:
            label, probs, emotion_eng, emotion_kor = predict_emotion(
                text, model, tokenizer, device
            )
            
            print(f"\n📝 텍스트: {text}")
            print(f"🎭 예측 감정: {emotion_kor} ({emotion_eng}) [라벨: {label}]")
            print(f"📊 확률 분포:")
            for i, (eng, kor) in enumerate([('joy', '기쁨'), ('sad', '슬픔'), 
                                            ('anxiety', '불안'), ('anger', '분노'), 
                                            ('neutral', '중립')]):
                bar = '█' * int(probs[i] * 50)
                print(f"   {kor:4s}: {bar} {probs[i]:.4f}")
    else:
        # 대화형 모드
        print("\n" + "=" * 80)
        print("💬 대화형 감정 분석 모드")
        print("=" * 80)
        print("텍스트를 입력하세요 (종료: 'quit' 또는 'exit')")
        
        while True:
            print("\n" + "-" * 80)
            text = input("📝 입력: ").strip()
            
            if text.lower() in ['quit', 'exit', 'q']:
                print("👋 종료합니다.")
                break
            
            if not text:
                continue
            
            label, probs, emotion_eng, emotion_kor = predict_emotion(
                text, model, tokenizer, device
            )
            
            print(f"\n🎭 예측 감정: {emotion_kor} ({emotion_eng})")
            print(f"📊 확률 분포:")
            for i, (eng, kor) in enumerate([('joy', '기쁨'), ('sad', '슬픔'), 
                                            ('anxiety', '불안'), ('anger', '분노'), 
                                            ('neutral', '중립')]):
                bar = '█' * int(probs[i] * 30)
                print(f"   {kor:4s}: {bar:30s} {probs[i]:.2%}")


if __name__ == '__main__':
    main()
