"""
감성대화말뭉치 전처리 스크립트
Training.json + Validation.json → K-Fold용 전체 데이터셋 생성
"""
import json
import pandas as pd
from pathlib import Path
from typing import List, Dict
import re

# 감정 코드 매핑 (14개 → 5개 클래스)
EMOTION_MAPPING = {
    # anger (분노)
    'E10': 'anger',  # 분노
    'E18': 'anger',  # 짜증
    'E19': 'anger',  # 툴툴거림
    
    # sad (슬픔)
    'E22': 'sad',    # 슬픔
    'E40': 'sad',    # 실망
    'E49': 'sad',    # 억울함
    'E56': 'sad',    # 괴로움
    
    # anxiety (불안)
    'E25': 'anxiety',  # 당황
    'E30': 'anxiety',  # 두려움
    'E31': 'anxiety',  # 긴장
    'E35': 'anxiety',  # 걱정
    'E37': 'anxiety',  # 안절부절못함
    'E50': 'anxiety',  # 초조
    
    # joy (기쁨)
    'E64': 'joy',     # 기쁨
    
    # neutral (중립)
    'E66': 'neutral'  # 편안
}

# 라벨 매핑
LABEL2ID = {'joy': 0, 'sad': 1, 'anxiety': 2, 'anger': 3, 'neutral': 4}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}


def load_json_data(file_path: str) -> List[Dict]:
    """JSON 파일 로드"""
    print(f"📂 로딩: {file_path}")
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    print(f"   ✅ {len(data):,}개 대화 로드")
    return data


def extract_conversations(data: List[Dict]) -> List[Dict]:
    """대화 데이터에서 텍스트 추출"""
    conversations = []
    
    for item in data:
        try:
            # 감정 타입 추출
            emotion_type = item['profile']['emotion']['type']
            
            # 매핑되지 않은 감정은 제외
            if emotion_type not in EMOTION_MAPPING:
                continue
            
            emotion_label = EMOTION_MAPPING[emotion_type]
            
            # 대화 내용 추출 (HS01, HS02, HS03 - 사용자 발화)
            content = item['talk']['content']
            
            # 각 사용자 발화를 개별 샘플로
            for key in ['HS01', 'HS02', 'HS03']:
                if key in content:
                    text = content[key].strip()
                    
                    # 빈 텍스트 제외
                    if not text:
                        continue
                    
                    conversations.append({
                        'text': text,
                        'emotion': emotion_label,
                        'label_id': LABEL2ID[emotion_label],
                        'emotion_code': emotion_type
                    })
        
        except (KeyError, TypeError) as e:
            continue
    
    return conversations


def clean_text(text: str) -> str:
    """텍스트 정제"""
    # 연속된 공백 제거
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def preprocess_corpus():
    """전체 전처리 파이프라인"""
    print("=" * 80)
    print("📊 감성대화말뭉치 전처리 시작")
    print("=" * 80)
    
    # 경로 설정
    raw_dir = Path(__file__).parent / 'raw'
    processed_dir = Path(__file__).parent / 'processed'
    processed_dir.mkdir(exist_ok=True)
    
    training_file = raw_dir / '감성대화말뭉치(최종데이터)_Training.json'
    validation_file = raw_dir / '감성대화말뭉치(최종데이터)_Validation.json'
    
    # 1. Training.json 로드 및 추출
    print("\n" + "=" * 80)
    print("📁 Training 데이터 처리")
    print("=" * 80)
    training_data = load_json_data(str(training_file))
    training_conversations = extract_conversations(training_data)
    print(f"   ✅ {len(training_conversations):,}개 샘플 추출")
    
    # 2. Validation.json 로드 및 추출
    print("\n" + "=" * 80)
    print("📁 Validation 데이터 처리")
    print("=" * 80)
    validation_data = load_json_data(str(validation_file))
    validation_conversations = extract_conversations(validation_data)
    print(f"   ✅ {len(validation_conversations):,}개 샘플 추출")
    
    # 3. 데이터 합치기
    print("\n" + "=" * 80)
    print("🔗 데이터 통합")
    print("=" * 80)
    all_conversations = training_conversations + validation_conversations
    print(f"   ✅ 총 {len(all_conversations):,}개 샘플")
    
    # 4. DataFrame 생성
    df = pd.DataFrame(all_conversations)
    
    # 텍스트 정제
    print("\n🧹 텍스트 정제 중...")
    df['text'] = df['text'].apply(clean_text)
    
    # 중복 제거
    before_count = len(df)
    df = df.drop_duplicates(subset=['text'], keep='first')
    after_count = len(df)
    print(f"   - 중복 제거: {before_count:,} → {after_count:,} ({before_count - after_count:,}개 제거)")
    
    # 5. 클래스 분포 확인
    print("\n" + "=" * 80)
    print("📊 클래스 분포")
    print("=" * 80)
    print(f"{'클래스':<15} {'샘플 수':>10} {'비율':>10}")
    print("-" * 40)
    for label in sorted(LABEL2ID.keys()):
        count = (df['emotion'] == label).sum()
        percentage = count / len(df) * 100
        print(f"{label:<15} {count:>10,} {percentage:>9.1f}%")
    print("-" * 40)
    print(f"{'전체':<15} {len(df):>10,} {100.0:>9.1f}%")
    
    # 6. 저장
    output_file = processed_dir / 'emotion_corpus_full.csv'
    df.to_csv(output_file, index=False, encoding='utf-8')
    
    print("\n" + "=" * 80)
    print("💾 저장 완료")
    print("=" * 80)
    print(f"   - 파일: {output_file}")
    print(f"   - 크기: {output_file.stat().st_size / (1024*1024):.2f} MB")
    print(f"   - 샘플 수: {len(df):,}")
    print(f"   - 컬럼: {list(df.columns)}")
    
    # 7. 메타데이터 저장
    metadata = {
        'total_samples': len(df),
        'num_classes': len(LABEL2ID),
        'class_distribution': df['emotion'].value_counts().to_dict(),
        'label2id': LABEL2ID,
        'id2label': ID2LABEL,
        'emotion_mapping': EMOTION_MAPPING,
        'columns': list(df.columns),
        'source_files': [
            '감성대화말뭉치(최종데이터)_Training.json',
            '감성대화말뭉치(최종데이터)_Validation.json'
        ]
    }
    
    metadata_file = processed_dir / 'emotion_corpus_metadata.json'
    with open(metadata_file, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    
    print(f"   - 메타데이터: {metadata_file}")
    
    print("\n" + "=" * 80)
    print("✅ 전처리 완료!")
    print("=" * 80)
    print(f"\n💡 K-Fold 학습 실행:")
    print(f"   python training/main_kfold.py --data_path {output_file} --k_folds 5 --epochs 10")
    
    return df


if __name__ == '__main__':
    preprocess_corpus()
