"""
새로운 한국어 대화 데이터셋 전처리
- 한국어_단발성_대화_데이터셋.xlsx
- 한국어_연속적_대화_데이터셋.xlsx

기존 감성대화말뭉치와 통합하여 학습용 데이터 생성
"""
import pandas as pd
import numpy as np
from pathlib import Path
import json
from datetime import datetime

# 감정 라벨 매핑 (기존 데이터셋과 통일)
EMOTION_MAP = {
    '행복': 'joy',
    '기쁨': 'joy',
    '슬픔': 'sad',
    '불안': 'anxiety',
    '공포': 'anxiety',  # 공포를 불안으로 매핑
    '당황': 'anxiety',
    '분노': 'anger',
    '화남': 'anger',
    '혐오': 'anger',  # 혐오를 분노로 매핑
    '놀람': 'neutral',  # 놀람을 중립으로 매핑
    '중립': 'neutral',
    '상처': 'sad',
}

# 최종 5개 감정 클래스
EMOTION_CLASSES = ['joy', 'sad', 'anxiety', 'anger', 'neutral']
LABEL_TO_ID = {emotion: idx for idx, emotion in enumerate(EMOTION_CLASSES)}


def preprocess_single_conversation():
    """단발성 대화 데이터셋 전처리"""
    print("\n" + "=" * 80)
    print("1️⃣ 한국어_단발성_대화_데이터셋.xlsx 전처리")
    print("=" * 80)
    
    df = pd.read_excel('raw/한국어_단발성_대화_데이터셋.xlsx')
    print(f"✅ 원본 데이터 로드: {len(df):,} samples")
    
    # 필요한 컬럼만 선택 (Sentence, Emotion)
    df = df[['Sentence', 'Emotion']].copy()
    
    # 결측치 제거
    df = df.dropna()
    print(f"   - 결측치 제거 후: {len(df):,} samples")
    
    # 빈 문자열 제거
    df = df[df['Sentence'].str.strip() != '']
    df = df[df['Emotion'].str.strip() != '']
    print(f"   - 빈 문자열 제거 후: {len(df):,} samples")
    
    # 감정 라벨 매핑
    df['emotion'] = df['Emotion'].map(EMOTION_MAP)
    
    # 매핑되지 않은 감정 확인
    unmapped = df[df['emotion'].isna()]['Emotion'].unique()
    if len(unmapped) > 0:
        print(f"   ⚠️ 매핑되지 않은 감정: {unmapped}")
        df = df.dropna(subset=['emotion'])
    
    print(f"   - 감정 매핑 후: {len(df):,} samples")
    
    # 컬럼명 변경
    df = df.rename(columns={'Sentence': 'text'})
    df = df[['text', 'emotion']].copy()
    
    # label_id 추가
    df['label_id'] = df['emotion'].map(LABEL_TO_ID)
    
    # 감정 분포 출력
    print(f"\n📊 감정 분포:")
    for emotion in EMOTION_CLASSES:
        count = (df['emotion'] == emotion).sum()
        percentage = count / len(df) * 100
        print(f"   - {emotion}: {count:,} ({percentage:.1f}%)")
    
    return df


def preprocess_continuous_conversation():
    """연속적 대화 데이터셋 전처리"""
    print("\n" + "=" * 80)
    print("2️⃣ 한국어_연속적_대화_데이터셋.xlsx 전처리")
    print("=" * 80)
    
    df = pd.read_excel('raw/한국어_연속적_대화_데이터셋.xlsx')
    print(f"✅ 원본 데이터 로드: {len(df):,} samples")
    
    # 첫 번째 헤더 행 제거 (dialog #, 발화, 감정 행)
    df = df[df['Unnamed: 0'] != 'dialog #'].copy()
    
    # 필요한 컬럼만 선택 (발화, 감정)
    # Unnamed: 1 = 발화, Unnamed: 2 = 감정
    df = df[['Unnamed: 1', 'Unnamed: 2']].copy()
    df.columns = ['text', 'emotion_raw']
    
    # 결측치 제거
    df = df.dropna()
    print(f"   - 결측치 제거 후: {len(df):,} samples")
    
    # 빈 문자열 제거
    df = df[df['text'].str.strip() != '']
    df = df[df['emotion_raw'].str.strip() != '']
    print(f"   - 빈 문자열 제거 후: {len(df):,} samples")
    
    # 오타 제거 (ㅍ, 분, ㅈ중립, 분ㄴ, 중림, ㄴ중립, 줄 등)
    valid_emotions = set(EMOTION_MAP.keys())
    df = df[df['emotion_raw'].isin(valid_emotions)]
    print(f"   - 유효한 감정만 선택 후: {len(df):,} samples")
    
    # 감정 라벨 매핑
    df['emotion'] = df['emotion_raw'].map(EMOTION_MAP)
    
    # 매핑되지 않은 감정 확인
    unmapped = df[df['emotion'].isna()]['emotion_raw'].unique()
    if len(unmapped) > 0:
        print(f"   ⚠️ 매핑되지 않은 감정: {unmapped}")
        df = df.dropna(subset=['emotion'])
    
    print(f"   - 감정 매핑 후: {len(df):,} samples")
    
    # 필요한 컬럼만 선택
    df = df[['text', 'emotion']].copy()
    
    # label_id 추가
    df['label_id'] = df['emotion'].map(LABEL_TO_ID)
    
    # 감정 분포 출력
    print(f"\n📊 감정 분포:")
    for emotion in EMOTION_CLASSES:
        count = (df['emotion'] == emotion).sum()
        percentage = count / len(df) * 100
        print(f"   - {emotion}: {count:,} ({percentage:.1f}%)")
    
    return df


def load_existing_data():
    """기존 감성대화말뭉치 데이터 로드"""
    print("\n" + "=" * 80)
    print("3️⃣ 기존 감성대화말뭉치 데이터 로드")
    print("=" * 80)
    
    processed_path = Path('processed/emotion_corpus_full.csv')
    
    if processed_path.exists():
        df = pd.read_csv(processed_path)
        print(f"✅ 기존 데이터 로드: {len(df):,} samples")
        
        # 감정 분포 출력
        print(f"\n📊 감정 분포:")
        for emotion in EMOTION_CLASSES:
            count = (df['emotion'] == emotion).sum()
            percentage = count / len(df) * 100
            print(f"   - {emotion}: {count:,} ({percentage:.1f}%)")
        
        return df
    else:
        print("⚠️ 기존 데이터가 없습니다. 새 데이터만 사용합니다.")
        return None


def merge_and_save(df_existing, df_single, df_continuous):
    """데이터 통합 및 저장"""
    print("\n" + "=" * 80)
    print("4️⃣ 데이터 통합 및 저장")
    print("=" * 80)
    
    # 데이터 통합
    dfs = []
    sources = []
    
    if df_existing is not None:
        dfs.append(df_existing)
        sources.append('emotion_corpus')
        print(f"   - 기존 감성대화말뭉치: {len(df_existing):,} samples")
    
    if df_single is not None:
        dfs.append(df_single)
        sources.append('single_conversation')
        print(f"   - 단발성 대화: {len(df_single):,} samples")
    
    if df_continuous is not None:
        dfs.append(df_continuous)
        sources.append('continuous_conversation')
        print(f"   - 연속적 대화: {len(df_continuous):,} samples")
    
    df_merged = pd.concat(dfs, ignore_index=True)
    print(f"\n✅ 통합 완료: {len(df_merged):,} samples")
    
    # 중복 제거 (text 기준)
    original_len = len(df_merged)
    df_merged = df_merged.drop_duplicates(subset=['text'], keep='first')
    duplicates = original_len - len(df_merged)
    if duplicates > 0:
        print(f"   - 중복 제거: {duplicates:,} samples")
        print(f"   - 최종: {len(df_merged):,} samples")
    
    # 최종 감정 분포
    print(f"\n📊 최종 감정 분포:")
    for emotion in EMOTION_CLASSES:
        count = (df_merged['emotion'] == emotion).sum()
        percentage = count / len(df_merged) * 100
        print(f"   - {emotion}: {count:,} ({percentage:.1f}%)")
    
    # 저장
    output_path = Path('processed/emotion_corpus_merged.csv')
    df_merged.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"\n💾 저장 완료: {output_path}")
    
    # 메타데이터 저장
    metadata = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'total_samples': len(df_merged),
        'sources': sources,
        'emotion_distribution': {
            emotion: {
                'count': int((df_merged['emotion'] == emotion).sum()),
                'percentage': float((df_merged['emotion'] == emotion).sum() / len(df_merged) * 100)
            }
            for emotion in EMOTION_CLASSES
        },
        'emotion_classes': EMOTION_CLASSES,
        'label_mapping': LABEL_TO_ID,
        'duplicates_removed': duplicates
    }
    
    metadata_path = Path('processed/emotion_corpus_merged_metadata.json')
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    
    print(f"📄 메타데이터 저장: {metadata_path}")
    
    return df_merged


def main():
    """메인 함수"""
    print("\n" + "=" * 80)
    print("🚀 새로운 한국어 대화 데이터셋 전처리 시작")
    print("=" * 80)
    
    # 1. 단발성 대화 데이터셋 전처리
    df_single = preprocess_single_conversation()
    
    # 2. 연속적 대화 데이터셋 전처리
    df_continuous = preprocess_continuous_conversation()
    
    # 3. 기존 데이터 로드
    df_existing = load_existing_data()
    
    # 4. 통합 및 저장
    df_merged = merge_and_save(df_existing, df_single, df_continuous)
    
    print("\n" + "=" * 80)
    print("✅ 전처리 완료!")
    print("=" * 80)
    print(f"\n📊 최종 결과:")
    print(f"   - 총 샘플 수: {len(df_merged):,}")
    print(f"   - 감정 클래스: {len(EMOTION_CLASSES)}개")
    print(f"   - 저장 경로: processed/emotion_corpus_merged.csv")
    print(f"\n💡 학습 시 사용:")
    print(f"   python training/train_krbert_hf.py --data_path data/processed/emotion_corpus_merged.csv")


if __name__ == '__main__':
    main()
