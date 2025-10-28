"""
각 클래스당 1000개씩 샘플링된 데이터 생성
빠른 모델 비교 실험용
"""
import pandas as pd
import os

# 전체 데이터 로드
print("📂 전체 데이터 로드 중...")
df = pd.read_csv('processed/emotion_corpus_full.csv')
print(f"✅ 전체 데이터: {len(df):,} samples")

# 클래스별 분포 확인
print("\n📊 원본 클래스 분포:")
for emotion_id in sorted(df['label_id'].unique()):
    emotion_name = df[df['label_id'] == emotion_id]['emotion'].iloc[0]
    count = len(df[df['label_id'] == emotion_id])
    print(f"  - {emotion_name} (id={emotion_id}): {count:,} samples")

# 각 클래스당 1000개씩 샘플링
print("\n🔀 각 클래스당 1000개씩 샘플링 중...")
sampled_dfs = []

for emotion_id in sorted(df['label_id'].unique()):
    emotion_data = df[df['label_id'] == emotion_id]
    emotion_name = emotion_data['emotion'].iloc[0]
    
    # 1000개 샘플링 (클래스에 1000개 미만이면 전체 사용)
    n_samples = min(1000, len(emotion_data))
    sampled = emotion_data.sample(n=n_samples, random_state=42)
    sampled_dfs.append(sampled)
    
    print(f"  ✓ {emotion_name}: {n_samples} samples")

# 결합 및 섞기
sampled_df = pd.concat(sampled_dfs, ignore_index=True)
sampled_df = sampled_df.sample(frac=1, random_state=42).reset_index(drop=True)

print(f"\n✅ 샘플링 완료: 총 {len(sampled_df):,} samples")

# 샘플링된 클래스 분포 확인
print("\n📊 샘플링된 클래스 분포:")
for emotion_id in sorted(sampled_df['label_id'].unique()):
    emotion_name = sampled_df[sampled_df['label_id'] == emotion_id]['emotion'].iloc[0]
    count = len(sampled_df[sampled_df['label_id'] == emotion_id])
    percentage = (count / len(sampled_df)) * 100
    print(f"  - {emotion_name} (id={emotion_id}): {count:,} ({percentage:.1f}%)")

# 저장
output_path = 'processed/emotion_corpus_sampled_1k.csv'
sampled_df.to_csv(output_path, index=False, encoding='utf-8-sig')
print(f"\n💾 저장 완료: {output_path}")

# 샘플 데이터 확인
print("\n📝 샘플 데이터 (처음 3개):")
for idx, row in sampled_df.head(3).iterrows():
    print(f"  {idx+1}. [{row['emotion']}] {row['text'][:50]}...")
