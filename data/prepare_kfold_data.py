"""
K-Fold Cross Validation용 데이터 준비
train.csv와 val.csv를 합쳐서 전체 데이터셋 생성
"""
import pandas as pd
import os
from pathlib import Path

def prepare_kfold_data():
    """train.csv와 val.csv를 합쳐서 전체 데이터 생성"""
    
    # 경로 설정
    processed_dir = Path(__file__).parent / 'processed'
    train_path = processed_dir / 'train.csv'
    val_path = processed_dir / 'val.csv'
    output_path = processed_dir / 'full_data.csv'
    
    print("=" * 80)
    print("📂 K-Fold용 데이터 준비")
    print("=" * 80)
    
    # 데이터 로드
    print(f"\n📥 데이터 로드 중...")
    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    
    print(f"   - Train: {len(train_df)} samples")
    print(f"   - Val: {len(val_df)} samples")
    
    # 데이터 합치기
    full_df = pd.concat([train_df, val_df], ignore_index=True)
    print(f"\n✅ 전체 데이터: {len(full_df)} samples")
    
    # 클래스 분포 확인
    print(f"\n📊 클래스 분포:")
    label_col = 'label_id' if 'label_id' in full_df.columns else 'label'
    for label, count in full_df[label_col].value_counts().sort_index().items():
        percentage = count / len(full_df) * 100
        print(f"   - Label {label}: {count:,} samples ({percentage:.1f}%)")
    
    # 저장
    full_df.to_csv(output_path, index=False, encoding='utf-8')
    print(f"\n💾 저장 완료: {output_path}")
    print(f"   - 크기: {output_path.stat().st_size / (1024*1024):.2f} MB")
    
    print("\n" + "=" * 80)
    print("✅ 데이터 준비 완료!")
    print("=" * 80)
    print(f"\n💡 K-Fold 학습 실행:")
    print(f"   python training/main_kfold.py --data_path {output_path} --k_folds 5 --epochs 10")

if __name__ == '__main__':
    prepare_kfold_data()
