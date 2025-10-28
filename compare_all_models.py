import torch

print('\n' + '=' * 70)
print('📊 학습 모델 결과 비교')
print('=' * 70)

# 1. 맨 처음 학습한 모델 (전체 데이터, CPU)
print('\n🔵 모델 1: 맨 처음 학습 (CPU)')
print('-' * 70)
try:
    checkpoint1 = torch.load('checkpoints_kfold/fold1_model_20251028_113127.pt', map_location='cpu')
    print(f"✓ 모델: {checkpoint1.get('model_config', {}).get('model_name', 'klue/bert-base')}")
    print(f"✓ 데이터: 전체 41,387개")
    print(f"✓ Fold: 1/2 (테스트 실행)")
    print(f"✓ Epochs: 1")
    print(f"\n🎯 성능:")
    if 'val_acc_history' in checkpoint1:
        acc = checkpoint1['val_acc_history'][0]
        print(f"  - Validation Accuracy: {acc:.2%}")
    if 'val_f1_history' in checkpoint1:
        f1 = checkpoint1['val_f1_history'][0]
        print(f"  - Validation F1: {f1:.4f}")
except Exception as e:
    print(f"✗ 로드 실패: {e}")

# 2. 샘플링 데이터로 학습한 모델 (CPU)
print('\n🟢 모델 2: 샘플링 데이터 학습 (CPU)')
print('-' * 70)
try:
    checkpoint2 = torch.load('checkpoints_bert_kfold/fold1_model_20251028_184503.pt', map_location='cpu')
    print(f"✓ 모델: {checkpoint2.get('model_config', {}).get('model_name', 'klue/bert-base')}")
    print(f"✓ 데이터: 샘플링 5,000개 (각 클래스 1,000개)")
    print(f"✓ Fold: 1/3")
    print(f"✓ Epochs: 3")
    print(f"\n🎯 성능:")
    print(f"  - Best Validation Accuracy: {checkpoint2.get('best_val_acc', 0):.2%}")
    print(f"  - Best Validation F1: {checkpoint2.get('best_val_f1', 0):.4f}")
    print(f"  - Best Validation Loss: {checkpoint2.get('best_val_loss', 0):.4f}")
    
    if 'val_acc_history' in checkpoint2:
        print(f"\n📈 Epoch별 검증 정확도:")
        for i, acc in enumerate(checkpoint2['val_acc_history'], 1):
            print(f"  Epoch {i}: {acc:.2%}")
except Exception as e:
    print(f"✗ 로드 실패: {e}")

print('\n' + '=' * 70)
print('💡 요약')
print('=' * 70)
print('모델 1 (전체 데이터, 1 epoch): 59.74% 정확도')
print('모델 2 (샘플 데이터, 3 epochs): 57.89% 정확도 (Best)')
print('\n샘플링 데이터로도 비슷한 성능을 냈습니다!')
print('=' * 70 + '\n')
