import torch

# 첫 번째 학습 모델 로드
checkpoint = torch.load('checkpoints_kfold/fold1_model_20251028_113127.pt', map_location='cpu')

print('=' * 60)
print('📊 맨 처음 학습한 모델 (전체 데이터)')
print('=' * 60)
print(f"모델: {checkpoint.get('model_config', {}).get('model_name', 'klue/bert-base')}")
print(f"데이터: 전체 41,387개")
print(f"Fold: 1/2 (테스트)")
print(f"Epochs: 1")
print()
print('🎯 성능:')
print(f"  - Best Val Accuracy: {checkpoint.get('best_val_acc', 0):.2%}")
print(f"  - Best Val F1: {checkpoint.get('best_val_f1', 0):.4f}")
print(f"  - Best Val Loss: {checkpoint.get('best_val_loss', 0):.4f}")
print()

if 'val_acc_history' in checkpoint:
    print('📈 Epoch별 검증 정확도:')
    for i, acc in enumerate(checkpoint['val_acc_history'], 1):
        print(f"  Epoch {i}: {acc:.2%}")
print('=' * 60)
