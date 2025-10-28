"""
KLUE-BERT vs KLUE-RoBERTa 성능 비교 실험
3-Fold Cross Validation, 3 Epochs
"""

import subprocess
import sys
import os
from datetime import datetime
import json

def run_experiment(model_name: str, model_type: str):
    """모델 학습 실험 실행"""
    print("\n" + "=" * 80)
    print(f"🔬 실험 시작: {model_type}")
    print(f"📦 모델: {model_name}")
    print(f"�️ 디바이스: CPU (강제)")
    print(f"📊 데이터: 샘플링 5,000개 (각 클래스 1,000개)")
    print(f"�🕐 시작 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80 + "\n")
    
    output_dir = f'checkpoints_{model_type.lower()}_kfold'
    
    # 출력 디렉토리 생성
    os.makedirs(output_dir, exist_ok=True)
    
    # CPU 강제 사용 (GPU 비활성화)
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = '-1'
    
    # 실험 설정 (샘플링된 데이터 사용)
    cmd = [
        sys.executable,
        'training/main_kfold.py',
        '--data_path', 'data/processed/emotion_corpus_sampled_1k.csv',
        '--model_name', model_name,
        '--k_folds', '3',
        '--epochs', '3',
        '--batch_size', '16',
        '--learning_rate', '2e-5',
        '--max_length', '128',
        '--early_stopping_patience', '3',
        '--output_dir', output_dir
    ]
    
    start_time = datetime.now()
    
    try:
        result = subprocess.run(
            cmd,
            check=True,
            capture_output=False,
            text=True,
            env=env  # CPU 강제 사용
        )
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds() / 60
        
        print("\n" + "=" * 80)
        print(f"✅ {model_type} 실험 완료!")
        print(f"⏱️  소요 시간: {duration:.2f}분")
        print(f"🕐 종료 시간: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 80 + "\n")
        
        return {
            'model_type': model_type,
            'model_name': model_name,
            'status': 'success',
            'duration_minutes': duration,
            'output_dir': output_dir
        }
        
    except subprocess.CalledProcessError as e:
        print(f"\n❌ {model_type} 실험 실패: {e}")
        return {
            'model_type': model_type,
            'model_name': model_name,
            'status': 'failed',
            'error': str(e)
        }

def main():
    """메인 함수"""
    print("\n" + "=" * 80)
    print("🎯 KLUE-BERT vs KLUE-RoBERTa 비교 실험")
    print("=" * 80)
    print("📋 실험 설정:")
    print("  - 데이터: 샘플링 5,000개 (각 클래스 1,000개)")
    print("  - K-Folds: 3")
    print("  - Epochs: 3 (각 Fold)")
    print("  - Batch Size: 16")
    print("  - Learning Rate: 2e-5")
    print("  - 디바이스: CPU")
    print("=" * 80 + "\n")
    
    # 실험 모델 목록
    experiments = [
        ('klue/bert-base', 'BERT'),
        ('klue/roberta-base', 'RoBERTa')
    ]
    
    results = []
    total_start = datetime.now()
    
    # 각 모델 실험
    for model_name, model_type in experiments:
        result = run_experiment(model_name, model_type)
        results.append(result)
        
        # 중간 휴식 (GPU 쿨다운)
        if model_type == 'BERT':
            print("\n⏸️  GPU 쿨다운 (10초)...\n")
            import time
            time.sleep(10)
    
    total_end = datetime.now()
    total_duration = (total_end - total_start).total_seconds() / 60
    
    # 결과 요약
    print("\n" + "=" * 80)
    print("📊 실험 결과 요약")
    print("=" * 80)
    
    for result in results:
        print(f"\n🔹 {result['model_type']}")
        print(f"  - 모델: {result['model_name']}")
        print(f"  - 상태: {result['status']}")
        if result['status'] == 'success':
            print(f"  - 소요 시간: {result['duration_minutes']:.2f}분")
            print(f"  - 출력 경로: {result['output_dir']}")
        else:
            print(f"  - 오류: {result.get('error', 'Unknown')}")
    
    print(f"\n⏱️  전체 실험 시간: {total_duration:.2f}분")
    print("=" * 80)
    
    # 결과 저장
    results_file = f'experiment_results_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump({
            'experiments': results,
            'total_duration_minutes': total_duration,
            'start_time': total_start.isoformat(),
            'end_time': total_end.isoformat()
        }, f, indent=2, ensure_ascii=False)
    
    print(f"\n📝 결과 저장: {results_file}")
    
    print("\n💡 다음 단계:")
    print("  1. 각 모델의 학습 결과 비교")
    print("  2. 검증 정확도가 높은 모델 선택")
    print("  3. 선택된 모델로 전체 학습 (5-Fold, 10 Epochs)")
    print("\n")

if __name__ == '__main__':
    main()
