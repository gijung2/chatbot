# 🐋 Docker로 RTX 5070 GPU 학습 가이드

## 📋 사전 요구사항

1. **Docker Desktop 설치**: https://www.docker.com/products/docker-desktop/
2. **WSL 2 활성화**: Docker 설치 시 자동 설정
3. **NVIDIA Container Toolkit 설치** (WSL2 내부)

---

## 🚀 빠른 시작

### 1단계: Docker 이미지 빌드
```powershell
docker-compose -f docker-compose.training.yml build
```

### 2단계: GPU 테스트
```powershell
docker run --rm --gpus all nvidia/cuda:12.4.0-base-ubuntu22.04 nvidia-smi
```

### 3단계: 학습 컨테이너 실행
```powershell
docker-compose -f docker-compose.training.yml run --rm training
```

### 4단계: 컨테이너 내부에서 학습 실행
```bash
# 컨테이너 내부 (/workspace)

# GPU 확인
python -c "import torch; print('CUDA:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0))"

# 통합 데이터로 학습 (131K samples, 2-fold CV)
python training/train_krbert_hf.py \
    --data_path data/processed/emotion_corpus_merged.csv \
    --model_name snunlp/KR-Medium \
    --epochs 12 \
    --batch_size 64 \
    --k_folds 2

# 또는 기존 데이터만 (41K samples)
python training/train_krbert_hf.py \
    --data_path data/processed/emotion_corpus_full.csv \
    --model_name snunlp/KR-Medium \
    --epochs 12 \
    --batch_size 64 \
    --k_folds 2
```

---

## 🛠️ 상세 설치 가이드

### A. Docker Desktop 설치 (Windows)

1. **다운로드**:
   - https://www.docker.com/products/docker-desktop/
   - "Download for Windows" 클릭

2. **설치 옵션**:
   - ✅ **Use WSL 2 instead of Hyper-V** (필수!)
   - ✅ Add shortcut to desktop

3. **설치 완료 후**:
   - 컴퓨터 재부팅
   - Docker Desktop 실행
   - 우측 하단 트레이에서 Docker 아이콘 확인 (고래 아이콘)

4. **확인**:
   ```powershell
   docker --version
   # Docker version 24.0.x 이상 표시되어야 함
   ```

---

### B. NVIDIA Container Toolkit 설치 (WSL2 내부)

Windows에서 Docker + GPU를 사용하려면 WSL2 내부에서 설정해야 합니다.

#### 1. WSL2 Ubuntu 실행
```powershell
# PowerShell에서 실행
wsl -d Ubuntu
```

#### 2. NVIDIA Container Toolkit 설치 (Ubuntu 내부)
```bash
# NVIDIA Docker 저장소 설정
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg

curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list | \
    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
    sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

# 패키지 설치
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit

# Docker 설정
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker

# WSL에서 나가기
exit
```

#### 3. Docker Desktop 재시작
- Docker Desktop 우클릭 → Quit Docker Desktop
- Docker Desktop 재실행

---

### C. GPU 테스트

```powershell
# CUDA 컨테이너로 GPU 확인
docker run --rm --gpus all nvidia/cuda:12.4.0-base-ubuntu22.04 nvidia-smi
```

**성공 시 출력**:
```
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 576.88       Driver Version: 576.88       CUDA Version: 12.9     |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|                               |                      |               MIG M. |
|===============================+======================+======================|
|   0  NVIDIA GeForce ...  Off  | 00000000:01:00.0  On |                  N/A |
|  0%   32C    P8     8W / 220W |    123MiB / 12288MiB |      0%      Default |
+-----------------------------------------------------------------------------+
```

---

## 📦 학습 실행

### 방법 1: Docker Compose 사용 (권장)

```powershell
# 이미지 빌드 (최초 1회)
docker-compose -f docker-compose.training.yml build

# 컨테이너 실행 및 접속
docker-compose -f docker-compose.training.yml run --rm training

# 컨테이너 내부에서
python training/train_krbert_hf.py \
    --data_path data/processed/emotion_corpus_merged.csv \
    --epochs 12 \
    --batch_size 64 \
    --k_folds 2
```

### 방법 2: Docker 직접 실행

```powershell
# 이미지 빌드
docker build -t chatbot-training -f Dockerfile.training .

# 컨테이너 실행
docker run --rm --gpus all -it \
    -v ${PWD}:/workspace \
    chatbot-training bash

# 컨테이너 내부에서 학습
python training/train_krbert_hf.py \
    --data_path data/processed/emotion_corpus_merged.csv \
    --epochs 12 \
    --batch_size 64 \
    --k_folds 2
```

---

## ⏱️ 예상 학습 시간

| 데이터셋 | 샘플 수 | RTX 5070 (Docker) | Colab GPU T4 |
|---------|---------|------------------|--------------|
| 통합 (merged) | 131,091 | **1-1.5시간** | 2-3시간 |
| 기존 (full) | 41,387 | **30-40분** | 1-2시간 |

---

## 🔧 문제 해결

### 1. "docker: command not found"
**원인**: Docker Desktop 미설치  
**해결**: Docker Desktop 설치 후 재부팅

### 2. "could not select device driver with capabilities: [[gpu]]"
**원인**: NVIDIA Container Toolkit 미설치  
**해결**: WSL2 내부에서 toolkit 설치 (위 B 섹션 참조)

### 3. "CUDA error: no kernel image available"
**원인**: 아직도 SM 12.0 미지원  
**해결**: PyTorch nightly 버전 사용:
```dockerfile
# Dockerfile.training 수정
FROM pytorch/pytorch:nightly-cuda12.4-cudnn9-runtime
```

### 4. Out of Memory (OOM)
**해결**: 배치 크기 줄이기
```bash
python training/train_krbert_hf.py \
    --batch_size 32  # 64에서 32로 감소
```

---

## 📊 학습 결과 확인

학습 완료 후 모델은 호스트 컴퓨터에 자동 저장됩니다:

```
chatbot/
├── checkpoints_krbert/
│   ├── fold1_best_model_20251102_XXXXXX/
│   │   ├── config.json
│   │   ├── pytorch_model.bin
│   │   └── tokenizer files...
│   ├── fold2_best_model_20251102_XXXXXX/
│   └── kfold_summary_20251102_XXXXXX.json
```

---

## 💡 추가 팁

### GPU 메모리 모니터링
```bash
# 컨테이너 내부에서
watch -n 1 nvidia-smi
```

### 백그라운드 실행
```powershell
# 학습을 백그라운드에서 실행
docker-compose -f docker-compose.training.yml run -d training \
    python training/train_krbert_hf.py --data_path data/processed/emotion_corpus_merged.csv --epochs 12 --batch_size 64 --k_folds 2

# 로그 확인
docker logs -f chatbot-training
```

### 컨테이너 정리
```powershell
# 모든 중지된 컨테이너 삭제
docker container prune

# 사용하지 않는 이미지 삭제
docker image prune
```

---

## 🎯 성공 기준

✅ `nvidia-smi`가 Docker 컨테이너 내부에서 정상 작동  
✅ PyTorch에서 `torch.cuda.is_available()` = True  
✅ 모델 학습이 GPU에서 실행 (nvidia-smi에서 GPU 사용률 증가)  
✅ 학습 속도가 CPU 대비 8배 이상 빠름

---

## 📞 지원

문제가 발생하면:
1. Docker Desktop 재시작
2. WSL2 재시작: `wsl --shutdown` 후 재실행
3. NVIDIA 드라이버 최신 버전 확인
4. 위 문제 해결 섹션 참조
