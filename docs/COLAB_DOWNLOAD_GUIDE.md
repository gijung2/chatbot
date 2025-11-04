# 📥 Colab에서 학습한 모델 다운로드하기

모델 학습은 완료했지만 `best_model_fold{N}_{timestamp}.zip` 파일이 다운로드되지 않은 경우 해결 방법입니다.

---

## 🔧 방법 1: 다운로드 셀 다시 실행 (가장 쉬움)

### 1단계: Colab에서 다운로드 셀 찾기

노트북에서 **"9️⃣ 모델 다운로드"** 섹션의 셀을 찾습니다.

### 2단계: 셀 실행

```python
# 이 셀을 실행하세요
from google.colab import files
import shutil

# 최고 성능 모델 압축
best_model_path = best_fold['model_path']
zip_filename = f'best_model_fold{best_fold["fold"]}_{timestamp}.zip'

print(f"📦 압축 중: {best_model_path}")
shutil.make_archive(zip_filename.replace('.zip', ''), 'zip', best_model_path)
print(f"✅ 압축 완료: {zip_filename}")

# 다운로드
files.download(zip_filename)
```

### 3단계: 자동 다운로드

브라우저에서 자동으로 파일이 다운로드됩니다.

---

## 🔧 방법 2: 파일 브라우저에서 수동 다운로드

### 1단계: Colab 파일 브라우저 열기

1. Colab 왼쪽 사이드바의 **📁 폴더 아이콘** 클릭
2. 파일 목록이 표시됩니다

### 2단계: 모델 폴더 찾기

다음 형식의 폴더를 찾습니다:
```
fold1_best_model_20251104_XXXXXX/
fold2_best_model_20251104_XXXXXX/
```

### 3단계: 폴더 전체 다운로드

**옵션 A: ZIP 압축 후 다운로드** (권장)

Colab에서 새 코드 셀을 만들어 실행:

```python
import shutil
from google.colab import files

# 모델 폴더 이름 (실제 폴더명으로 변경)
model_folder = "fold1_best_model_20251104_165817"

# ZIP 압축
shutil.make_archive(f'{model_folder}', 'zip', model_folder)

# 다운로드
files.download(f'{model_folder}.zip')
```

**옵션 B: 개별 파일 다운로드** (느림)

각 파일에 마우스 우클릭 → **Download**

필요한 파일:
- `config.json`
- `pytorch_model.bin` (또는 `model.safetensors`)
- `tokenizer_config.json`
- `vocab.txt`
- `special_tokens_map.json`
- `tokenizer.json`

---

## 🔧 방법 3: Google Drive 연동 (추천 - 재사용 가능)

세션이 끊겨도 파일이 유지되고 다시 다운로드할 수 있습니다.

### 1단계: Drive 마운트 (노트북 실행 중에)

```python
from google.colab import drive
drive.mount('/content/drive')
```

### 2단계: 모델을 Drive로 복사

```python
import shutil

# 최고 성능 모델 경로
best_model_path = best_fold['model_path']

# Drive 경로 (MyDrive에 저장)
drive_path = f'/content/drive/MyDrive/chatbot_models/{best_model_path.split("/")[-1]}'

# 폴더 복사
shutil.copytree(best_model_path, drive_path)

print(f"✅ 모델을 Google Drive에 저장했습니다!")
print(f"   경로: {drive_path}")
```

### 3단계: Google Drive에서 다운로드

1. https://drive.google.com 접속
2. `MyDrive/chatbot_models/` 폴더로 이동
3. 모델 폴더 우클릭 → **다운로드**

---

## 🔧 방법 4: 직접 압축 명령어 실행

### 새 코드 셀에서 실행:

```python
# 1. 필요한 모듈 임포트
from google.colab import files
import shutil
import os

# 2. 저장된 모델 폴더 확인
print("📁 현재 폴더의 모델 목록:")
models = [d for d in os.listdir('.') if 'fold' in d and 'best_model' in d]
for i, model in enumerate(models, 1):
    print(f"{i}. {model}")

# 3. 최고 성능 모델 찾기 (또는 수동으로 번호 선택)
# 자동: JSON 파일에서 읽기
import json
with open('kfold_summary.json', 'r') as f:
    summary = json.load(f)

best_fold_num = summary['best_fold']
best_models = [m for m in models if f'fold{best_fold_num}_' in m]

if best_models:
    best_model = best_models[0]
    print(f"\n🏆 최고 성능 모델: {best_model}")
    
    # 4. ZIP 압축
    zip_name = f"{best_model}"
    print(f"\n📦 압축 중...")
    shutil.make_archive(zip_name, 'zip', best_model)
    print(f"✅ 압축 완료: {zip_name}.zip")
    
    # 5. 다운로드
    print(f"\n📥 다운로드 시작...")
    files.download(f'{zip_name}.zip')
    print(f"✅ 다운로드 완료!")
else:
    print("❌ 모델을 찾을 수 없습니다.")
    print("💡 수동으로 선택하세요:")
    print("   model_name = models[0]  # 번호 변경")
    print("   shutil.make_archive(model_name, 'zip', model_name)")
    print("   files.download(f'{model_name}.zip')")
```

---

## 🔧 방법 5: 모든 Fold 한번에 다운로드

모든 Fold를 한번에 압축하여 다운로드:

```python
from google.colab import files
import shutil
import os

# 모든 모델 폴더 찾기
models = [d for d in os.listdir('.') if 'fold' in d and 'best_model' in d]

print(f"📁 발견된 모델: {len(models)}개")
for model in models:
    print(f"   - {model}")

# 모든 모델을 하나의 폴더로 모으기
all_models_dir = "all_trained_models"
os.makedirs(all_models_dir, exist_ok=True)

for model in models:
    dest = os.path.join(all_models_dir, model)
    shutil.copytree(model, dest)
    print(f"✅ 복사: {model}")

# JSON 파일도 포함
shutil.copy('kfold_summary.json', all_models_dir)

# 압축
print("\n📦 전체 압축 중...")
shutil.make_archive('all_trained_models', 'zip', all_models_dir)
print("✅ 압축 완료: all_trained_models.zip")

# 다운로드
print("\n📥 다운로드 시작...")
files.download('all_trained_models.zip')
print("✅ 모든 모델 다운로드 완료!")
```

---

## 📊 다운로드된 파일 확인

다운로드가 완료되면 다음 파일이 있어야 합니다:

### 최소 필수 파일 (모델 폴더 내):
```
fold1_best_model_20251104_XXXXXX/
├── config.json              ← 모델 구조 설정
├── pytorch_model.bin        ← 학습된 가중치 (110MB)
├── tokenizer_config.json    ← 토크나이저 설정
├── vocab.txt               ← 어휘 사전
├── special_tokens_map.json ← 특수 토큰
└── tokenizer.json          ← 토크나이저 (옵션)
```

### 추가 파일:
```
kfold_summary.json  ← 학습 결과 요약
```

---

## ❓ 문제 해결

### ❌ "NameError: name 'best_fold' is not defined"

**원인:** 학습 셀을 실행하지 않았거나 변수가 초기화되지 않음

**해결:**
1. 노트북 전체를 처음부터 다시 실행: **런타임 > 모두 실행**
2. 또는 수동으로 모델 이름 지정:
```python
# 실제 폴더명으로 변경
model_folder = "fold1_best_model_20251104_165817"
shutil.make_archive(model_folder, 'zip', model_folder)
files.download(f'{model_folder}.zip')
```

### ❌ "shutil.Error: Directory not found"

**원인:** 모델 폴더가 없음

**해결:**
1. 파일 브라우저에서 폴더 존재 확인
2. 폴더 목록 출력:
```python
import os
models = [d for d in os.listdir('.') if os.path.isdir(d) and 'fold' in d]
print("발견된 모델:", models)
```

### ❌ 다운로드가 시작되지 않음

**원인:** 브라우저가 다운로드를 차단

**해결:**
1. 브라우저 주소창 오른쪽의 다운로드 차단 아이콘 클릭
2. "허용" 선택
3. 셀 다시 실행

---

## 🎯 다운로드 후 할 일

### 1. 압축 해제

```powershell
# PowerShell
cd C:\Users\rlarl\OneDrive\Desktop\chatbot\checkpoints_kfold
Expand-Archive -Path "best_model_fold1_20251104_XXXXXX.zip" -DestinationPath ".\"
```

### 2. 모델 테스트

```powershell
# 프로젝트 루트에서
python test_model_integration.py
```

### 3. 챗봇 실행

```powershell
python fastapi_app/main.py
```

---

## 💡 꿀팁: 자동 백업 설정

다음부터는 학습 시작 전에 Drive를 마운트하고 자동 백업 설정:

```python
# 노트북 시작 시 실행
from google.colab import drive
drive.mount('/content/drive')

# 학습 완료 후 자동으로 Drive에 저장되도록 output_dir 설정
output_dir = '/content/drive/MyDrive/chatbot_models'
```

이렇게 하면 세션이 끊겨도 파일이 안전하게 보관됩니다! 🔒

---

## 📞 추가 도움

- 모델 통합 가이드: `MODEL_INTEGRATION_GUIDE.md`
- 모델 테스트: `python test_model_integration.py`
- API 문서: http://localhost:8000/docs (서버 실행 후)
