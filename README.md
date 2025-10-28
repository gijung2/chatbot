# 🧠 심리상담 전문 아바타 시스템
**Psychological Counseling Avatar System**

실시간 감정 분석을 통한 심리상담 전문 아바타 생성 시스템입니다.

## 🚀 주요 기능

### 💡 **심리상담 전문 감정 분석**
- 한국어 특화 심리 패턴 인식
- 자살사고, 우울, 불안, 분노, 트라우마 감지
- 위험도 3단계 분류 (Low/Medium/High)
- 실시간 위기 상황 자동 대응

### 🎨 **감정별 맞춤 아바타 생성**
- 5가지 감정별 아름다운 아바타 (기쁨, 슬픔, 불안, 분노, 중립)
- 감정 강도에 따른 시각적 변화
- 고해상도 그라데이션 배경
- 특수 효과 (별, 빗방울, 화염 등)

### 💬 **실시간 채팅 인터페이스**
- 메시지 입력 시 즉시 감정 분석
- 아바타 자동 업데이트
- 상담사 스타일 공감적 응답
- 위험 상황 시 응급 연락처 제공

## 📂 프로젝트 구조

```
심리상담-아바타-시스템/
├── lightweight_psychological_api.py    # 메인 API 서버 (포트 8003)
├── simple_chat_demo.html              # 채팅 인터페이스
├── demo_server.py                     # 데모 서버 (포트 8080)
├── requirements_minimal.txt           # 최소 의존성 (권장)
├── requirements.txt                   # 전체 의존성 (고급 기능용)
├── Dockerfile                         # Docker 컨테이너 설정
├── data/                             # 감정 데이터셋
└── frontend/                         # React 프론트엔드 (선택사항)
```

## 🛠️ 설치 및 실행

### **1. 기본 설치 (권장)**
```bash
# 의존성 설치
pip install -r requirements_minimal.txt

# API 서버 시작
python lightweight_psychological_api.py

# 데모 서버 시작 (새 터미널에서)
python demo_server.py
```

### **2. 접속**
- **데모 페이지**: http://localhost:8080
- **API 문서**: http://localhost:8003

## 🔧 API 사용법

### **감정 분석 + 아바타 생성**
```python
import requests

response = requests.post('http://localhost:8003/generate_avatar', 
    json={'text': '너무 우울해서 죽고싶어요'})

result = response.json()
print(f"감정: {result['emotion']}")
print(f"위험도: {result['risk_level']}")
print(f"아바타: {result['avatar_image']}")  # Base64 이미지
```

### **감정 분석만**
```python
response = requests.post('http://localhost:8003/analyze', 
    json={'text': '시험이 다가와서 불안해요'})

analysis = response.json()['analysis']
print(f"감정: {analysis['emotion']}")
print(f"강도: {analysis['intensity']}")
print(f"패턴: {analysis['detected_patterns']}")
```

## 🚨 응급 상황 대응

시스템이 다음과 같은 고위험 상황을 감지하면 자동으로 응급 연락처를 제공합니다:

- **자살사고**: "죽고싶", "사라지고싶", "끝내고싶"
- **심각한 우울**: "의미없", "가치없", "소용없"
- **극심한 절망**: "포기", "한계", "견딜수없"

### **응급 연락처**
- **자살예방상담전화**: 109 (24시간)
- **정신건강상담전화**: 1577-0199 (24시간)
- **청소년전화**: 1388 (24시간)

## 🧪 테스트 문장들

### **기쁨** 😊
- "오늘 정말 행복한 일이 생겼어요!"
- "시험에 합격해서 너무 기뻐요!"

### **슬픔** 😢
- "너무 힘들어서 눈물이 나요"
- "요즘 우울해서 잠을 못 자요"

### **불안** 😰
- "시험이 다가와서 너무 불안해요"
- "계속 걱정이 되어서 잠이 안 와요"

### **분노** 😠
- "정말 화가 나서 참을 수 없어요"
- "그 사람이 나를 무시해서 짜증나요"

### **중립** 😐
- "그냥 평범한 하루였어요"
- "특별한 일은 없었어요"

## 💻 개발 정보

- **언어**: Python 3.11+
- **프레임워크**: Flask
- **이미지 처리**: Pillow
- **감정 분석**: 정규표현식 기반 패턴 매칭
- **아키텍처**: RESTful API + 실시간 웹 인터페이스

## ⚠️ 중요 안내

이 시스템은 **심리상담의 보조 도구**로 설계되었으며, 전문 상담사나 의료진의 진단을 대체할 수 없습니다. 실제 위기 상황에서는 즉시 전문가의 도움을 받으시기 바랍니다.

## 📝 라이선스

이 프로젝트는 MIT 라이선스 하에 제공됩니다.

---
**개발**: 심리상담 아바타 시스템 팀  
**버전**: 1.0.0 (경량 안정화 버전)  
**마지막 업데이트**: 2025년 10월 27일