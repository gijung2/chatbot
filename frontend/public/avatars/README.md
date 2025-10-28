# Avatar Images Directory

이 디렉토리는 Colab에서 생성된 감정별 아바타 이미지들을 저장합니다.

## 디렉토리 구조

```
frontend/public/avatars/
├── joy/
│   ├── style_1.png
│   ├── style_2.png
│   └── style_3.png
├── sad/
│   ├── style_1.png
│   ├── style_2.png
│   └── style_3.png
├── anxiety/
│   ├── style_1.png
│   ├── style_2.png
│   └── style_3.png
├── anger/
│   ├── style_1.png
│   ├── style_2.png
│   └── style_3.png
├── neutral/
│   ├── style_1.png
│   ├── style_2.png
│   └── style_3.png
└── metadata.json
```

## 사용 방법

1. **Colab에서 이미지 생성**: `generate_avatars_colab.py` 실행
2. **파일 다운로드**: 생성된 모든 이미지 파일들 다운로드
3. **파일 배치**: 위 구조에 맞게 파일들을 정리
4. **웹 서버 재시작**: 정적 파일 서빙 시작

## 파일 명명 규칙

- 감정별 폴더: `joy`, `sad`, `anxiety`, `anger`, `neutral`
- 이미지 파일: `style_1.png`, `style_2.png`, `style_3.png`
- 해상도: 512x512 픽셀
- 포맷: PNG (투명 배경 지원)

## 웹에서 접근

```
http://localhost:3000/avatars/joy/style_1.png
http://localhost:3000/avatars/sad/style_2.png
http://localhost:3000/avatars/anxiety/style_3.png
```
