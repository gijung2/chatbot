# 🎮 Unity 감정 기반 표정 변경 연동 가이드

## ✅ 결론: 완전히 가능합니다!

현재 구축된 FastAPI 서버와 KR-BERT 모델을 활용하여 Unity에서 실시간으로 감정 기반 표정 변경이 가능합니다.

---

## 🏗️ 현재 시스템 구조

```
┌─────────────────────────────────────────────────────────┐
│  Unity (C#)                                             │
│  - HTTP 요청 (UnityWebRequest)                          │
│  - JSON 파싱                                            │
│  - 표정 변경 로직                                        │
└───────────────────┬─────────────────────────────────────┘
                    │ HTTP POST
                    ↓
┌─────────────────────────────────────────────────────────┐
│  FastAPI 서버 (Python)                                  │
│  http://localhost:8000                                  │
│                                                         │
│  📍 현재 구축된 API:                                     │
│  ├─ POST /emotion/analyze        (감정 분석만)          │
│  ├─ POST /chat/message            (채팅 + 감정)         │
│  └─ POST /avatar/generate         (아바타 생성)         │
└───────────────────┬─────────────────────────────────────┘
                    │
                    ↓
┌─────────────────────────────────────────────────────────┐
│  KR-BERT 모델                                            │
│  - 감정 분류: joy, sad, anxiety, anger, neutral         │
│  - 신뢰도 반환                                           │
└─────────────────────────────────────────────────────────┘
```

---

## 📂 이미 준비된 리소스

### 1. 아바타 이미지 파일들 ✅
```
avatar/mark_free_ko/
├── joy.png         # 기쁨 표정
├── sad.png         # 슬픔 표정
├── anxious.png     # 불안 표정
├── angry.png       # 분노 표정
├── neutral.png     # 중립 표정
└── background.png  # 배경
```

### 2. Live2D 모델 ✅
```
avatar/mark_free_ko/
├── mark_free_t03.can3   # Cubism 3 모델
├── mark_free_t04.cmo3   # Cubism 4 모델
└── runtime/             # 런타임 파일들
```

### 3. FastAPI 아바타 API ✅
- POST /avatar/generate
- POST /avatar/generate/image
- POST /chat/message (감정 분석 포함)

---

## 🎯 Unity 연동 방법

## 방법 1: REST API 연동 ⭐ (가장 간단)

### Unity C# 코드

```csharp
using UnityEngine;
using UnityEngine.Networking;
using System.Collections;
using System.Text;

public class EmotionAvatarController : MonoBehaviour
{
    // API URL
    private const string API_URL = "http://localhost:8000/chat/message";
    
    // 표정 스프라이트들
    public Sprite joySprite;
    public Sprite sadSprite;
    public Sprite anxietySprite;
    public Sprite angerSprite;
    public Sprite neutralSprite;
    
    // 아바타 UI 이미지
    public UnityEngine.UI.Image avatarImage;
    
    // 사용자 입력
    public void SendMessage(string userMessage)
    {
        StartCoroutine(GetEmotionFromAPI(userMessage));
    }
    
    IEnumerator GetEmotionFromAPI(string message)
    {
        // JSON 요청 데이터
        string jsonData = JsonUtility.ToJson(new ChatRequest 
        { 
            message = message,
            session_id = "unity-session-" + System.DateTime.Now.Ticks
        });
        
        // HTTP POST 요청
        using (UnityWebRequest www = UnityWebRequest.Post(API_URL, "POST"))
        {
            byte[] bodyRaw = Encoding.UTF8.GetBytes(jsonData);
            www.uploadHandler = new UploadHandlerRaw(bodyRaw);
            www.downloadHandler = new DownloadHandlerBuffer();
            www.SetRequestHeader("Content-Type", "application/json");
            
            yield return www.SendWebRequest();
            
            if (www.result == UnityWebRequest.Result.Success)
            {
                // JSON 파싱
                string responseText = www.downloadHandler.text;
                ChatResponse response = JsonUtility.FromJson<ChatResponse>(responseText);
                
                Debug.Log($"감정: {response.emotion}, 신뢰도: {response.confidence}");
                
                // ⭐ 표정 변경
                ChangeEmotion(response.emotion);
                
                // 응답 텍스트 표시
                ShowBotResponse(response.response);
            }
            else
            {
                Debug.LogError($"API 오류: {www.error}");
            }
        }
    }
    
    // ⭐ 핵심: 감정에 따라 표정 변경
    void ChangeEmotion(string emotion)
    {
        Sprite newSprite = null;
        
        switch (emotion.ToLower())
        {
            case "joy":
                newSprite = joySprite;
                break;
            case "sad":
                newSprite = sadSprite;
                break;
            case "anxiety":
                newSprite = anxietySprite;
                break;
            case "anger":
                newSprite = angerSprite;
                break;
            case "neutral":
                newSprite = neutralSprite;
                break;
        }
        
        if (newSprite != null)
        {
            avatarImage.sprite = newSprite;
            Debug.Log($"표정 변경: {emotion}");
        }
    }
    
    void ShowBotResponse(string response)
    {
        // UI에 봇 응답 표시
        Debug.Log($"봇 응답: {response}");
    }
}

// JSON 직렬화용 클래스
[System.Serializable]
public class ChatRequest
{
    public string message;
    public string session_id;
}

[System.Serializable]
public class ChatResponse
{
    public string response;
    public string emotion;
    public float confidence;
    public string avatar_url;
    public string[] suggestions;
}
```

---

## 방법 2: Live2D Cubism SDK 사용 (고급)

### Unity에서 Live2D 파라미터 제어

```csharp
using Live2D.Cubism.Core;
using Live2D.Cubism.Framework;

public class Live2DEmotionController : MonoBehaviour
{
    public CubismModel cubismModel;
    
    // API에서 받은 감정으로 Live2D 파라미터 변경
    public void ChangeEmotionLive2D(string emotion)
    {
        var parameters = cubismModel.Parameters;
        
        switch (emotion.ToLower())
        {
            case "joy":
                SetParameter(parameters, "ParamMouthForm", 1.0f);  // 웃는 입
                SetParameter(parameters, "ParamEyeLOpen", 1.0f);   // 눈 크게
                SetParameter(parameters, "ParamEyeROpen", 1.0f);
                SetParameter(parameters, "ParamBrowLY", 0.5f);     // 눈썹 올림
                SetParameter(parameters, "ParamBrowRY", 0.5f);
                break;
                
            case "sad":
                SetParameter(parameters, "ParamMouthForm", -0.8f); // 슬픈 입
                SetParameter(parameters, "ParamEyeLOpen", 0.3f);   // 눈 작게
                SetParameter(parameters, "ParamEyeROpen", 0.3f);
                SetParameter(parameters, "ParamBrowLY", -0.5f);    // 눈썹 내림
                SetParameter(parameters, "ParamBrowRY", -0.5f);
                break;
                
            case "anxiety":
                SetParameter(parameters, "ParamMouthForm", -0.3f);
                SetParameter(parameters, "ParamEyeLOpen", 0.8f);
                SetParameter(parameters, "ParamEyeROpen", 0.8f);
                SetParameter(parameters, "ParamBrowLAngle", -0.5f); // 눈썹 각도
                SetParameter(parameters, "ParamBrowRAngle", 0.5f);
                break;
                
            case "anger":
                SetParameter(parameters, "ParamMouthForm", -0.5f);
                SetParameter(parameters, "ParamEyeLOpen", 0.5f);
                SetParameter(parameters, "ParamEyeROpen", 0.5f);
                SetParameter(parameters, "ParamBrowLAngle", -1.0f);
                SetParameter(parameters, "ParamBrowRAngle", 1.0f);
                break;
                
            case "neutral":
                // 기본 상태로 리셋
                SetParameter(parameters, "ParamMouthForm", 0.0f);
                SetParameter(parameters, "ParamEyeLOpen", 1.0f);
                SetParameter(parameters, "ParamEyeROpen", 1.0f);
                SetParameter(parameters, "ParamBrowLY", 0.0f);
                SetParameter(parameters, "ParamBrowRY", 0.0f);
                break;
        }
    }
    
    void SetParameter(CubismParameter[] parameters, string paramName, float value)
    {
        for (int i = 0; i < parameters.Length; i++)
        {
            if (parameters[i].Id == paramName)
            {
                parameters[i].Value = value;
                break;
            }
        }
    }
}
```

---

## 방법 3: WebSocket 실시간 연동 (가장 빠름)

### FastAPI에 WebSocket 추가

```python
# fastapi_app/main.py
from fastapi import WebSocket

@app.websocket("/ws/emotion")
async def emotion_websocket(websocket: WebSocket):
    await websocket.accept()
    
    try:
        while True:
            # Unity에서 메시지 수신
            data = await websocket.receive_text()
            
            # 감정 분석
            emotion_result = emotion_model.predict_emotion(data)
            
            # Unity로 결과 전송
            await websocket.send_json({
                "emotion": emotion_result['emotion'],
                "confidence": emotion_result['confidence'],
                "probabilities": emotion_result['probabilities']
            })
    except Exception as e:
        print(f"WebSocket 오류: {e}")
```

### Unity WebSocket 클라이언트

```csharp
using NativeWebSocket;

public class WebSocketEmotionClient : MonoBehaviour
{
    WebSocket websocket;
    
    async void Start()
    {
        websocket = new WebSocket("ws://localhost:8000/ws/emotion");
        
        websocket.OnMessage += (bytes) =>
        {
            var message = System.Text.Encoding.UTF8.GetString(bytes);
            var data = JsonUtility.FromJson<EmotionData>(message);
            
            // ⭐ 실시간 표정 변경
            ChangeEmotionLive2D(data.emotion);
        };
        
        await websocket.Connect();
    }
    
    async void Update()
    {
        #if !UNITY_WEBGL || UNITY_EDITOR
        websocket?.DispatchMessageQueue();
        #endif
    }
    
    public async void SendMessage(string text)
    {
        if (websocket.State == WebSocketState.Open)
        {
            await websocket.SendText(text);
        }
    }
}
```

---

## 🎨 Unity 프로젝트 구조 예시

```
Unity Project/
├── Assets/
│   ├── Scripts/
│   │   ├── EmotionAvatarController.cs     # API 연동
│   │   ├── Live2DEmotionController.cs     # Live2D 제어
│   │   └── ChatUIManager.cs               # UI 관리
│   │
│   ├── Resources/
│   │   └── Avatars/                       # 표정 이미지들
│   │       ├── joy.png
│   │       ├── sad.png
│   │       ├── anxiety.png
│   │       ├── anger.png
│   │       └── neutral.png
│   │
│   ├── Live2D/
│   │   └── mark_free/                     # Live2D 모델
│   │       ├── mark_free.model3.json
│   │       └── textures/
│   │
│   └── Scenes/
│       └── ChatScene.unity                # 메인 씬
```

---

## 🚀 실제 작동 흐름

```
1. Unity에서 사용자 입력
   ↓
   "오늘 정말 행복해요!"
   
2. Unity → FastAPI (HTTP POST)
   ↓
   POST http://localhost:8000/chat/message
   Body: {"message": "오늘 정말 행복해요!"}
   
3. FastAPI → KR-BERT 모델
   ↓
   감정 분석 결과: "joy" (92% 신뢰도)
   
4. FastAPI → Unity (JSON Response)
   ↓
   {
     "emotion": "joy",
     "confidence": 0.92,
     "response": "기쁜 마음이 느껴지네요!",
     "suggestions": [...]
   }
   
5. Unity에서 표정 변경
   ↓
   avatarImage.sprite = joySprite;
   또는
   Live2D 파라미터 변경 (ParamMouthForm = 1.0)
   
6. 화면에 표시
   ↓
   😊 아바타 표정 변경 완료!
```

---

## 📊 API 응답 예시

### POST http://localhost:8000/chat/message

**요청:**
```json
{
  "message": "오늘 정말 행복해요!",
  "session_id": "unity-session-123"
}
```

**응답:**
```json
{
  "response": "기쁜 마음이 느껴지네요! 긍정적인 에너지를 계속 유지하세요.",
  "emotion": "joy",
  "confidence": 0.92,
  "avatar_url": null,
  "suggestions": [
    "긍정적인 활동을 더 많이 시도해보세요",
    "이 감정을 일기로 기록해보세요"
  ]
}
```

---

## ✅ 체크리스트

### 이미 준비된 것 ✅
- ✅ KR-BERT 감정 분석 모델 (학습 완료)
- ✅ FastAPI 서버 (`/chat/message` API)
- ✅ 표정 이미지 파일 (joy, sad, anxiety, anger, neutral)
- ✅ Live2D 모델 파일 (`.can3`, `.cmo3`)
- ✅ JSON 응답 형식 (emotion, confidence 포함)

### Unity에서 구현할 것 📝
- [ ] UnityWebRequest로 HTTP 통신
- [ ] JSON 파싱 (JsonUtility 또는 Newtonsoft.Json)
- [ ] 표정 스프라이트 변경 로직
- [ ] (선택) Live2D Cubism SDK 설치
- [ ] (선택) WebSocket 실시간 통신

---

## 🔧 Unity 설정 방법

### 1. 스프라이트 Import
1. `avatar/mark_free_ko/` 폴더의 PNG 파일들을 Unity로 드래그
2. Texture Type을 "Sprite (2D and UI)"로 설정
3. Inspector에서 각 스프라이트를 `EmotionAvatarController`에 할당

### 2. UI 구성
1. Canvas 생성
2. Image 오브젝트 추가 (아바타 표시용)
3. Text 오브젝트 추가 (채팅 메시지용)
4. InputField 추가 (사용자 입력용)
5. Button 추가 (전송 버튼)

### 3. 스크립트 연결
1. `EmotionAvatarController.cs` 스크립트를 GameObject에 추가
2. Inspector에서 public 변수들 할당:
   - avatarImage → UI Image 오브젝트
   - joySprite → joy.png
   - sadSprite → sad.png
   - 등등...

---

## 🎯 테스트 시나리오

### 테스트 1: 기쁨 표정
```
입력: "오늘 정말 행복해요!"
예상 응답: emotion = "joy"
결과: 😊 표정으로 변경
```

### 테스트 2: 슬픔 표정
```
입력: "너무 슬프고 우울해요"
예상 응답: emotion = "sad"
결과: 😢 표정으로 변경
```

### 테스트 3: 불안 표정
```
입력: "걱정되고 불안해요"
예상 응답: emotion = "anxiety"
결과: 😰 표정으로 변경
```

### 테스트 4: 분노 표정
```
입력: "정말 짜증나고 화가 나요"
예상 응답: emotion = "anger"
결과: 😠 표정으로 변경
```

### 테스트 5: 중립 표정
```
입력: "그냥 평범한 하루네요"
예상 응답: emotion = "neutral"
결과: 😐 표정으로 변경
```

---

## 🐛 문제 해결

### 1. CORS 오류
**증상:** Unity에서 API 호출 시 "CORS policy" 오류

**해결:**
```python
# fastapi_app/main.py에 이미 설정되어 있음
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
```

### 2. JSON 파싱 오류
**증상:** JsonUtility가 null 반환

**해결:**
```csharp
// Newtonsoft.Json 사용
using Newtonsoft.Json;

ChatResponse response = JsonConvert.DeserializeObject<ChatResponse>(responseText);
```

### 3. 서버 연결 실패
**증상:** "Failed to connect to localhost:8000"

**해결:**
1. FastAPI 서버가 실행 중인지 확인
2. Windows 방화벽 확인
3. localhost 대신 127.0.0.1 사용

---

## 📚 추가 리소스

### Unity Package Manager
- Live2D Cubism SDK: https://www.live2d.com/en/download/cubism-sdk/
- Newtonsoft.Json: Package Manager에서 설치
- NativeWebSocket: https://github.com/endel/NativeWebSocket

### 참고 문서
- Unity UnityWebRequest: https://docs.unity3d.com/ScriptReference/Networking.UnityWebRequest.html
- FastAPI WebSocket: https://fastapi.tiangolo.com/advanced/websockets/
- Live2D Cubism Manual: https://docs.live2d.com/

---

## 🎉 최종 요약

**현재 시스템은 Unity 연동을 위한 모든 준비가 완료되었습니다!**

**필요한 것:**
1. Unity 프로젝트 생성
2. HTTP 통신 스크립트 작성 (위 코드 사용)
3. 표정 이미지 Import
4. UI 구성

**작동 방식:**
- Unity → FastAPI 서버로 텍스트 전송
- KR-BERT 모델이 감정 분석 (joy, sad, anxiety, anger, neutral)
- Unity가 감정에 맞는 표정으로 변경

**예상 개발 시간:**
- 기본 연동: 2-3시간
- Live2D 연동: 1-2일
- WebSocket 실시간: 3-4시간

---

작성일: 2025년 11월 12일
버전: 1.0
