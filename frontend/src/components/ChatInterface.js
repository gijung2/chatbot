// 실시간 감정 채팅 인터페이스
import React, { useState, useEffect, useRef } from 'react';

const ChatInterface = ({ 
  onMessageSend, 
  messages = [], 
  isConnected = false,
  currentEmotion = 'neutral',
  isAnalyzing = false,
  sessionId = null
}) => {
  const [inputMessage, setInputMessage] = useState('');
  const [isTyping, setIsTyping] = useState(false);
  const [messageHistory, setMessageHistory] = useState([]);
  const messagesEndRef = useRef(null);
  const inputRef = useRef(null);

  // 메시지 히스토리 업데이트
  useEffect(() => {
    if (messages && messages.length > 0) {
      setMessageHistory(messages);
    }
  }, [messages]);

  // 메시지 자동 스크롤
  useEffect(() => {
    scrollToBottom();
  }, [messageHistory]);

  // 입력 포커스
  useEffect(() => {
    if (isConnected && inputRef.current) {
      inputRef.current.focus();
    }
  }, [isConnected]);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  const handleSubmit = (e) => {
    e.preventDefault();
    
    if (!inputMessage.trim() || !isConnected) {
      return;
    }

    const message = {
      id: Date.now(),
      text: inputMessage.trim(),
      timestamp: new Date(),
      sender: 'user',
      emotion: currentEmotion,
      sessionId: sessionId
    };

    // 메시지 전송
    if (onMessageSend) {
      onMessageSend(message);
    }

    // 로컬 히스토리 업데이트
    setMessageHistory(prev => [...prev, message]);
    setInputMessage('');
    
    // 타이핑 시뮬레이션
    setIsTyping(true);
    setTimeout(() => {
      setIsTyping(false);
    }, 1500);
  };

  const handleInputChange = (e) => {
    setInputMessage(e.target.value);
  };

  const handleKeyPress = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      handleSubmit(e);
    }
  };

  const getEmotionIcon = (emotion) => {
    const icons = {
      joy: '😊',
      sad: '😢',
      anxiety: '😰',
      anger: '😠',
      neutral: '😐'
    };
    return icons[emotion] || '💭';
  };

  const getEmotionColor = (emotion) => {
    const colors = {
      joy: '#FFD700',
      sad: '#87CEEB',
      anxiety: '#FFA500',
      anger: '#DC143C',
      neutral: '#808080'
    };
    return colors[emotion] || '#808080';
  };

  const formatTime = (timestamp) => {
    if (!timestamp) return '';
    const date = new Date(timestamp);
    return date.toLocaleTimeString('ko-KR', { 
      hour: '2-digit', 
      minute: '2-digit',
      second: '2-digit'
    });
  };

  const generateBotResponse = (userMessage) => {
    // 간단한 응답 생성 (실제로는 서버에서 처리)
    const responses = {
      joy: [
        "기쁘신 마음이 느껴집니다! 😊 더 자세히 이야기해주세요.",
        "좋은 일이 있으셨나 보네요! 🌟 어떤 일인지 궁금해요.",
        "긍정적인 에너지가 전해집니다! ✨"
      ],
      sad: [
        "마음이 힘드시겠어요. 😔 제가 들어드릴게요.",
        "슬픈 일이 있으셨나요? 💙 천천히 말씀해주세요.",
        "괜찮습니다. 함께 이야기해봐요. 🤗"
      ],
      anxiety: [
        "불안하신 마음이 느껴집니다. 😰 어떤 것이 걱정되시나요?",
        "걱정이 많으시군요. 🌸 하나씩 정리해볼까요?",
        "긴장하지 마세요. 함께 해결해봐요. 💪"
      ],
      anger: [
        "화가 나신 것 같네요. 😤 무엇 때문인지 말씀해주세요.",
        "분노가 느껴집니다. 🔥 감정을 표현해주세요.",
        "힘든 상황이신가요? 저에게 털어놓으세요. 💭"
      ],
      neutral: [
        "네, 잘 들었습니다. 😐 더 자세히 설명해주시겠어요?",
        "그렇군요. 🤔 어떻게 생각하시나요?",
        "이해했습니다. 💭 계속 이야기해주세요."
      ]
    };

    const emotionResponses = responses[currentEmotion] || responses.neutral;
    const randomResponse = emotionResponses[Math.floor(Math.random() * emotionResponses.length)];
    
    return {
      id: Date.now() + 1,
      text: randomResponse,
      timestamp: new Date(),
      sender: 'therapist',
      emotion: currentEmotion,
      sessionId: sessionId
    };
  };

  // 자동 응답 시뮬레이션 (WebSocket 없이도 작동)
  useEffect(() => {
    if (messageHistory.length > 0) {
      const lastMessage = messageHistory[messageHistory.length - 1];
      if (lastMessage.sender === 'user') {
        const timer = setTimeout(() => {
          const botResponse = generateBotResponse(lastMessage);
          setMessageHistory(prev => [...prev, botResponse]);
        }, 1500);

        return () => clearTimeout(timer);
      }
    }
  }, [messageHistory, currentEmotion, sessionId]);

  return (
    <div className="chat-interface">
      {/* 채팅 헤더 */}
      <div className="chat-header">
        <div className="connection-status">
          <div className={`status-indicator ${isConnected ? 'connected' : 'disconnected'}`}></div>
          <span>{isConnected ? '연결됨' : '연결 끊김'}</span>
        </div>
        
        <div className="current-emotion">
          <span className="emotion-icon">{getEmotionIcon(currentEmotion)}</span>
          <span className="emotion-text">현재 감정: {currentEmotion}</span>
          {isAnalyzing && <span className="analyzing">분석 중...</span>}
        </div>
        
        {sessionId && (
          <div className="session-info">
            세션: {sessionId.slice(0, 8)}...
          </div>
        )}
      </div>

      {/* 메시지 목록 */}
      <div className="messages-container">
        {messageHistory.length === 0 ? (
          <div className="welcome-message">
            <h3>실시간 감정 상담 채팅</h3>
            <p>안녕하세요! 마음 편하게 이야기해주세요. 🌟</p>
            <p>당신의 감정을 실시간으로 분석하여 맞춤형 상담을 제공합니다.</p>
          </div>
        ) : (
          messageHistory.map((message, index) => (
            <div 
              key={message.id || index}
              className={`message ${message.sender}`}
            >
              <div className="message-content">
                <div 
                  className="message-bubble"
                  style={{
                    borderLeft: message.sender === 'user' 
                      ? `4px solid ${getEmotionColor(message.emotion)}` 
                      : '4px solid #e0e0e0'
                  }}
                >
                  <div className="message-text">{message.text}</div>
                  <div className="message-meta">
                    <span className="message-time">
                      {formatTime(message.timestamp)}
                    </span>
                    {message.sender === 'user' && (
                      <span className="message-emotion">
                        {getEmotionIcon(message.emotion)}
                      </span>
                    )}
                  </div>
                </div>
              </div>
            </div>
          ))
        )}
        
        {isTyping && (
          <div className="message therapist typing">
            <div className="message-content">
              <div className="message-bubble">
                <div className="typing-indicator">
                  <span></span>
                  <span></span>
                  <span></span>
                </div>
              </div>
            </div>
          </div>
        )}
        
        <div ref={messagesEndRef} />
      </div>

      {/* 입력 영역 */}
      <form onSubmit={handleSubmit} className="message-input-form">
        <div className="input-container">
          <textarea
            ref={inputRef}
            value={inputMessage}
            onChange={handleInputChange}
            onKeyPress={handleKeyPress}
            placeholder={isConnected ? "메시지를 입력하세요..." : "서버에 연결하는 중..."}
            disabled={!isConnected}
            rows="2"
            className="message-input"
          />
          <button 
            type="submit" 
            disabled={!inputMessage.trim() || !isConnected}
            className="send-button"
          >
            전송
          </button>
        </div>
        
        <div className="input-footer">
          <small>Enter: 전송 | Shift + Enter: 줄바꿈</small>
          {isAnalyzing && (
            <small className="analyzing-text">
              ⚡ 실시간 감정 분석 중...
            </small>
          )}
        </div>
      </form>

      <style jsx>{`
        .chat-interface {
          display: flex;
          flex-direction: column;
          height: 600px;
          background: white;
          border-radius: 15px;
          box-shadow: 0 10px 30px rgba(0, 0, 0, 0.1);
          overflow: hidden;
        }

        .chat-header {
          display: flex;
          justify-content: space-between;
          align-items: center;
          padding: 15px 20px;
          background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
          color: white;
          flex-wrap: wrap;
          gap: 10px;
        }

        .connection-status {
          display: flex;
          align-items: center;
          gap: 8px;
        }

        .status-indicator {
          width: 12px;
          height: 12px;
          border-radius: 50%;
          animation: pulse 2s infinite;
        }

        .status-indicator.connected {
          background: #2ecc71;
        }

        .status-indicator.disconnected {
          background: #e74c3c;
        }

        @keyframes pulse {
          0% { opacity: 1; }
          50% { opacity: 0.5; }
          100% { opacity: 1; }
        }

        .current-emotion {
          display: flex;
          align-items: center;
          gap: 8px;
          font-size: 14px;
        }

        .emotion-icon {
          font-size: 16px;
        }

        .analyzing {
          color: #f1c40f;
          font-size: 12px;
          animation: blink 1s infinite;
        }

        @keyframes blink {
          0%, 50% { opacity: 1; }
          51%, 100% { opacity: 0.5; }
        }

        .session-info {
          font-size: 12px;
          opacity: 0.8;
        }

        .messages-container {
          flex: 1;
          overflow-y: auto;
          padding: 20px;
          background: #f8f9fa;
        }

        .welcome-message {
          text-align: center;
          padding: 40px 20px;
          color: #666;
        }

        .welcome-message h3 {
          color: #333;
          margin-bottom: 15px;
        }

        .welcome-message p {
          margin: 10px 0;
          line-height: 1.6;
        }

        .message {
          margin-bottom: 20px;
          display: flex;
        }

        .message.user {
          justify-content: flex-end;
        }

        .message.therapist {
          justify-content: flex-start;
        }

        .message-content {
          max-width: 70%;
        }

        .message-bubble {
          padding: 12px 16px;
          border-radius: 18px;
          background: white;
          box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
          position: relative;
        }

        .message.user .message-bubble {
          background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
          color: white;
          border-bottom-right-radius: 6px;
        }

        .message.therapist .message-bubble {
          background: white;
          border-bottom-left-radius: 6px;
        }

        .message-text {
          line-height: 1.5;
          word-wrap: break-word;
        }

        .message-meta {
          display: flex;
          justify-content: space-between;
          align-items: center;
          margin-top: 8px;
          font-size: 12px;
          opacity: 0.7;
        }

        .typing-indicator {
          display: flex;
          gap: 4px;
          align-items: center;
        }

        .typing-indicator span {
          width: 8px;
          height: 8px;
          border-radius: 50%;
          background: #bbb;
          animation: typing 1.4s infinite;
        }

        .typing-indicator span:nth-child(2) {
          animation-delay: 0.2s;
        }

        .typing-indicator span:nth-child(3) {
          animation-delay: 0.4s;
        }

        @keyframes typing {
          0%, 60%, 100% {
            transform: translateY(0);
          }
          30% {
            transform: translateY(-10px);
          }
        }

        .message-input-form {
          padding: 20px;
          background: white;
          border-top: 1px solid #e0e0e0;
        }

        .input-container {
          display: flex;
          gap: 12px;
          align-items: flex-end;
        }

        .message-input {
          flex: 1;
          padding: 12px 16px;
          border: 2px solid #e0e0e0;
          border-radius: 20px;
          resize: none;
          font-family: inherit;
          font-size: 14px;
          line-height: 1.4;
          transition: border-color 0.3s ease;
        }

        .message-input:focus {
          outline: none;
          border-color: #667eea;
        }

        .message-input:disabled {
          background: #f5f5f5;
          color: #999;
        }

        .send-button {
          padding: 12px 20px;
          background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
          color: white;
          border: none;
          border-radius: 20px;
          cursor: pointer;
          font-weight: 600;
          transition: all 0.3s ease;
        }

        .send-button:hover:not(:disabled) {
          transform: translateY(-2px);
          box-shadow: 0 4px 12px rgba(102, 126, 234, 0.4);
        }

        .send-button:disabled {
          background: #ccc;
          cursor: not-allowed;
          transform: none;
          box-shadow: none;
        }

        .input-footer {
          display: flex;
          justify-content: space-between;
          align-items: center;
          margin-top: 8px;
          font-size: 12px;
          color: #666;
        }

        .analyzing-text {
          color: #f39c12;
          font-weight: 500;
        }

        /* 스크롤바 스타일링 */
        .messages-container::-webkit-scrollbar {
          width: 6px;
        }

        .messages-container::-webkit-scrollbar-track {
          background: #f1f1f1;
          border-radius: 3px;
        }

        .messages-container::-webkit-scrollbar-thumb {
          background: #c1c1c1;
          border-radius: 3px;
        }

        .messages-container::-webkit-scrollbar-thumb:hover {
          background: #a8a8a8;
        }
      `}</style>
    </div>
  );
};

export default ChatInterface;
