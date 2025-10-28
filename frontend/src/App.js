// 메인 애플리케이션 컴포넌트
import React, { useState, useEffect, useCallback } from 'react';
import RealtimeAvatar from './components/RealtimeAvatar';
import ChatInterface from './components/ChatInterface';
import socketService from './services/socketService';
import avatarApiService from './services/avatarApiService';

function App() {
  // 연결 상태
  const [isConnected, setIsConnected] = useState(false);
  const [connectionError, setConnectionError] = useState(null);
  
  // 감정 상태
  const [currentEmotion, setCurrentEmotion] = useState('neutral');
  const [emotionIntensity, setEmotionIntensity] = useState(0.5);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  
  // 아바타 상태
  const [currentAvatar, setCurrentAvatar] = useState(null);
  const [isAvatarTransitioning, setIsAvatarTransitioning] = useState(false);
  
  // 채팅 상태
  const [messages, setMessages] = useState([]);
  const [sessionId, setSessionId] = useState(null);
  
  // 시스템 상태
  const [systemStats, setSystemStats] = useState({});
  const [isInitialized, setIsInitialized] = useState(false);

  // 초기화
  useEffect(() => {
    initializeApp();
    
    return () => {
      cleanup();
    };
  }, []);

  const initializeApp = async () => {
    try {
      console.log('🚀 실시간 아바타 상담 시스템 초기화 중...');
      
      // 빠른 초기화를 위해 API 체크를 더 간단하게
      try {
        const apiHealth = await avatarApiService.checkHealth();
        console.log('📡 API 서버 상태:', apiHealth);
      } catch (error) {
        console.warn('⚠️ API 서버 연결 실패, 계속 진행:', error);
      }
      
      // 소켓 연결 설정 (백그라운드에서 처리)
      setupSocketConnection();
      
      // 기본 아바타 생성 (백그라운드에서 처리)
      generateDefaultAvatar().catch(error => {
        console.warn('⚠️ 기본 아바타 생성 실패, 계속 진행:', error);
      });
      
      // 즉시 초기화 완료로 처리
      setIsInitialized(true);
      console.log('✅ 시스템 초기화 완료');
      
    } catch (error) {
      console.error('❌ 초기화 실패:', error);
      // 에러가 있어도 앱을 시작할 수 있도록 함
      setIsInitialized(true);
      setConnectionError('일부 기능이 제한될 수 있습니다.');
    }
  };

  const setupSocketConnection = () => {
    console.log('🔌 WebSocket 연결 설정 중...');
    
    // 연결 이벤트
    socketService.onConnect(() => {
      console.log('✅ WebSocket 연결됨');
      setIsConnected(true);
      setConnectionError(null);
    });

    socketService.onDisconnect(() => {
      console.log('❌ WebSocket 연결 끊김');
      setIsConnected(false);
    });

    socketService.onError((error) => {
      console.error('🚨 WebSocket 오류:', error);
      // WebSocket 오류가 있어도 앱 사용은 가능하도록 함
      setConnectionError('실시간 기능이 제한될 수 있습니다.');
    });

    // 감정 업데이트 이벤트
    socketService.onEmotionUpdate((data) => {
      console.log('💭 감정 업데이트:', data);
      handleEmotionUpdate(data);
    });

    // 아바타 업데이트 이벤트
    socketService.onAvatarUpdate((data) => {
      console.log('🎭 아바타 업데이트:', data);
      handleAvatarUpdate(data);
    });

    // 세션 업데이트 이벤트
    socketService.onSessionUpdate((data) => {
      console.log('📊 세션 업데이트:', data);
      setSessionId(data.session_id);
      setSystemStats(data.stats || {});
    });

    // 메시지 이벤트
    socketService.onMessage((data) => {
      console.log('💬 메시지 수신:', data);
      handleNewMessage(data);
    });

    // 연결 시작 (에러가 있어도 계속 진행)
    socketService.connect().catch(error => {
      console.warn('⚠️ WebSocket 초기 연결 실패, 나중에 재시도됩니다:', error);
    });
  };

  const cleanup = () => {
    console.log('🧹 시스템 정리 중...');
    socketService.disconnect();
  };

  const generateDefaultAvatar = async () => {
    try {
      console.log('🎨 기본 아바타 생성 중...');
      const avatarData = await avatarApiService.generateAvatar('neutral', {
        description: '상담을 위한 중립적인 아바타입니다.',
        style: 'professional'
      });
      
      setCurrentAvatar(avatarData);
      console.log('✅ 기본 아바타 생성 완료');
      
    } catch (error) {
      console.error('❌ 기본 아바타 생성 실패:', error);
      // 기본 아바타 생성 실패시 컴포넌트 자체에서 fallback 생성
    }
  };

  const handleEmotionUpdate = useCallback(async (emotionData) => {
    const { emotion, intensity, confidence } = emotionData;
    
    console.log(`😊 감정 변화: ${emotion} (강도: ${intensity}, 신뢰도: ${confidence})`);
    
    setIsAnalyzing(true);
    setCurrentEmotion(emotion);
    setEmotionIntensity(intensity);
    
    // 감정 변화가 충분할 때만 새 아바타 생성
    if (intensity > 0.6 && confidence > 0.7) {
      await generateEmotionAvatar(emotion, intensity);
    }
    
    setTimeout(() => setIsAnalyzing(false), 1000);
  }, []);

  const handleAvatarUpdate = useCallback((avatarData) => {
    console.log('🎭 아바타 전환 시작');
    setIsAvatarTransitioning(true);
    
    setTimeout(() => {
      setCurrentAvatar(avatarData);
      setIsAvatarTransitioning(false);
      console.log('✅ 아바타 전환 완료');
    }, 500);
  }, []);

  const handleNewMessage = useCallback((messageData) => {
    const formattedMessage = {
      id: messageData.id || Date.now(),
      text: messageData.text,
      timestamp: new Date(messageData.timestamp),
      sender: messageData.sender,
      emotion: messageData.emotion,
      sessionId: messageData.session_id
    };
    
    setMessages(prev => [...prev, formattedMessage]);
  }, []);

  const generateEmotionAvatar = async (emotion, intensity) => {
    try {
      console.log(`🎨 ${emotion} 아바타 생성 중... (강도: ${intensity})`);
      
      const avatarData = await avatarApiService.generateAvatar(emotion, {
        intensity: intensity,
        style: 'therapist',
        description: `${emotion} 감정에 맞는 상담사 아바타`
      });
      
      // WebSocket으로 아바타 업데이트 전송
      socketService.sendAvatarUpdate(avatarData);
      
    } catch (error) {
      console.error('❌ 감정 아바타 생성 실패:', error);
    }
  };

  const handleMessageSend = useCallback(async (message) => {
    try {
      console.log('📤 메시지 전송:', message.text);
      
      // 로컬 메시지 추가
      setMessages(prev => [...prev, message]);
      
      // 감정 분석 시작
      setIsAnalyzing(true);
      
      // 서버로 메시지 전송 (감정 분석 포함)
      const analysisResult = await avatarApiService.analyzeEmotion(message.text);
      
      if (analysisResult && analysisResult.emotion) {
        // WebSocket으로 감정 업데이트 전송
        socketService.sendEmotionUpdate({
          emotion: analysisResult.emotion,
          intensity: analysisResult.intensity || 0.5,
          confidence: analysisResult.confidence || 0.8,
          text: message.text,
          timestamp: message.timestamp
        });
      }
      
      // WebSocket으로 메시지 전송
      socketService.sendMessage(message);
      
    } catch (error) {
      console.error('❌ 메시지 전송 실패:', error);
    } finally {
      setIsAnalyzing(false);
    }
  }, []);

  const handleRetryConnection = () => {
    console.log('🔄 연결 재시도 중...');
    setConnectionError(null);
    socketService.connect();
  };

  if (!isInitialized) {
    return (
      <div className="app-loading">
        <div className="loading-container">
          <div className="loading-spinner"></div>
          <h2>실시간 아바타 상담 시스템</h2>
          <p>시스템을 초기화하고 있습니다...</p>
          {connectionError && (
            <div className="error-message">
              <p>⚠️ {connectionError}</p>
              <button onClick={handleRetryConnection}>다시 시도</button>
            </div>
          )}
        </div>
      </div>
    );
  }

  return (
    <div className="app">
      <header className="app-header">
        <h1>🤖 실시간 감정 아바타 상담</h1>
        <div className="header-status">
          <div className={`connection-badge ${isConnected ? 'connected' : 'disconnected'}`}>
            {isConnected ? '🟢 연결됨' : '🔴 연결 끊김'}
          </div>
          {sessionId && (
            <div className="session-badge">
              세션: {sessionId.slice(0, 8)}
            </div>
          )}
        </div>
      </header>

      <main className="app-main">
        <div className="content-container">
          {/* 아바타 섹션 */}
          <div className="avatar-section">
            <RealtimeAvatar
              currentEmotion={currentEmotion}
              intensity={emotionIntensity}
              avatarData={currentAvatar}
              isTransitioning={isAvatarTransitioning}
              transitionDuration={1.0}
              onTransitionComplete={() => console.log('✅ 아바타 전환 완료')}
            />
            
            {/* 시스템 정보 */}
            <div className="system-info">
              <div className="info-item">
                <span className="info-label">현재 감정:</span>
                <span className="info-value">{currentEmotion}</span>
              </div>
              <div className="info-item">
                <span className="info-label">감정 강도:</span>
                <span className="info-value">{Math.round(emotionIntensity * 100)}%</span>
              </div>
              {systemStats.total_messages && (
                <div className="info-item">
                  <span className="info-label">총 메시지:</span>
                  <span className="info-value">{systemStats.total_messages}</span>
                </div>
              )}
            </div>
          </div>

          {/* 채팅 섹션 */}
          <div className="chat-section">
            <ChatInterface
              onMessageSend={handleMessageSend}
              messages={messages}
              isConnected={isConnected}
              currentEmotion={currentEmotion}
              isAnalyzing={isAnalyzing}
              sessionId={sessionId}
            />
          </div>
        </div>
      </main>

      {/* 연결 오류 모달 */}
      {connectionError && (
        <div className="error-modal">
          <div className="error-content">
            <h3>⚠️ 연결 오류</h3>
            <p>{connectionError}</p>
            <div className="error-actions">
              <button onClick={handleRetryConnection}>다시 연결</button>
              <button onClick={() => setConnectionError(null)}>닫기</button>
            </div>
          </div>
        </div>
      )}

      <style jsx>{`
        .app {
          min-height: 100vh;
          background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
          font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        }

        .app-loading {
          display: flex;
          justify-content: center;
          align-items: center;
          min-height: 100vh;
          background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        }

        .loading-container {
          text-align: center;
          color: white;
          padding: 40px;
          background: rgba(255, 255, 255, 0.1);
          border-radius: 20px;
          backdrop-filter: blur(10px);
        }

        .loading-spinner {
          width: 50px;
          height: 50px;
          border: 4px solid rgba(255, 255, 255, 0.3);
          border-top: 4px solid white;
          border-radius: 50%;
          animation: spin 1s linear infinite;
          margin: 0 auto 20px;
        }

        @keyframes spin {
          0% { transform: rotate(0deg); }
          100% { transform: rotate(360deg); }
        }

        .app-header {
          display: flex;
          justify-content: space-between;
          align-items: center;
          padding: 20px 40px;
          background: rgba(255, 255, 255, 0.1);
          backdrop-filter: blur(10px);
          color: white;
          border-bottom: 1px solid rgba(255, 255, 255, 0.2);
        }

        .app-header h1 {
          margin: 0;
          font-size: 1.8em;
          font-weight: 600;
        }

        .header-status {
          display: flex;
          gap: 15px;
          align-items: center;
        }

        .connection-badge, .session-badge {
          padding: 6px 12px;
          border-radius: 20px;
          font-size: 0.9em;
          font-weight: 500;
        }

        .connection-badge.connected {
          background: rgba(46, 204, 113, 0.2);
          color: #2ecc71;
          border: 1px solid rgba(46, 204, 113, 0.3);
        }

        .connection-badge.disconnected {
          background: rgba(231, 76, 60, 0.2);
          color: #e74c3c;
          border: 1px solid rgba(231, 76, 60, 0.3);
        }

        .session-badge {
          background: rgba(255, 255, 255, 0.2);
          color: white;
          border: 1px solid rgba(255, 255, 255, 0.3);
        }

        .app-main {
          padding: 30px 40px;
        }

        .content-container {
          display: grid;
          grid-template-columns: 400px 1fr;
          gap: 30px;
          max-width: 1400px;
          margin: 0 auto;
        }

        .avatar-section {
          display: flex;
          flex-direction: column;
          gap: 20px;
        }

        .system-info {
          background: rgba(255, 255, 255, 0.95);
          border-radius: 15px;
          padding: 20px;
          box-shadow: 0 10px 30px rgba(0, 0, 0, 0.1);
        }

        .info-item {
          display: flex;
          justify-content: space-between;
          align-items: center;
          margin-bottom: 12px;
          padding-bottom: 8px;
          border-bottom: 1px solid #eee;
        }

        .info-item:last-child {
          margin-bottom: 0;
          border-bottom: none;
        }

        .info-label {
          font-weight: 500;
          color: #666;
        }

        .info-value {
          font-weight: 600;
          color: #333;
        }

        .chat-section {
          display: flex;
          flex-direction: column;
        }

        .error-modal {
          position: fixed;
          top: 0;
          left: 0;
          right: 0;
          bottom: 0;
          background: rgba(0, 0, 0, 0.5);
          display: flex;
          justify-content: center;
          align-items: center;
          z-index: 1000;
        }

        .error-content {
          background: white;
          padding: 30px;
          border-radius: 15px;
          box-shadow: 0 20px 40px rgba(0, 0, 0, 0.3);
          text-align: center;
          max-width: 400px;
          margin: 20px;
        }

        .error-content h3 {
          margin: 0 0 15px 0;
          color: #e74c3c;
        }

        .error-content p {
          margin: 0 0 20px 0;
          color: #666;
          line-height: 1.5;
        }

        .error-actions {
          display: flex;
          gap: 10px;
          justify-content: center;
        }

        .error-actions button {
          padding: 10px 20px;
          border: none;
          border-radius: 8px;
          cursor: pointer;
          font-weight: 500;
          transition: all 0.3s ease;
        }

        .error-actions button:first-child {
          background: #3498db;
          color: white;
        }

        .error-actions button:first-child:hover {
          background: #2980b9;
        }

        .error-actions button:last-child {
          background: #ecf0f1;
          color: #7f8c8d;
        }

        .error-actions button:last-child:hover {
          background: #d5dbdb;
        }

        .error-message {
          margin-top: 20px;
          padding: 15px;
          background: rgba(231, 76, 60, 0.1);
          border: 1px solid rgba(231, 76, 60, 0.3);
          border-radius: 8px;
          color: #e74c3c;
        }

        .error-message button {
          margin-top: 10px;
          padding: 8px 16px;
          background: #e74c3c;
          color: white;
          border: none;
          border-radius: 5px;
          cursor: pointer;
        }

        .error-message button:hover {
          background: #c0392b;
        }

        /* 반응형 디자인 */
        @media (max-width: 1024px) {
          .content-container {
            grid-template-columns: 1fr;
            gap: 20px;
          }
          
          .app-header {
            padding: 15px 20px;
            flex-direction: column;
            gap: 10px;
          }
          
          .app-main {
            padding: 20px;
          }
        }

        @media (max-width: 768px) {
          .app-header h1 {
            font-size: 1.4em;
          }
          
          .header-status {
            flex-direction: column;
            gap: 8px;
          }
          
          .error-content {
            margin: 10px;
            padding: 20px;
          }
        }
      `}</style>
    </div>
  );
}

export default App;
