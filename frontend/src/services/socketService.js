// WebSocket 연결 관리 서비스
import { io } from 'socket.io-client';

class SocketService {
  constructor() {
    this.socket = null;
    this.isConnected = false;
    this.connectionCallbacks = [];
    this.disconnectionCallbacks = [];
    this.eventHandlers = new Map();
  }

  connect(serverUrl = 'http://localhost:8006') {
    if (this.socket && this.isConnected) {
      console.log('Already connected to WebSocket');
      return Promise.resolve();
    }

    return new Promise((resolve, reject) => {
      try {
        this.socket = io(serverUrl, {
          transports: ['websocket', 'polling'],
          timeout: 10000,
          reconnection: true,
          reconnectionAttempts: 5,
          reconnectionDelay: 1000
        });

        this.socket.on('connect', () => {
          console.log('✅ WebSocket 연결 성공');
          this.isConnected = true;
          this.connectionCallbacks.forEach(callback => callback());
          resolve();
        });

        this.socket.on('disconnect', (reason) => {
          console.log('❌ WebSocket 연결 끊어짐:', reason);
          this.isConnected = false;
          this.disconnectionCallbacks.forEach(callback => callback(reason));
        });

        this.socket.on('connect_error', (error) => {
          console.error('❌ WebSocket 연결 오류:', error);
          reject(error);
        });

        // 기본 이벤트 핸들러 설정
        this.setupDefaultHandlers();

      } catch (error) {
        console.error('Socket 초기화 실패:', error);
        reject(error);
      }
    });
  }

  setupDefaultHandlers() {
    if (!this.socket) return;

    // 연결 상태 확인
    this.socket.on('connection_status', (data) => {
      console.log('연결 상태:', data);
    });

    // 아바타 업데이트
    this.socket.on('avatar_update', (data) => {
      console.log('아바타 업데이트:', data);
      this.emit('avatarUpdate', data);
    });

    // 아바타 전환
    this.socket.on('avatar_transition', (data) => {
      console.log('아바타 전환:', data);
      this.emit('avatarTransition', data);
    });

    // 감정 처리 완료
    this.socket.on('emotion_processed', (data) => {
      console.log('감정 처리 완료:', data);
      this.emit('emotionProcessed', data);
    });

    // 감정 분석 결과
    this.socket.on('emotion_analysis', (data) => {
      console.log('감정 분석 결과:', data);
      this.emit('emotionAnalysis', data);
    });

    // 세션 참가 완료
    this.socket.on('session_joined', (data) => {
      console.log('세션 참가 완료:', data);
      this.emit('sessionJoined', data);
    });

    // 전환 완료
    this.socket.on('avatar_transition_complete', (data) => {
      console.log('아바타 전환 완료:', data);
      this.emit('transitionComplete', data);
    });

    // 전환 실패
    this.socket.on('avatar_transition_failed', (data) => {
      console.error('아바타 전환 실패:', data);
      this.emit('transitionFailed', data);
    });
  }

  // 세션 참가
  joinSession(userId, sessionType = 'therapy') {
    if (!this.isConnected) {
      console.error('WebSocket이 연결되지 않았습니다');
      return;
    }

    this.socket.emit('join_session', {
      user_id: userId,
      session_type: sessionType
    });
  }

  // 감정 업데이트 전송
  updateEmotion(emotionData) {
    if (!this.isConnected) {
      console.error('WebSocket이 연결되지 않았습니다');
      return;
    }

    this.socket.emit('emotion_update', {
      emotion: emotionData.emotion,
      intensity: emotionData.intensity,
      context: emotionData.context || 'chatbot',
      confidence: emotionData.confidence || 1.0,
      user_id: emotionData.userId
    });
  }

  // 아바타 전환 요청
  requestAvatarTransition(userId, emotion, intensity = 0.5) {
    if (!this.isConnected) {
      console.error('WebSocket이 연결되지 않았습니다');
      return;
    }

    this.socket.emit('request_avatar_transition', {
      user_id: userId,
      emotion: emotion,
      intensity: intensity
    });
  }

  // 감정 분석 요청
  getEmotionAnalysis(userId) {
    if (!this.isConnected) {
      console.error('WebSocket이 연결되지 않았습니다');
      return;
    }

    this.socket.emit('get_emotion_analysis', {
      user_id: userId
    });
  }

  // 이벤트 리스너 등록
  on(eventName, callback) {
    if (!this.eventHandlers.has(eventName)) {
      this.eventHandlers.set(eventName, []);
    }
    this.eventHandlers.get(eventName).push(callback);
  }

  // 이벤트 리스너 제거
  off(eventName, callback) {
    if (this.eventHandlers.has(eventName)) {
      const handlers = this.eventHandlers.get(eventName);
      const index = handlers.indexOf(callback);
      if (index > -1) {
        handlers.splice(index, 1);
      }
    }
  }

  // 이벤트 발생
  emit(eventName, data) {
    if (this.eventHandlers.has(eventName)) {
      this.eventHandlers.get(eventName).forEach(callback => {
        try {
          callback(data);
        } catch (error) {
          console.error(`이벤트 핸들러 오류 (${eventName}):`, error);
        }
      });
    }
  }

  // 연결 상태 확인
  isSocketConnected() {
    return this.isConnected && this.socket && this.socket.connected;
  }

  // 연결 해제
  disconnect() {
    if (this.socket) {
      this.socket.disconnect();
      this.socket = null;
      this.isConnected = false;
    }
  }

  // 연결 콜백 등록
  onConnect(callback) {
    this.connectionCallbacks.push(callback);
  }

  onDisconnect(callback) {
    this.disconnectionCallbacks.push(callback);
  }
}

// 싱글톤 인스턴스 생성
const socketService = new SocketService();

export default socketService;
