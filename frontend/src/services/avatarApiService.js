// Avatar API 호출 서비스
import axios from 'axios';

class AvatarApiService {
  constructor() {
    this.baseURL = 'http://localhost:8002';
    this.emotionServerURL = 'http://localhost:5000';
    
    // Axios 인스턴스 생성
    this.api = axios.create({
      baseURL: this.baseURL,
      timeout: 10000,
      headers: {
        'Content-Type': 'application/json'
      }
    });

    this.emotionApi = axios.create({
      baseURL: this.emotionServerURL,
      timeout: 5000,
      headers: {
        'Content-Type': 'application/json'
      }
    });
  }

  // 아바타 생성
  async generateAvatar(emotionData) {
    try {
      const response = await this.api.post('/api/avatar', {
        emotion: emotionData.emotion || 'neutral',
        intensity: emotionData.intensity || 0.5,
        style: emotionData.style || 'default'
      });

      if (response.data.success) {
        return {
          success: true,
          avatar: response.data,
          imageBase64: response.data.image_base64,
          emotion: response.data.emotion,
          description: response.data.description
        };
      } else {
        throw new Error(response.data.message || 'Avatar generation failed');
      }
    } catch (error) {
      console.error('아바타 생성 실패:', error);
      return {
        success: false,
        error: error.message,
        fallback: this.getFallbackAvatar(emotionData.emotion)
      };
    }
  }

  // 감정 분석 (감정 서버가 있는 경우)
  async analyzeEmotion(text) {
    try {
      const response = await this.emotionApi.post('/analyze', {
        text: text
      });

      return {
        success: true,
        emotion: response.data.emotion,
        confidence: response.data.confidence,
        details: response.data.details || {}
      };
    } catch (error) {
      console.warn('감정 분석 서버 사용 불가, 키워드 분석 사용:', error);
      // 클라이언트 사이드 키워드 기반 감정 분석
      return this.analyzeEmotionKeywords(text);
    }
  }

  // 클라이언트 사이드 키워드 기반 감정 분석
  analyzeEmotionKeywords(text) {
    const emotions = {
      joy: ['기쁘', '행복', '좋', '즐거', '웃', '사랑', '고마', '감사', '만족', '기분좋', '신나', '최고'],
      sad: ['슬프', '우울', '힘들', '괴로', '아프', '눈물', '절망', '외로', '그리', '안타까', '서러'],
      anxiety: ['불안', '걱정', '두려', '긴장', '초조', '떨려', '무서', '심각', '조급', '답답'],
      anger: ['화나', '짜증', '분노', '열받', '빡쳐', '미치', '싫어', '짜증나', '억울', '분해'],
      neutral: ['그냥', '보통', '괜찮', '별로', '그럭저럭', '일반', '평범']
    };

    const text_lower = text.toLowerCase();
    const emotionScores = {};

    // 각 감정별 키워드 매칭
    Object.keys(emotions).forEach(emotion => {
      let score = 0;
      emotions[emotion].forEach(keyword => {
        const matches = (text_lower.match(new RegExp(keyword, 'g')) || []).length;
        score += matches;
      });
      emotionScores[emotion] = score;
    });

    // 가장 높은 점수의 감정 찾기
    const maxEmotion = Object.keys(emotionScores).reduce((a, b) => 
      emotionScores[a] > emotionScores[b] ? a : b
    );

    const maxScore = emotionScores[maxEmotion];
    const confidence = maxScore > 0 ? Math.min(maxScore * 0.3 + 0.3, 1.0) : 0.3;

    return {
      success: true,
      emotion: maxScore > 0 ? maxEmotion : 'neutral',
      confidence: confidence,
      details: {
        scores: emotionScores,
        method: 'keyword_analysis',
        text_length: text.length
      }
    };
  }

  // 사용 가능한 감정 목록 조회
  async getAvailableEmotions() {
    try {
      const response = await this.api.get('/api/emotions');
      return {
        success: true,
        emotions: response.data.emotions,
        usage: response.data.usage
      };
    } catch (error) {
      console.error('감정 목록 조회 실패:', error);
      return {
        success: false,
        error: error.message,
        fallback: this.getDefaultEmotions()
      };
    }
  }

  // 서버 상태 확인
  async checkHealth() {
    try {
      const response = await this.api.get('/api/health');
      return {
        success: true,
        status: response.data.status,
        service: response.data.service
      };
    } catch (error) {
      console.error('서버 상태 확인 실패:', error);
      return {
        success: false,
        error: error.message
      };
    }
  }

  // 기본 감정 목록
  getDefaultEmotions() {
    return {
      joy: {
        description: "기쁨 - 밝고 긍정적인 상담사",
        color: [255, 223, 85],
        expression: "smile"
      },
      sad: {
        description: "슬픔 - 공감하고 위로하는 상담사",
        color: [135, 206, 235],
        expression: "gentle"
      },
      anxiety: {
        description: "불안 - 안정감을 주는 상담사",
        color: [255, 165, 79],
        expression: "calm"
      },
      anger: {
        description: "분노 - 차분하고 이해심 있는 상담사",
        color: [220, 20, 60],
        expression: "understanding"
      },
      neutral: {
        description: "중립 - 전문적이고 신뢰할 수 있는 상담사",
        color: [128, 128, 128],
        expression: "professional"
      }
    };
  }

  // Fallback 아바타 (서버 오류시)
  getFallbackAvatar(emotion = 'neutral') {
    return {
      success: true,
      emotion: emotion,
      description: `${emotion} 감정 상담사 (기본)`,
      source: 'fallback',
      image_base64: this.generateFallbackImage(emotion)
    };
  }

  // 간단한 fallback 이미지 생성 (SVG)
  generateFallbackImage(emotion) {
    const colors = {
      joy: '#FFD700',
      sad: '#87CEEB', 
      anxiety: '#FFA500',
      anger: '#DC143C',
      neutral: '#808080'
    };

    const color = colors[emotion] || colors.neutral;
    
    const svg = `
      <svg width="200" height="200" xmlns="http://www.w3.org/2000/svg">
        <circle cx="100" cy="100" r="80" fill="${color}" stroke="#333" stroke-width="3"/>
        <circle cx="75" cy="80" r="8" fill="#000"/>
        <circle cx="125" cy="80" r="8" fill="#000"/>
        <path d="M 70 120 Q 100 140 130 120" stroke="#000" stroke-width="3" fill="none"/>
        <text x="100" y="170" text-anchor="middle" font-family="Arial" font-size="12" fill="#333">${emotion}</text>
      </svg>
    `;

    return btoa(svg);
  }

  // 감정 강도에 따른 색상 조정
  getEmotionColor(emotion, intensity = 0.5) {
    const baseColors = {
      joy: [255, 223, 85],
      sad: [135, 206, 235],
      anxiety: [255, 165, 79],
      anger: [220, 20, 60],
      neutral: [128, 128, 128]
    };

    const color = baseColors[emotion] || baseColors.neutral;
    
    // 강도에 따라 색상 조정
    const adjustedColor = color.map(c => 
      Math.round(c * (0.5 + intensity * 0.5))
    );

    return `rgb(${adjustedColor.join(',')})`;
  }
}

// 싱글톤 인스턴스 생성
const avatarApiService = new AvatarApiService();

export default avatarApiService;
