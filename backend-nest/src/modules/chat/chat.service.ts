import { Injectable } from '@nestjs/common';
import { EmotionService } from '../emotion/emotion.service';

@Injectable()
export class ChatService {
  constructor(private emotionService: EmotionService) {}

  async processMessage(userId: string, message: string): Promise<any> {
    try {
      const startTime = Date.now();

      // 1. 감정 분석
      const emotionResult = await this.emotionService.analyze(message);
      const emotionLatency = Date.now() - startTime;

      // 2. 아바타 상태 매핑 (병렬 처리로 성능 최적화)
      const [avatarResult, avatarStateResult] = await Promise.all([
        this.emotionService.generateAvatar(
          message,
          emotionResult.data.emotion,
        ),
        this.emotionService.mapEmotionToAvatarState(
          emotionResult.data.emotion,
          emotionResult.data.confidence,
          emotionResult.data.risk_level,
        ),
      ]);

      const totalLatency = Date.now() - startTime;

      // 3. 응답 생성
      const response = this.generateResponse(emotionResult.data);

      return {
        success: true,
        userId,
        message,
        emotion: emotionResult.data.emotion,
        confidence: emotionResult.data.confidence,
        riskLevel: emotionResult.data.risk_level,
        riskMessage: emotionResult.data.risk_message,
        avatar: avatarResult.data.image_base64,
        avatarState: avatarStateResult.data, // Live2D 파라미터
        response,
        timestamp: new Date().toISOString(),
        performance: {
          emotionLatency: `${emotionLatency}ms`,
          totalLatency: `${totalLatency}ms`,
          avatarTransitionDuration: `${avatarStateResult.data.transition_duration}ms`,
        },
      };
    } catch (error) {
      return {
        success: false,
        error: error.message,
        timestamp: new Date().toISOString(),
      };
    }
  }

  private generateResponse(emotionData: any): string {
    const responses = {
      joy: '행복한 감정이 느껴지네요! 😊 긍정적인 에너지를 계속 유지하세요!',
      sad: '힘든 시간을 보내고 계시는군요. 😢 제가 함께 있어드릴게요.',
      anxiety:
        '불안한 마음이 드시는군요. 😟 천천히 심호흡을 해보는 건 어떨까요?',
      anger: '화가 나셨나봐요. 😠 감정을 표현하는 것도 중요해요.',
      neutral: '평온한 하루를 보내고 계시네요. 😌',
    };

    return responses[emotionData.emotion] || '어떤 감정이신가요?';
  }
}
