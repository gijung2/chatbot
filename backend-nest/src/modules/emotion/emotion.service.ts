import { Injectable, HttpException, HttpStatus } from '@nestjs/common';
import { ConfigService } from '@nestjs/config';
import axios from 'axios';

@Injectable()
export class EmotionService {
  private readonly mlServiceUrl: string;

  constructor(private configService: ConfigService) {
    this.mlServiceUrl = this.configService.get<string>(
      'ML_SERVICE_URL',
      'http://localhost:8000',
    );
  }

  async analyze(text: string): Promise<any> {
    try {
      const response = await axios.post(
        `${this.mlServiceUrl}/api/v1/analyze`,
        { text },
        { timeout: 10000 },
      );

      return {
        success: true,
        data: response.data,
        timestamp: new Date().toISOString(),
      };
    } catch (error) {
      throw new HttpException(
        {
          success: false,
          message: 'ML 서비스 호출 실패',
          error: error.message,
        },
        HttpStatus.SERVICE_UNAVAILABLE,
      );
    }
  }

  async getHistory(userId: string): Promise<any> {
    // TODO: 데이터베이스에서 사용자별 감정 분석 히스토리 조회
    return {
      success: true,
      userId,
      history: [],
      message: '히스토리 조회 기능 구현 예정',
    };
  }

  async generateAvatar(text: string, emotion?: string): Promise<any> {
    try {
      const response = await axios.post(
        `${this.mlServiceUrl}/api/v1/generate-avatar`,
        { text, emotion, style: 'gradient' },
        { timeout: 10000 },
      );

      return {
        success: true,
        data: response.data,
      };
    } catch (error) {
      throw new HttpException(
        {
          success: false,
          message: '아바타 생성 실패',
          error: error.message,
        },
        HttpStatus.SERVICE_UNAVAILABLE,
      );
    }
  }

  async mapEmotionToAvatarState(
    emotion: string,
    confidence: number,
    riskLevel: string = 'low',
  ): Promise<any> {
    try {
      const response = await axios.post(
        `${this.mlServiceUrl}/api/v1/map-emotion`,
        {
          emotion,
          confidence,
          risk_level: riskLevel,
        },
        { timeout: 3000 }, // 200ms 목표 달성 위해 짧은 타임아웃
      );

      return {
        success: true,
        data: response.data,
        latency: response.headers['x-response-time'] || 'N/A',
      };
    } catch (error) {
      throw new HttpException(
        {
          success: false,
          message: '아바타 상태 매핑 실패',
          error: error.message,
        },
        HttpStatus.SERVICE_UNAVAILABLE,
      );
    }
  }
}
