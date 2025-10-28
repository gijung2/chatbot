import { Injectable } from '@nestjs/common';

@Injectable()
export class AppService {
  getHello(): object {
    return {
      message: 'Welcome to Emotion Analysis Chatbot Backend API',
      version: '1.0.0',
      endpoints: {
        docs: '/api/docs',
        health: '/health',
        chat: '/api/chat',
        emotion: '/api/emotion',
        analytics: '/api/analytics',
        user: '/api/user',
      },
    };
  }
}
