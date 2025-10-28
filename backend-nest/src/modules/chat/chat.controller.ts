import { Controller, Get, Post, Body, Param } from '@nestjs/common';
import { ApiTags, ApiOperation } from '@nestjs/swagger';
import { ChatService } from './chat.service';

@ApiTags('chat')
@Controller('api/chat')
export class ChatController {
  constructor(private chatService: ChatService) {}

  @Post('message')
  @ApiOperation({ summary: '메시지 전송 (HTTP)' })
  async sendMessage(
    @Body() body: { userId: string; message: string },
  ): Promise<any> {
    return this.chatService.processMessage(body.userId, body.message);
  }

  @Get('history/:userId')
  @ApiOperation({ summary: '채팅 히스토리 조회' })
  async getHistory(@Param('userId') userId: string): Promise<any> {
    return {
      success: true,
      userId,
      history: [],
      message: '히스토리 조회 기능 구현 예정',
    };
  }
}
