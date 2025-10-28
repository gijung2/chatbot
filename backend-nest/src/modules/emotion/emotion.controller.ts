import { Controller, Post, Body, Get, Param } from '@nestjs/common';
import { ApiTags, ApiOperation, ApiResponse } from '@nestjs/swagger';
import { EmotionService } from './emotion.service';
import { AnalyzeEmotionDto } from './dto/analyze-emotion.dto';

@ApiTags('emotion')
@Controller('api/emotion')
export class EmotionController {
  constructor(private readonly emotionService: EmotionService) {}

  @Post('analyze')
  @ApiOperation({ summary: '감정 분석' })
  @ApiResponse({ status: 200, description: '감정 분석 성공' })
  async analyze(@Body() dto: AnalyzeEmotionDto) {
    return this.emotionService.analyze(dto.text);
  }

  @Get('history/:userId')
  @ApiOperation({ summary: '사용자 감정 분석 히스토리 조회' })
  async getHistory(@Param('userId') userId: string) {
    return this.emotionService.getHistory(userId);
  }
}
