'use client'

import { useEffect, useState } from 'react'

interface AvatarCardProps {
  emotion: string
  confidence: number
  isConnected: boolean
  avatarState?: any // Live2D 파라미터
}

export default function AvatarCard({ emotion, confidence, isConnected, avatarState }: AvatarCardProps) {
  const [avatarImage, setAvatarImage] = useState<string | null>(null)
  const [transitionProgress, setTransitionProgress] = useState(0)

  // 감정 변경 시 부드러운 전환 애니메이션
  useEffect(() => {
    if (avatarState) {
      // 200ms 이내 전환 애니메이션
      const duration = avatarState.transition_duration || 200
      const steps = 20
      const interval = duration / steps

      let currentStep = 0
      const timer = setInterval(() => {
        currentStep++
        setTransitionProgress(currentStep / steps)
        
        if (currentStep >= steps) {
          clearInterval(timer)
          setTransitionProgress(1)
        }
      }, interval)

      return () => clearInterval(timer)
    }
  }, [avatarState])

  // 감정별 이모지 매핑
  const emotionEmojis: Record<string, string> = {
    joy: '😊',
    sad: '😢',
    anxiety: '😰',
    anger: '😠',
    neutral: '😐',
  }

  // 감정별 한글명
  const emotionNames: Record<string, string> = {
    joy: '기쁨',
    sad: '슬픔',
    anxiety: '불안',
    anger: '분노',
    neutral: '중립',
  }

  // 감정별 배경색
  const emotionColors: Record<string, string> = {
    joy: 'from-yellow-300 to-yellow-400',
    sad: 'from-blue-300 to-blue-400',
    anxiety: 'from-purple-300 to-purple-400',
    anger: 'from-red-300 to-red-400',
    neutral: 'from-gray-300 to-gray-400',
  }

  // 감정별 메시지
  const emotionMessages: Record<string, string> = {
    joy: '좋은 마음을 느끼네요',
    sad: '슬픈 마음을 이해해요',
    anxiety: '불안한 마음을 달래드려요',
    anger: '화난 마음을 이해해요',
    neutral: '편안한 마음을 느껴요',
  }

  const emoji = emotionEmojis[emotion] || '😐'
  const emotionName = emotionNames[emotion] || '중립'
  const bgColor = emotionColors[emotion] || 'from-gray-300 to-gray-400'
  const message = emotionMessages[emotion] || '함께 이야기해요'

  // Live2D 파라미터 기반 애니메이션 스타일
  const getAnimationStyle = () => {
    if (!avatarState || !avatarState.parameters) return {}
    
    const params = avatarState.parameters
    return {
      transform: `
        rotate(${params.head_tilt * 10}deg) 
        translateX(${params.body_rotation * 20}px)
        scale(${1 + (params.eye_smile * 0.05)})
      `,
      transition: `transform ${avatarState.transition_duration || 200}ms ease-out`,
    }
  }

  // 빠른 감정 선택 버튼들
  const quickEmotions = [
    { key: 'joy', label: '기쁨', emoji: '😊' },
    { key: 'sad', label: '슬픔', emoji: '😢' },
    { key: 'anxiety', label: '불안', emoji: '😰' },
    { key: 'anger', label: '분노', emoji: '😠' },
  ]

  return (
    <div className="bg-white/95 backdrop-blur rounded-3xl shadow-2xl p-6 h-full flex flex-col">
      {/* 아바타 디스플레이 */}
      <div className={`bg-gradient-to-br ${bgColor} rounded-2xl p-8 flex items-center justify-center mb-6 aspect-square relative overflow-hidden`}>
        <div 
          className="text-9xl transition-all duration-200" 
          style={getAnimationStyle()}
        >
          {emoji}
        </div>
        
        {/* 특수 제스처 표시 */}
        {avatarState?.special_gesture && (
          <div className="absolute top-4 right-4 bg-red-500 text-white text-xs px-3 py-1 rounded-full animate-pulse">
            ⚠️ {avatarState.alert_level}
          </div>
        )}
        
        {/* 전환 진행도 표시 (디버그용) */}
        {transitionProgress < 1 && transitionProgress > 0 && (
          <div className="absolute bottom-4 left-4 right-4 bg-white/30 rounded-full h-2">
            <div 
              className="bg-white h-full rounded-full transition-all"
              style={{ width: `${transitionProgress * 100}%` }}
            />
          </div>
        )}
      </div>

      {/* 감정 상태 정보 */}
      <div className="text-center mb-6">
        <h2 className="text-2xl font-bold text-gray-800 mb-2">
          {emotionName} 상담사
        </h2>
        <p className="text-sm text-gray-600 mb-3">
          {message}
        </p>
        {confidence > 0 && (
          <div className="bg-gray-100 rounded-full px-4 py-2 inline-block">
            <span className="text-xs text-gray-700">
              신뢰도: {(confidence * 100).toFixed(1)}%
            </span>
          </div>
        )}
        
        {/* Live2D 파라미터 표시 (디버그용) */}
        {avatarState && (
          <div className="mt-3 text-xs text-gray-400">
            <div>표정: {avatarState.expression}</div>
            <div>애니메이션: {avatarState.animation}</div>
            <div>전환시간: {avatarState.transition_duration}ms</div>
          </div>
        )}
      </div>

      {/* 빠른 감정 선택 */}
      <div className="mb-4">
        <p className="text-xs text-gray-500 mb-3 text-center">빠른 감정 선택</p>
        <div className="grid grid-cols-2 gap-2">
          {quickEmotions.map((emo) => (
            <button
              key={emo.key}
              className={`
                py-3 px-2 rounded-xl text-sm font-medium
                transition-all duration-200
                ${emotion === emo.key
                  ? 'bg-gradient-to-br from-purple-500 to-blue-500 text-white shadow-lg scale-105'
                  : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
                }
              `}
            >
              <span className="text-xl">{emo.emoji}</span>
              <span className="ml-2 text-xs">{emo.label}</span>
            </button>
          ))}
        </div>
      </div>

      {/* 연결 상태 */}
      <div className="mt-auto pt-4 border-t border-gray-200">
        <div className="flex items-center justify-between text-xs">
          <span className="text-gray-500">상태</span>
          <span className={`px-3 py-1 rounded-full ${isConnected ? 'bg-green-100 text-green-700' : 'bg-red-100 text-red-700'}`}>
            {isConnected ? '● 연결됨' : '● 오프라인'}
          </span>
        </div>
        <div className="mt-2 text-center">
          <p className="text-xs text-gray-400">
            오늘 01:{new Date().getMinutes().toString().padStart(2, '0')}
          </p>
        </div>
      </div>
    </div>
  )
}
