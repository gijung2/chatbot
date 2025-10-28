'use client'

import { useState, useEffect, useRef } from 'react'
import { useSocket } from '@/hooks/useSocket'

export default function ChatPage() {
  const [messages, setMessages] = useState<any[]>([])
  const [userId] = useState(() => `user-${Date.now()}`)
  const [currentEmotion, setCurrentEmotion] = useState('neutral')
  const [emotionConfidence, setEmotionConfidence] = useState(0)
  const [avatarImage, setAvatarImage] = useState<string | null>(null)
  const [connectionTime, setConnectionTime] = useState(0)
  const [isTyping, setIsTyping] = useState(false)
  const messagesEndRef = useRef<HTMLDivElement>(null)

  const { socket, isConnected, sendMessage } = useSocket()

  const emotions = {
    neutral: { 
      icon: '😐', 
      name: '중립', 
      korean: '중립',
      color: '#E0E0E0',
      bgGradient: 'from-gray-100 to-gray-200'
    },
    joy: { 
      icon: '😊', 
      name: 'Joy', 
      korean: '기쁨',
      color: '#FFD700',
      bgGradient: 'from-yellow-100 to-yellow-200'
    },
    sad: { 
      icon: '😢', 
      name: 'Sad', 
      korean: '슬픔',
      color: '#6495ED',
      bgGradient: 'from-blue-100 to-blue-200'
    },
    anxiety: { 
      icon: '😰', 
      name: 'Anxiety', 
      korean: '불안',
      color: '#9370DB',
      bgGradient: 'from-purple-100 to-purple-200'
    },
    anger: { 
      icon: '😠', 
      name: 'Anger', 
      korean: '분노',
      color: '#DC143C',
      bgGradient: 'from-red-100 to-red-200'
    },
  }

  useEffect(() => {
    if (!socket) return

    socket.on('receiveMessage', (data: any) => {
      setIsTyping(false)
      setMessages((prev) => [...prev, data])
      
      if (data.emotion) {
        setCurrentEmotion(data.emotion)
        setEmotionConfidence(data.confidence || 0)
        
        if (data.avatar) {
          setAvatarImage(`data:image/png;base64,${data.avatar}`)
        }
      }
    })

    // 환영 메시지
    setTimeout(() => {
      setMessages([{
        isUser: false,
        response: '안녕하세요! 저는 공감적 상호작용이 가능한 심리 상담 챗봇입니다. 어떤 이야기를 나누고 싶으신가요?',
        timestamp: new Date().toISOString()
      }])
    }, 800)

    return () => {
      socket.off('receiveMessage')
    }
  }, [socket])

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages])

  useEffect(() => {
    const timer = setInterval(() => {
      setConnectionTime(prev => prev + 1)
    }, 1000)
    return () => clearInterval(timer)
  }, [])

  const formatTime = (seconds: number) => {
    const mins = Math.floor(seconds / 60)
    const secs = seconds % 60
    return `${mins}m ${secs}s`
  }

  const handleSendMessage = async (text: string) => {
    const userMessage = {
      userId,
      message: text,
      timestamp: new Date().toISOString(),
      isUser: true,
    }
    
    setMessages((prev) => [...prev, userMessage])
    setIsTyping(true)
    sendMessage(userId, text)
  }

  const currentEmotionData = emotions[currentEmotion as keyof typeof emotions] || emotions.neutral

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 via-white to-purple-50 flex items-center justify-center p-4">
      <div className="w-full max-w-md bg-white rounded-3xl shadow-2xl overflow-hidden border border-gray-100">
        
        {/* Header */}
        <div className="bg-white border-b border-gray-100 p-6">
          <h1 className="text-2xl font-bold text-gray-800 text-center mb-2">
            공감적 상호작용 심리 상담 챗봇
          </h1>
          <p className="text-sm text-gray-500 text-center">
            실시간 감정 분석 기반 아바타 시각화 PoC 시스템
          </p>
        </div>

        {/* Avatar Card */}
        <div className="bg-gradient-to-br from-blue-50 to-purple-50 p-6">
          <div className="bg-white rounded-2xl p-6 shadow-lg">
            <h2 className="text-lg font-bold text-gray-800 text-center mb-4">
              상담사 아바타
            </h2>
            
            {/* Avatar Display */}
            <div className="flex justify-center mb-4">
              <div className={`w-48 h-48 rounded-full bg-gradient-to-br ${currentEmotionData.bgGradient} 
                            flex items-center justify-center shadow-inner border-4 border-gray-200`}>
                {avatarImage ? (
                  <img src={avatarImage} alt="avatar" className="w-full h-full object-cover rounded-full" />
                ) : (
                  <div className="text-8xl">{currentEmotionData.icon}</div>
                )}
              </div>
            </div>

            {/* Status */}
            <div className="flex justify-between items-center text-sm">
              <div className="flex items-center gap-2">
                <span className="font-semibold text-gray-600">연결 상태:</span>
                <span className={`font-bold ${isConnected ? 'text-green-500' : 'text-red-500'}`}>
                  {isConnected ? '연결됨' : '연결 안됨'}
                </span>
              </div>
              <div className="flex items-center gap-2">
                <span className="font-semibold text-gray-600">지연시간:</span>
                <span className="font-bold text-blue-500">
                  {connectionTime < 1 ? '0ms' : formatTime(connectionTime)}
                </span>
              </div>
            </div>
          </div>
        </div>

        {/* Chat Section */}
        <div className="bg-gradient-to-br from-purple-500 via-blue-500 to-purple-600 p-4">
          <div className="bg-white rounded-2xl p-4 shadow-lg">
            <div className="flex items-center justify-between mb-3">
              <h3 className="text-lg font-bold text-gray-800">심리 상담 챗봇</h3>
              <div className="flex items-center gap-2">
                <div className={`w-2 h-2 rounded-full ${isConnected ? 'bg-green-500 animate-pulse' : 'bg-red-500'}`} />
                <span className="text-sm font-semibold text-green-500">
                  {isConnected ? '연결됨' : '연결 안됨'}
                </span>
                <span className="text-xs text-gray-500">0ms</span>
              </div>
            </div>

            {/* Messages */}
            <div className="h-64 overflow-y-auto mb-4 space-y-3 scrollbar-thin">
              {messages.map((msg, index) => {
                const time = new Date(msg.timestamp).toLocaleTimeString('ko-KR', { 
                  hour: '2-digit', 
                  minute: '2-digit' 
                })

                if (msg.isUser) {
                  return (
                    <div key={index} className="flex justify-end animate-slide-in-right">
                      <div className="max-w-[75%] bg-gradient-to-r from-purple-500 to-blue-500 text-white px-4 py-2 rounded-2xl rounded-tr-sm">
                        <p className="text-sm">{msg.message}</p>
                        <p className="text-xs opacity-80 mt-1 text-right">{time}</p>
                      </div>
                    </div>
                  )
                }

                return (
                  <div key={index} className="flex justify-start animate-slide-in-left">
                    <div className="max-w-[75%] bg-gray-100 text-gray-800 px-4 py-2 rounded-2xl rounded-tl-sm">
                      <p className="text-sm whitespace-pre-wrap">{msg.response}</p>
                      {msg.riskMessage && (
                        <p className="text-xs text-purple-600 mt-2 font-medium">💡 {msg.riskMessage}</p>
                      )}
                      <p className="text-xs text-gray-500 mt-1">{time}</p>
                    </div>
                  </div>
                )
              })}

              {isTyping && (
                <div className="flex justify-start">
                  <div className="bg-gray-100 text-gray-800 px-4 py-2 rounded-2xl rounded-tl-sm">
                    <div className="flex gap-1">
                      <span className="w-2 h-2 bg-gray-400 rounded-full animate-bounce" style={{ animationDelay: '0ms' }} />
                      <span className="w-2 h-2 bg-gray-400 rounded-full animate-bounce" style={{ animationDelay: '150ms' }} />
                      <span className="w-2 h-2 bg-gray-400 rounded-full animate-bounce" style={{ animationDelay: '300ms' }} />
                    </div>
                  </div>
                </div>
              )}

              <div ref={messagesEndRef} />
            </div>

            {/* Input */}
            <div className="flex gap-2">
              <input
                type="text"
                placeholder="Type your message..."
                disabled={!isConnected}
                onKeyDown={(e) => {
                  if (e.key === 'Enter') {
                    const value = e.currentTarget.value.trim()
                    if (value) {
                      handleSendMessage(value)
                      e.currentTarget.value = ''
                    }
                  }
                }}
                className="flex-1 px-4 py-2 border-2 border-gray-200 rounded-full 
                         focus:border-purple-400 focus:outline-none text-sm
                         disabled:bg-gray-100 disabled:cursor-not-allowed"
              />
              <button
                disabled={!isConnected}
                onClick={(e) => {
                  const input = e.currentTarget.previousElementSibling as HTMLInputElement
                  const value = input.value.trim()
                  if (value) {
                    handleSendMessage(value)
                    input.value = ''
                  }
                }}
                className="bg-gradient-to-r from-cyan-400 to-cyan-500 text-white p-2 rounded-full 
                         hover:from-cyan-500 hover:to-cyan-600 transition-all disabled:opacity-50
                         disabled:cursor-not-allowed"
              >
                <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} 
                        d="M12 19l9 2-9-18-9 18 9-2zm0 0v-8" />
                </svg>
              </button>
            </div>
          </div>
        </div>

        {/* Footer - Talk with Us Button */}
        <div className="bg-white p-4 border-t border-gray-100">
          <button className="w-full bg-black text-white py-3 rounded-full font-semibold 
                           flex items-center justify-center gap-2 hover:bg-gray-800 transition-all">
            <svg className="w-6 h-6 text-cyan-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} 
                    d="M18 9v3m0 0v3m0-3h3m-3 0h-3m-2-5a4 4 0 11-8 0 4 4 0 018 0zM3 20a6 6 0 0112 0v1H3v-1z" />
            </svg>
            Talk with Us
          </button>
          <p className="text-xs text-gray-500 text-center mt-2">
            Ready to assist
          </p>
        </div>
      </div>
    </div>
  )
}
