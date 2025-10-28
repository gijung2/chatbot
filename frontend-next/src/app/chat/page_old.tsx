'use client'

import { useState, useEffect, useRef } from 'react'
import { useSocket } from '@/hooks/useSocket'
import ChatMessage from '@/components/ChatMessage'
import ChatInput from '@/components/ChatInput'
import AvatarCard from '@/components/AvatarCard'

export default function ChatPage() {
  const [messages, setMessages] = useState<any[]>([])
  const [userId] = useState(() => `user-${Date.now()}`)
  const [currentEmotion, setCurrentEmotion] = useState('neutral')
  const [emotionConfidence, setEmotionConfidence] = useState(0)
  const [avatarState, setAvatarState] = useState<any>(null)
  const messagesEndRef = useRef<HTMLDivElement>(null)

  const { socket, isConnected, sendMessage } = useSocket()

  useEffect(() => {
    if (!socket) return

    socket.on('receiveMessage', (data: any) => {
      setMessages((prev) => [...prev, data])
      
      // 감정 상태 업데이트 (200ms 이내 반영)
      if (data.emotion) {
        setCurrentEmotion(data.emotion)
        setEmotionConfidence(data.confidence || 0)
        
        // 아바타 상태 업데이트
        if (data.avatarState) {
          setAvatarState(data.avatarState)
        }
      }
    })

    return () => {
      socket.off('receiveMessage')
    }
  }, [socket])

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages])

  const handleSendMessage = async (text: string) => {
    const userMessage = {
      userId,
      message: text,
      timestamp: new Date().toISOString(),
      isUser: true,
    }
    
    setMessages((prev) => [...prev, userMessage])
    sendMessage(userId, text)
  }

  return (
    <div className="flex h-screen bg-gradient-to-br from-purple-400 via-purple-300 to-blue-300">
      {/* 왼쪽: 아바타 카드 */}
      <div className="w-80 p-4 flex-shrink-0">
        <AvatarCard 
          emotion={currentEmotion}
          confidence={emotionConfidence}
          isConnected={isConnected}
          avatarState={avatarState}
        />
      </div>

      {/* 오른쪽: 채팅 영역 */}
      <div className="flex-1 flex flex-col p-4 pl-0">
        <div className="bg-white/90 backdrop-blur-sm rounded-2xl shadow-2xl flex flex-col h-full overflow-hidden">
          {/* Header */}
          <div className="bg-gradient-to-r from-purple-500 to-blue-500 text-white p-4 flex items-center justify-between">
            <div>
              <h1 className="text-xl font-bold flex items-center gap-2">
                � 실시간 아바타 상담 채팅
              </h1>
              <p className="text-xs opacity-90 mt-1">
                {isConnected ? '🟢 API 오프라인' : '🔴 API 오프라인'}
              </p>
            </div>
            <div className="bg-white/20 px-3 py-1 rounded-full text-sm">
              {currentEmotion}
            </div>
          </div>

          {/* Messages */}
          <div className="flex-1 overflow-y-auto p-6 space-y-3">
            {messages.map((msg, index) => (
              <ChatMessage key={index} message={msg} />
            ))}
            <div ref={messagesEndRef} />
          </div>

          {/* Input */}
          <ChatInput onSend={handleSendMessage} disabled={!isConnected} />
        </div>
      </div>
    </div>
  )
}
