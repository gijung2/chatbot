export default function ChatMessage({ message }: { message: any }) {
  if (message.isUser) {
    return (
      <div className="flex justify-end animate-slide-in-right">
        <div className="bg-gradient-to-br from-purple-500 to-blue-500 text-white rounded-2xl rounded-tr-sm px-5 py-3 max-w-md shadow-md">
          <p className="text-sm leading-relaxed">{message.message}</p>
          <p className="text-xs opacity-70 mt-1 text-right">
            오후 {new Date(message.timestamp).toLocaleTimeString('ko-KR', { 
              hour: '2-digit', 
              minute: '2-digit',
              hour12: false 
            })}
          </p>
        </div>
      </div>
    )
  }

  const emotionColors: Record<string, string> = {
    joy: 'bg-gradient-to-br from-yellow-50 to-yellow-100 border-yellow-300',
    sad: 'bg-gradient-to-br from-blue-50 to-blue-100 border-blue-300',
    anxiety: 'bg-gradient-to-br from-purple-50 to-purple-100 border-purple-300',
    anger: 'bg-gradient-to-br from-red-50 to-red-100 border-red-300',
    neutral: 'bg-gradient-to-br from-gray-50 to-gray-100 border-gray-300',
  }

  const bgColor = emotionColors[message.emotion] || 'bg-gradient-to-br from-gray-50 to-gray-100 border-gray-300'

  return (
    <div className="flex justify-start animate-slide-in-left">
      <div className="max-w-md">
        {/* 봇 이름 표시 */}
        <div className="flex items-center gap-2 mb-1 ml-1">
          <span className="text-xs font-medium text-gray-600">봇상담</span>
        </div>
        
        <div className={`${bgColor} border rounded-2xl rounded-tl-sm px-5 py-3 shadow-md`}>
          {/* 감정 정보 헤더 */}
          {message.emotion && (
            <div className="flex items-center gap-2 mb-2 pb-2 border-b border-gray-200">
              {message.avatar && (
                <img
                  src={`data:image/png;base64,${message.avatar}`}
                  alt="avatar"
                  className="w-10 h-10 rounded-full border-2 border-white shadow-sm"
                />
              )}
              <div className="flex-1">
                <p className="font-bold text-sm text-gray-800">
                  {message.emotion}
                </p>
                <p className="text-xs text-gray-500">
                  신뢰도 {(message.confidence * 100).toFixed(1)}% · {message.riskLevel}
                </p>
              </div>
            </div>
          )}
          
          {/* 응답 메시지 */}
          <p className="text-sm text-gray-800 leading-relaxed mb-2">
            {message.response}
          </p>
          
          {/* 위험도 메시지 */}
          {message.riskMessage && (
            <div className="mt-3 pt-3 border-t border-gray-200">
              <p className="text-xs text-gray-600 bg-white/50 rounded-lg px-3 py-2">
                💡 {message.riskMessage}
              </p>
            </div>
          )}
          
          {/* 타임스탬프 */}
          <p className="text-xs text-gray-400 mt-2 text-right">
            오후 {new Date(message.timestamp || Date.now()).toLocaleTimeString('ko-KR', { 
              hour: '2-digit', 
              minute: '2-digit',
              hour12: false 
            })}
          </p>
        </div>
      </div>
    </div>
  )
}
