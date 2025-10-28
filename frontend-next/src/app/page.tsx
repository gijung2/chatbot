import Link from 'next/link'

export default function Home() {
  return (
    <main className="flex min-h-screen flex-col items-center justify-center p-24">
      <div className="z-10 max-w-5xl w-full items-center justify-between font-mono text-sm">
        <h1 className="text-4xl font-bold text-center mb-8">
          🤖 감정 분석 챗봇
        </h1>
        
        <p className="text-center mb-12 text-lg text-gray-600">
          AI 기반 감정 분석으로 당신의 마음을 이해합니다
        </p>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-12">
          <div className="p-6 border rounded-lg hover:shadow-lg transition">
            <h2 className="text-2xl font-bold mb-4">💬 실시간 채팅</h2>
            <p className="text-gray-600 mb-4">
              WebSocket 기반 실시간 감정 분석 채팅
            </p>
            <Link 
              href="/chat" 
              className="inline-block bg-blue-500 text-white px-6 py-2 rounded hover:bg-blue-600"
            >
              시작하기
            </Link>
          </div>

          <div className="p-6 border rounded-lg hover:shadow-lg transition">
            <h2 className="text-2xl font-bold mb-4">📊 분석 대시보드</h2>
            <p className="text-gray-600 mb-4">
              감정 분석 통계 및 히스토리 조회
            </p>
            <Link 
              href="/analytics" 
              className="inline-block bg-green-500 text-white px-6 py-2 rounded hover:bg-green-600"
            >
              보기
            </Link>
          </div>
        </div>

        <div className="text-center text-sm text-gray-500">
          <p>🚀 Next.js 15 + NestJS + FastAPI</p>
          <p>🔧 TypeScript + TailwindCSS + Socket.io</p>
        </div>
      </div>
    </main>
  )
}
