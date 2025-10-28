// React 애플리케이션 진입점
import React from 'react';
import ReactDOM from 'react-dom/client';
import App from './App';

// 전역 스타일
const globalStyles = `
  * {
    margin: 0;
    padding: 0;
    box-sizing: border-box;
  }

  body {
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    line-height: 1.6;
    color: #333;
    background: #f5f5f5;
    -webkit-font-smoothing: antialiased;
    -moz-osx-font-smoothing: grayscale;
  }

  * {
    scrollbar-width: thin;
    scrollbar-color: #c1c1c1 #f1f1f1;
  }

  *::-webkit-scrollbar {
    width: 8px;
    height: 8px;
  }

  *::-webkit-scrollbar-track {
    background: #f1f1f1;
    border-radius: 4px;
  }

  *::-webkit-scrollbar-thumb {
    background: #c1c1c1;
    border-radius: 4px;
    transition: background 0.3s ease;
  }

  *::-webkit-scrollbar-thumb:hover {
    background: #a8a8a8;
  }

  button {
    font-family: inherit;
    cursor: pointer;
    transition: all 0.3s ease;
  }

  button:focus {
    outline: 2px solid #667eea;
    outline-offset: 2px;
  }

  input, textarea {
    font-family: inherit;
  }

  .visually-hidden {
    position: absolute;
    width: 1px;
    height: 1px;
    padding: 0;
    margin: -1px;
    overflow: hidden;
    clip: rect(0, 0, 0, 0);
    white-space: nowrap;
    border: 0;
  }

  /* 애니메이션 클래스 */
  .fade-in {
    animation: fadeIn 0.3s ease;
  }

  .fade-out {
    animation: fadeOut 0.3s ease;
  }

  .slide-up {
    animation: slideUp 0.3s ease;
  }

  .slide-down {
    animation: slideDown 0.3s ease;
  }

  .scale-in {
    animation: scaleIn 0.3s ease;
  }

  .pulse {
    animation: pulse 2s infinite;
  }

  @keyframes fadeIn {
    from { opacity: 0; }
    to { opacity: 1; }
  }

  @keyframes fadeOut {
    from { opacity: 1; }
    to { opacity: 0; }
  }

  @keyframes slideUp {
    from { transform: translateY(20px); opacity: 0; }
    to { transform: translateY(0); opacity: 1; }
  }

  @keyframes slideDown {
    from { transform: translateY(-20px); opacity: 0; }
    to { transform: translateY(0); opacity: 1; }
  }

  @keyframes scaleIn {
    from { transform: scale(0.9); opacity: 0; }
    to { transform: scale(1); opacity: 1; }
  }

  @keyframes pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.7; }
  }

  /* 접근성 개선 */
  @media (prefers-reduced-motion: reduce) {
    *,
    *::before,
    *::after {
      animation-duration: 0.01ms !important;
      animation-iteration-count: 1 !important;
      transition-duration: 0.01ms !important;
    }
  }

  /* 다크 모드 대응 */
  @media (prefers-color-scheme: dark) {
    body {
      background: #1a1a1a;
      color: #e0e0e0;
    }
  }

  /* 고대비 모드 대응 */
  @media (prefers-contrast: high) {
    button {
      border: 2px solid currentColor;
    }
  }

  /* 터치 디바이스 최적화 */
  @media (hover: none) and (pointer: coarse) {
    button {
      min-height: 44px;
      min-width: 44px;
    }
  }

  /* 프린트 스타일 */
  @media print {
    .no-print {
      display: none !important;
    }
    
    * {
      background: white !important;
      color: black !important;
      box-shadow: none !important;
    }
  }
`;

// 전역 스타일 적용
const styleSheet = document.createElement('style');
styleSheet.textContent = globalStyles;
document.head.appendChild(styleSheet);

// React 애플리케이션 초기화
const root = ReactDOM.createRoot(document.getElementById('root'));

root.render(
  <React.StrictMode>
    <App />
  </React.StrictMode>
);

// 서비스 워커 등록 (PWA 기능)
if ('serviceWorker' in navigator) {
  window.addEventListener('load', () => {
    navigator.serviceWorker.register('/sw.js')
      .then((registration) => {
        console.log('SW registered: ', registration);
      })
      .catch((registrationError) => {
        console.log('SW registration failed: ', registrationError);
      });
  });
}

// 전역 에러 핸들링
window.addEventListener('error', (event) => {
  console.error('전역 에러 발생:', event.error);
});

window.addEventListener('unhandledrejection', (event) => {
  console.error('처리되지 않은 Promise 거부:', event.reason);
});

// 온라인/오프라인 상태 모니터링
window.addEventListener('online', () => {
  console.log('네트워크 연결됨');
});

window.addEventListener('offline', () => {
  console.log('네트워크 연결 끊김');
});

// 성능 모니터링
if ('performance' in window && 'measure' in window.performance) {
  window.addEventListener('load', () => {
    setTimeout(() => {
      const perfData = window.performance.timing;
      const loadTime = perfData.loadEventEnd - perfData.navigationStart;
      console.log(`페이지 로딩 시간: ${loadTime}ms`);
    }, 0);
  });
}

// 키보드 네비게이션 개선
document.addEventListener('keydown', (event) => {
  // ESC 키로 모달 닫기
  if (event.key === 'Escape') {
    const modals = document.querySelectorAll('.error-modal, .modal');
    modals.forEach(modal => {
      if (modal.style.display !== 'none') {
        modal.style.display = 'none';
      }
    });
  }
});

// 포커스 표시 개선
document.addEventListener('keydown', (event) => {
  if (event.key === 'Tab') {
    document.body.classList.add('using-keyboard');
  }
});

document.addEventListener('mousedown', () => {
  document.body.classList.remove('using-keyboard');
});

// 디버그 정보 (개발 환경에서만)
if (process.env.NODE_ENV === 'development') {
  console.log('🚀 실시간 아바타 상담 시스템 개발 모드');
  console.log('📡 API 서버: http://localhost:8002');
  console.log('🔌 WebSocket 서버: http://localhost:8003');
  console.log('⚛️ React 개발 서버: http://localhost:3000');
}
