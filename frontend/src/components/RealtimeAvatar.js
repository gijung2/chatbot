// 실시간 아바타 디스플레이 컴포넌트
import React, { useState, useEffect, useRef } from 'react';

const RealtimeAvatar = ({ 
  currentEmotion = 'neutral', 
  intensity = 0.5, 
  avatarData = null,
  isTransitioning = false,
  transitionDuration = 1.0,
  onTransitionComplete = null
}) => {
  const [displayImage, setDisplayImage] = useState(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState(null);
  const [animationClass, setAnimationClass] = useState('');
  const imageRef = useRef(null);
  const canvasRef = useRef(null);

  // 아바타 데이터 변경시 이미지 업데이트
  useEffect(() => {
    if (avatarData && avatarData.image_base64) {
      updateAvatarImage(avatarData);
    }
  }, [avatarData]);

  // 전환 애니메이션 처리
  useEffect(() => {
    if (isTransitioning) {
      setAnimationClass('transitioning');
      
      const timer = setTimeout(() => {
        setAnimationClass('');
        if (onTransitionComplete) {
          onTransitionComplete();
        }
      }, transitionDuration * 1000);

      return () => clearTimeout(timer);
    }
  }, [isTransitioning, transitionDuration, onTransitionComplete]);

  const updateAvatarImage = async (data) => {
    try {
      setIsLoading(true);
      setError(null);

      if (data.image_base64) {
        const imageUrl = `data:image/png;base64,${data.image_base64}`;
        
        // 이미지 로드 확인
        const img = new Image();
        img.onload = () => {
          setDisplayImage(imageUrl);
          setIsLoading(false);
          
          // Canvas에 이미지 그리기 (후처리용)
          if (canvasRef.current) {
            const canvas = canvasRef.current;
            const ctx = canvas.getContext('2d');
            canvas.width = 300;
            canvas.height = 300;
            
            ctx.clearRect(0, 0, canvas.width, canvas.height);
            ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
            
            // 감정 강도에 따른 필터 적용
            applyEmotionFilter(ctx, currentEmotion, intensity);
          }
        };
        
        img.onerror = () => {
          setError('이미지 로드 실패');
          setIsLoading(false);
          generateFallbackAvatar();
        };
        
        img.src = imageUrl;
      } else {
        generateFallbackAvatar();
      }
    } catch (error) {
      console.error('아바타 이미지 업데이트 실패:', error);
      setError(error.message);
      setIsLoading(false);
      generateFallbackAvatar();
    }
  };

  const generateFallbackAvatar = () => {
    // Canvas로 간단한 아바타 생성
    if (canvasRef.current) {
      const canvas = canvasRef.current;
      const ctx = canvas.getContext('2d');
      canvas.width = 300;
      canvas.height = 300;
      
      // 배경
      const gradient = ctx.createLinearGradient(0, 0, 300, 300);
      gradient.addColorStop(0, getEmotionColor(currentEmotion, intensity, 0.3));
      gradient.addColorStop(1, getEmotionColor(currentEmotion, intensity, 0.1));
      
      ctx.fillStyle = gradient;
      ctx.fillRect(0, 0, 300, 300);
      
      // 얼굴 원
      ctx.beginPath();
      ctx.arc(150, 150, 80, 0, 2 * Math.PI);
      ctx.fillStyle = getEmotionColor(currentEmotion, intensity, 0.8);
      ctx.fill();
      ctx.strokeStyle = '#333';
      ctx.lineWidth = 3;
      ctx.stroke();
      
      // 눈
      ctx.beginPath();
      ctx.arc(125, 130, 8, 0, 2 * Math.PI);
      ctx.arc(175, 130, 8, 0, 2 * Math.PI);
      ctx.fillStyle = '#000';
      ctx.fill();
      
      // 입 (감정별)
      drawEmotionMouth(ctx, currentEmotion, intensity);
      
      // 감정 라벨
      ctx.font = '16px Arial';
      ctx.fillStyle = '#333';
      ctx.textAlign = 'center';
      ctx.fillText(getEmotionLabel(currentEmotion), 150, 250);
      ctx.fillText(`강도: ${Math.round(intensity * 100)}%`, 150, 270);
      
      // Canvas를 이미지로 변환
      const dataURL = canvas.toDataURL('image/png');
      setDisplayImage(dataURL);
      setIsLoading(false);
    }
  };

  const applyEmotionFilter = (ctx, emotion, intensity) => {
    // 감정에 따른 색상 필터 적용
    const filters = {
      joy: () => {
        // 밝기 증가
        ctx.globalCompositeOperation = 'screen';
        ctx.fillStyle = `rgba(255, 255, 0, ${intensity * 0.1})`;
        ctx.fillRect(0, 0, ctx.canvas.width, ctx.canvas.height);
      },
      sad: () => {
        // 파란색 톤
        ctx.globalCompositeOperation = 'multiply';
        ctx.fillStyle = `rgba(100, 150, 200, ${intensity * 0.1})`;
        ctx.fillRect(0, 0, ctx.canvas.width, ctx.canvas.height);
      },
      anxiety: () => {
        // 약간의 흔들림 효과
        const shake = intensity * 2;
        ctx.translate(Math.random() * shake - shake/2, Math.random() * shake - shake/2);
      },
      anger: () => {
        // 빨간색 톤
        ctx.globalCompositeOperation = 'multiply';
        ctx.fillStyle = `rgba(200, 100, 100, ${intensity * 0.1})`;
        ctx.fillRect(0, 0, ctx.canvas.width, ctx.canvas.height);
      }
    };

    ctx.globalCompositeOperation = 'source-over';
    
    if (filters[emotion]) {
      filters[emotion]();
    }
    
    ctx.globalCompositeOperation = 'source-over';
  };

  const drawEmotionMouth = (ctx, emotion, intensity) => {
    ctx.strokeStyle = '#000';
    ctx.lineWidth = 4;
    ctx.lineCap = 'round';
    
    const centerX = 150;
    const centerY = 170;
    const width = 30 + (intensity * 20);
    
    ctx.beginPath();
    
    switch (emotion) {
      case 'joy':
        // 웃는 입
        ctx.arc(centerX, centerY - 10, width/2, 0, Math.PI);
        break;
      case 'sad':
        // 슬픈 입
        ctx.arc(centerX, centerY + 10, width/2, Math.PI, 2 * Math.PI);
        break;
      case 'anxiety':
        // 물결 모양
        ctx.moveTo(centerX - width/2, centerY);
        ctx.quadraticCurveTo(centerX, centerY - 5, centerX + width/2, centerY);
        break;
      case 'anger':
        // 찡그린 입
        ctx.moveTo(centerX - width/2, centerY);
        ctx.lineTo(centerX + width/2, centerY);
        break;
      default: // neutral
        // 일자 입
        ctx.moveTo(centerX - width/3, centerY);
        ctx.lineTo(centerX + width/3, centerY);
    }
    
    ctx.stroke();
  };

  const getEmotionColor = (emotion, intensity, alpha = 1) => {
    const colors = {
      joy: [255, 223, 85],
      sad: [135, 206, 235],
      anxiety: [255, 165, 79],
      anger: [220, 20, 60],
      neutral: [128, 128, 128]
    };
    
    const color = colors[emotion] || colors.neutral;
    const adjustedColor = color.map(c => Math.round(c * (0.5 + intensity * 0.5)));
    
    return `rgba(${adjustedColor.join(',')}, ${alpha})`;
  };

  const getEmotionLabel = (emotion) => {
    const labels = {
      joy: '기쁨',
      sad: '슬픔',
      anxiety: '불안',
      anger: '분노',
      neutral: '중립'
    };
    
    return labels[emotion] || '알 수 없음';
  };

  return (
    <div className={`realtime-avatar ${animationClass}`}>
      <div className="avatar-container">
        {isLoading && (
          <div className="loading-spinner">
            <div className="spinner"></div>
            <p>아바타 로딩 중...</p>
          </div>
        )}
        
        {error && (
          <div className="error-message">
            <p>⚠️ {error}</p>
            <button onClick={generateFallbackAvatar}>
              기본 아바타 생성
            </button>
          </div>
        )}
        
        {displayImage && !isLoading && (
          <div className="avatar-display">
            <img 
              ref={imageRef}
              src={displayImage} 
              alt={`${getEmotionLabel(currentEmotion)} 상담사`}
              className="avatar-image"
            />
            <div className="emotion-indicator">
              <div 
                className="emotion-bar"
                style={{
                  backgroundColor: getEmotionColor(currentEmotion, intensity, 0.7),
                  width: `${intensity * 100}%`
                }}
              ></div>
            </div>
          </div>
        )}
        
        <canvas 
          ref={canvasRef} 
          style={{ display: 'none' }}
          width={300}
          height={300}
        ></canvas>
      </div>
      
      <div className="avatar-info">
        <h3>{getEmotionLabel(currentEmotion)} 상담사</h3>
        <p>감정 강도: {Math.round(intensity * 100)}%</p>
        {avatarData && (
          <p className="avatar-description">
            {avatarData.description}
          </p>
        )}
      </div>

      <style jsx>{`
        .realtime-avatar {
          display: flex;
          flex-direction: column;
          align-items: center;
          padding: 20px;
          background: rgba(255, 255, 255, 0.95);
          border-radius: 20px;
          box-shadow: 0 10px 30px rgba(0, 0, 0, 0.1);
          transition: all 0.3s ease;
          max-width: 350px;
          margin: 0 auto;
        }

        .realtime-avatar.transitioning {
          transform: scale(1.05);
          box-shadow: 0 15px 40px rgba(0, 0, 0, 0.2);
        }

        .avatar-container {
          position: relative;
          width: 300px;
          height: 300px;
          margin-bottom: 20px;
        }

        .loading-spinner {
          display: flex;
          flex-direction: column;
          align-items: center;
          justify-content: center;
          height: 100%;
        }

        .spinner {
          width: 50px;
          height: 50px;
          border: 4px solid #f3f3f3;
          border-top: 4px solid #3498db;
          border-radius: 50%;
          animation: spin 1s linear infinite;
          margin-bottom: 10px;
        }

        @keyframes spin {
          0% { transform: rotate(0deg); }
          100% { transform: rotate(360deg); }
        }

        .error-message {
          display: flex;
          flex-direction: column;
          align-items: center;
          justify-content: center;
          height: 100%;
          color: #e74c3c;
          text-align: center;
        }

        .error-message button {
          margin-top: 10px;
          padding: 8px 16px;
          background: #3498db;
          color: white;
          border: none;
          border-radius: 5px;
          cursor: pointer;
        }

        .avatar-display {
          position: relative;
          width: 100%;
          height: 100%;
        }

        .avatar-image {
          width: 100%;
          height: 100%;
          object-fit: cover;
          border-radius: 15px;
          transition: all 0.5s ease;
        }

        .emotion-indicator {
          position: absolute;
          bottom: 10px;
          left: 10px;
          right: 10px;
          height: 8px;
          background: rgba(0, 0, 0, 0.2);
          border-radius: 4px;
          overflow: hidden;
        }

        .emotion-bar {
          height: 100%;
          transition: all 0.3s ease;
          border-radius: 4px;
        }

        .avatar-info {
          text-align: center;
          color: #333;
        }

        .avatar-info h3 {
          margin: 0 0 10px 0;
          font-size: 1.4em;
          font-weight: 600;
        }

        .avatar-info p {
          margin: 5px 0;
          color: #666;
        }

        .avatar-description {
          font-style: italic;
          font-size: 0.9em;
          max-width: 300px;
        }
      `}</style>
    </div>
  );
};

export default RealtimeAvatar;
