"""
아바타 상태 매핑 서비스
감정 분석 결과를 Live2D 파라미터로 매핑
"""
from typing import Dict, Any
import time


class AvatarStateMapper:
    """감정을 아바타 상태(표정/제스처)로 매핑"""
    
    def __init__(self):
        # 감정별 Live2D 파라미터 매핑
        self.emotion_to_parameters = {
            "joy": {
                "expression": "happy",
                "mouth_open": 0.7,
                "eye_smile": 0.8,
                "eyebrow_angle": 0.3,
                "body_rotation": 0.0,
                "head_tilt": 0.1,
                "animation": "wave_hand",
                "color": "#FFD700",
                "emoji": "😊"
            },
            "sad": {
                "expression": "sad",
                "mouth_open": 0.2,
                "eye_smile": 0.0,
                "eyebrow_angle": -0.4,
                "body_rotation": 0.0,
                "head_tilt": -0.2,
                "animation": "look_down",
                "color": "#6495ED",
                "emoji": "😢"
            },
            "anxiety": {
                "expression": "worried",
                "mouth_open": 0.3,
                "eye_smile": 0.1,
                "eyebrow_angle": -0.5,
                "body_rotation": 0.1,
                "head_tilt": 0.0,
                "animation": "fidget",
                "color": "#9370DB",
                "emoji": "😰"
            },
            "anger": {
                "expression": "angry",
                "mouth_open": 0.5,
                "eye_smile": 0.0,
                "eyebrow_angle": -0.7,
                "body_rotation": 0.0,
                "head_tilt": 0.0,
                "animation": "shake_head",
                "color": "#DC143C",
                "emoji": "😠"
            },
            "neutral": {
                "expression": "neutral",
                "mouth_open": 0.3,
                "eye_smile": 0.3,
                "eyebrow_angle": 0.0,
                "body_rotation": 0.0,
                "head_tilt": 0.0,
                "animation": "idle",
                "color": "#808080",
                "emoji": "😐"
            }
        }
        
        # 신뢰도에 따른 애니메이션 강도 조절
        self.confidence_thresholds = {
            "high": 0.8,      # 강한 표현
            "medium": 0.5,    # 중간 표현
            "low": 0.3        # 약한 표현
        }
    
    def map_emotion_to_avatar_state(
        self, 
        emotion: str, 
        confidence: float,
        risk_level: str = "low"
    ) -> Dict[str, Any]:
        """
        감정을 아바타 상태로 매핑
        
        Args:
            emotion: 감정 레이블 (joy, sad, anxiety, anger, neutral)
            confidence: 신뢰도 (0.0 ~ 1.0)
            risk_level: 위험도 레벨
            
        Returns:
            아바타 파라미터 딕셔너리
        """
        # 기본 파라미터 가져오기
        base_params = self.emotion_to_parameters.get(
            emotion, 
            self.emotion_to_parameters["neutral"]
        ).copy()
        
        # 신뢰도에 따라 표현 강도 조절
        intensity = self._calculate_intensity(confidence)
        
        # 파라미터 조정
        adjusted_params = {
            "emotion": emotion,
            "confidence": confidence,
            "risk_level": risk_level,
            "expression": base_params["expression"],
            "parameters": {
                "mouth_open": base_params["mouth_open"] * intensity,
                "eye_smile": base_params["eye_smile"] * intensity,
                "eyebrow_angle": base_params["eyebrow_angle"] * intensity,
                "body_rotation": base_params["body_rotation"],
                "head_tilt": base_params["head_tilt"] * intensity,
            },
            "animation": base_params["animation"],
            "transition_duration": self._calculate_transition_duration(confidence),
            "color": base_params["color"],
            "emoji": base_params["emoji"],
            "timestamp": time.time(),
        }
        
        # 위험도가 높으면 특별한 제스처 추가
        if risk_level in ["high", "critical"]:
            adjusted_params["special_gesture"] = "attention_required"
            adjusted_params["alert_level"] = risk_level
        
        return adjusted_params
    
    def _calculate_intensity(self, confidence: float) -> float:
        """신뢰도에 따른 표현 강도 계산"""
        if confidence >= self.confidence_thresholds["high"]:
            return 1.0  # 100% 강도
        elif confidence >= self.confidence_thresholds["medium"]:
            return 0.7  # 70% 강도
        elif confidence >= self.confidence_thresholds["low"]:
            return 0.5  # 50% 강도
        else:
            return 0.3  # 30% 강도 (최소)
    
    def _calculate_transition_duration(self, confidence: float) -> float:
        """
        전환 애니메이션 지속 시간 계산 (ms)
        목표: p50 ≤ 200ms, p95 ≤ 400ms
        """
        # 신뢰도가 높을수록 빠르게 전환
        if confidence >= 0.8:
            return 150  # 150ms (매우 빠름)
        elif confidence >= 0.5:
            return 200  # 200ms (빠름)
        else:
            return 300  # 300ms (중간)
    
    def get_idle_state(self) -> Dict[str, Any]:
        """대기 상태 파라미터 반환"""
        return self.map_emotion_to_avatar_state("neutral", 1.0, "low")
    
    def interpolate_states(
        self, 
        from_state: Dict[str, Any], 
        to_state: Dict[str, Any], 
        progress: float
    ) -> Dict[str, Any]:
        """
        두 상태 사이를 부드럽게 보간
        
        Args:
            from_state: 시작 상태
            to_state: 목표 상태
            progress: 진행도 (0.0 ~ 1.0)
            
        Returns:
            보간된 상태
        """
        progress = max(0.0, min(1.0, progress))  # 클램핑
        
        interpolated = to_state.copy()
        interpolated["parameters"] = {}
        
        from_params = from_state.get("parameters", {})
        to_params = to_state.get("parameters", {})
        
        for key in to_params:
            from_val = from_params.get(key, 0.0)
            to_val = to_params.get(key, 0.0)
            # 선형 보간
            interpolated["parameters"][key] = from_val + (to_val - from_val) * progress
        
        return interpolated


# 전역 인스턴스
avatar_mapper = AvatarStateMapper()
