"""
Configuration settings for ML serving
"""
from pydantic_settings import BaseSettings
from pathlib import Path

class Settings(BaseSettings):
    # Application
    APP_NAME: str = "Emotion Analysis ML Service"
    VERSION: str = "1.0.0"
    DEBUG: bool = False
    
    # Server
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    
    # Model
    MODEL_PATH: str = "../checkpoints_kfold/fold1_model_20251028_113127.pt"
    MODEL_NAME: str = "klue/bert-base"
    NUM_LABELS: int = 5
    DROPOUT_RATE: float = 0.3
    MAX_LENGTH: int = 128
    
    # Device
    DEVICE: str = "cpu"  # auto, cuda, cpu (CPU로 강제 설정 - CUDA 없음)
    
    # Cache
    ENABLE_CACHE: bool = True
    CACHE_SIZE: int = 1000
    
    # Logging
    LOG_LEVEL: str = "INFO"
    
    class Config:
        env_file = ".env"
        case_sensitive = True

settings = Settings()
