import os
from typing import List
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class Settings:
    # API Settings
    API_V1_STR: str = "/v1"
    
    # Server Settings
    HOST: str = os.getenv("HOST", "0.0.0.0")
    PORT: int = int(os.getenv("PORT", "8000"))
    DEBUG_MODE: bool = os.getenv("DEBUG_MODE", "False").lower() == "true"
    
    # CORS Settings
    CORS_ORIGINS: List[str] = [
        "http://localhost:3000",  # React frontend
        "http://localhost:8000",  # Backend for development
        os.getenv("FRONTEND_URL", ""),  # Production frontend URL
    ]
    
    # Model Settings
    MODEL_ID: str = os.getenv(
        "MODEL_ID", 
        "lllyasviel/control_v11p_sd15_scribble"  # Default model
    )
    DEVICE: str = os.getenv("DEVICE", "cuda" if os.path.exists("/dev/nvidia0") else "cpu")
    
    # Generation Settings
    NUM_INFERENCE_STEPS: int = int(os.getenv("NUM_INFERENCE_STEPS", "30"))
    GUIDANCE_SCALE: float = float(os.getenv("GUIDANCE_SCALE", "7.5"))
    OUTPUT_IMAGE_SIZE: int = int(os.getenv("OUTPUT_IMAGE_SIZE", "512"))
    
    # Storage Settings
    UPLOAD_DIR: str = os.getenv("UPLOAD_DIR", "uploads")
    OUTPUT_DIR: str = os.getenv("OUTPUT_DIR", "outputs")

# Create instance of Settings
settings = Settings()

# Create upload and output directories if they don't exist
os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
os.makedirs(settings.OUTPUT_DIR, exist_ok=True)