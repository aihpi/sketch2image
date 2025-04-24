import os
from typing import List, Dict
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
    DEFAULT_MODEL_ID: str = os.getenv(
        "DEFAULT_MODEL_ID", 
        "controlnet_sd15_scribble"  # Default model ID
    )
    
    # Available Models (from sketch2image.md)
    AVAILABLE_MODELS = {
        "controlnet_sd15_scribble": {
            "name": "Stable Diffusion 1.5 + ControlNet (Scribble)",
            "huggingface_id": "lllyasviel/control_v11p_sd15_scribble",
            "base_model": "runwayml/stable-diffusion-v1-5",
            "inference_speed": "Moderate (5-15s on GPU, ~30s on CPU)",
            "recommended_for": ["Balanced quality and speed", "General purpose"]
        },
        "controlnet_sdxl_scribble": {
            "name": "Stable Diffusion XL + ControlNet (Scribble)",
            "huggingface_id": "xinsir/controlnet-scribble-sdxl-1.0",
            "base_model": "stabilityai/stable-diffusion-xl-base-1.0",
            "inference_speed": "Slow (10-30s per image)",
            "recommended_for": ["High quality", "Detailed outputs"]
        },
        "t2i_adapter_sdxl": {
            "name": "Stable Diffusion XL + T2I-Adapter (Sketch)",
            "huggingface_id": "TencentARC/t2i-adapter-sketch-sdxl-1.0",
            "base_model": "stabilityai/stable-diffusion-xl-base-1.0",
            "inference_speed": "Moderate-Slow (several seconds on GPU)",
            "recommended_for": ["High quality", "Sharp details"]
        },
        "pix2pix_turbo": {
            "name": "Pix2Pix-Turbo (One-Step Sketch-to-Image)",
            "huggingface_id": "qninhdt/img2img-turbo",
            "base_model": "",  # Different architecture
            "inference_speed": "Fast (~1s on most GPUs)",
            "recommended_for": ["Speed", "Quick iterations"]
        }
    }
    
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