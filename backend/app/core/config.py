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
        "controlnet_scribble"  # Default to ControlNet Scribble
    )
    
    # Available Models
    AVAILABLE_MODELS = {
    "controlnet_scribble": {
        "name": "SD 1.5 + ControlNet Scribble",
        "huggingface_id": "lllyasviel/sd-controlnet-scribble",
        "base_model": "runwayml/stable-diffusion-v1-5",
        "inference_speed": "Fast (5-15s on GPU)",
        "recommended_for": ["Simple sketches", "Fast generation"],
        "preprocessing": {
            "type": "simple_inverter",
            "detect_resolution": 768,
            "image_resolution": 768,
        },
        "config": {
            "pipeline_type": "StableDiffusionControlNetPipeline",
            "model_type": "ControlNetModel",
            "needs_safety_checker": False,
            "num_inference_steps": 20,
            "guidance_scale": 7.5,
            "scheduler": "DDIMScheduler",
            "default_negative_prompt": "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality"
        }
    },
    "t2i_adapter_sdxl": {
        "name": "SDXL + T2I-Adapter Sketch",
        "huggingface_id": "TencentARC/t2i-adapter-sketch-sdxl-1.0",
        "base_model": "stabilityai/stable-diffusion-xl-base-1.0",
        "inference_speed": "Medium (10-30s on GPU)",
        "recommended_for": ["Detailed sketches", "High quality output"],
        "preprocessing": {
            "type": "simple_inverter",
            "detect_resolution": 768,
            "image_resolution": 768,
        },
        "config": {
            "pipeline_type": "StableDiffusionXLAdapterPipeline",
            "model_type": "T2IAdapter",
            "needs_safety_checker": False,
            "adapter_conditioning_scale": 0.9,
            "adapter_conditioning_factor": 0.9,
            "num_inference_steps": 40,
            "guidance_scale": 7.5,
            "adapter_type": "full_adapter_xl",
            "vae": "madebyollin/sdxl-vae-fp16-fix",
            "scheduler": "DDIMScheduler", # Changed to DDIM
            "default_negative_prompt": "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality"
        }
    }
}
    
    DEVICE: str = os.getenv("DEVICE", "cuda" if os.path.exists("/dev/nvidia0") else "cpu")
    
    # Generation Settings
    NUM_INFERENCE_STEPS: int = int(os.getenv("NUM_INFERENCE_STEPS", "40"))
    GUIDANCE_SCALE: float = float(os.getenv("GUIDANCE_SCALE", "7.5"))
    OUTPUT_IMAGE_SIZE: int = int(os.getenv("OUTPUT_IMAGE_SIZE", "512"))
    
    # Storage Settings
    UPLOAD_DIR: str = os.getenv("UPLOAD_DIR", "uploads")
    OUTPUT_DIR: str = os.getenv("OUTPUT_DIR", "outputs")
    PREPROCESSED_DIR: str = os.getenv("PREPROCESSED_DIR", "preprocessed")

# Create instance of Settings
settings = Settings()

# Create required directories if they don't exist
os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
os.makedirs(settings.OUTPUT_DIR, exist_ok=True)
os.makedirs(settings.PREPROCESSED_DIR, exist_ok=True)