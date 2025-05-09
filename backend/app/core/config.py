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
        "t2i_adapter_sdxl"  # Default to T2I adapter
    )
    
    # Enable sketch preprocessing with PidiNet
    USE_PIDINET_PREPROCESSING: bool = os.getenv("USE_PIDINET_PREPROCESSING", "True").lower() == "true"
    PIDINET_DETECT_RESOLUTION: int = int(os.getenv("PIDINET_DETECT_RESOLUTION", "1024"))
    PIDINET_IMAGE_RESOLUTION: int = int(os.getenv("PIDINET_IMAGE_RESOLUTION", "1024"))
    PIDINET_APPLY_FILTER: bool = os.getenv("PIDINET_APPLY_FILTER", "True").lower() == "true"
    
    # Available Models
    AVAILABLE_MODELS = {
        "controlnet_sd15_scribble": {
            "name": "Stable Diffusion 1.5 + ControlNet (Scribble)",
            "huggingface_id": "lllyasviel/control_v11p_sd15_scribble",
            "base_model": "runwayml/stable-diffusion-v1-5",
            "inference_speed": "Moderate (5-15s on GPU, ~30s on CPU)",
            "recommended_for": ["Balanced quality and speed", "General purpose"],
            "preprocessing": {
                "type": "scribble",
                "invert": False,  # Don't invert, use white background and black lines
                "normalize": True
            },
            "config": {
                "pipeline_type": "StableDiffusionControlNetPipeline",
                "model_type": "ControlNetModel",
                "needs_safety_checker": True
            }
        },
        "controlnet_sd15_softedge": {
            "name": "Stable Diffusion 1.5 + ControlNet (SoftEdge)",
            "huggingface_id": "lllyasviel/control_v11p_sd15_softedge",
            "base_model": "runwayml/stable-diffusion-v1-5",
            "inference_speed": "Moderate (5-15s on GPU, ~30s on CPU)",
            "recommended_for": ["Detailed drawings", "Soft sketches", "Less harsh lines"],
            "preprocessing": {
                "type": "pidinet",  # Use PidiNet preprocessing
                "invert": False,    # No inversion needed
                "normalize": True
            },
            "config": {
                "pipeline_type": "StableDiffusionControlNetPipeline",
                "model_type": "ControlNetModel",
                "needs_safety_checker": True
            }
        },
        "controlnet_sdxl_scribble": {
            "name": "Stable Diffusion XL + ControlNet (Scribble)",
            "huggingface_id": "xinsir/controlnet-scribble-sdxl-1.0",
            "base_model": "stabilityai/stable-diffusion-xl-base-1.0",
            "inference_speed": "Slow (10-30s per image)",
            "recommended_for": ["High quality", "Detailed outputs"],
            "preprocessing": {
                "type": "scribble",
                "invert": False,  # Don't invert, use white background and black lines
                "normalize": True
            },
            "config": {
                "pipeline_type": "StableDiffusionXLControlNetPipeline",
                "model_type": "ControlNetModel",
                "needs_safety_checker": True
            }
        },
        "t2i_adapter_sdxl": {
            "name": "SDXL + T2I-Adapter (Sketch)",
            "huggingface_id": "Adapter/t2iadapter",
            "sub_folder": "sketch_sdxl_1.0",
            "base_model": "stabilityai/stable-diffusion-xl-base-1.0",
            "vae_model": "madebyollin/sdxl-vae-fp16-fix",
            "model_type": "t2i_adapter",
            "inference_speed": "Moderate-Slow (10-40s on GPU)",
            "recommended_for": ["High quality robot versions", "Best results with PidiNet preprocessing"],
            "preprocessing": {
                "type": "pidinet",  # Use PidiNet preprocessing
                "invert": False,     # This model works better with inverted input (black background, white lines)
                "normalize": True
            },
            "config": {
                "pipeline_type": "StableDiffusionXLAdapterPipeline",
                "model_type": "T2IAdapter",
                "needs_safety_checker": True,
                "adapter_conditioning_scale": 0.9,
                "adapter_conditioning_factor": 0.9,
                "num_inference_steps": 40,
                "guidance_scale": 7.5,
                "custom_scheduler": True,
                "default_negative_prompt": "disfigured, extra digit, fewer digits, cropped, worst quality, low quality"
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