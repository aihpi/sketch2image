import os
from typing import List, Dict, Any, Callable
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
    
    # Get Hugging Face token from environment
    HUGGINGFACE_TOKEN: str = os.getenv("HUGGINGFACE_TOKEN", "")
    
    # Enhanced model configuration
    AVAILABLE_MODELS = {
        "controlnet_sd15_scribble": {
            "name": "Stable Diffusion 1.5 + ControlNet (Scribble)",
            "huggingface_id": "lllyasviel/control_v11p_sd15_scribble",
            "base_model": "runwayml/stable-diffusion-v1-5",
            "model_type": "controlnet_sd15",
            "inference_speed": "Moderate (5-15s on GPU, ~30s on CPU)",
            "recommended_for": ["Balanced quality and speed", "General purpose"],
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
            "model_type": "controlnet_sdxl",
            "inference_speed": "Slow (10-30s per image)",
            "recommended_for": ["High quality", "Detailed outputs"],
            "config": {
                "pipeline_type": "StableDiffusionXLControlNetPipeline",
                "model_type": "ControlNetModel",
                "needs_safety_checker": True
            }
        },
        "t2i_adapter_sdxl": {
            "name": "SDXL + T2I-Adapter (Sketch)",
            "huggingface_id": "TencentARC/t2i-adapter-sketch-sdxl-1.0",
            "base_model": "stabilityai/stable-diffusion-xl-base-1.0",
            "vae_model": "madebyollin/sdxl-vae-fp16-fix",  # Add VAE model
            "model_type": "t2i_adapter",
            "inference_speed": "Moderate-Slow (10-20s on GPU)",
            "recommended_for": ["High quality", "Sharp details", "Accurate sketch interpretation"],
            "config": {
                "pipeline_type": "StableDiffusionXLAdapterPipeline",
                "model_type": "T2IAdapter",
                "adapter_type": "full_adapter_xl",  # Add adapter type
                "needs_safety_checker": True,
                "adapter_conditioning_scale": 0.9,  # Updated based on tutorial
                "adapter_conditioning_factor": 0.9, # Added based on tutorial
                "num_inference_steps": 40,          # Increased steps for better quality
                "guidance_scale": 7.5,              # Keeping the default guidance scale
                "custom_scheduler": True            # Flag to use custom scheduler
            }
        },
        "pix2pix_turbo": {
            "name": "Pix2Pix-Turbo (One-Step Sketch-to-Image)",
            "huggingface_id": "qninhdt/img2img-turbo",
            "model_type": "pix2pix",
            "inference_speed": "Fast (~1s on most GPUs)",
            "recommended_for": ["Speed", "Quick iterations"],
            "config": {
                "pipeline_type": "AutoPipelineForImage2Image",
                "needs_safety_checker": False,
                "strength": 0.8,
                "num_inference_steps": 1
            }
        },
        "controlnet_canny": {
            "name": "ControlNet with Canny Edge Detection",
            "huggingface_id": "lllyasviel/sd-controlnet-canny",
            "base_model": "runwayml/stable-diffusion-v1-5",
            "model_type": "controlnet_sd15",
            "inference_speed": "Moderate (5-15s on GPU)",
            "recommended_for": ["Edge-based sketches", "Detailed line art"],
            "config": {
                "pipeline_type": "StableDiffusionControlNetPipeline",
                "model_type": "ControlNetModel",
                "needs_safety_checker": True
            }
        },
        "kandinsky": {
            "name": "Kandinsky 2.2",
            "huggingface_id": "kandinsky-community/kandinsky-2-2-decoder",
            "prior_model_id": "kandinsky-community/kandinsky-2-2-prior",
            "model_type": "kandinsky",
            "inference_speed": "Moderate (7-20s on GPU)",
            "recommended_for": ["Artistic style", "Creative sketches"],
            "config": {
                "pipeline_type": "KandinskyPipeline",
                "needs_safety_checker": False,
                "prior_guidance_scale": 4.0
            }
        },
        "sdxl_controlnet": {
            "name": "SDXL with ControlNet Adapters",
            "huggingface_id": "diffusers/controlnet-canny-sdxl-1.0",
            "base_model": "stabilityai/stable-diffusion-xl-base-1.0",
            "model_type": "controlnet_sdxl",
            "inference_speed": "Slow (15-30s on GPU)",
            "recommended_for": ["High-resolution", "Professional quality"],
            "config": {
                "pipeline_type": "StableDiffusionXLControlNetPipeline",
                "model_type": "ControlNetModel",
                "needs_safety_checker": True
            }
        },
        "stability_sd2": {
            "name": "Stable Diffusion 2.1",
            "huggingface_id": "stabilityai/stable-diffusion-2-1",
            "model_type": "stable_diffusion",
            "inference_speed": "Moderate (5-15s on GPU)",
            "recommended_for": ["General purpose", "High quality images"],
            "config": {
                "pipeline_type": "StableDiffusionPipeline",
                "needs_safety_checker": True
            }
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