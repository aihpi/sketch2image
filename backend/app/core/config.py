import os
from typing import List, Dict
from dotenv import load_dotenv

load_dotenv()

class Settings:
    API_V1_STR: str = "/v1"
    
    HOST: str = os.getenv("HOST", "0.0.0.0")
    PORT: int = int(os.getenv("PORT", "8000"))
    DEBUG_MODE: bool = os.getenv("DEBUG_MODE", "False").lower() == "true"
    
    # Public URL for QR code generation (should be set to your domain in production)
    PUBLIC_URL: str = os.getenv("PUBLIC_URL", "http://localhost:3000")
    
    CORS_ORIGINS: List[str] = [
        "http://localhost:3000",
        "http://localhost:8000", 
        os.getenv("FRONTEND_URL", ""),  
    ]
    
    DEFAULT_MODEL_ID: str = os.getenv(
        "DEFAULT_MODEL_ID", 
        "controlnet_scribble"  
    )
    
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
            "scheduler": "DDIMScheduler",
            "default_negative_prompt": "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality"
        }
    },
    "flux_canny": {
        "name": "FLUX.1-Canny-dev",
        "huggingface_id": "black-forest-labs/FLUX.1-Canny-dev",
        "base_model": "black-forest-labs/FLUX.1-Canny-dev",
        "inference_speed": "Slow (30-60s on GPU)",
        "recommended_for": ["High-quality sketches", "Premium output", "Complex scenes"],
        "preprocessing": {
            "type": "canny_detector",
            "detect_resolution": 1024,
            "image_resolution": 1024,
            "low_threshold": 50,
            "high_threshold": 200,
        },
        "config": {
            "pipeline_type": "FluxControlPipeline",
            "model_type": "FluxControl",
            "needs_safety_checker": False,
            "num_inference_steps": 50,
            "guidance_scale": 30.0,
            "output_height": 1024,
            "output_width": 1024,
            "torch_dtype": "bfloat16",
            "default_negative_prompt": ""
        }
    }
}
    
    DEVICE: str = os.getenv("DEVICE", "cuda" if os.path.exists("/dev/nvidia0") else "cpu")
    
    NUM_INFERENCE_STEPS: int = int(os.getenv("NUM_INFERENCE_STEPS", "40"))
    GUIDANCE_SCALE: float = float(os.getenv("GUIDANCE_SCALE", "7.5"))
    OUTPUT_IMAGE_SIZE: int = int(os.getenv("OUTPUT_IMAGE_SIZE", "512"))
    
    # Dataset directories (direct save)
    DATASET_DIR: str = os.getenv("DATASET_DIR", "dataset")
    DATASET_SKETCH_DIR: str = os.path.join(DATASET_DIR, "sketch")
    DATASET_RESULT_DIR: str = os.path.join(DATASET_DIR, "result")
    DATASET_METADATA_DIR: str = os.path.join(DATASET_DIR, "metadata")

settings = Settings()

# Create dataset directories
os.makedirs(settings.DATASET_SKETCH_DIR, exist_ok=True)
os.makedirs(settings.DATASET_RESULT_DIR, exist_ok=True)
os.makedirs(settings.DATASET_METADATA_DIR, exist_ok=True)