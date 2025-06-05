import os
import torch
from PIL import Image, ImageOps, ImageEnhance
import numpy as np
from diffusers import (
    StableDiffusionControlNetPipeline,
    ControlNetModel,
    DDIMScheduler,
    StableDiffusionXLAdapterPipeline,
    T2IAdapter,
    AutoencoderKL
)
from diffusers.utils import load_image
import torchvision.transforms as T
from app.core.config import settings

model_cache = {}

def simple_sketch_inverter(image, target_resolution=768):
    """
    Simple processor that inverts sketches (dark lines on white background → white lines on dark background)
    and resizes to the desired resolution
    """
    try:
        if isinstance(image, str):
            image = Image.open(image)
        
        width, height = image.size
        scale_factor = target_resolution / max(width, height)
        new_width = int(width * scale_factor)
        new_height = int(height * scale_factor)
        
        image = image.resize((new_width, new_height), Image.LANCZOS)
        
        canvas = Image.new('RGB', (target_resolution, target_resolution), color='black')
        
        paste_x = (target_resolution - new_width) // 2
        paste_y = (target_resolution - new_height) // 2
        
        image = image.convert("L")
        
        inverted_image = ImageOps.invert(image)
        
        inverted_image = inverted_image.convert("RGB")
        
        canvas.paste(inverted_image, (paste_x, paste_y))
        
        return canvas
        
    except Exception as e:
        print(f"Error in simple sketch inverter: {str(e)}")
        if isinstance(image, str):
            return Image.open(image).convert("RGB")
        return image.convert("RGB")

def load_model_pipeline(model_id):
    """Load and cache model pipeline based on the model ID"""
    global model_cache
    
    if model_id in model_cache:
        return model_cache[model_id]
    
    if model_id not in settings.AVAILABLE_MODELS:
        raise ValueError(f"Unknown model ID: {model_id}")
    
    model_info = settings.AVAILABLE_MODELS[model_id]
    huggingface_id = model_info["huggingface_id"]
    
    is_cuda_available = torch.cuda.is_available()
    device = settings.DEVICE if is_cuda_available and settings.DEVICE == 'cuda' else 'cpu'
    torch_dtype = torch.float16 if device == "cuda" else torch.float32
   
    print(f"Device set to: {device}")
    
    try:
        print(f"Loading model: {model_id} ({huggingface_id})")
        
        if model_id == "controlnet_scribble":
            controlnet = ControlNetModel.from_pretrained(
                huggingface_id,
                torch_dtype=torch_dtype
            )
            
            pipeline = StableDiffusionControlNetPipeline.from_pretrained(
                "runwayml/stable-diffusion-v1-5",
                controlnet=controlnet,
                safety_checker=None,
                torch_dtype=torch_dtype
            )
            
            pipeline.scheduler = DDIMScheduler.from_config(pipeline.scheduler.config)
            
        elif model_id == "t2i_adapter_sdxl":
            adapter = T2IAdapter.from_pretrained(
                huggingface_id,
                torch_dtype=torch_dtype,
                adapter_type="full_adapter_xl",
                variant="fp16" if device == "cuda" else None
            )
            
            vae = AutoencoderKL.from_pretrained(
                "madebyollin/sdxl-vae-fp16-fix",
                torch_dtype=torch_dtype
            )
            
            ddim_scheduler = DDIMScheduler.from_pretrained(
                model_info["base_model"], 
                subfolder="scheduler"
            )
            
            pipeline = StableDiffusionXLAdapterPipeline.from_pretrained(
                model_info["base_model"],
                adapter=adapter,
                vae=vae,
                scheduler=ddim_scheduler,
                torch_dtype=torch_dtype,
                variant="fp16" if device == "cuda" else None
            )
        
        pipeline = pipeline.to(device)
        
        if device == "cuda":
            try:
                if hasattr(pipeline, "enable_xformers_memory_efficient_attention"):
                    pipeline.enable_xformers_memory_efficient_attention()
                    print("Enabled xformers memory efficient attention")
                else:
                    if hasattr(pipeline, "enable_model_cpu_offload"):
                        pipeline.enable_model_cpu_offload()
                        print("Enabled model CPU offload")
                    else:
                        pipeline.enable_attention_slicing()
                        print("Enabled attention slicing")
            except Exception as e:
                print(f"Warning: Could not optimize memory usage: {str(e)}")
                if hasattr(pipeline, "enable_attention_slicing"):
                    pipeline.enable_attention_slicing()
                    print("Enabled attention slicing")
        
        model_cache[model_id] = pipeline
        print(f"Model {model_id} loaded successfully!")
        
        return pipeline
        
    except Exception as e:
        print(f"Error loading model {model_id}: {str(e)}")
        raise e

def preprocess_sketch(sketch_path, model_id):
    """
    Preprocess the sketch to match the expected format for the specific model
    """
    if model_id not in settings.AVAILABLE_MODELS:
        raise ValueError(f"Unknown model ID: {model_id}")
    
    model_info = settings.AVAILABLE_MODELS[model_id]
    preprocessing = model_info.get("preprocessing", {})
    
    sketch_filename = os.path.basename(sketch_path)
    preprocessed_path = os.path.join(settings.PREPROCESSED_DIR, f"{model_id}_{sketch_filename}")
    
    original_image = Image.open(sketch_path)
    
    detect_resolution = preprocessing.get("detect_resolution", 768)
    
    print(f"Using simple sketch inverter for model {model_id}")
    processed_image = simple_sketch_inverter(
        original_image,
        target_resolution=detect_resolution
    )
    
    processed_image.save(preprocessed_path)
    print(f"Saved preprocessed sketch for {model_id} to {preprocessed_path}")
    
    return processed_image

async def generate_image_from_sketch(
    sketch_path, 
    output_path, 
    prompt, 
    model_id="controlnet_scribble",
    negative_prompt=""
):
    """Generate an image from a sketch using the selected model"""
    try:
        pipe = load_model_pipeline(model_id)
        
        sketch_image = preprocess_sketch(sketch_path, model_id)
        
        model_info = settings.AVAILABLE_MODELS[model_id]
        config = model_info.get("config", {})
        
        if not negative_prompt and "default_negative_prompt" in config:
            negative_prompt = config["default_negative_prompt"]
        
        params = {
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "image": sketch_image,
        }
        
        num_inference_steps = config.get("num_inference_steps", 20)
        guidance_scale = config.get("guidance_scale", 7.5)
        
        params["num_inference_steps"] = num_inference_steps
        params["guidance_scale"] = guidance_scale
        
        if model_id == "t2i_adapter_sdxl":
            params["adapter_conditioning_scale"] = config.get("adapter_conditioning_scale", 0.9)
            params["adapter_conditioning_factor"] = config.get("adapter_conditioning_factor", 0.9)
        
        print(f"Generating with model {model_id} using parameters: {params}")
        
        output = pipe(**params)
        
        if hasattr(output, "images") and len(output.images) > 0:
            output_image = output.images[0]
        else:
            output_image = output[0] if isinstance(output, (list, tuple)) else output
        
        output_image.save(output_path)
        
        return output_path
    
    except Exception as e:
        print(f"Error generating image with model {model_id}: {str(e)}")
        raise e