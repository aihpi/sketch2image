import os
import torch
from PIL import Image
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, StableDiffusionXLControlNetPipeline
from diffusers import StableDiffusionXLAdapterPipeline, T2IAdapter
from diffusers.utils import load_image
import numpy as np
from app.core.config import settings

# Cache for loaded models
model_cache = {}

def load_model_pipeline(model_id):
    """Load and cache model pipeline based on the model ID"""
    global model_cache
    
    if model_id in model_cache:
        return model_cache[model_id]
    
    # Check if model exists in available models
    if model_id not in settings.AVAILABLE_MODELS:
        raise ValueError(f"Unknown model ID: {model_id}")
    
    model_info = settings.AVAILABLE_MODELS[model_id]
    huggingface_id = model_info["huggingface_id"]
    
    # Check if CUDA is available
    is_cuda_available = torch.cuda.is_available()
    device = settings.DEVICE if is_cuda_available and settings.DEVICE == 'cuda' else 'cpu'
   
    print(f"Device set to: {device}")
    print(f"Is CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"Current CUDA device: {torch.cuda.current_device()}")
        print(f"Device name: {torch.cuda.get_device_name()}")
        print(f"Memory allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
        print(f"Memory reserved: {torch.cuda.memory_reserved() / 1e9:.2f} GB")
    
    try:
        print(f"Loading model: {model_id} ({huggingface_id})")
        
        # Different loading logic based on model type
        if model_id == "controlnet_sd15_scribble":
            # Stable Diffusion 1.5 + ControlNet
            controlnet = ControlNetModel.from_pretrained(
                huggingface_id,
                torch_dtype=torch.float16 if device == "cuda" else torch.float32
            )
            
            pipeline = StableDiffusionControlNetPipeline.from_pretrained(
                "runwayml/stable-diffusion-v1-5",
                controlnet=controlnet,
                torch_dtype=torch.float16 if device == "cuda" else torch.float32
            )
            
        elif model_id == "controlnet_sdxl_scribble":
            # Stable Diffusion XL + ControlNet
            controlnet = ControlNetModel.from_pretrained(
                huggingface_id,
                torch_dtype=torch.float16 if device == "cuda" else torch.float32
            )
            
            pipeline = StableDiffusionXLControlNetPipeline.from_pretrained(
                "stabilityai/stable-diffusion-xl-base-1.0",
                controlnet=controlnet,
                torch_dtype=torch.float16 if device == "cuda" else torch.float32
            )
            
        elif model_id == "t2i_adapter_sdxl":
            # Stable Diffusion XL + T2I-Adapter
            adapter = T2IAdapter.from_pretrained(
                huggingface_id,
                torch_dtype=torch.float16 if device == "cuda" else torch.float32
            )
            
            pipeline = StableDiffusionXLAdapterPipeline.from_pretrained(
                "stabilityai/stable-diffusion-xl-base-1.0",
                adapter=adapter,
                torch_dtype=torch.float16 if device == "cuda" else torch.float32
            )
            
        elif model_id == "pix2pix_turbo":
            # Different architecture, using a simpler import for now
            # In a real implementation, you'd need to handle this model differently
            # This is a placeholder implementation
            from diffusers import AutoPipelineForImage2Image
            
            pipeline = AutoPipelineForImage2Image.from_pretrained(
                huggingface_id,
                torch_dtype=torch.float16 if device == "cuda" else torch.float32
            )
        
        # Move to device (GPU if available)
        pipeline = pipeline.to(device)
        
        # Optimize memory usage for GPU
        if device == "cuda":
            try:
                # Try to enable xformers first
                pipeline.enable_xformers_memory_efficient_attention()
                print("Enabled xformers memory efficient attention")
            except:
                # Fall back to attention slicing if xformers fails
                pipeline.enable_attention_slicing()
                print("Enabled attention slicing")
        
        # Cache the pipeline
        model_cache[model_id] = pipeline
        print(f"Model {model_id} loaded successfully!")
        
        return pipeline
        
    except Exception as e:
        print(f"Error loading model {model_id}: {str(e)}")
        raise e

def preprocess_sketch(sketch_path):
    """Preprocess the sketch to match the expected format for the model"""
    # Load the sketch image
    image = load_image(sketch_path)
    
    # Ensure the image has the right size
    image = image.resize((settings.OUTPUT_IMAGE_SIZE, settings.OUTPUT_IMAGE_SIZE))
    
    # Convert to numpy array for processing
    image_np = np.array(image)
    
    # If the image has an alpha channel, composite it on a white background
    if image_np.shape[2] == 4:
        # Create a white background
        white_background = np.ones((image_np.shape[0], image_np.shape[1], 3), dtype=np.uint8) * 255
        
        # Extract RGB and alpha channels
        rgb = image_np[:, :, :3]
        alpha = image_np[:, :, 3:4] / 255.0
        
        # Composite RGB over white background using alpha
        image_np = rgb * alpha + white_background * (1 - alpha)
        image_np = image_np.astype(np.uint8)
    
    # Convert back to PIL Image
    image = Image.fromarray(image_np)
    
    return image

async def generate_image_from_sketch(
    sketch_path, 
    output_path, 
    prompt, 
    model_id="controlnet_sd15_scribble", 
    negative_prompt=""
):
    """Generate an image from a sketch using the selected model"""
    try:
        # Load the selected model pipeline
        pipe = load_model_pipeline(model_id)
        
        # Preprocess the sketch
        sketch_image = preprocess_sketch(sketch_path)
        
        # Different generation logic based on model type
        if model_id in ["controlnet_sd15_scribble", "controlnet_sdxl_scribble"]:
            # ControlNet models (SD 1.5 or SDXL)
            output_image = pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                image=sketch_image,
                num_inference_steps=settings.NUM_INFERENCE_STEPS,
                guidance_scale=settings.GUIDANCE_SCALE,
            ).images[0]
            
        elif model_id == "t2i_adapter_sdxl":
            # T2I-Adapter model
            output_image = pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                image=sketch_image,
                num_inference_steps=settings.NUM_INFERENCE_STEPS,
                guidance_scale=settings.GUIDANCE_SCALE,
                adapter_conditioning_scale=1.0,  # Specific to adapter models
            ).images[0]
            
        elif model_id == "pix2pix_turbo":
            # Pix2Pix-Turbo model - uses different parameters
            output_image = pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                image=sketch_image,
                strength=0.8,  # Controls how much to respect the original image
                guidance_scale=settings.GUIDANCE_SCALE,
                num_inference_steps=1,  # Pix2Pix-Turbo uses just one step
            ).images[0]
        
        # Save the output image
        output_image.save(output_path)
        
        return output_path
    
    except Exception as e:
        # Log the error
        print(f"Error generating image with model {model_id}: {str(e)}")
        raise e