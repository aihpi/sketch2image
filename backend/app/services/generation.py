import os
import torch
from PIL import Image
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel
from diffusers.utils import load_image
import numpy as np
from app.core.config import settings

# Load models lazily to save memory
pipeline = None

def get_pipeline():
    """Initialize and return the pipeline, loading models if needed"""
    global pipeline
    
    if pipeline is None:
        print("Starting to load ControlNet model...")
        print(f"Using model ID: {settings.MODEL_ID}")
        print(f"Device: {settings.DEVICE}")
        
        # Check if CUDA is available
        is_cuda_available = torch.cuda.is_available()
        print(f"CUDA available: {is_cuda_available}")
        if is_cuda_available:
            print(f"CUDA Device: {torch.cuda.get_device_name(0)}")
            print(f"CUDA Version: {torch.version.cuda}")
        
        # Use the specified device, fallback to CPU if CUDA is not available
        device = settings.DEVICE if is_cuda_available and settings.DEVICE == 'cuda' else 'cpu'
        print(f"Using device: {device}")
        
        # Load the ControlNet model
        try:
            print("Downloading/loading ControlNet model...")
            controlnet = ControlNetModel.from_pretrained(
                settings.MODEL_ID,
                torch_dtype=torch.float16 if device == "cuda" else torch.float32
            )
            print("ControlNet model loaded successfully!")
            
            # Load the Stable Diffusion pipeline with ControlNet
            print("Downloading/loading Stable Diffusion pipeline...")
            pipeline = StableDiffusionControlNetPipeline.from_pretrained(
                "runwayml/stable-diffusion-v1-5",
                controlnet=controlnet,
                torch_dtype=torch.float16 if device == "cuda" else torch.float32
            )
            print("Stable Diffusion pipeline loaded successfully!")
            
            # Move to device (GPU if available)
            print(f"Moving pipeline to device: {device}")
            pipeline = pipeline.to(device)
            
            # Optimize memory usage for GPU
            if device == "cuda":
                print("Enabling memory-efficient attention for CUDA...")
                pipeline.enable_xformers_memory_efficient_attention()
                # Alternatively use this if xformers is not available
                # pipeline.enable_attention_slicing()
            
            print("Pipeline initialization complete!")
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            raise e
    
    return pipeline

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

async def generate_image_from_sketch(sketch_path, output_path, prompt, negative_prompt=""):
    """Generate an image from a sketch using the ControlNet model"""
    try:
        # Get the pipeline
        pipe = get_pipeline()
        
        # Preprocess the sketch
        sketch_image = preprocess_sketch(sketch_path)
        
        # Generate the image
        output_image = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=sketch_image,
            num_inference_steps=settings.NUM_INFERENCE_STEPS,
            guidance_scale=settings.GUIDANCE_SCALE,
        ).images[0]
        
        # Save the output image
        output_image.save(output_path)
        
        return output_path
    
    except Exception as e:
        # Log the error
        print(f"Error generating image: {str(e)}")
        raise e