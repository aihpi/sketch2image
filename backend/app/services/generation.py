import os
import torch
from PIL import Image, ImageOps, ImageEnhance
import numpy as np
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler
from diffusers import StableDiffusionXLAdapterPipeline, T2IAdapter, DDPMScheduler
from diffusers.utils import load_image
import torchvision.transforms as T
from app.core.config import settings

# Cache for loaded models
model_cache = {}
# Cache for PidiNet processor
pidinet_processor = None

def get_pidinet_processor():
    """Get or initialize the PidiNet processor for sketch preprocessing"""
    global pidinet_processor
    
    if pidinet_processor is None:
        try:
            from controlnet_aux.pidi import PidiNetDetector
            
            # Initialize the processor
            processor = PidiNetDetector.from_pretrained("lllyasviel/Annotators")
            
            # Move to GPU if available
            if torch.cuda.is_available():
                processor = processor.to("cuda")
                
            pidinet_processor = processor
            print("PidiNet processor initialized successfully")
        except Exception as e:
            print(f"Error initializing PidiNet processor: {str(e)}")
            # Return None to indicate failure
            return None
    
    return pidinet_processor

def preprocess_with_pidinet(image, detect_resolution=1024, image_resolution=1024, apply_filter=False, enhance_contrast=1.5, safe_steps=2):
    """
    Preprocess an image using PidiNet to create a sketch-like format
    """
    try:
        # Get the PidiNet processor
        processor = get_pidinet_processor()
        if processor is None:
            print("PidiNet processor not available, returning original image")
            if isinstance(image, str):
                return Image.open(image).convert("L")
            return image.convert("L")
        
        # If image is a path, load it
        if isinstance(image, str):
            image = Image.open(image)
        
        # Process image with PidiNet
        processed_image = processor(
            image,
            detect_resolution=detect_resolution,
            image_resolution=image_resolution,
            apply_filter=apply_filter,
            safe_steps=safe_steps
        )
        
        # Enhance contrast if requested
        if enhance_contrast > 0:
            enhancer = ImageEnhance.Contrast(processed_image)
            processed_image = enhancer.enhance(enhance_contrast)
            
        return processed_image
    except Exception as e:
        print(f"Error preprocessing image with PidiNet: {str(e)}")
        # Fall back to original image
        if isinstance(image, str):
            image = Image.open(image).convert("L")
        else:
            image = image.convert("L")
            
        return image

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
    torch_dtype = torch.float16 if device == "cuda" else torch.float32
   
    print(f"Device set to: {device}")
    
    try:
        print(f"Loading model: {model_id} ({huggingface_id})")
        
        if model_id == "controlnet_scribble":
            # Load ControlNet Scribble model
            controlnet = ControlNetModel.from_pretrained(
                huggingface_id,
                torch_dtype=torch_dtype
            )
            
            # Load pipeline with stable diffusion v1-5
            pipeline = StableDiffusionControlNetPipeline.from_pretrained(
                "runwayml/stable-diffusion-v1-5",
                controlnet=controlnet,
                safety_checker=None,
                torch_dtype=torch_dtype
            )
            
            # Use UniPC scheduler as in the example
            pipeline.scheduler = UniPCMultistepScheduler.from_config(pipeline.scheduler.config)
            
        elif model_id == "t2i_adapter_sdxl":
            # Load the T2I adapter for SDXL
            adapter = T2IAdapter.from_pretrained(
                huggingface_id,
                torch_dtype=torch_dtype,
                adapter_type="full_adapter_xl"
            )
            
            # Create the scheduler
            scheduler = DDPMScheduler.from_pretrained(
                model_info["base_model"], 
                subfolder="scheduler"
            )
            
            # Create the pipeline
            pipeline = StableDiffusionXLAdapterPipeline.from_pretrained(
                model_info["base_model"],
                adapter=adapter,
                scheduler=scheduler,
                torch_dtype=torch_dtype,
                variant="fp16" if device == "cuda" else None
            )
        
        # Move to device (GPU if available)
        pipeline = pipeline.to(device)
        
        # Optimize memory usage for GPU
        if device == "cuda":
            try:
                # Try to enable xformers first
                if hasattr(pipeline, "enable_xformers_memory_efficient_attention"):
                    pipeline.enable_xformers_memory_efficient_attention()
                    print("Enabled xformers memory efficient attention")
                else:
                    # Try to enable model CPU offload as shown in the example
                    if hasattr(pipeline, "enable_model_cpu_offload"):
                        pipeline.enable_model_cpu_offload()
                        print("Enabled model CPU offload")
                    else:
                        # Fall back to attention slicing if neither is available
                        pipeline.enable_attention_slicing()
                        print("Enabled attention slicing")
            except Exception as e:
                print(f"Warning: Could not optimize memory usage: {str(e)}")
                # Still enable attention slicing as fallback
                if hasattr(pipeline, "enable_attention_slicing"):
                    pipeline.enable_attention_slicing()
                    print("Enabled attention slicing")
        
        # Cache the pipeline
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
    # Get the model info and preprocessing settings
    if model_id not in settings.AVAILABLE_MODELS:
        raise ValueError(f"Unknown model ID: {model_id}")
    
    model_info = settings.AVAILABLE_MODELS[model_id]
    preprocessing = model_info.get("preprocessing", {})
    
    # Create a path for the preprocessed image
    sketch_filename = os.path.basename(sketch_path)
    preprocessed_path = os.path.join(settings.PREPROCESSED_DIR, f"{model_id}_{sketch_filename}")
    
    # Load the original image
    original_image = Image.open(sketch_path)
    
    # Get preprocessing parameters
    detect_resolution = preprocessing.get("detect_resolution", 1024)
    image_resolution = preprocessing.get("image_resolution", 1024)
    apply_filter = preprocessing.get("apply_filter", False)
    enhance_contrast = preprocessing.get("enhance_contrast", 1.5)
    safe_steps = preprocessing.get("safe_steps", 2)
    
    # Always use PidiNet for preprocessing
    print(f"Using PidiNet for sketch preprocessing with model {model_id}")
    processed_image = preprocess_with_pidinet(
        original_image,
        detect_resolution=detect_resolution,
        image_resolution=image_resolution,
        apply_filter=apply_filter,
        enhance_contrast=enhance_contrast,
        safe_steps=safe_steps
    )
    
    # Save the preprocessed image for debugging/reference
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
        # Load the selected model pipeline
        pipe = load_model_pipeline(model_id)
        
        # Preprocess the sketch for the specific model
        sketch_image = preprocess_sketch(sketch_path, model_id)
        
        # Get model configuration
        model_info = settings.AVAILABLE_MODELS[model_id]
        config = model_info.get("config", {})
        
        # Use default negative prompt if not provided and available in config
        if not negative_prompt and "default_negative_prompt" in config:
            negative_prompt = config["default_negative_prompt"]
        
        # Prepare common parameters
        params = {
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "image": sketch_image,
        }
        
        # Get specific parameters from config
        num_inference_steps = config.get("num_inference_steps", 20)
        guidance_scale = config.get("guidance_scale", 7.5)
        
        params["num_inference_steps"] = num_inference_steps
        params["guidance_scale"] = guidance_scale
        
        # Add model-specific parameters 
        if model_id == "t2i_adapter_sdxl":
            params["adapter_conditioning_scale"] = config.get("adapter_conditioning_scale", 0.9)
            params["adapter_conditioning_factor"] = config.get("adapter_conditioning_factor", 0.9)
        
        print(f"Generating with model {model_id} using parameters: {params}")
        
        # Generate the image
        output = pipe(**params)
        
        # Get the output image
        if hasattr(output, "images") and len(output.images) > 0:
            output_image = output.images[0]
        else:
            # Fallback in case the output format changes
            output_image = output[0] if isinstance(output, (list, tuple)) else output
        
        # Save the output image
        output_image.save(output_path)
        
        return output_path
    
    except Exception as e:
        # Log the error
        print(f"Error generating image with model {model_id}: {str(e)}")
        raise e