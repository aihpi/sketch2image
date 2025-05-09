import os
import torch
from PIL import Image
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, StableDiffusionXLControlNetPipeline
from diffusers import StableDiffusionXLAdapterPipeline, T2IAdapter, AutoPipelineForImage2Image
from diffusers import EulerAncestralDiscreteScheduler, AutoencoderKL
from diffusers.utils import load_image
import numpy as np
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

def preprocess_with_pidinet(image, detect_resolution=None, image_resolution=None, apply_filter=None):
    """
    Preprocess an image using PidiNet to create a sketch-like format
    """
    try:
        # Get the PidiNet processor
        processor = get_pidinet_processor()
        if processor is None:
            print("PidiNet processor not available, returning original image")
            if isinstance(image, str):
                return Image.open(image)
            return image
        
        # If image is a path, load it
        if isinstance(image, str):
            image = Image.open(image)
        
        # Set default parameters from settings if not provided
        if detect_resolution is None:
            detect_resolution = settings.PIDINET_DETECT_RESOLUTION
        if image_resolution is None:
            image_resolution = settings.PIDINET_IMAGE_RESOLUTION
        if apply_filter is None:
            apply_filter = settings.PIDINET_APPLY_FILTER
        
        # Process image
        processed_image = processor(
            image,
            detect_resolution=detect_resolution,
            image_resolution=image_resolution,
            apply_filter=apply_filter
        ).convert("L")
        
        return processed_image
    except Exception as e:
        print(f"Error preprocessing image with PidiNet: {str(e)}")
        # Fall back to original image
        if isinstance(image, str):
            return Image.open(image)
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
                torch_dtype=torch_dtype
            )
            
            pipeline = StableDiffusionControlNetPipeline.from_pretrained(
                "runwayml/stable-diffusion-v1-5",
                controlnet=controlnet,
                torch_dtype=torch_dtype
            )
            
        elif model_id == "controlnet_sdxl_scribble":
            # Stable Diffusion XL + ControlNet
            controlnet = ControlNetModel.from_pretrained(
                huggingface_id,
                torch_dtype=torch_dtype
            )
            
            pipeline = StableDiffusionXLControlNetPipeline.from_pretrained(
                "stabilityai/stable-diffusion-xl-base-1.0",
                controlnet=controlnet,
                torch_dtype=torch_dtype
            )
            
        elif model_id == "t2i_adapter_sdxl":
            # Stable Diffusion XL + T2I-Adapter - Exactly like in the tutorial
            
            # 1. Load the T2I adapter
            adapter = T2IAdapter.from_pretrained(
                huggingface_id,
                subfolder=model_info.get("sub_folder", ""),
                torch_dtype=torch_dtype
            )
            
            # 2. Load the VAE if specified
            vae = None
            if "vae_model" in model_info:
                vae = AutoencoderKL.from_pretrained(
                    model_info["vae_model"],
                    torch_dtype=torch_dtype
                )
            
            # 3. Load the custom scheduler if specified
            scheduler = None
            if model_info.get("config", {}).get("custom_scheduler", False):
                scheduler = EulerAncestralDiscreteScheduler.from_pretrained(
                    model_info["base_model"],
                    subfolder="scheduler"
                )
            
            # 4. Set up the pipeline with all components
            kwargs = {
                "adapter": adapter,
                "torch_dtype": torch_dtype
            }
            
            if vae is not None:
                kwargs["vae"] = vae
                
            if scheduler is not None:
                kwargs["scheduler"] = scheduler
                
            if "sdxl" in model_info.get("base_model", "").lower():
                kwargs["variant"] = "fp16" if device == "cuda" else None
            
            pipeline = StableDiffusionXLAdapterPipeline.from_pretrained(
                model_info["base_model"],
                **kwargs
            )
            
        elif model_id == "pix2pix_turbo":
            # Pix2Pix-Turbo model - uses different parameters
            pipeline = AutoPipelineForImage2Image.from_pretrained(
                huggingface_id,
                torch_dtype=torch_dtype
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
                    # Fall back to attention slicing if xformers fails
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

def preprocess_sketch(sketch_path):
    """Preprocess the sketch to match the expected format for the model"""
    # Check if we should use PidiNet preprocessing
    if settings.USE_PIDINET_PREPROCESSING:
        print("Using PidiNet for sketch preprocessing")
        # Create a path for the preprocessed image
        sketch_filename = os.path.basename(sketch_path)
        preprocessed_path = os.path.join(settings.PREPROCESSED_DIR, f"pidinet_{sketch_filename}")
        
        # Load the original image
        original_image = Image.open(sketch_path)
        
        # Preprocess the image with PidiNet
        processed_image = preprocess_with_pidinet(original_image)
        
        # Save the preprocessed image for debugging/reference
        processed_image.save(preprocessed_path)
        print(f"Saved preprocessed PidiNet sketch to {preprocessed_path}")
        
        # Ensure the image has the right size
        processed_image = processed_image.resize((settings.OUTPUT_IMAGE_SIZE, settings.OUTPUT_IMAGE_SIZE))
        
        return processed_image
    
    # If not using PidiNet, use the original preprocessing
    # Load the sketch image
    image = load_image(sketch_path)
    
    # Ensure the image has the right size
    image = image.resize((settings.OUTPUT_IMAGE_SIZE, settings.OUTPUT_IMAGE_SIZE))
    
    # Convert to numpy array for processing
    image_np = np.array(image)
    
    # If the image has an alpha channel, composite it on a white background
    if len(image_np.shape) == 3 and image_np.shape[2] == 4:
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
    model_id="t2i_adapter_sdxl",  # Default to T2I adapter
    negative_prompt=""
):
    """Generate an image from a sketch using the selected model"""
    try:
        # Load the selected model pipeline
        pipe = load_model_pipeline(model_id)
        
        # Preprocess the sketch
        sketch_image = preprocess_sketch(sketch_path)
        
        # Get model configuration
        model_info = settings.AVAILABLE_MODELS[model_id]
        config = model_info.get("config", {})
        model_type = model_info.get("model_type", "")
        
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
        num_inference_steps = config.get("num_inference_steps", settings.NUM_INFERENCE_STEPS)
        guidance_scale = config.get("guidance_scale", settings.GUIDANCE_SCALE)
        
        params["num_inference_steps"] = num_inference_steps
        params["guidance_scale"] = guidance_scale
        
        # Add model-specific parameters 
        if model_id == "t2i_adapter_sdxl":
            # Exactly match tutorial parameters
            params["adapter_conditioning_scale"] = config.get("adapter_conditioning_scale", 0.9)
            params["adapter_conditioning_factor"] = config.get("adapter_conditioning_factor", 0.9)
            
        elif model_id == "pix2pix_turbo":
            # Pix2Pix-Turbo model - uses different parameters
            params["strength"] = config.get("strength", 0.8)
            params["num_inference_steps"] = 1  # Always 1 for pix2pix_turbo
        
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