import os
import torch
from PIL import Image, ImageOps
import numpy as np
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, StableDiffusionXLControlNetPipeline
from diffusers import StableDiffusionXLAdapterPipeline, T2IAdapter
from diffusers import EulerAncestralDiscreteScheduler, AutoencoderKL
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

def normalize_sketch(sketch):
    """Normalize sketch to have proper contrast and detail"""
    sketch = sketch.convert("L")  # Convert to grayscale
    tensor = T.ToTensor()(sketch)
    return T.ToPILImage()(tensor)

def preprocess_with_pidinet(image, detect_resolution=None, image_resolution=None, apply_filter=None, invert=False):
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
        
        # Invert if needed
        if invert:
            processed_image = ImageOps.invert(processed_image)
            
        return processed_image
    except Exception as e:
        print(f"Error preprocessing image with PidiNet: {str(e)}")
        # Fall back to original image
        if isinstance(image, str):
            image = Image.open(image).convert("L")
        else:
            image = image.convert("L")
            
        # Invert if needed
        if invert:
            image = ImageOps.invert(image)
            
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
        if model_id in ["controlnet_sd15_scribble", "controlnet_sd15_softedge"]:
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
            # Stable Diffusion XL + T2I-Adapter
            
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

def preprocess_sketch(sketch_path, model_id):
    """
    Preprocess the sketch to match the expected format for the specific model
    Each model may have different preprocessing requirements
    """
    # Get the model info and preprocessing settings
    if model_id not in settings.AVAILABLE_MODELS:
        raise ValueError(f"Unknown model ID: {model_id}")
    
    model_info = settings.AVAILABLE_MODELS[model_id]
    preprocessing = model_info.get("preprocessing", {})
    
    # Determine preprocessing type
    preprocess_type = preprocessing.get("type", "scribble")
    invert = preprocessing.get("invert", False)
    normalize = preprocessing.get("normalize", True)
    
    # Create a path for the preprocessed image
    sketch_filename = os.path.basename(sketch_path)
    preprocessed_path = os.path.join(settings.PREPROCESSED_DIR, f"{model_id}_{sketch_filename}")
    
    # Load the original image
    original_image = Image.open(sketch_path)
    
    # Process based on type
    if preprocess_type == "pidinet" and settings.USE_PIDINET_PREPROCESSING:
        print(f"Using PidiNet for sketch preprocessing with model {model_id}")
        processed_image = preprocess_with_pidinet(
            original_image,
            invert=invert
        )
    else:
        # Basic scribble processing
        print(f"Using basic scribble preprocessing for model {model_id}")
        processed_image = original_image.convert("L")
        
        # Invert if needed (some models expect black background with white lines)
        if invert:
            processed_image = ImageOps.invert(processed_image)
    
    # Normalize if requested
    if normalize:
        processed_image = normalize_sketch(processed_image)
    
    # Resize to the expected output size
    processed_image = processed_image.resize((settings.OUTPUT_IMAGE_SIZE, settings.OUTPUT_IMAGE_SIZE))
    
    # Save the preprocessed image for debugging/reference
    processed_image.save(preprocessed_path)
    print(f"Saved preprocessed sketch for {model_id} to {preprocessed_path}")
    
    return processed_image

async def generate_image_from_sketch(
    sketch_path, 
    output_path, 
    prompt, 
    model_id="t2i_adapter_sdxl",
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
        num_inference_steps = config.get("num_inference_steps", settings.NUM_INFERENCE_STEPS)
        guidance_scale = config.get("guidance_scale", settings.GUIDANCE_SCALE)
        
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