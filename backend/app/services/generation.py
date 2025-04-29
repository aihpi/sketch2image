import os
import torch
from PIL import Image
import numpy as np
import importlib
from app.core.config import settings

# Cache for loaded models
model_cache = {}

def get_huggingface_token():
    """Get Hugging Face token from environment variable or settings"""
    # First check environment variable
    token = os.environ.get("HUGGINGFACE_TOKEN", None)
    
    # Then fallback to settings if available
    if token is None and hasattr(settings, "HUGGINGFACE_TOKEN"):
        token = settings.HUGGINGFACE_TOKEN
        
    return token

def get_class_from_name(class_name):
    """Dynamically import and return a class from diffusers based on its name"""
    try:
        module = importlib.import_module('diffusers')
        return getattr(module, class_name)
    except (ImportError, AttributeError) as e:
        # If not found in main diffusers module, try specific submodules
        try:
            # Try pipelines module
            module = importlib.import_module('diffusers.pipelines')
            return getattr(module, class_name)
        except (ImportError, AttributeError):
            # As a fallback, try other common modules
            for submodule in ['models', 'schedulers', 'utils']:
                try:
                    module = importlib.import_module(f'diffusers.{submodule}')
                    return getattr(module, class_name)
                except (ImportError, AttributeError):
                    pass
            # If we get here, we couldn't find the class
            raise ValueError(f"Could not find class {class_name} in diffusers package: {str(e)}")

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
        config = model_info.get("config", {})
        
        # Check if this model requires authentication
        requires_auth = config.get("requires_auth", False)
        token = get_huggingface_token() if requires_auth else None
        
        if requires_auth and token is None:
            raise ValueError(
                f"Model {model_id} requires Hugging Face authentication, but no token was provided. "
                "Please set the HUGGINGFACE_TOKEN environment variable or add it to your config."
            )
        
        # Special case for Kandinsky model
        if model_id == "kandinsky":
            # Kandinsky needs special handling as it's not structured like other models
            from diffusers import KandinskyPipeline, KandinskyPriorPipeline
            
            # The Kandinsky model requires a different loading approach
            # First, load the prior pipeline
            prior = KandinskyPriorPipeline.from_pretrained(
                model_info["prior_model_id"],
                torch_dtype=torch_dtype,
                use_auth_token=token
            ).to(device)
            
            # Then load the image pipeline
            pipeline = KandinskyPipeline.from_pretrained(
                huggingface_id,
                torch_dtype=torch_dtype,
                use_auth_token=token
            ).to(device)
            
            # Store the prior pipeline for later use
            pipeline.prior_pipeline = prior
        else:
            # Get the pipeline and model classes
            pipeline_class_name = config.get("pipeline_type")
            model_class_name = config.get("model_type")
            
            if not pipeline_class_name:
                raise ValueError(f"Pipeline type not specified for model {model_id}")
            
            # Dynamically import the classes
            PipelineClass = get_class_from_name(pipeline_class_name)
            
            # Special handling for different model types
            if model_class_name == "T2IAdapter":
                # For T2I Adapter models
                ModelClass = get_class_from_name(model_class_name)
                
                # Get additional parameters
                adapter_type = config.get("adapter_type", "full_adapter_xl")
                
                # Load the component model
                model_component = ModelClass.from_pretrained(
                    huggingface_id,
                    torch_dtype=torch_dtype,
                    adapter_type=adapter_type,  # Add adapter_type parameter
                    use_auth_token=token
                )
                
                # Load VAE if specified
                vae = None
                if "vae_model" in model_info:
                    from diffusers import AutoencoderKL
                    vae = AutoencoderKL.from_pretrained(
                        model_info["vae_model"],
                        torch_dtype=torch_dtype,
                        use_auth_token=token
                    )
                
                # Load scheduler
                scheduler = None
                if config.get("custom_scheduler", False):
                    from diffusers import EulerAncestralDiscreteScheduler
                    scheduler = EulerAncestralDiscreteScheduler.from_pretrained(
                        model_info["base_model"],
                        subfolder="scheduler"
                    )
                
                # Load the base model with the component
                base_model = model_info.get("base_model", "")
                if not base_model:
                    raise ValueError(f"Base model not specified for model {model_id}")
                
                # Create the keyword arguments dictionary with the correct parameters
                kwargs = {
                    "adapter": model_component,
                    "torch_dtype": torch_dtype
                }
                
                # Add optional components if available
                if vae is not None:
                    kwargs["vae"] = vae
                if scheduler is not None:
                    kwargs["scheduler"] = scheduler
                
                # Add auth token if needed
                if token:
                    kwargs["use_auth_token"] = token
                
                # For SDXL models, add the variant parameter
                if "sdxl" in base_model.lower():
                    kwargs["variant"] = "fp16"
                
                # Load the pipeline with the properly named parameters
                pipeline = PipelineClass.from_pretrained(
                    base_model,
                    **kwargs
                )
            elif model_class_name == "ControlNetModel":
                # For ControlNet models
                ModelClass = get_class_from_name(model_class_name)
                
                # Load the component model
                model_component = ModelClass.from_pretrained(
                    huggingface_id,
                    torch_dtype=torch_dtype,
                    use_auth_token=token
                )
                
                # Load the base model with the component
                base_model = model_info.get("base_model", "")
                if not base_model:
                    raise ValueError(f"Base model not specified for model {model_id}")
                
                # Create the keyword arguments dictionary with the correct parameter name
                kwargs = {
                    "controlnet": model_component,  # Use "controlnet" for ControlNetModel
                    "torch_dtype": torch_dtype
                }
                
                # Add auth token if needed
                if token:
                    kwargs["use_auth_token"] = token
                
                # Load the pipeline with the properly named parameter
                pipeline = PipelineClass.from_pretrained(
                    base_model,
                    **kwargs
                )
            else:
                # For models that don't need separate components (like AutoPipeline)
                pipeline = PipelineClass.from_pretrained(
                    huggingface_id,
                    torch_dtype=torch_dtype,
                    use_auth_token=token
                )
        
        # Move to device (GPU if available)
        pipeline = pipeline.to(device)
        
        # Optimize memory usage for GPU
        if device == "cuda":
            try:
                # Try to enable xformers first if the model supports it
                if hasattr(pipeline, "enable_xformers_memory_efficient_attention"):
                    pipeline.enable_xformers_memory_efficient_attention()
                    print("Enabled xformers memory efficient attention")
                else:
                    # Fall back to attention slicing if xformers not supported
                    if hasattr(pipeline, "enable_attention_slicing"):
                        pipeline.enable_attention_slicing()
                        print("Enabled attention slicing")
            except Exception as e:
                print(f"Warning: Could not optimize memory usage: {str(e)}")
        
        # Cache the pipeline
        model_cache[model_id] = pipeline
        print(f"Model {model_id} loaded successfully!")
        
        return pipeline
        
    except Exception as e:
        print(f"Error loading model {model_id}: {str(e)}")
        raise e

def preprocess_sketch(sketch_path):
    """Preprocess the sketch to match the expected format for the model"""
    from diffusers.utils import load_image
    
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
    model_id="controlnet_sd15_scribble", 
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
        
        # Prepare common parameters
        params = {
            "prompt": prompt,
            "negative_prompt": negative_prompt or "disfigured, extra digit, fewer digits, cropped, worst quality, low quality",
            "image": sketch_image,
        }
        
        # Set model-specific parameters from config
        if "num_inference_steps" not in config:
            params["num_inference_steps"] = settings.NUM_INFERENCE_STEPS
            
        if "guidance_scale" not in config:
            params["guidance_scale"] = settings.GUIDANCE_SCALE
            
        # Add any model-specific parameters from config
        for key, value in config.items():
            if key not in ["pipeline_type", "model_type", "needs_safety_checker", "requires_auth", "custom_scheduler", "adapter_type"]:
                params[key] = value
                
        # Special handling for Kandinsky model
        if model_type == "kandinsky" and hasattr(pipe, "prior_pipeline"):
            print("Using Kandinsky model with prior")
            # Generate image embeddings from the prompt
            prior_output = pipe.prior_pipeline(
                prompt,
                guidance_scale=config.get("prior_guidance_scale", 4.0)
            )
            
            # Kandinsky uses different parameter names
            kandinsky_params = {
                "image_embeds": prior_output.image_embeds,
                "negative_image_embeds": prior_output.negative_image_embeds,
                "image": sketch_image,
                "height": settings.OUTPUT_IMAGE_SIZE,
                "width": settings.OUTPUT_IMAGE_SIZE,
                "guidance_scale": config.get("guidance_scale", settings.GUIDANCE_SCALE),
                "num_inference_steps": config.get("num_inference_steps", settings.NUM_INFERENCE_STEPS)
            }
            
            # Replace params with kandinsky-specific ones
            params = kandinsky_params
        
        # Special handling for ControlNet vs T2I Adapter models
        elif model_type in ["controlnet_sd15", "controlnet_sdxl"]:
            # ControlNet models may need specific parameters
            pass  # Already handled by the common parameters
            
        elif model_type == "t2i_adapter":
            # T2I adapter models need specific parameters
            if "adapter_conditioning_scale" not in params:
                params["adapter_conditioning_scale"] = config.get("adapter_conditioning_scale", 1.0)
            if "adapter_conditioning_factor" not in params:
                params["adapter_conditioning_factor"] = config.get("adapter_conditioning_factor", 1.0)
                
        elif model_type == "pix2pix":
            # Pix2Pix models might need strength parameter
            if "strength" not in params:
                params["strength"] = 0.8
            # Also might use fewer steps
            if params.get("num_inference_steps", 0) > 1:
                params["num_inference_steps"] = 1
                
        # Handling for standard Stable Diffusion models
        elif model_type == "stable_diffusion":
            # Use img2img pipeline if we have a sketch
            if hasattr(pipe, "img2img") and sketch_image is not None:
                # If this is a standard SD pipeline with img2img capability
                params["strength"] = 0.8  # Controls how much to respect the original image
                if hasattr(pipe, "img2img"):
                    pipe = pipe.img2img
        
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