import os
import json
import time
from dotenv import load_dotenv
import torch
from PIL import Image, ImageOps, ImageEnhance
import numpy as np
from diffusers import (
    StableDiffusionControlNetPipeline,
    ControlNetModel,
    DDIMScheduler,
    StableDiffusionXLAdapterPipeline,
    T2IAdapter,
    AutoencoderKL,
    FluxControlPipeline
)
from diffusers.utils import load_image
import torchvision.transforms as T
from app.core.config import settings
from app.services.progress_tracker import update_progress, create_progress_gif

load_dotenv()
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

def decode_latents_to_image(pipe, latents):
    """Decode latents to PIL Image for intermediate visualization"""
    try:
        # Skip intermediate image generation for FLUX models
        pipeline_type = type(pipe).__name__
        if pipeline_type == "FluxControlPipeline":
            print("Skipping intermediate image generation for FLUX model (incompatible latent format)")
            return None
        
        # Check if we have a VAE to decode with
        if hasattr(pipe, 'vae') and pipe.vae is not None:
            # Make sure latents are in the right format
            if len(latents.shape) == 4:
                # Take first image if batch
                latent = latents[0:1] if latents.shape[0] > 1 else latents
            else:
                latent = latents.unsqueeze(0)
            
            # Decode latents
            with torch.no_grad():
                if hasattr(pipe.vae, 'decode'):
                    image = pipe.vae.decode(latent / pipe.vae.config.scaling_factor, return_dict=False)[0]
                else:
                    # Fallback for different VAE types
                    image = pipe.vae.decode(latent).sample
                
                # Convert to PIL Image
                image = (image / 2 + 0.5).clamp(0, 1)
                image = image.cpu().permute(0, 2, 3, 1).float().numpy()
                image = (image * 255).round().astype("uint8")[0]
                return Image.fromarray(image)
    except Exception as e:
        print(f"Error decoding latents: {e}")
        return None

def get_pipeline_callback_params(pipe, generation_id: str, total_steps: int, image_index: int = 0, total_images: int = 1):
    """
    Get the appropriate callback parameters for different pipeline types
    Returns a tuple: (callback_params_dict, supports_callbacks)
    """
    pipeline_type = type(pipe).__name__
    
    def legacy_callback(step: int, timestep: int, latents):
        """Legacy callback function for older pipelines"""
        if total_images > 1:
            overall_step = (image_index * total_steps) + step + 1
            overall_total = total_steps * total_images
            stage = f"generating image {image_index + 1}/{total_images}"
        else:
            overall_step = step + 1
            overall_total = total_steps
            stage = "generating"
        
        # Decode intermediate image every few steps (skip for FLUX)
        intermediate_image = None
        if step > 0 and pipeline_type != "FluxControlPipeline":
            intermediate_image = decode_latents_to_image(pipe, latents)
        
        update_progress(generation_id, overall_step, overall_total, stage, intermediate_image=intermediate_image)
    
    def modern_callback_on_step_end(pipe_obj, step: int, timestep: int, callback_kwargs):
        """Modern callback function for newer pipelines"""
        if total_images > 1:
            overall_step = (image_index * total_steps) + step + 1
            overall_total = total_steps * total_images
            stage = f"generating image {image_index + 1}/{total_images}"
        else:
            overall_step = step + 1
            overall_total = total_steps
            stage = "generating"
        
        # Decode intermediate image every few steps (skip for FLUX)
        intermediate_image = None
        if step > 0 and pipeline_type != "FluxControlPipeline":
            if 'latents' in callback_kwargs:
                intermediate_image = decode_latents_to_image(pipe_obj, callback_kwargs['latents'])
        
        update_progress(generation_id, overall_step, overall_total, stage, intermediate_image=intermediate_image)
        return callback_kwargs
    
    # Pipeline-specific callback handling
    if pipeline_type == "StableDiffusionControlNetPipeline":
        # ControlNet supports both old and new callbacks, prefer new one
        return {
            "callback_on_step_end": modern_callback_on_step_end
        }, True
        
    elif pipeline_type == "StableDiffusionXLAdapterPipeline":
        # T2I Adapter SDXL uses legacy callback system
        return {
            "callback": legacy_callback,
            "callback_steps": 1
        }, True
        
    elif pipeline_type == "FluxControlPipeline":
        # Flux supports modern callbacks
        return {
            "callback_on_step_end": modern_callback_on_step_end
        }, True
        
    else:
        # Unknown pipeline type, try modern first, then legacy
        print(f"Unknown pipeline type: {pipeline_type}, attempting modern callback")
        return {
            "callback_on_step_end": modern_callback_on_step_end
        }, True

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
    
    # Set torch_dtype based on model requirements
    if model_id == "flux_canny":
        torch_dtype = torch.bfloat16 if device == "cuda" else torch.float32
    else:
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
            
        elif model_id == "flux_canny":
            pipeline = FluxControlPipeline.from_pretrained(
                huggingface_id,
                torch_dtype=torch_dtype,
                device_map="balanced"
            )
        
        if model_id != "flux_canny":
            pipeline = pipeline.to(device)
        
        if device == "cuda":
            try:
                if model_id != "flux_canny" and hasattr(pipeline, "enable_xformers_memory_efficient_attention"):
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
    
    original_image = Image.open(sketch_path)
    
    detect_resolution = preprocessing.get("detect_resolution", 768)
    
    print(f"Using simple sketch inverter for model {model_id}")
    processed_image = simple_sketch_inverter(
        original_image,
        target_resolution=detect_resolution
    )
    
    return processed_image

def save_results_and_metadata(sketch_hash: str, images: list, metadata: dict, generation_time: float):
    """Save generated images and update metadata"""
    result_files = []
    
    # Save generated images
    for i, image in enumerate(images):
        if len(images) == 1:
            result_filename = f"{sketch_hash}.png"
        else:
            result_filename = f"{sketch_hash}_{i+1}.png"
        
        result_path = os.path.join(settings.DATASET_RESULT_DIR, result_filename)
        image.save(result_path)
        result_files.append(result_filename)
        print(f"Saved result {i+1} to {result_path}")
    
    # Create progress GIF from intermediate images
    # gif_path = os.path.join(settings.DATASET_RESULT_DIR, f"gif/{sketch_hash}_progress.gif")
    # if create_progress_gif(sketch_hash, gif_path):
        # print(f"Created progress GIF: {gif_path}")
        # metadata["progress_gif"] = f"{sketch_hash}_progress.gif"
    
    # Update metadata with file info
    metadata["file_info"] = {
        "result_count": len(images),
        "generation_time": generation_time
    }
    
    # Save metadata
    metadata_path = os.path.join(settings.DATASET_METADATA_DIR, f"{sketch_hash}.json")
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"Saved metadata for {sketch_hash}: {len(images)} results, {generation_time:.2f}s")

def generate_controlnet_batch(pipe, sketch_image, prompt, negative_prompt, config, num_images, generation_id):
    """
    Generate multiple images using ControlNet with batch processing for efficiency
    """
    print(f"Generating {num_images} images with ControlNet batch processing")
    
    # Create batch inputs
    batch_prompt = [prompt] * num_images
    batch_negative_prompt = [negative_prompt] * num_images if negative_prompt else None
    batch_image = [sketch_image] * num_images
    
    # Generate different seeds for variation
    generators = []
    device = pipe.device
    for i in range(num_images):
        generator = torch.Generator(device=device)
        generator.manual_seed(torch.randint(0, 2**42, (1,)).item())
        generators.append(generator)
    
    total_steps = config.get("num_inference_steps", 20)
    callback_params, supports_callbacks = get_pipeline_callback_params(pipe, generation_id, total_steps, 0, 1)
    
    params = {
        "prompt": batch_prompt,
        "image": batch_image,
        "num_inference_steps": total_steps,
        "guidance_scale": config.get("guidance_scale", 7.5),
        "generator": generators
    }
    
    # Add callback parameters if supported
    if supports_callbacks:
        params.update(callback_params)
    
    if batch_negative_prompt:
        params["negative_prompt"] = batch_negative_prompt
    
    print(f"Starting batch generation with ControlNet...")
    try:
        output = pipe(**params)
    except TypeError as e:
        if "callback" in str(e):
            print(f"Callback not supported for this pipeline version, continuing without progress updates: {e}")
            # Retry without callback
            params_no_callback = {k: v for k, v in params.items() if not k.startswith('callback')}
            output = pipe(**params_no_callback)
        else:
            raise e
    
    if hasattr(output, "images"):
        return output.images
    else:
        return output if isinstance(output, list) else [output]

def generate_t2i_sequence(pipe, sketch_image, prompt, negative_prompt, config, num_images, generation_id):
    """
    Generate multiple images using T2I Adapter with optimized sequential generation
    """
    print(f"Generating {num_images} images with T2I sequential processing")
    
    images = []
    device = pipe.device
    total_steps_per_image = config.get("num_inference_steps", 40)
    
    for i in range(num_images):
        print(f"Generating image {i+1}/{num_images}")
        
        # Get pipeline-specific callback parameters
        callback_params, supports_callbacks = get_pipeline_callback_params(pipe, generation_id, total_steps_per_image, i, num_images)
        
        # Generate different seed for each image
        generator = torch.Generator(device=device)
        generator.manual_seed(torch.randint(0, 2**32, (1,)).item())
        
        params = {
            "prompt": prompt,
            "image": sketch_image,
            "num_inference_steps": total_steps_per_image,
            "guidance_scale": config.get("guidance_scale", 7.5),
            "generator": generator,
            "adapter_conditioning_scale": config.get("adapter_conditioning_scale", 0.9),
            "adapter_conditioning_factor": config.get("adapter_conditioning_factor", 0.9)
        }
        
        # Add callback parameters if supported
        if supports_callbacks:
            params.update(callback_params)
        
        if negative_prompt:
            params["negative_prompt"] = negative_prompt
        
        try:
            output = pipe(**params)
        except TypeError as e:
            if "callback" in str(e):
                print(f"Callback not supported for this pipeline version, continuing without progress updates: {e}")
                # Retry without callback
                params_no_callback = {k: v for k, v in params.items() if not k.startswith('callback')}
                output = pipe(**params_no_callback)
                # Manual progress update since callback failed
                overall_step = (i + 1) * total_steps_per_image
                overall_total = total_steps_per_image * num_images
                update_progress(generation_id, overall_step, overall_total, f"completed image {i+1}/{num_images}")
            else:
                raise e
        
        if hasattr(output, "images") and len(output.images) > 0:
            images.append(output.images[0])
        else:
            images.append(output[0] if isinstance(output, (list, tuple)) else output)
    
    return images

def generate_image_from_sketch(
    sketch_path: str,
    prompt: str, 
    model_id: str,
    negative_prompt: str,
    sketch_hash: str,
    metadata: dict
):
    """Generate multiple images from a sketch using the selected model"""
    try:
        start_time = time.time()
        
        # Update progress: loading model
        model_info = settings.AVAILABLE_MODELS[model_id]
        total_steps = model_info.get("config", {}).get("num_inference_steps", 20)
        update_progress(sketch_hash, 0, total_steps, "loading model")
        
        pipe = load_model_pipeline(model_id)
        
        # Update progress: preprocessing
        update_progress(sketch_hash, 0, total_steps, "preprocessing sketch")
        sketch_image = preprocess_sketch(sketch_path, model_id)
        
        config = model_info.get("config", {})
        
        # Determine number of images to generate (3 for controlnet/t2i, 1 for flux)
        num_images = 1 if model_id == "flux_canny" else 3
        
        print(f"Generating {num_images} images with model {model_id}")
        
        # Update progress: starting generation
        if num_images > 1:
            total_generation_steps = total_steps * num_images
        else:
            total_generation_steps = total_steps
            
        update_progress(sketch_hash, 0, total_generation_steps, "starting generation")
        
        if model_id == "controlnet_scribble":
            # Use batch generation for ControlNet
            images = generate_controlnet_batch(
                pipe, sketch_image, prompt, negative_prompt, config, num_images, sketch_hash
            )
        elif model_id == "t2i_adapter_sdxl":
            # Use optimized sequential generation for T2I
            images = generate_t2i_sequence(
                pipe, sketch_image, prompt, negative_prompt, config, num_images, sketch_hash
            )
        elif model_id == "flux_canny":
            # Single image generation for Flux
            callback_params, supports_callbacks = get_pipeline_callback_params(pipe, sketch_hash, total_steps)
            
            params = {
                "prompt": prompt,
                "control_image": sketch_image,
                "height": config.get("output_height", 1024),
                "width": config.get("output_width", 1024),
                "num_inference_steps": config.get("num_inference_steps", 50),
                "guidance_scale": config.get("guidance_scale", 30.0)
            }
            
            # Add callback parameters if supported
            if supports_callbacks:
                params.update(callback_params)
            
            if negative_prompt:
                params["negative_prompt"] = negative_prompt
            
            try:
                output = pipe(**params)
            except TypeError as e:
                if "callback" in str(e):
                    print(f"Callback not supported for this pipeline version, continuing without progress updates: {e}")
                    # Retry without callback
                    params_no_callback = {k: v for k, v in params.items() if not k.startswith('callback')}
                    output = pipe(**params_no_callback)
                    # Manual progress update since callback failed
                    update_progress(sketch_hash, total_steps, total_steps, "completed generation")
                else:
                    raise e
            
            if hasattr(output, "images") and len(output.images) > 0:
                images = [output.images[0]]
            else:
                images = [output[0] if isinstance(output, (list, tuple)) else output]
        
        generation_time = time.time() - start_time
        
        # Update progress: saving results
        final_step = total_generation_steps if num_images > 1 else total_steps
        update_progress(sketch_hash, final_step, final_step, "saving results")
        
        # Save results and metadata
        save_results_and_metadata(sketch_hash, images, metadata, generation_time)
        
        print(f"Generation completed for {sketch_hash} in {generation_time:.2f}s")
        
    except Exception as e:
        print(f"Error generating images with model {model_id}: {str(e)}")
        # Update progress with error
        update_progress(sketch_hash, 0, 1, "error")
        raise e