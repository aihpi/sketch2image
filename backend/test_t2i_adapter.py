# Simplified T2I Adapter Testing Script (Fixed for Better Sketch Adherence)

# Install required packages (uncomment to run)
# !pip install diffusers==0.23.1 transformers accelerate torch controlnet-aux==0.0.7 safetensors

import os
import io
import time
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image, ImageOps
import requests
from tqdm.auto import tqdm
import torchvision.transforms as T
from torchvision.transforms import ToTensor

# Free up CUDA memory
torch.cuda.empty_cache()

# Set up GPU with memory optimization
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Create directory for saving outputs
os.makedirs("model_outputs", exist_ok=True)

def free_memory():
    if device == "cuda":
        torch.cuda.empty_cache()
        print(f"Freed GPU memory. Available: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

def download_image(url):
    response = requests.get(url, stream=True)
    return Image.open(io.BytesIO(response.content))

def load_image(path_or_url):
    return download_image(path_or_url) if path_or_url.startswith('http') else Image.open(path_or_url)

def normalize_sketch(sketch, invert=True):
    """Normalize a sketch with optional inversion for different models
    Some models (like T2I Adapter) work better with white lines on black background
    Others (like ControlNet Scribble) work better with black lines on white background
    """
    sketch = sketch.resize((1024, 1024), Image.BILINEAR).convert("L")
    tensor = T.ToTensor()(sketch)
    sketch = T.ToPILImage()(tensor)
    
    if invert:
        sketch = ImageOps.invert(sketch)
        
    return sketch

def preprocess_with_pidinet(image, detect_resolution=1024, image_resolution=1024, apply_filter=True, invert=True):
    """Process an image with PidiNet for edge detection, with optional inversion
    
    Args:
        image: PIL image or path to image
        detect_resolution: Resolution for edge detection
        image_resolution: Output resolution
        apply_filter: Whether to apply filtering to the edges
        invert: Whether to invert the colors (for T2I models that expect white lines on black)
    """
    try:
        from controlnet_aux.pidi import PidiNetDetector
        processor = PidiNetDetector.from_pretrained("lllyasviel/Annotators")
        if device == "cuda":
            processor = processor.to(device)
        sketch = processor(
            image, 
            detect_resolution=detect_resolution, 
            image_resolution=image_resolution,
            apply_filter=apply_filter
        ).convert("L")
        
        sketch = sketch.resize((1024, 1024), Image.BILINEAR)
        
        # Invert if needed for model compatibility
        if invert:
            sketch = ImageOps.invert(sketch)
            
        return sketch
    except Exception as e:
        print(f"Error with PidiNet: {e}")
        sketch = image.convert("L").resize((1024, 1024), Image.BILINEAR)
        if invert:
            sketch = ImageOps.invert(sketch)
        return sketch

def test_t2i_adapter(image_path, prompt, 
                     num_inference_steps=40,
                     guidance_scale=7.5,
                     adapter_conditioning_scale=0.9,
                     adapter_conditioning_factor=0.9,
                     save_path="model_outputs",
                     use_pidinet=True,
                     invert_sketch=True):

    print(f"Loading image from: {image_path}")
    original_image = load_image(image_path)

    if use_pidinet:
        print("Converting to sketch with PidiNet...")
        try:
            sketch = preprocess_with_pidinet(original_image, invert=invert_sketch)
            sketch_path = os.path.join(save_path, "input_sketch_pidinet.png")
            sketch.save(sketch_path)
            print(f"Saved sketch to {sketch_path}")
        except Exception as e:
            print(f"PidiNet failed: {e}\nFalling back to grayscale conversion")
            sketch = normalize_sketch(original_image.convert("L"), invert=invert_sketch)
            sketch_path = os.path.join(save_path, "input_sketch_grayscale.png")
            sketch.save(sketch_path)
    else:
        print("Using standard grayscale conversion...")
        sketch = normalize_sketch(original_image.convert("L"), invert=invert_sketch)
        sketch_path = os.path.join(save_path, "input_sketch_standard.png")
        sketch.save(sketch_path)

    free_memory()

    print("Loading T2I Adapter model...")
    try:
        from diffusers import (
            T2IAdapter,
            StableDiffusionXLAdapterPipeline,
            AutoencoderKL,
            EulerAncestralDiscreteScheduler
        )

        adapter = T2IAdapter.from_pretrained(
            "Adapter/t2iadapter", 
            subfolder="sketch_sdxl_1.0",
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            adapter_type="full_adapter_xl"
        )

        vae = AutoencoderKL.from_pretrained(
            "madebyollin/sdxl-vae-fp16-fix",
            torch_dtype=torch.float16 if device == "cuda" else torch.float32
        )

        model_id = "stabilityai/stable-diffusion-xl-base-1.0"
        euler_a = EulerAncestralDiscreteScheduler.from_pretrained(model_id, subfolder="scheduler")

        pipeline = StableDiffusionXLAdapterPipeline.from_pretrained(
            model_id,
            adapter=adapter,
            vae=vae,
            scheduler=euler_a,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            variant="fp16" if device == "cuda" else None
        ).to(device)

        if device == "cuda":
            try:
                pipeline.enable_xformers_memory_efficient_attention()
            except:
                pipeline.enable_attention_slicing()

        negative_prompt = "disfigured, extra digit, fewer digits, cropped, worst quality, low quality"

        print("Generating image...")
        generator = torch.manual_seed(42)
        start_time = time.time()

        sketch_tensor = ToTensor()(sketch).unsqueeze(0).to(device)

        result = pipeline(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=sketch_tensor,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            adapter_conditioning_scale=adapter_conditioning_scale,
            adapter_conditioning_factor=adapter_conditioning_factor,
            generator=generator
        ).images[0]

        generation_time = time.time() - start_time
        print(f"Generation time: {generation_time:.2f} seconds")

        timestamp = int(time.time())
        output_path = os.path.join(save_path, f"output_{timestamp}.png")
        result.save(output_path)
        print(f"Saved result to {output_path}")

        plt.figure(figsize=(20, 6))
        plt.subplot(1, 3, 1)
        plt.imshow(original_image)
        plt.title("Original Image")
        plt.axis('off')

        plt.subplot(1, 3, 2)
        plt.imshow(sketch, cmap='gray')
        plt.title("Processed Sketch")
        plt.axis('off')

        plt.subplot(1, 3, 3)
        plt.imshow(result)
        plt.title(f"Generated - {generation_time:.2f}s")
        plt.axis('off')

        plt.tight_layout()
        plt.savefig(os.path.join(save_path, f"comparison_{timestamp}.png"))
        plt.show()

        return result, sketch

    except Exception as e:
        print(f"Error with T2I Adapter: {e}")
        import traceback
        traceback.print_exc()
        return None, None

def test_controlnet_scribble(image_path, prompt, 
                         num_inference_steps=30,
                         guidance_scale=7.5,
                         save_path="model_outputs",
                         use_pidinet=True,
                         invert_sketch=False,  # ControlNet scribble expects black lines on white
                         use_sdxl=False):  # Use SDXL or SD 1.5

    print(f"Loading image from: {image_path}")
    original_image = load_image(image_path)

    if use_pidinet:
        print("Converting to sketch with PidiNet...")
        try:
            sketch = preprocess_with_pidinet(original_image, invert=invert_sketch)
            sketch_path = os.path.join(save_path, "input_sketch_pidinet.png")
            sketch.save(sketch_path)
            print(f"Saved sketch to {sketch_path}")
        except Exception as e:
            print(f"PidiNet failed: {e}\nFalling back to grayscale conversion")
            sketch = normalize_sketch(original_image.convert("L"), invert=invert_sketch)
            sketch_path = os.path.join(save_path, "input_sketch_grayscale.png")
            sketch.save(sketch_path)
    else:
        print("Using standard grayscale conversion...")
        sketch = normalize_sketch(original_image.convert("L"), invert=invert_sketch)
        sketch_path = os.path.join(save_path, "input_sketch_standard.png")
        sketch.save(sketch_path)

    free_memory()

    print("Loading ControlNet Scribble model...")
    try:
        if use_sdxl:
            from diffusers import StableDiffusionXLControlNetPipeline, ControlNetModel
            
            controlnet = ControlNetModel.from_pretrained(
                "xinsir/controlnet-scribble-sdxl-1.0",
                torch_dtype=torch.float16 if device == "cuda" else torch.float32
            )
            
            pipeline = StableDiffusionXLControlNetPipeline.from_pretrained(
                "stabilityai/stable-diffusion-xl-base-1.0",
                controlnet=controlnet,
                torch_dtype=torch.float16 if device == "cuda" else torch.float32
            ).to(device)
        else:
            from diffusers import StableDiffusionControlNetPipeline, ControlNetModel
            
            controlnet = ControlNetModel.from_pretrained(
                "lllyasviel/control_v11p_sd15_scribble",
                torch_dtype=torch.float16 if device == "cuda" else torch.float32
            )
            
            pipeline = StableDiffusionControlNetPipeline.from_pretrained(
                "runwayml/stable-diffusion-v1-5",
                controlnet=controlnet,
                torch_dtype=torch.float16 if device == "cuda" else torch.float32
            ).to(device)

        if device == "cuda":
            try:
                pipeline.enable_xformers_memory_efficient_attention()
            except:
                pipeline.enable_attention_slicing()

        negative_prompt = "disfigured, extra digit, fewer digits, cropped, worst quality, low quality"

        print("Generating image...")
        generator = torch.manual_seed(42)
        start_time = time.time()

        # For ControlNet, we directly use the image not a tensor
        result = pipeline(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=sketch,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            generator=generator
        ).images[0]

        generation_time = time.time() - start_time
        print(f"Generation time: {generation_time:.2f} seconds")

        timestamp = int(time.time())
        output_path = os.path.join(save_path, f"output_{timestamp}.png")
        result.save(output_path)
        print(f"Saved result to {output_path}")

        plt.figure(figsize=(20, 6))
        plt.subplot(1, 3, 1)
        plt.imshow(original_image)
        plt.title("Original Image")
        plt.axis('off')

        plt.subplot(1, 3, 2)
        plt.imshow(sketch, cmap='gray')
        plt.title("Processed Sketch")
        plt.axis('off')

        plt.subplot(1, 3, 3)
        plt.imshow(result)
        plt.title(f"Generated - {generation_time:.2f}s")
        plt.axis('off')

        plt.tight_layout()
        plt.savefig(os.path.join(save_path, f"comparison_{timestamp}.png"))
        plt.show()

        return result, sketch

    except Exception as e:
        print(f"Error with ControlNet: {e}")
        import traceback
        traceback.print_exc()
        return None, None

def test_controlnet_softedge(image_path, prompt, 
                         num_inference_steps=30,
                         guidance_scale=7.5,
                         save_path="model_outputs",
                         use_pidinet=True):

    print(f"Loading image from: {image_path}")
    original_image = load_image(image_path)

    # For softedge, we always want PidiNet but without inversion
    # (black lines on white background)
    if use_pidinet:
        print("Converting to sketch with PidiNet...")
        try:
            sketch = preprocess_with_pidinet(original_image, invert=False)
            sketch_path = os.path.join(save_path, "input_sketch_pidinet.png")
            sketch.save(sketch_path)
            print(f"Saved sketch to {sketch_path}")
        except Exception as e:
            print(f"PidiNet failed: {e}\nFalling back to grayscale conversion")
            sketch = normalize_sketch(original_image.convert("L"), invert=False)
            sketch_path = os.path.join(save_path, "input_sketch_grayscale.png")
            sketch.save(sketch_path)
    else:
        print("Using standard grayscale conversion...")
        sketch = normalize_sketch(original_image.convert("L"), invert=False)
        sketch_path = os.path.join(save_path, "input_sketch_standard.png")
        sketch.save(sketch_path)

    free_memory()

    print("Loading ControlNet SoftEdge model...")
    try:
        from diffusers import StableDiffusionControlNetPipeline, ControlNetModel
        
        controlnet = ControlNetModel.from_pretrained(
            "lllyasviel/control_v11p_sd15_softedge",
            torch_dtype=torch.float16 if device == "cuda" else torch.float32
        )
        
        pipeline = StableDiffusionControlNetPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            controlnet=controlnet,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32
        ).to(device)

        if device == "cuda":
            try:
                pipeline.enable_xformers_memory_efficient_attention()
            except:
                pipeline.enable_attention_slicing()

        negative_prompt = "disfigured, extra digit, fewer digits, cropped, worst quality, low quality"

        print("Generating image...")
        generator = torch.manual_seed(42)
        start_time = time.time()

        # For ControlNet, we directly use the image not a tensor
        result = pipeline(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=sketch,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            generator=generator
        ).images[0]

        generation_time = time.time() - start_time
        print(f"Generation time: {generation_time:.2f} seconds")

        timestamp = int(time.time())
        output_path = os.path.join(save_path, f"output_{timestamp}.png")
        result.save(output_path)
        print(f"Saved result to {output_path}")

        plt.figure(figsize=(20, 6))
        plt.subplot(1, 3, 1)
        plt.imshow(original_image)
        plt.title("Original Image")
        plt.axis('off')

        plt.subplot(1, 3, 2)
        plt.imshow(sketch, cmap='gray')
        plt.title("Processed Sketch")
        plt.axis('off')

        plt.subplot(1, 3, 3)
        plt.imshow(result)
        plt.title(f"Generated - {generation_time:.2f}s")
        plt.axis('off')

        plt.tight_layout()
        plt.savefig(os.path.join(save_path, f"comparison_{timestamp}.png"))
        plt.show()

        return result, sketch

    except Exception as e:
        print(f"Error with ControlNet SoftEdge: {e}")
        import traceback
        traceback.print_exc()
        return None, None

# Function to run all models for comparison
def compare_models(image_path, prompt, save_path="model_outputs"):
    # Create comparison directory
    comparison_dir = os.path.join(save_path, "comparison")
    os.makedirs(comparison_dir, exist_ok=True)
    
    # Test each model
    print("=== Testing T2I Adapter SDXL ===")
    t2i_result, t2i_sketch = test_t2i_adapter(
        image_path, 
        prompt,
        save_path=comparison_dir,
        use_pidinet=True,
        invert_sketch=True  # T2I adapter works best with white on black
    )
    
    print("\n=== Testing ControlNet Scribble SD 1.5 ===")
    cn_scribble_result, cn_scribble_sketch = test_controlnet_scribble(
        image_path,
        prompt,
        save_path=comparison_dir,
        use_pidinet=False,
        invert_sketch=False,  # ControlNet scribble works best with black on white
        use_sdxl=False
    )
    
    print("\n=== Testing ControlNet SoftEdge SD 1.5 ===")
    cn_softedge_result, cn_softedge_sketch = test_controlnet_softedge(
        image_path,
        prompt,
        save_path=comparison_dir,
        use_pidinet=True
    )
    
    print("\n=== Testing ControlNet Scribble SDXL ===")
    cn_sdxl_result, cn_sdxl_sketch = test_controlnet_scribble(
        image_path,
        prompt,
        save_path=comparison_dir,
        use_pidinet=False,
        invert_sketch=False,
        use_sdxl=True
    )
    
    # Compare all results
    try:
        plt.figure(figsize=(20, 15))
        
        # Original image
        plt.subplot(3, 3, 1)
        original_image = load_image(image_path)
        plt.imshow(original_image)
        plt.title("Original Image")
        plt.axis('off')
        
        # T2I Adapter
        plt.subplot(3, 3, 4)
        plt.imshow(t2i_sketch, cmap='gray')
        plt.title("T2I Adapter Sketch (Inverted)")
        plt.axis('off')
        
        plt.subplot(3, 3, 7)
        if t2i_result is not None:
            plt.imshow(t2i_result)
        else:
            plt.text(0.5, 0.5, "Generation Failed", ha='center', va='center')
        plt.title("T2I Adapter SDXL Result")
        plt.axis('off')
        
        # ControlNet Scribble
        plt.subplot(3, 3, 5)
        plt.imshow(cn_scribble_sketch, cmap='gray')
        plt.title("ControlNet Scribble Sketch")
        plt.axis('off')
        
        plt.subplot(3, 3, 8)
        if cn_scribble_result is not None:
            plt.imshow(cn_scribble_result)
        else:
            plt.text(0.5, 0.5, "Generation Failed", ha='center', va='center')
        plt.title("ControlNet Scribble SD 1.5 Result")
        plt.axis('off')
        
        # ControlNet SoftEdge
        plt.subplot(3, 3, 6)
        plt.imshow(cn_softedge_sketch, cmap='gray')
        plt.title("ControlNet SoftEdge Sketch")
        plt.axis('off')
        
        plt.subplot(3, 3, 9)
        if cn_softedge_result is not None:
            plt.imshow(cn_softedge_result)
        else:
            plt.text(0.5, 0.5, "Generation Failed", ha='center', va='center')
        plt.title("ControlNet SoftEdge SD 1.5 Result")
        plt.axis('off')
        
        # SDXL ControlNet Scribble
        plt.subplot(3, 3, 2)
        plt.imshow(cn_sdxl_sketch, cmap='gray')
        plt.title("SDXL ControlNet Sketch")
        plt.axis('off')
        
        plt.subplot(3, 3, 3)
        if cn_sdxl_result is not None:
            plt.imshow(cn_sdxl_result)
        else:
            plt.text(0.5, 0.5, "Generation Failed", ha='center', va='center')
        plt.title("ControlNet Scribble SDXL Result")
        plt.axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(comparison_dir, f"model_comparison.png"))
        plt.show()
        
    except Exception as e:
        print(f"Error during comparison: {e}")

# Example usage
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test and compare sketch-to-image models")
    parser.add_argument("--image", type=str, required=True, help="Path to input image")
    parser.add_argument("--prompt", type=str, default="a detailed high-quality digital illustration", 
                       help="Prompt for image generation")
    parser.add_argument("--model", type=str, choices=["t2i", "scribble", "softedge", "sdxl", "compare"], 
                       default="compare", help="Which model to test (or compare all)")
    args = parser.parse_args()
    
    if args.model == "t2i":
        test_t2i_adapter(args.image, args.prompt, use_pidinet=True, invert_sketch=True)
    elif args.model == "scribble":
        test_controlnet_scribble(args.image, args.prompt, use_pidinet=False, invert_sketch=False, use_sdxl=False)
    elif args.model == "softedge":
        test_controlnet_softedge(args.image, args.prompt, use_pidinet=True)
    elif args.model == "sdxl":
        test_controlnet_scribble(args.image, args.prompt, use_pidinet=False, invert_sketch=False, use_sdxl=True)
    else:
        compare_models(args.image, args.prompt)