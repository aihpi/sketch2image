#!/usr/bin/env python3
"""
Batch Testing Script for Sketch-to-Image Generation

This script loads sketches from a specified folder and generates images using all available
models and styles with a consistent prompt. It documents generation times and saves results
in a structured format for easy comparison.

Usage:
    python batch_test.py --sketches-dir ./test_sketches --output-dir ./test_results --prompt "a detailed scene with mountains and trees" --seed 42
"""

import os
import sys
import time
import argparse
import json
import shutil
from datetime import datetime
from typing import List, Dict, Any
import csv
from PIL import Image
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from app.core.config import settings
from app.services.generation import preprocess_sketch, load_model_pipeline
from app.api.routes import AVAILABLE_STYLES

def parse_args():
    parser = argparse.ArgumentParser(description="Batch test sketch-to-image generation")
    parser.add_argument(
        "--sketches-dir", 
        type=str, 
        required=True,
        help="Directory containing sketch files to test"
    )
    parser.add_argument(
        "--output-dir", 
        type=str, 
        default="batch_test_results",
        help="Directory to save the generated images and results"
    )
    parser.add_argument(
        "--prompt-file", 
        type=str, 
        help="JSON file mapping sketch filenames to prompts"
    )
    parser.add_argument(
        "--prompt", 
        type=str, 
        default="a detailed scene with high quality",
        help="Default prompt to use for sketches without a specific prompt in the prompt file"
    )
    parser.add_argument(
        "--negative-prompt", 
        type=str, 
        default="low quality, bad anatomy, worst quality, low resolution",
        help="Negative prompt to use for all generations"
    )
    parser.add_argument(
        "--device",
        type=str,
        choices=["cuda", "cpu"],
        default=None,
        help="Device to use for generation (default: use settings.DEVICE)"
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip generation if the output file already exists"
    )
    parser.add_argument(
        "--create-comparison-grid",
        action="store_true",
        help="Create a grid image comparing all results for each sketch"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for generation (default: 42)"
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        help="Specific models to test (e.g., --models controlnet_scribble t2i_adapter_sdxl flux_canny). If not specified, all models are tested."
    )
    return parser.parse_args()

def setup_test_directories(base_dir: str) -> Dict[str, str]:
    """Create directory structure for test results"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(base_dir, f"batch_test_{timestamp}")
    
    os.makedirs(output_dir, exist_ok=True)
    
    model_dirs = {}
    for model_id in settings.AVAILABLE_MODELS:
        model_dir = os.path.join(output_dir, model_id)
        os.makedirs(model_dir, exist_ok=True)
        model_dirs[model_id] = model_dir
    
    comparison_dir = os.path.join(output_dir, "comparisons")
    os.makedirs(comparison_dir, exist_ok=True)
    
    return {
        "base": output_dir,
        "models": model_dirs,
        "comparisons": comparison_dir
    }

def get_sketch_files(sketches_dir: str) -> List[str]:
    """Get all image files in the sketches directory"""
    allowed_extensions = ['.png', '.jpg', '.jpeg', '.bmp']
    
    sketch_files = []
    for file in os.listdir(sketches_dir):
        file_path = os.path.join(sketches_dir, file)
        if os.path.isfile(file_path):
            _, ext = os.path.splitext(file)
            if ext.lower() in allowed_extensions:
                sketch_files.append(file_path)
    
    if not sketch_files:
        raise ValueError(f"No valid image files found in {sketches_dir}")
    
    return sorted(sketch_files)

def create_results_csv(output_dir: str, headers: List[str]) -> str:
    """Create a CSV file to store test results"""
    csv_path = os.path.join(output_dir, "generation_times.csv")
    with open(csv_path, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(headers)
    return csv_path

def append_to_csv(csv_path: str, row: List[Any]) -> None:
    """Append a row to the CSV file"""
    with open(csv_path, 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(row)

def create_comparison_grid(sketch_path: str, images_dict: Dict[str, Dict[str, str]], output_path: str) -> None:
    """Create a grid image comparing results across models and styles"""
    original_sketch = Image.open(sketch_path)
    
    num_models = len(images_dict)
    
    if num_models == 0:
        print(f"Warning: No model results found for {sketch_path}. Cannot create comparison grid.")
        return
        
    first_model = next(iter(images_dict.values()))
    styles = list(first_model.keys())
    num_styles = len(styles)
    
    if num_styles == 0:
        print(f"Warning: No style results found for {sketch_path}. Cannot create comparison grid.")
        return
    
    fig, axes = plt.subplots(1 + num_models, 1 + num_styles, figsize=(4 * (1 + num_styles), 4 * (1 + num_models)))
    
    if num_models == 0 or num_styles == 0:
        print(f"Warning: Not enough data to create a comparison grid for {sketch_path}")
        return
    
    if num_models == 1 and num_styles == 1:
        axes = np.array([[axes]])
    elif num_models == 1:
        axes = axes.reshape(1, -1)
    elif num_styles == 1:
        axes = axes.reshape(-1, 1)
    
    for ax in axes.flat:
        ax.axis('off')
    
    try:
        axes[0, 0].imshow(original_sketch)
        axes[0, 0].set_title("Original Sketch", fontsize=12)
        
        for j, style in enumerate(styles):
            axes[0, j + 1].text(0.5, 0.5, style, fontsize=14, ha='center', va='center', wrap=True)
        
        for i, model_id in enumerate(images_dict.keys()):
            model_name = settings.AVAILABLE_MODELS[model_id]['name']
            axes[i + 1, 0].text(0.5, 0.5, model_name, fontsize=14, ha='center', va='center', wrap=True)
            
            for j, style in enumerate(styles):
                image_path = images_dict[model_id].get(style)
                if image_path and os.path.exists(image_path):
                    img = Image.open(image_path)
                    axes[i + 1, j + 1].imshow(img)
                else:
                    axes[i + 1, j + 1].text(0.5, 0.5, "No Image", fontsize=12, ha='center', va='center', color='red')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        
        print(f"Saved comparison grid to {output_path}")
    except Exception as e:
        print(f"Error creating comparison grid: {e}")
        plt.close(fig)

def generate_and_save_image(
    sketch_path: str,
    output_path: str,
    model_id: str,
    style: Dict[str, Any],
    prompt: str,
    negative_prompt: str,
    device: str,
    seed: int = 42
) -> tuple:
    """
    Generate an image from a sketch and save it, returning generation time and model config
    """
    start_loading = time.time()
    pipe = load_model_pipeline(model_id)
    loading_time = time.time() - start_loading
    
    sketch_image = preprocess_sketch(sketch_path, model_id)
    
    full_prompt = f"{prompt}, {style.name}, best quality, extremely detailed"
    
    generator = torch.Generator(device=device).manual_seed(seed)
    
    # Get model-specific configuration from config.py
    model_info = settings.AVAILABLE_MODELS[model_id]
    config = model_info.get("config", {})
    
    # Use model-specific settings from config.py
    num_inference_steps = config.get("num_inference_steps", 20)
    guidance_scale = config.get("guidance_scale", 7.5)
    
    params = {
        "prompt": full_prompt,
        "image": sketch_image,
        "num_inference_steps": num_inference_steps,
        "guidance_scale": guidance_scale,
        "generator": generator
    }

    # Handle model-specific parameters
    if model_id == "t2i_adapter_sdxl":
        params["adapter_conditioning_scale"] = config.get("adapter_conditioning_scale", 0.9)
        params["adapter_conditioning_factor"] = config.get("adapter_conditioning_factor", 0.9)
        params["negative_prompt"] = negative_prompt
    
    elif model_id == "flux_canny":
        params["control_image"] = params.pop("image")
        params["height"] = config.get("output_height", 1024)
        params["width"] = config.get("output_width", 1024)
        # Flux doesn't use negative prompts the same way
    else:
        # For controlnet_scribble and other models that support negative prompts
        params["negative_prompt"] = negative_prompt

    start_time = time.time()
    output = pipe(**params)
    generation_time = time.time() - start_time
    
    if hasattr(output, "images") and len(output.images) > 0:
        output_image = output.images[0]
    else:
        output_image = output[0] if isinstance(output, (list, tuple)) else output
    
    output_image.save(output_path)
    
    return generation_time, loading_time, num_inference_steps, guidance_scale

def create_summary_plots(csv_path: str, output_dir: str) -> None:
    """Create summary plots based on generation time data"""
    pass  # Removed as requested

def save_test_config(output_dir: str, args: argparse.Namespace) -> None:
    """Save the test configuration for reproducibility"""
    config = {
        "timestamp": datetime.now().isoformat(),
        "args": vars(args),
        "cuda_available": torch.cuda.is_available(),
        "device_used": args.device if args.device else settings.DEVICE,
        "available_models": {k: v["name"] for k, v in settings.AVAILABLE_MODELS.items()},
        "models_tested": args.models if args.models else list(settings.AVAILABLE_MODELS.keys()),
        "available_styles": [s.name for s in AVAILABLE_STYLES],
        "base_seed": args.seed,
        "seed_strategy": "different_seed_per_sketch"
    }
    
    if args.prompt_file and os.path.exists(args.prompt_file):
        try:
            with open(args.prompt_file, 'r') as f:
                prompt_mapping = json.load(f)
            config["prompt_mapping"] = prompt_mapping
        except:
            config["prompt_mapping_error"] = f"Failed to load prompt file: {args.prompt_file}"
    
    if torch.cuda.is_available():
        config["cuda_device_name"] = torch.cuda.get_device_name(0)
        config["cuda_device_count"] = torch.cuda.device_count()
        config["cuda_memory_allocated"] = torch.cuda.memory_allocated(0)
        config["cuda_memory_reserved"] = torch.cuda.memory_reserved(0)
    
    config_path = os.path.join(output_dir, "test_config.json")
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)

def main():
    args = parse_args()
    
    if args.device:
        settings.DEVICE = args.device
    
    # Determine which models to test
    models_to_test = args.models if args.models else list(settings.AVAILABLE_MODELS.keys())
    
    # Validate models
    invalid_models = [m for m in models_to_test if m not in settings.AVAILABLE_MODELS]
    if invalid_models:
        print(f"Error: Invalid models specified: {invalid_models}")
        print(f"Available models: {list(settings.AVAILABLE_MODELS.keys())}")
        exit(1)
    
    print(f"Testing models: {models_to_test}")
    
    dirs = setup_test_directories(args.output_dir)
    
    sketch_files = get_sketch_files(args.sketches_dir)
    print(f"Found {len(sketch_files)} sketch files to process")
    
    prompt_mapping = {}
    if args.prompt_file:
        try:
            with open(args.prompt_file, 'r') as f:
                prompt_mapping = json.load(f)
            print(f"Loaded {len(prompt_mapping)} prompts from {args.prompt_file}")
        except Exception as e:
            print(f"Error loading prompt file: {str(e)}")
            print(f"Using default prompt for all sketches: '{args.prompt}'")
    else:
        print(f"No prompt file provided. Using default prompt for all sketches: '{args.prompt}'")
    
    csv_headers = ["Sketch", "Prompt", "Model ID", "Model Name", "Style ID", "Style Name", "Seed", "Inference Steps", "Guidance Scale", "Generation Time (s)", "Loading Time (s)"]
    csv_path = create_results_csv(dirs["base"], csv_headers)
    
    save_test_config(dirs["base"], args)
    
    print(f"Using base seed: {args.seed} (each sketch will have a unique derived seed)")
    
    sketch_results = {}
    sketch_seeds = {}
    
    for sketch_path in tqdm(sketch_files, desc="Processing sketches"):
        sketch_name = os.path.basename(sketch_path)
        
        sketch_seed = args.seed + abs(hash(sketch_name)) % 10000
        sketch_seeds[sketch_name] = sketch_seed
        
        sketch_prompt = prompt_mapping.get(sketch_name, args.prompt)
        print(f"\nProcessing sketch: {sketch_name}")
        print(f"Using prompt: '{sketch_prompt}'")
        print(f"Using seed: {sketch_seed} (derived from base seed {args.seed})")
        
        sketch_results[sketch_path] = {}
        
        for model_id in models_to_test:  # Changed from settings.AVAILABLE_MODELS to models_to_test
            model_dir = dirs["models"][model_id]
            sketch_results[sketch_path][model_id] = {}
            
            for style in AVAILABLE_STYLES:
                style_id = style.id
                style_name = style.name
                
                print(f"  Generating with model={model_id}, style={style_id}")
                
                output_filename = f"{os.path.splitext(sketch_name)[0]}_{style_id}.png"
                output_path = os.path.join(model_dir, output_filename)
                
                if args.skip_existing and os.path.exists(output_path):
                    print(f"    Skipping existing file: {output_path}")
                    sketch_results[sketch_path][model_id][style_id] = output_path
                    continue
                
                try:
                    generation_time, loading_time, inference_steps, guidance_scale = generate_and_save_image(
                        sketch_path=sketch_path,
                        output_path=output_path,
                        model_id=model_id,
                        style=style,
                        prompt=sketch_prompt,
                        negative_prompt=args.negative_prompt,
                        device=settings.DEVICE,
                        seed=sketch_seed
                    )
                    
                    print(f"    Generated in {generation_time:.2f} seconds")
                    
                    sketch_results[sketch_path][model_id][style_id] = output_path
                    
                    model_name = settings.AVAILABLE_MODELS[model_id]['name']
                    
                    append_to_csv(csv_path, [
                        sketch_name,
                        sketch_prompt,
                        model_id,
                        model_name,
                        style_id,
                        style_name,
                        sketch_seed,
                        inference_steps,
                        guidance_scale,
                        f"{generation_time:.4f}",
                        f"{loading_time:.4f}"
                    ])
                    
                except Exception as e:
                    print(f"    Error generating image: {str(e)}")
                    model_name = settings.AVAILABLE_MODELS[model_id]['name']
                    append_to_csv(csv_path, [
                        sketch_name,
                        sketch_prompt, 
                        model_id,
                        model_name,
                        style_id,
                        style_name,
                        sketch_seed,
                        "ERROR",
                        "ERROR",
                        "ERROR",
                        "ERROR"
                    ])
        
        if args.create_comparison_grid:
            grid_filename = f"{os.path.splitext(sketch_name)[0]}_comparison_grid.png"
            grid_path = os.path.join(dirs["comparisons"], grid_filename)
            try:
                create_comparison_grid(sketch_path, sketch_results[sketch_path], grid_path)
            except Exception as e:
                print(f"Error creating comparison grid for {sketch_name}: {str(e)}")
    
    seeds_path = os.path.join(dirs["base"], "sketch_seeds.json")
    with open(seeds_path, 'w') as f:
        json.dump(sketch_seeds, f, indent=2)
    print(f"Saved sketch seed mapping to {seeds_path}")
    
    print(f"\nAll done! Results saved to {dirs['base']}")
    print(f"Generation times recorded in {csv_path}")
    print(f"Models tested: {models_to_test}")

if __name__ == "__main__":
    main()