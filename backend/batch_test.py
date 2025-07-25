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
        "--num-inference-steps", 
        type=int, 
        default=20,
        help="Number of inference steps for each generation"
    )
    parser.add_argument(
        "--guidance-scale", 
        type=float, 
        default=7.5,
        help="Guidance scale for each generation"
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
    
    preprocessed_dir = os.path.join(output_dir, "preprocessed")
    os.makedirs(preprocessed_dir, exist_ok=True)
    
    return {
        "base": output_dir,
        "models": model_dirs,
        "comparisons": comparison_dir,
        "preprocessed": preprocessed_dir
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
    num_inference_steps: int,
    guidance_scale: float,
    preprocessed_dir: str,
    device: str,
    seed: int = 42
) -> float:
    """
    Generate an image from a sketch and save it, returning generation time
    """
    start_loading = time.time()
    pipe = load_model_pipeline(model_id)
    loading_time = time.time() - start_loading
    
    sketch_filename = os.path.basename(sketch_path)
    preprocessed_path = os.path.join(preprocessed_dir, f"{model_id}_{style.id}_{sketch_filename}")
    
    sketch_image = preprocess_sketch(sketch_path, model_id)
    sketch_image.save(preprocessed_path)
    
    full_prompt = f"{prompt}, {style.name}, best quality, extremely detailed"
    
    generator = torch.Generator(device=device).manual_seed(seed)
    
    params = {
        "prompt": full_prompt,
        "image": sketch_image,
        "num_inference_steps": num_inference_steps,
        "guidance_scale": guidance_scale,
        "generator": generator
    }

    if model_id != "flux_canny":
        params["negative_prompt"] = negative_prompt

    model_info = settings.AVAILABLE_MODELS[model_id]
    config = model_info.get("config", {})

    if model_id == "t2i_adapter_sdxl":
        params["adapter_conditioning_scale"] = config.get("adapter_conditioning_scale", 0.9)
        params["adapter_conditioning_factor"] = config.get("adapter_conditioning_factor", 0.9)
    
    elif model_id == "flux_canny":
        params["control_image"] = params.pop("image")
        params["height"] = config.get("output_height", 1024)
        params["width"] = config.get("output_width", 1024)

    start_time = time.time()
    output = pipe(**params)
    generation_time = time.time() - start_time
    
    if hasattr(output, "images") and len(output.images) > 0:
        output_image = output.images[0]
    else:
        output_image = output[0] if isinstance(output, (list, tuple)) else output
    
    output_image.save(output_path)
    
    return generation_time, loading_time

def create_summary_plots(csv_path: str, output_dir: str) -> None:
    """Create summary plots based on generation time data"""
    data = []
    with open(csv_path, 'r', newline='') as file:
        reader = csv.DictReader(file)
        for row in reader:
            if row['Generation Time (s)'] == "ERROR":
                continue
                
            row['Generation Time (s)'] = float(row['Generation Time (s)'])
            row['Loading Time (s)'] = float(row['Loading Time (s)'])
            data.append(row)
    
    if not data:
        print("No data available for summary plots")
        return
    
    plots_dir = os.path.join(output_dir, "summary_plots")
    os.makedirs(plots_dir, exist_ok=True)
    
    
    model_times = {}
    for row in data:
        model_id = row['Model ID']
        if model_id not in model_times:
            model_times[model_id] = []
        model_times[model_id].append(row['Generation Time (s)'])
    
    fig, ax = plt.subplots(figsize=(10, 6))
    models = list(model_times.keys())
    avg_times = [np.mean(model_times[model]) for model in models]
    std_times = [np.std(model_times[model]) for model in models]
    
    model_names = [settings.AVAILABLE_MODELS[model]['name'] for model in models]
    
    ax.bar(model_names, avg_times, yerr=std_times, capsize=10)
    ax.set_ylabel('Average Generation Time (s)')
    ax.set_title('Average Generation Time by Model')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'model_times.png'))
    plt.close(fig)
    
    style_model_times = {}
    for row in data:
        model_id = row['Model ID']
        style_id = row['Style ID']
        key = f"{model_id}_{style_id}"
        if key not in style_model_times:
            style_model_times[key] = []
        style_model_times[key].append(row['Generation Time (s)'])
    
    styles = list(set(row['Style ID'] for row in data))
    x = np.arange(len(models))
    width = 0.8 / len(styles)
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    for i, style in enumerate(styles):
        style_name = next((s.name for s in AVAILABLE_STYLES if s.id == style), style)
        style_times = []
        for model in models:
            key = f"{model}_{style}"
            if key in style_model_times:
                style_times.append(np.mean(style_model_times[key]))
            else:
                style_times.append(0)
        
        ax.bar(x + i * width - width * (len(styles) - 1) / 2, style_times, width, label=style_name)
    
    ax.set_ylabel('Average Generation Time (s)')
    ax.set_title('Generation Time by Model and Style')
    ax.set_xticks(x)
    ax.set_xticklabels(model_names, rotation=45, ha='right')
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'style_model_times.png'))
    plt.close(fig)
    
    total_model_times = {model: sum(times) for model, times in model_times.items()}
    fig, ax = plt.subplots(figsize=(10, 10))
    model_names = [settings.AVAILABLE_MODELS[model]['name'] for model in total_model_times.keys()]
    ax.pie(
        total_model_times.values(), 
        labels=model_names, 
        autopct='%1.1f%%',
        startangle=90
    )
    ax.axis('equal')
    plt.title('Total Generation Time Distribution by Model')
    plt.savefig(os.path.join(plots_dir, 'time_distribution_pie.png'))
    plt.close(fig)
    
    fig, ax = plt.subplots(figsize=(12, 8))
    box_data = [model_times[model] for model in models]
    ax.boxplot(box_data, labels=model_names)
    ax.set_ylabel('Generation Time (s)')
    ax.set_title('Distribution of Generation Times by Model')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'time_boxplot.png'))
    plt.close(fig)
    
    if len(set(row['Sketch'] for row in data)) > 1:
        fig, ax = plt.subplots(figsize=(12, 8))
        sketches = list(set(row['Sketch'] for row in data))
        
        for sketch in sketches:
            sketch_data = [row for row in data if row['Sketch'] == sketch]
            times = [row['Generation Time (s)'] for row in sketch_data]
            avg_time = np.mean(times)
            ax.scatter([sketch] * len(sketch_data), times, alpha=0.5, label=None)
            ax.scatter([sketch], [avg_time], color='red', s=100, label=f"{sketch} (avg)")
        
        ax.set_ylabel('Generation Time (s)')
        ax.set_title('Generation Times by Sketch')
        plt.xticks(rotation=90)
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, 'sketch_times.png'))
        plt.close(fig)
    
    print(f"Created summary plots in {plots_dir}")

def save_test_config(output_dir: str, args: argparse.Namespace) -> None:
    """Save the test configuration for reproducibility"""
    config = {
        "timestamp": datetime.now().isoformat(),
        "args": vars(args),
        "cuda_available": torch.cuda.is_available(),
        "device_used": args.device if args.device else settings.DEVICE,
        "available_models": {k: v["name"] for k, v in settings.AVAILABLE_MODELS.items()},
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
    
    csv_headers = ["Sketch", "Prompt", "Model ID", "Style ID", "Style Name", "Seed", "Generation Time (s)", "Loading Time (s)"]
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
        
        for model_id in settings.AVAILABLE_MODELS:
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
                    generation_time, loading_time = generate_and_save_image(
                        sketch_path=sketch_path,
                        output_path=output_path,
                        model_id=model_id,
                        style=style,
                        prompt=sketch_prompt,
                        negative_prompt=args.negative_prompt,
                        num_inference_steps=args.num_inference_steps,
                        guidance_scale=args.guidance_scale,
                        preprocessed_dir=dirs["preprocessed"],
                        device=settings.DEVICE,
                        seed=sketch_seed
                    )
                    
                    print(f"    Generated in {generation_time:.2f} seconds")
                    
                    sketch_results[sketch_path][model_id][style_id] = output_path
                    
                    append_to_csv(csv_path, [
                        sketch_name,
                        sketch_prompt,
                        model_id,
                        style_id,
                        style_name,
                        sketch_seed,
                        f"{generation_time:.4f}",
                        f"{loading_time:.4f}"
                    ])
                    
                except Exception as e:
                    print(f"    Error generating image: {str(e)}")
                    append_to_csv(csv_path, [
                        sketch_name,
                        sketch_prompt, 
                        model_id,
                        style_id,
                        style_name,
                        sketch_seed,  
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
    
    try:
        create_summary_plots(csv_path, dirs["base"])
    except Exception as e:
        print(f"Error creating summary plots: {str(e)}")
    
    print(f"\nAll done! Results saved to {dirs['base']}")
    print(f"Generation times recorded in {csv_path}")

if __name__ == "__main__":
    main()