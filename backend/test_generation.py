#!/usr/bin/env python3
"""
Test script for the sketch-to-image generation functionality.
This script tests the image generation pipeline independently from the API.
"""

import os
import sys
import argparse
from PIL import Image
import matplotlib.pyplot as plt
import torch
from app.services.generation import preprocess_sketch, load_model_pipeline

def parse_args():
    parser = argparse.ArgumentParser(description="Test sketch-to-image generation")
    parser.add_argument(
        "--sketch", 
        type=str, 
        required=True,
        help="Path to the sketch image file"
    )
    parser.add_argument(
        "--output", 
        type=str, 
        default="test_output.png",
        help="Path to save the generated image"
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=["controlnet_sd15_scribble", "controlnet_sd15_softedge", "controlnet_sdxl_scribble", "t2i_adapter_sdxl"],
        default="t2i_adapter_sdxl",
        help="Model to use for image generation"
    )
    parser.add_argument(
        "--style", 
        type=str, 
        choices=["photorealistic", "anime", "oil_painting", "watercolor", "sketch"],
        default="photorealistic",
        help="Style for the generated image"
    )
    parser.add_argument(
        "--description", 
        type=str, 
        default="",
        help="Optional description for the generated image"
    )
    parser.add_argument(
        "--steps", 
        type=int, 
        default=40,
        help="Number of inference steps"
    )
    parser.add_argument(
        "--guidance", 
        type=float, 
        default=7.5,
        help="Guidance scale"
    )
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Check if sketch file exists
    if not os.path.exists(args.sketch):
        print(f"Error: Sketch file {args.sketch} not found")
        sys.exit(1)
    
    # Map style to prompt prefix
    style_map = {
        "photorealistic": "a photorealistic image of",
        "anime": "an anime style drawing of",
        "oil_painting": "an oil painting of",
        "watercolor": "a watercolor painting of",
        "sketch": "a detailed sketch of"
    }
    
    # Build prompt
    prompt_prefix = style_map[args.style]
    if args.description:
        prompt = f"{prompt_prefix} {args.description}"
    else:
        prompt = f"{prompt_prefix} a scene"
    
    print(f"Using model: {args.model}")
    print(f"Using prompt: '{prompt}'")
    
    try:
        # Get the pipeline
        print("Loading model pipeline...")
        pipe = load_model_pipeline(args.model)
        
        # Preprocess the sketch
        print("Preprocessing sketch...")
        sketch_image = preprocess_sketch(args.sketch, args.model)
        
        # Set model-specific parameters
        params = {
            "prompt": prompt,
            "negative_prompt": "low quality, bad anatomy, worst quality, low resolution",
            "image": sketch_image,
            "num_inference_steps": args.steps,
            "guidance_scale": args.guidance,
        }
        
        # Add model-specific parameters
        if args.model == "t2i_adapter_sdxl":
            params["adapter_conditioning_scale"] = 0.9
            params["adapter_conditioning_factor"] = 0.9
        
        # Generate the image
        print(f"Generating image with {args.steps} steps and guidance {args.guidance}...")
        output = pipe(**params)
        
        # Get the output image
        if hasattr(output, "images") and len(output.images) > 0:
            output_image = output.images[0]
        else:
            # Fallback in case the output format changes
            output_image = output[0] if isinstance(output, (list, tuple)) else output
        
        # Save the output image
        output_image.save(args.output)
        print(f"Image generated successfully and saved to {args.output}")
        
        # Display the images if possible
        try:
            plt.figure(figsize=(12, 6))
            plt.subplot(1, 2, 1)
            plt.imshow(sketch_image, cmap='gray')
            plt.title("Preprocessed Sketch")
            plt.axis("off")
            
            plt.subplot(1, 2, 2)
            plt.imshow(output_image)
            plt.title("Generated Image")
            plt.axis("off")
            
            plt.tight_layout()
            plt.show()
            
        except ImportError:
            print("Note: Install matplotlib to visualize the results")
            
    except Exception as e:
        print(f"Error during generation: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()