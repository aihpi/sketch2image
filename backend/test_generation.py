#!/usr/bin/env python3
"""
Test script for the sketch-to-image generation functionality.
This script tests the image generation pipeline independently from the API.
"""

import os
import sys
import argparse
from PIL import Image
import torch
from app.services.generation import get_pipeline, preprocess_sketch

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
        default=20,
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
    
    print(f"Using prompt: '{prompt}'")
    
    try:
        # Get the pipeline
        print("Loading model pipeline...")
        pipe = get_pipeline()
        
        # Preprocess the sketch
        print("Preprocessing sketch...")
        sketch_image = preprocess_sketch(args.sketch)
        
        # Generate the image
        print(f"Generating image with {args.steps} steps and guidance {args.guidance}...")
        output_image = pipe(
            prompt=prompt,
            negative_prompt="low quality, bad anatomy, worst quality, low resolution",
            image=sketch_image,
            num_inference_steps=args.steps,
            guidance_scale=args.guidance,
        ).images[0]
        
        # Save the output image
        output_image.save(args.output)
        print(f"Image generated successfully and saved to {args.output}")
        
        # Display the images if possible
        try:
            import matplotlib.pyplot as plt
            
            plt.figure(figsize=(12, 6))
            plt.subplot(1, 2, 1)
            plt.imshow(sketch_image)
            plt.title("Input Sketch")
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
        sys.exit(1)

if __name__ == "__main__":
    main()
