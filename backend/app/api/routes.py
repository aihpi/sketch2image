import os
import hashlib
import json
import time
from datetime import datetime
from typing import List
from fastapi import APIRouter, HTTPException, UploadFile, File, Form, BackgroundTasks
from fastapi.responses import FileResponse, StreamingResponse
from app.core.config import settings
from app.services.generation import generate_image_from_sketch, load_model_pipeline
from app.services.progress_tracker import update_progress, get_progress, remove_progress
from app.models.image import ImageResponse, StyleOption, ModelOption
import asyncio

router = APIRouter()

AVAILABLE_STYLES = [
    StyleOption(id="photorealistic", name="Photorealistic", prompt_prefix="a photorealistic image of"),
    StyleOption(id="anime", name="Anime", prompt_prefix="an anime style drawing of"),
    StyleOption(id="oil_painting", name="Oil Painting", prompt_prefix="an oil painting of"),
    StyleOption(id="watercolor", name="Watercolor", prompt_prefix="a watercolor painting of"),
    StyleOption(id="sketch", name="Detailed Sketch", prompt_prefix="a detailed sketch of"),
]

def generate_sketch_hash(sketch_content: bytes, model_id: str, style_id: str, description: str) -> str:
    """Generate a unique hash for this generation request (includes timestamp for uniqueness)"""
    hasher = hashlib.sha256()
    hasher.update(sketch_content)
    hasher.update(model_id.encode('utf-8'))
    hasher.update(style_id.encode('utf-8'))
    hasher.update(description.encode('utf-8'))
    # Add current timestamp to ensure uniqueness for each generation
    hasher.update(str(time.time()).encode('utf-8'))
    return hasher.hexdigest()[:16]  # Use first 16 characters for shorter filenames

def save_generation_metadata(sketch_hash: str, metadata: dict):
    """Save generation metadata to JSON file"""
    metadata_path = os.path.join(settings.DATASET_METADATA_DIR, f"{sketch_hash}.json")
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)

@router.get("/styles", response_model=List[StyleOption])
async def get_styles():
    """Get available style options for image generation"""
    return AVAILABLE_STYLES

@router.get("/models", response_model=List[ModelOption])
async def get_models():
    """Get available model options for image generation"""
    models = []
    for model_id, model_data in settings.AVAILABLE_MODELS.items():
        description = f"Inference speed: {model_data['inference_speed']}"
        
        models.append(
            ModelOption(
                id=model_id,
                name=model_data["name"],
                description=description,
                huggingface_id=model_data["huggingface_id"],
                inference_speed=model_data["inference_speed"],
                recommended_for=model_data["recommended_for"]
            )
        )
    return models

@router.get("/progress/{generation_id}")
async def get_progress_stream(generation_id: str):
    """Stream progress updates for a generation using Server-Sent Events"""
    
    async def event_stream():
        yield f"data: {json.dumps({'type': 'connected', 'generation_id': generation_id})}\n\n"
        
        start_time = time.time()
        last_progress = None
        
        while True:
            metadata_path = os.path.join(settings.DATASET_METADATA_DIR, f"{generation_id}.json")
            if os.path.exists(metadata_path):
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                
                if "file_info" in metadata:
                    yield f"data: {json.dumps({'type': 'completed', 'generation_id': generation_id})}\n\n"
                    remove_progress(generation_id)
                    break
            
            current_progress = get_progress(generation_id)
            
            if current_progress and current_progress != last_progress:
                # Calculate ETA if not provided
                if current_progress.get('eta_seconds') is None and current_progress['current_step'] > 0:
                    elapsed = time.time() - start_time
                    steps_per_second = current_progress['current_step'] / elapsed
                    remaining_steps = current_progress['total_steps'] - current_progress['current_step']
                    eta_seconds = int(remaining_steps / steps_per_second) if steps_per_second > 0 else None
                    current_progress['eta_seconds'] = eta_seconds
                
                progress_data = {
                    'type': 'progress',
                    'generation_id': generation_id,
                    **current_progress
                }
                yield f"data: {json.dumps(progress_data)}\n\n"
                last_progress = current_progress.copy()
            
            await asyncio.sleep(0.5)  # Check every 500ms
            
            # Timeout after 5 minutes
            if time.time() - start_time > 300:
                yield f"data: {json.dumps({'type': 'timeout', 'generation_id': generation_id})}\n\n"
                remove_progress(generation_id)
                break

    return StreamingResponse(
        event_stream(),
        media_type="text/plain",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "Content-Type": "text/event-stream",
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Headers": "Cache-Control"
        }
    )

@router.post("/generate", response_model=ImageResponse)
async def generate_image(
    background_tasks: BackgroundTasks,
    sketch_file: UploadFile = File(...),
    style_id: str = Form(...),
    model_id: str = Form(None),
    description: str = Form(None),
):
    """Generate images from a sketch with the specified style and model"""
    
    style = next((s for s in AVAILABLE_STYLES if s.id == style_id), None)
    if not style:
        raise HTTPException(status_code=400, detail=f"Invalid style ID: {style_id}")
    
    if not model_id:
        model_id = settings.DEFAULT_MODEL_ID
    
    if model_id not in settings.AVAILABLE_MODELS:
        raise HTTPException(status_code=400, detail=f"Invalid model ID: {model_id}")
    
    model_info = settings.AVAILABLE_MODELS[model_id]
    
    # Read sketch content
    sketch_content = await sketch_file.read()
    
    # Generate hash-based ID
    sketch_hash = generate_sketch_hash(sketch_content, model_id, style_id, description or "")
    
    total_steps = model_info.get("config", {}).get("num_inference_steps", 20)
    update_progress(sketch_hash, 0, total_steps, "initializing")
    
    # Save sketch directly to dataset
    sketch_path = os.path.join(settings.DATASET_SKETCH_DIR, f"{sketch_hash}.png")
    with open(sketch_path, "wb") as f:
        f.write(sketch_content)
    
    try:
        if description:
            prompt = f"{description}, {style.name} style, best quality, extremely detailed"
        else:
            prompt = f"a scene, {style.name}, best quality, extremely detailed"
        
        negative_prompt = model_info.get("config", {}).get("default_negative_prompt", 
                                                        "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality")
        
        # Create initial metadata
        metadata = {
            "sketch_hash": sketch_hash,
            "prompt": prompt,
            "model_id": model_id,
            "style_id": style_id,
            "timestamp": datetime.now().isoformat(),
            "generation_params": {
                "num_inference_steps": model_info.get("config", {}).get("num_inference_steps", 20),
                "guidance_scale": model_info.get("config", {}).get("guidance_scale", 7.5),
                "negative_prompt": negative_prompt
            }
        }
        
        background_tasks.add_task(
            generate_image_from_sketch,
            sketch_path=sketch_path,
            prompt=prompt,
            model_id=model_id,
            negative_prompt=negative_prompt,
            sketch_hash=sketch_hash,
            metadata=metadata,
        )
        
        return ImageResponse(
            generation_id=sketch_hash,
            status="processing",
            message=f"Image generation started using {model_info['name']}. Connect to progress stream for real-time updates."
        )
    
    except Exception as e:
        remove_progress(sketch_hash)
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/status/{generation_id}", response_model=ImageResponse)
async def get_generation_status(generation_id: str):
    """Check the status of an image generation request"""
    # Check metadata to see if generation is complete
    metadata_path = os.path.join(settings.DATASET_METADATA_DIR, f"{generation_id}.json")
    
    if os.path.exists(metadata_path):
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        if "file_info" in metadata:
            # Generation completed
            result_count = metadata["file_info"]["result_count"]
            if result_count == 1:
                image_url = f"/api/images/{generation_id}"
            else:
                image_url = f"/api/images/{generation_id}_1"
                
            return ImageResponse(
                generation_id=generation_id,
                status="completed",
                message=f"Image generation completed. Generated {result_count} images.",
                image_url=image_url
            )
    
    return ImageResponse(
        generation_id=generation_id,
        status="processing",
        message="Images are still being generated"
    )

@router.get("/images/{image_id}")
async def get_generated_image(image_id: str):
    """Get the generated image by ID"""
    # Handle both single image and multi-image cases
    if "_" in image_id and image_id.split("_")[-1].isdigit():
        # Multi-image case: generation_id_1, generation_id_2, etc.
        result_path = os.path.join(settings.DATASET_RESULT_DIR, f"{image_id}.png")
    else:
        # Single image case: generation_id
        result_path = os.path.join(settings.DATASET_RESULT_DIR, f"{image_id}.png")
    
    if not os.path.exists(result_path):
        raise HTTPException(status_code=404, detail="Generated image not found")
    
    # Add cache-busting headers to ensure fresh images are served
    headers = {
        "Cache-Control": "no-cache, no-store, must-revalidate",
        "Pragma": "no-cache",
        "Expires": "0"
    }
    
    return FileResponse(result_path, headers=headers)