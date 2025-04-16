import os
import uuid
from typing import List
from fastapi import APIRouter, HTTPException, UploadFile, File, Form, BackgroundTasks
from fastapi.responses import FileResponse
from app.core.config import settings
from app.services.generation import generate_image_from_sketch
from app.models.image import ImageResponse, StyleOption

router = APIRouter()

# Available style options
AVAILABLE_STYLES = [
    StyleOption(id="photorealistic", name="Photorealistic", prompt_prefix="a photorealistic image of"),
    StyleOption(id="anime", name="Anime", prompt_prefix="an anime style drawing of"),
    StyleOption(id="oil_painting", name="Oil Painting", prompt_prefix="an oil painting of"),
    StyleOption(id="watercolor", name="Watercolor", prompt_prefix="a watercolor painting of"),
    StyleOption(id="sketch", name="Detailed Sketch", prompt_prefix="a detailed sketch of"),
]

@router.get("/styles", response_model=List[StyleOption])
async def get_styles():
    """Get available style options for image generation"""
    return AVAILABLE_STYLES

@router.post("/generate", response_model=ImageResponse)
async def generate_image(
    background_tasks: BackgroundTasks,
    sketch_file: UploadFile = File(...),
    style_id: str = Form(...),
    description: str = Form(None),
):
    """Generate an image from a sketch with the specified style"""
    
    # Validate style_id
    style = next((s for s in AVAILABLE_STYLES if s.id == style_id), None)
    if not style:
        raise HTTPException(status_code=400, detail=f"Invalid style ID: {style_id}")
    
    # Generate a unique ID for this generation
    generation_id = str(uuid.uuid4())
    
    # Create file paths for input and output
    os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
    os.makedirs(settings.OUTPUT_DIR, exist_ok=True)
    
    sketch_path = os.path.join(settings.UPLOAD_DIR, f"{generation_id}_sketch.png")
    output_path = os.path.join(settings.OUTPUT_DIR, f"{generation_id}_output.png")
    
    # Save uploaded sketch to file
    with open(sketch_path, "wb") as f:
        content = await sketch_file.read()
        f.write(content)
    
    try:
        # Build the prompt based on style and optional description
        prompt = style.prompt_prefix
        if description:
            prompt += f" {description}"
        else:
            prompt += " a scene"  # Generic fallback
        
        # Generate the image (this will run in the background)
        background_tasks.add_task(
            generate_image_from_sketch,
            sketch_path=sketch_path,
            output_path=output_path,
            prompt=prompt,
            negative_prompt="low quality, bad anatomy, worst quality, low resolution",
        )
        
        # Return response with generation ID
        return ImageResponse(
            generation_id=generation_id,
            status="processing",
            message="Image generation started. Check status endpoint for completion."
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/status/{generation_id}", response_model=ImageResponse)
async def get_generation_status(generation_id: str):
    """Check the status of an image generation request"""
    output_path = os.path.join(settings.OUTPUT_DIR, f"{generation_id}_output.png")
    
    if os.path.exists(output_path):
        return ImageResponse(
            generation_id=generation_id,
            status="completed",
            message="Image generation completed",
            image_url=f"/api/images/{generation_id}"
        )
    else:
        return ImageResponse(
            generation_id=generation_id,
            status="processing",
            message="Image is still being generated"
        )

@router.get("/images/{generation_id}")
async def get_generated_image(generation_id: str):
    """Get the generated image by ID"""
    output_path = os.path.join(settings.OUTPUT_DIR, f"{generation_id}_output.png")
    
    if not os.path.exists(output_path):
        raise HTTPException(status_code=404, detail="Generated image not found")
    
    return FileResponse(output_path)