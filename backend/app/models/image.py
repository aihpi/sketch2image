from typing import Optional
from pydantic import BaseModel

class StyleOption(BaseModel):
    """Style option for image generation"""
    id: str
    name: str
    prompt_prefix: str

class ImageResponse(BaseModel):
    """Response model for image generation"""
    generation_id: str
    status: str  # "processing" or "completed"
    message: str
    image_url: Optional[str] = None