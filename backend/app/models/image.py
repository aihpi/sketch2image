from typing import Optional, List, Dict, Any
from pydantic import BaseModel

class StyleOption(BaseModel):
    """Style option for image generation"""
    id: str
    name: str
    prompt_prefix: str

class ModelOption(BaseModel):
    """Model option for image generation"""
    id: str
    name: str
    description: str
    huggingface_id: str
    inference_speed: str
    recommended_for: List[str]

class ImageResponse(BaseModel):
    """Response model for image generation"""
    generation_id: str
    status: str  # "processing" or "completed"
    message: str
    image_url: Optional[str] = None

class EnhancedPromptResponse(BaseModel):
    """Response model for prompt enhancement"""
    original_prompt: str
    enhanced_prompt: str