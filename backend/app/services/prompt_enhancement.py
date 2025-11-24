import os
from PIL import Image
from vllm import LLM, SamplingParams
from app.core.config import settings

# Global model cache
_qwen_model = None

def initialize_qwen_model():
    """Initialize Qwen VLM model (call once on startup)"""
    global _qwen_model
    if _qwen_model is None:
        print("Loading Qwen VLM model...")
        os.environ["CUDA_VISIBLE_DEVICES"] = "1"
        _qwen_model = LLM(
            model="Qwen/Qwen2-VL-7B-Instruct",
            tensor_parallel_size=1,
            max_model_len=2048,
            max_num_seqs=5,
            dtype="bfloat16",
            limit_mm_per_prompt={"image": 1},
            mm_processor_kwargs={
                "min_pixels": 28 * 28,
                "max_pixels": 1280 * 28 * 28,
            },
            trust_remote_code=True,
            seed=42,
        )
        print("Qwen VLM model loaded.")
    return _qwen_model

def enhance_prompt_with_qwen(sketch_path: str, user_description: str) -> str:
    """
    Use Qwen VLM to generate an enhanced description of the sketch
    based on the user's original description
    """
    try:
        # Load model
        llm = initialize_qwen_model()
        
        # Load sketch image
        image = Image.open(sketch_path)
        
        # Create prompt for Qwen
        system_prompt = (
            "You are an expert at describing sketches for AI image generation. "
            "Analyze the sketch and the user's description, then create a detailed, "
            "vivid description that will help generate a high-quality image. "
            "Keep the user's intent but add relevant visual details, composition, "
            "style, lighting, and atmosphere. Be concise but descriptive."
        )
        
        user_prompt = (
            f"User's description: {user_description}\n\n"
            f"Based on this sketch and description, write an enhanced prompt for AI image generation. "
            f"Output only the enhanced description, nothing else."
        )
        
        # Format prompt for Qwen2-VL
        placeholder = "<|image_pad|>"
        full_prompt = (
            f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
            f"<|im_start|>user\n<|vision_start|>{placeholder}<|vision_end|>"
            f"{user_prompt}<|im_end|>\n"
            "<|im_start|>assistant\n"
        )
        
        # Sampling parameters
        sampling_params = SamplingParams(
            temperature=0.7,
            max_tokens=256,
        )
        
        # Prepare input
        inputs = {
            "prompt": full_prompt,
            "multi_modal_data": {"image": image},
            "multi_modal_uuids": {"image": sketch_path},
        }
        
        # Generate
        print("Generating enhanced prompt with Qwen VLM...")
        outputs = llm.generate([inputs], sampling_params=sampling_params)
        enhanced_description = outputs[0].outputs[0].text.strip()
        
        print(f"Original: {user_description}")
        print(f"Enhanced: {enhanced_description}")
        
        return enhanced_description
        
    except Exception as e:
        print(f"Error enhancing prompt with Qwen: {str(e)}")
        # Return original description if enhancement fails
        return user_description