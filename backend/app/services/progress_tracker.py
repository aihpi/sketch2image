import time
from typing import Optional, Dict, Any
import threading

_progress_store: Dict[str, Dict[str, Any]] = {}
_lock = threading.Lock()

def update_progress(
    generation_id: str, 
    current_step: int, 
    total_steps: int, 
    stage: str = "generating", 
    eta_seconds: Optional[int] = None
):
    """Update progress for a generation"""
    with _lock:
        progress_data = {
            "current_step": current_step,
            "total_steps": total_steps,
            "percentage": int((current_step / total_steps) * 100) if total_steps > 0 else 0,
            "stage": stage,
            "eta_seconds": eta_seconds,
            "timestamp": time.time()
        }
        _progress_store[generation_id] = progress_data
        print(f"Progress update: {generation_id} - {stage} - Step {current_step}/{total_steps} ({progress_data['percentage']}%)")

def get_progress(generation_id: str) -> Optional[Dict[str, Any]]:
    """Get current progress for a generation"""
    with _lock:
        return _progress_store.get(generation_id)

def remove_progress(generation_id: str):
    """Remove progress data for a completed generation"""
    with _lock:
        if generation_id in _progress_store:
            del _progress_store[generation_id]

def get_all_progress() -> Dict[str, Dict[str, Any]]:
    """Get all current progress data (for debugging)"""
    with _lock:
        return _progress_store.copy()