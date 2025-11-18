"""
Database Schemas for the Animation AI app

Each Pydantic model maps to a MongoDB collection (lowercased class name).
"""
from typing import Optional, Dict, Any, Literal
from pydantic import BaseModel, Field

class Media(BaseModel):
    original_filename: str
    mime_type: str
    media_type: Literal["image", "video"]
    storage_path: str = Field(..., description="Server path to the uploaded asset")
    width: Optional[int] = None
    height: Optional[int] = None
    duration: Optional[float] = None

class AnimationJob(BaseModel):
    media_id: str
    status: Literal["queued", "processing", "completed", "failed"] = "queued"
    progress: int = 0
    params: Dict[str, Any] = Field(
        default_factory=lambda: {
            "pose": "auto",
            "expression": "neutral",
            "style": "original",
            "background": "keep",
            "speed": 1.0,
            "resolution": "1080p",
            "preserve_likeness": True,
            "lip_sync": False,
            "auto_rigging": True,
            "motion_template": None,
        }
    )
    output: Dict[str, Optional[str]] = Field(
        default_factory=lambda: {"mp4_url": None, "gif_url": None}
    )
    error: Optional[str] = None

class ChatMessage(BaseModel):
    job_id: str
    role: Literal["user", "assistant"]
    content: str

class Preset(BaseModel):
    name: str
    description: Optional[str] = None
    params: Dict[str, Any]
