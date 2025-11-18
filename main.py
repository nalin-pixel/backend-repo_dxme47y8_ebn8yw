import os
import uuid
import shutil
import asyncio
from datetime import datetime, timezone
from typing import Optional, Dict, Any, List

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from database import db
from schemas import Media, AnimationJob, ChatMessage, Preset

app = FastAPI(title="Animation AI Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Paths
UPLOAD_DIR = os.path.abspath("uploads")
OUTPUT_DIR = os.path.abspath("outputs")
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Utilities
ALLOWED_IMAGE = {"image/png", "image/jpeg", "image/webp"}
ALLOWED_VIDEO = {"video/mp4", "video/quicktime", "video/webm"}

RESOLUTION_MAP = {
    "720p": (1280, 720),
    "1080p": (1920, 1080),
    "2k": (2560, 1440),
    "4k": (3840, 2160),
}


def validate_resolution(value: str) -> str:
    key = value.lower()
    if key in RESOLUTION_MAP:
        return key
    if key in {"720", "1080", "1440", "2160"}:
        return {"720": "720p", "1080": "1080p", "1440": "2k", "2160": "4k"}[key]
    # Fallback
    return "1080p"


@app.get("/")
def root():
    return {"message": "Animation AI Backend running"}


@app.get("/test")
def test_database():
    response = {
        "backend": "✅ Running",
        "database": "❌ Not Connected",
        "collections": [],
    }
    try:
        if db is not None:
            response["database"] = "✅ Connected"
            response["collections"] = db.list_collection_names()
    except Exception as e:
        response["database"] = f"⚠️ {str(e)[:120]}"
    return response


# -------- Media Upload --------
@app.post("/upload")
async def upload_media(file: UploadFile = File(...)):
    if file.content_type not in ALLOWED_IMAGE | ALLOWED_VIDEO:
        raise HTTPException(status_code=400, detail="Unsupported file type")

    extension = os.path.splitext(file.filename)[1] or ""
    media_id = str(uuid.uuid4())
    storage_name = f"{media_id}{extension}"
    storage_path = os.path.join(UPLOAD_DIR, storage_name)

    with open(storage_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    media_doc = {
        "_id": media_id,
        "original_filename": file.filename,
        "mime_type": file.content_type,
        "media_type": "image" if file.content_type in ALLOWED_IMAGE else "video",
        "storage_path": storage_path,
        "created_at": datetime.now(timezone.utc),
    }
    db["media"].insert_one(media_doc)

    return {
        "media_id": media_id,
        "filename": file.filename,
        "mime_type": file.content_type,
        "media_type": media_doc["media_type"],
        "storage_path": storage_path,
    }


# -------- Jobs --------
class JobCreate(BaseModel):
    media_id: str
    params: Dict[str, Any] = Field(default_factory=dict)


async def simulate_processing(job_id: str):
    # Simulate job progress and generate dummy outputs
    for p in range(0, 101, 10):
        await asyncio.sleep(0.2)
        db["animationjob"].update_one({"_id": job_id}, {"$set": {"progress": p, "status": "processing"}})
    # Create dummy files
    mp4_path = os.path.join(OUTPUT_DIR, f"{job_id}.mp4")
    gif_path = os.path.join(OUTPUT_DIR, f"{job_id}.gif")
    with open(mp4_path, "wb") as f:
        f.write(b"\x00" * 1024)  # placeholder bytes
    with open(gif_path, "wb") as f:
        f.write(b"GIF89a")
    db["animationjob"].update_one(
        {"_id": job_id},
        {
            "$set": {
                "status": "completed",
                "progress": 100,
                "output": {
                    "mp4_url": f"/download/{job_id}.mp4",
                    "gif_url": f"/download/{job_id}.gif",
                },
                "updated_at": datetime.now(timezone.utc),
            }
        },
    )


@app.post("/jobs")
async def create_job(payload: JobCreate, background_tasks: BackgroundTasks):
    media = db["media"].find_one({"_id": payload.media_id})
    if not media:
        raise HTTPException(status_code=404, detail="Media not found")

    params = {
        "pose": payload.params.get("pose", "auto"),
        "expression": payload.params.get("expression", "neutral"),
        "style": payload.params.get("style", "original"),
        "background": payload.params.get("background", "keep"),
        "speed": float(payload.params.get("speed", 1.0)),
        "resolution": validate_resolution(str(payload.params.get("resolution", "1080p"))),
        "preserve_likeness": bool(payload.params.get("preserve_likeness", True)),
        "lip_sync": bool(payload.params.get("lip_sync", False)),
        "auto_rigging": bool(payload.params.get("auto_rigging", True)),
        "motion_template": payload.params.get("motion_template"),
    }

    job_id = str(uuid.uuid4())
    doc = {
        "_id": job_id,
        "media_id": payload.media_id,
        "status": "queued",
        "progress": 0,
        "params": params,
        "output": {"mp4_url": None, "gif_url": None},
        "created_at": datetime.now(timezone.utc),
        "updated_at": datetime.now(timezone.utc),
    }
    db["animationjob"].insert_one(doc)

    background_tasks.add_task(simulate_processing, job_id)

    return {"job_id": job_id, "status": "queued", "params": params}


@app.get("/jobs/{job_id}")
async def get_job(job_id: str):
    job = db["animationjob"].find_one({"_id": job_id}, {"_id": 0})
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    job["job_id"] = job_id
    return job


# -------- Download (serve generated files) --------
@app.get("/download/{filename}")
async def download_file(filename: str):
    path = os.path.join(OUTPUT_DIR, filename)
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="File not found")
    media_type = "video/mp4" if filename.endswith(".mp4") else "image/gif"
    return FileResponse(path, media_type=media_type, filename=filename)


# -------- Presets & Motion Templates --------
DEFAULT_PRESETS: List[Preset] = [
    Preset(
        name="Cinematic 4K",
        description="Film look, subtle motion, preserved likeness",
        params={
            "style": "cinematic",
            "speed": 1.0,
            "resolution": "4k",
            "preserve_likeness": True,
            "background": "keep",
        },
    ),
    Preset(
        name="Cartoon Pop",
        description="Bold outlines and punchy colors",
        params={
            "style": "cartoon",
            "speed": 1.1,
            "resolution": "1080p",
            "preserve_likeness": True,
            "background": "solid:yellow",
        },
    ),
]

MOTION_TEMPLATES = [
    {"id": "idle_wave", "name": "Idle Wave"},
    {"id": "talking_head", "name": "Talking Head"},
    {"id": "dance_loop", "name": "Dance Loop"},
]


@app.get("/presets")
async def get_presets():
    presets = list(db["preset"].find({}, {"_id": 0}))
    if not presets:
        # seed defaults
        for p in DEFAULT_PRESETS:
            db["preset"].insert_one({"name": p.name, "description": p.description, "params": p.params})
        presets = list(db["preset"].find({}, {"_id": 0}))
    return {"presets": presets}


class PresetCreate(BaseModel):
    name: str
    description: Optional[str] = None
    params: Dict[str, Any]


@app.post("/presets")
async def create_preset(payload: PresetCreate):
    db["preset"].insert_one(payload.model_dump())
    return {"ok": True}


@app.get("/motion-templates")
async def list_motion_templates():
    return {"templates": MOTION_TEMPLATES}


# -------- Chat: parse instructions and update params --------
class ChatInput(BaseModel):
    job_id: str
    message: str


def apply_nlp_to_params(current: Dict[str, Any], message: str) -> Dict[str, Any]:
    msg = message.lower()
    updated = current.copy()
    # style
    for key in ["cinematic", "cartoon", "anime", "realistic", "sketch"]:
        if key in msg:
            updated["style"] = key
    # speed
    if "faster" in msg:
        updated["speed"] = min(3.0, float(current.get("speed", 1.0)) + 0.25)
    if "slower" in msg:
        updated["speed"] = max(0.25, float(current.get("speed", 1.0)) - 0.25)
    # resolution
    if "4k" in msg or "ultra" in msg:
        updated["resolution"] = "4k"
    elif "1080" in msg:
        updated["resolution"] = "1080p"
    elif "720" in msg:
        updated["resolution"] = "720p"
    # background
    if "background" in msg:
        # naive extraction after word background
        try:
            after = msg.split("background", 1)[1].strip(":= .")
            if after:
                updated["background"] = after[:60]
        except Exception:
            pass
    # pose/expression
    for pose_key in ["idle", "run", "walk", "dance", "pose", "sit", "stand"]:
        if pose_key in msg:
            updated["pose"] = pose_key
    for expr_key in ["happy", "sad", "angry", "smile", "neutral", "surprised"]:
        if expr_key in msg:
            updated["expression"] = expr_key if expr_key != "smile" else "happy"
    # toggles
    if "lip" in msg and ("sync" in msg or "synch" in msg):
        updated["lip_sync"] = True
    if "auto-rig" in msg or "autorig" in msg or "auto rig" in msg:
        updated["auto_rigging"] = True
    if "preserve" in msg and "likeness" in msg:
        updated["preserve_likeness"] = True
    # motion templates
    for t in MOTION_TEMPLATES:
        if t["id"] in msg or t["name"].lower() in msg:
            updated["motion_template"] = t["id"]
    return updated


@app.post("/chat")
async def chat(payload: ChatInput):
    job = db["animationjob"].find_one({"_id": payload.job_id})
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    prev_params = job.get("params", {})
    new_params = apply_nlp_to_params(prev_params, payload.message)
    db["animationjob"].update_one({"_id": payload.job_id}, {"$set": {"params": new_params}})

    # Save messages
    db["chatmessage"].insert_one({
        "job_id": payload.job_id,
        "role": "user",
        "content": payload.message,
        "created_at": datetime.now(timezone.utc),
    })
    assistant_text = "Applied your changes. You can render again to see the update."
    db["chatmessage"].insert_one({
        "job_id": payload.job_id,
        "role": "assistant",
        "content": assistant_text,
        "created_at": datetime.now(timezone.utc),
    })

    return {"params": new_params, "assistant": assistant_text}


@app.get("/chat/{job_id}")
async def get_chat(job_id: str):
    msgs = list(db["chatmessage"].find({"job_id": job_id}, {"_id": 0}).sort("created_at", 1))
    return {"messages": msgs}


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
