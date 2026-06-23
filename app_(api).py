import os
import shutil
import uuid
from contextlib import asynccontextmanager
from typing import List, Optional

from fastapi import FastAPI, HTTPException, UploadFile, File, Form, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Importing your structured Master agent logic module
from backend.Agents.Master import WasteDispoMaster

# Global dictionary to maintain master agent instances per location or session
master_instances = {}

# We change the upload directory to a stable location inside the project
# that your Dashboard/Vision components can access consistently.
UPLOAD_DIR = "./backend/display"


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Handles application startup and shutdown events cleanly, 
    ensuring persistent workspace paths are verified.
    """
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    yield  
    
    print("\n🧹 Cleaning up agent sessions...")
    for loc, master in list(master_instances.items()):
        try:
            master.cleanup()
        except Exception as e:
            print(f"Error cleaning up session for {loc}: {e}")
    print("All backend master agents offline.")


app = FastAPI(
    title="SustainAi Waste Disposal API",
    description="REST API backend for autonomous environmental analysis and waste management.",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def get_master_agent(location: str) -> WasteDispoMaster:
    """Helper to reuse or create a WasteDispoMaster instance per location."""
    if location not in master_instances:
        master_instances[location] = WasteDispoMaster(default_location=location)
    return master_instances[location]


# --- Pydantic Schemas for Requests & Responses ---

class ChatRequest(BaseModel):
    user_input: str
    location: Optional[str] = "Chittarikkal, Kerala, India"


class ChatResponse(BaseModel):
    system_name: str
    master_name: str
    response: str
    mode: str
    knowledge_base: Optional[dict] = None


# --- REST API Endpoints ---

@app.get("/")
def root():
    return {
        "status": "online",
        "message": "Welcome to the SustainAi Waste Disposal API. Use /docs for interactive API testing UI."
    }


@app.post("/api/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    """
    Handles standard text-based chat interactions and passes back
    updated dashboard state context telemetry.
    """
    system_name = os.getenv("SUSTAINAI_SYSTEM_NAME", "SustainAi")
    master_name = os.getenv("SUSTAINAI_MASTER_NAME", "Lily")
    
    master = get_master_agent(request.location)
    user_input = request.user_input.strip()

    if not user_input:
        raise HTTPException(status_code=400, detail="User input cannot be empty.")

    if any(word in user_input.lower() for word in ["status", "processing", "working", "update"]):
        response = master.get_status_update()
        mode = "status"
    else:
        response = master.process_input(user_input)
        mode = "chat"

    return ChatResponse(
        system_name=system_name,
        master_name=master_name,
        response=response,
        mode=mode,
        # Returning current knowledge_base state to sync live dashboard data points
        knowledge_base=master.context.get("knowledge_base", {})
    )


@app.post("/api/speech", response_model=ChatResponse)
async def speech_endpoint(
    audio_file: UploadFile = File(...), 
    location: str = Form("Chittarikkal, Kerala, India")
):
    """
    Handles voice-based interactions.
    """
    system_name = os.getenv("SUSTAINAI_SYSTEM_NAME", "SustainAi")
    master_name = os.getenv("SUSTAINAI_MASTER_NAME", "Lily")
    master = get_master_agent(location)

    # Temporary holding path for incoming stream conversion
    temp_audio_dir = "./Storage"
    os.makedirs(temp_audio_dir, exist_ok=True)
    temp_audio_path = f"{temp_audio_dir}/{uuid.uuid4()}_{audio_file.filename}"
    
    with open(temp_audio_path, "wb") as buffer:
        shutil.copyfileobj(audio_file.file, buffer)

    try:
        # Placeholder transcript: adjust to direct STT pipeline bindings if needed
        user_in = f"[Transcribed text from {audio_file.filename}]" 
        response = master.process_input(user_in)

        return ChatResponse(
            system_name=system_name,
            master_name=master_name,
            response=response,
            mode="speech",
            knowledge_base=master.context.get("knowledge_base", {})
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Speech processing failed: {str(e)}")
    finally:
        if os.path.exists(temp_audio_path):
            os.remove(temp_audio_path)


@app.post("/api/analyze-images")
async def analyze_images_endpoint(
    images: List[UploadFile] = File(...),
    location: str = Form("Chittarikkal, Kerala, India")
):
    """
    Accepts multiple image file uploads, saves them into the persistent 
    workspace asset space used by Master Agent pipelines, updates the internal 
    Ecosystem Intelligence State, and returns synchronized structured data back.
    """
    master = get_master_agent(location)
    saved_file_paths = []

    # 1. Write the files directly into the active dashboard assets area ('./display')
    # instead of a volatile temporary uploads folder.
    for image in images:
        file_extension = os.path.splitext(image.filename)[1]
        unique_filename = f"upload_{uuid.uuid4().hex}{file_extension}"
        file_path = os.path.join(UPLOAD_DIR, unique_filename)
        
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(image.file, buffer)
        
        saved_file_paths.append(file_path)
        # Register the created files into master session context tracking
        master.context.setdefault("created_files", []).append(file_path)

    try:
        # 2. Run your agent's bulk image analyzer pipeline
        # This will write insights directly to master.context["knowledge_base"]
        upload_summaries = master._analyze_image_list(saved_file_paths)
        
        # 3. We NO LONGER delete files immediately in background tasks here.
        # This allows the vision agent descriptions, dashboards, and plot metrics 
        # to read them permanently until session.cleanup() handles them on shutdown.

        return {
            "status": "success",
            "location": location,
            "summaries": upload_summaries if upload_summaries else ["No visual abnormalities identified."],
            "knowledge_base": master.context.get("knowledge_base", {})
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Image analysis failed: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)