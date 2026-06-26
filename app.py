import os
import uuid
import shutil
import atexit
import threading
import time
from datetime import datetime
from flask import Flask, render_template, request, redirect, url_for, jsonify, session, send_file
from werkzeug.utils import secure_filename
from pydub import AudioSegment

from backend.Agents.Master import WasteDispoMaster

app = Flask(__name__)
app.secret_key = os.getenv("SECRET_KEY", "super_secret_sustainai_key_2026")

master_instances = {}
UPLOAD_DIR = "./backend/display"
DASHBOARD_PATH = os.path.abspath("backend/interface/dashboard.html")
os.makedirs(UPLOAD_DIR, exist_ok=True)

ui_state = {
    "chat_history": [],
    "speech_image_path": None,
    "speech_state": {"status": "Idle", "active": False, "error": None}
}

# ============================
# JOB QUEUE
# ============================
jobs = {}  # job_id -> {status, progress, result, error, created_at}

def cleanup_old_jobs():
    """Remove jobs older than 10 minutes to avoid memory bloat."""
    while True:
        time.sleep(60)
        now = time.time()
        for jid in list(jobs.keys()):
            if jobs[jid].get("created_at", 0) < now - 600:
                del jobs[jid]
                print(f"🧹 Cleaned up old job: {jid}")

threading.Thread(target=cleanup_old_jobs, daemon=True).start()

# ============================
# BACKGROUND PIPELINES
# ============================
def run_chat_pipeline(job_id, query, image_path, location):
    try:
        jobs[job_id] = {"status": "processing", "progress": "Initializing...", "created_at": time.time()}

        master = get_master_agent(location)
        if not master:
            jobs[job_id] = {"status": "error", "error": "Agent initialization failed.", "created_at": time.time()}
            return

        if image_path:
            query = f"{query} {image_path}"

        jobs[job_id] = {"status": "processing", "progress": "Processing your request...", "created_at": time.time()}

        ai_response = master.process_input(query)

        timestamp = datetime.now().strftime("%I:%M %p")
        ui_state["chat_history"].append({
            "role": "user",
            "timestamp": timestamp,
            "mode": "chat",
            "content": query  # Chat mode can show the full query (including image path) – that's fine
        })
        ui_state["chat_history"].append({
            "role": "assistant",
            "timestamp": timestamp,
            "mode": "chat",
            "content": ai_response
        })

        if os.path.exists(DASHBOARD_PATH):
            ui_state["dashboard_notification"] = True

        jobs[job_id] = {
            "status": "done",
            "response": ai_response,
            "created_at": time.time()
        }

    except Exception as e:
        jobs[job_id] = {"status": "error", "error": str(e), "created_at": time.time()}
        import traceback
        traceback.print_exc()

def run_voice_pipeline(job_id, audio_path, mime_type, location, image_path):
    try:
        jobs[job_id] = {"status": "processing", "progress": "Converting audio...", "created_at": time.time()}

        # Determine extension
        ext = 'wav' if 'wav' in mime_type else 'webm' if 'webm' in mime_type else 'ogg' if 'ogg' in mime_type else 'webm'
        wav_path = audio_path.replace(f".{ext}", ".wav")
        audio = AudioSegment.from_file(audio_path)
        audio = audio.set_channels(1).set_frame_rate(16000).set_sample_width(2)
        audio.export(wav_path, format="wav")

        jobs[job_id] = {"status": "processing", "progress": "Transcribing...", "created_at": time.time()}

        from backend.output.tts_and_sst import transcribe_audio_file, generate_tts_file
        user_text = transcribe_audio_file(wav_path)

        # Clean up temp files
        if os.path.exists(audio_path):
            os.remove(audio_path)
        if os.path.exists(wav_path):
            os.remove(wav_path)

        if not user_text:
            jobs[job_id] = {"status": "error", "error": "Could not transcribe audio. Please speak clearly.", "created_at": time.time()}
            return

        # Keep the clean transcript for the user
        clean_transcript = user_text

        # Append image path only for the master agent
        if image_path:
            user_text = f"{user_text} {image_path}"

        jobs[job_id] = {"status": "processing", "progress": "Contacting AI models...", "created_at": time.time()}

        master = get_master_agent(location)
        if not master:
            jobs[job_id] = {"status": "error", "error": "Agent initialization failed.", "created_at": time.time()}
            return

        ai_response = master.process_input(user_text)

        jobs[job_id] = {"status": "processing", "progress": "Generating speech...", "created_at": time.time()}

        tts_filename = f"resp_{uuid.uuid4().hex}.mp3"
        tts_path = os.path.join(UPLOAD_DIR, tts_filename)
        import asyncio
        tts_result = asyncio.run(generate_tts_file(ai_response, tts_path))

        timestamp = datetime.now().strftime("%I:%M %p")
        # Store the clean transcript (without image path) in chat history
        ui_state["chat_history"].append({
            "role": "user",
            "timestamp": timestamp,
            "mode": "speech",
            "content": f"🎤 {clean_transcript}"
        })
        ui_state["chat_history"].append({
            "role": "assistant",
            "timestamp": timestamp,
            "mode": "speech",
            "content": ai_response
        })

        if os.path.exists(DASHBOARD_PATH):
            ui_state["dashboard_notification"] = True

        jobs[job_id] = {
            "status": "done",
            "transcript": clean_transcript,  # Send clean transcript to frontend
            "response_text": ai_response,
            "audio_url": f"/display/{tts_filename}" if tts_result else None,
            "created_at": time.time()
        }

    except Exception as e:
        jobs[job_id] = {"status": "error", "error": str(e), "created_at": time.time()}
        import traceback
        traceback.print_exc()

# ============================
# SHUTDOWN CLEANUP LOGIC (kept)
# ============================
def cleanup_all_agents():
    print("\n🧹 Cleaning up all agent sessions on shutdown...")
    for location, master in master_instances.items():
        try:
            master.cleanup()
        except Exception as e:
            print(f"Error cleaning up {location}: {e}")

    try:
        if os.path.exists(UPLOAD_DIR):
            shutil.rmtree(UPLOAD_DIR)
            os.makedirs(UPLOAD_DIR, exist_ok=True)
    except Exception as e:
        print(f"Error clearing upload directory: {e}")

    print("Shutdown complete.")

atexit.register(cleanup_all_agents)

# ============================
# HELPER FUNCTIONS (unchanged)
# ============================
def get_master_agent(location: str = None) -> WasteDispoMaster:
    if not location:
        return None
    if location not in master_instances:
        print(f"🌟 Booting new Master Agent for {location}...")
        try:
            master_instances[location] = WasteDispoMaster(default_location=location)
        except Exception as e:
            print(f"Error booting agent: {e}")
            return None
    return master_instances[location]

def clear_master_cache(location: str):
    if location in master_instances:
        master = master_instances[location]
        cache = master._get_cache()
        cache.pop("env", None)
        print(f"🗑️ Cache cleared for location: {location}")

# ============================
# ROUTES (existing unchanged except additions)
# ============================
@app.route("/update-location", methods=["POST"])
def update_location():
    data = request.json
    new_location = data.get("location_name")
    old_location = session.get("location")
    if old_location and old_location != new_location:
        clear_master_cache(old_location)
        if old_location in master_instances:
            try:
                master_instances[old_location].cleanup()
            except Exception:
                pass
            del master_instances[old_location]
    session["location"] = new_location
    if new_location in master_instances:
        clear_master_cache(new_location)
    return jsonify({"success": True, "location": new_location})

@app.before_request
def initialize_session():
    if "mode" not in session:
        session["mode"] = "chat"
    if "location" not in session:
        session["location"] = None

@app.route("/")
def index():
    system_name = os.getenv("SUSTAINAI_SYSTEM_NAME", "SustainAi")
    active_mode = session.get("mode", "chat")
    active_mode_label = "Chat Mode" if active_mode == "chat" else "Voice Mode"
    dashboard_ready = os.path.exists(DASHBOARD_PATH)
    locationiq_key = os.getenv("LOCATIONIQ_KEY")
    show_dashboard_toast = ui_state.pop("dashboard_notification", False)
    return render_template(
        "index.html",
        system_name=system_name,
        active_mode=active_mode,
        active_mode_label=active_mode_label,
        speech_state=ui_state["speech_state"],
        chat_history=ui_state["chat_history"],
        dashboard_ready=dashboard_ready,
        locationiq_key=locationiq_key,
        show_dashboard_toast=show_dashboard_toast
    )

@app.route("/dashboard")
def dashboard():
    if os.path.exists(DASHBOARD_PATH):
        return send_file(DASHBOARD_PATH)
    return "Dashboard not generated yet.", 404

@app.route("/check-dashboard")
def check_dashboard():
    if os.path.exists(DASHBOARD_PATH):
        return jsonify({
            "ready": True,
            "last_modified": os.path.getmtime(DASHBOARD_PATH)
        })
    return jsonify({"ready": False, "last_modified": 0})

@app.route("/display/<filename>")
def serve_display_file(filename):
    file_path = os.path.join(UPLOAD_DIR, filename)
    if os.path.exists(file_path):
        return send_file(file_path)
    return "File not found", 404

@app.route("/set-mode", methods=["POST"])
def set_mode():
    mode = request.form.get("mode", "chat")
    session["mode"] = mode
    return redirect(url_for("index"))

@app.route("/clear-chat", methods=["POST"])
def clear_chat():
    ui_state["chat_history"] = []
    location = session.get("location")
    if location and location in master_instances:
        master = master_instances[location]
        master.cleanup()
        del master_instances[location]
    try:
        if os.path.exists(DASHBOARD_PATH):
            os.remove(DASHBOARD_PATH)
    except Exception:
        pass
    session.pop("dashboard_notification", None)
    return redirect(url_for("index"))

# --- Legacy synchronous endpoints (kept for fallback) ---
@app.route("/chat", methods=["POST"])
def chat():
    # This legacy route is kept for backward compatibility.
    location = session.get("location")
    if not location:
        return jsonify({"success": False, "error": "location_not_set"}), 400
    master = get_master_agent(location)
    if not master:
        return jsonify({"success": False, "error": "agent_init_failed"}), 500
    query = request.form.get("query", "").strip()
    image_file = request.files.get("image")
    if not query and not image_file:
        return jsonify({"success": False, "error": "empty_query"}), 400
    file_path_str = ""
    if image_file and image_file.filename:
        filename = secure_filename(f"upload_{uuid.uuid4().hex}_{image_file.filename}")
        save_path = os.path.join(UPLOAD_DIR, filename)
        image_file.save(save_path)
        file_path_str = f" {save_path}"
        master.context.setdefault("created_files", []).append(save_path)
    full_query = f"{query}{file_path_str}".strip()
    status_phrases = ["status update", "what is the status", "show status", "system status"]
    if any(phrase in full_query.lower() for phrase in status_phrases):
        ai_response = master.get_status_update()
    else:
        ai_response = master.process_input(full_query)
    timestamp = datetime.now().strftime("%I:%M %p")
    ui_state["chat_history"].append({
        "role": "user",
        "timestamp": timestamp,
        "mode": "chat",
        "content": full_query
    })
    ui_state["chat_history"].append({
        "role": "assistant",
        "timestamp": timestamp,
        "mode": "chat",
        "content": ai_response
    })
    if os.path.exists(DASHBOARD_PATH):
        ui_state["dashboard_notification"] = True
    return jsonify({"success": True, "response": ai_response})

@app.route("/upload-image", methods=["POST"])
def upload_image():
    image_file = request.files.get("image")
    if not image_file or not image_file.filename:
        return jsonify({"success": False, "error": "No file provided"}), 400
    filename = secure_filename(f"speech_{uuid.uuid4().hex}_{image_file.filename}")
    save_path = os.path.join(UPLOAD_DIR, filename)
    image_file.save(save_path)
    ui_state["speech_image_path"] = save_path
    preview_path = f"/display/{filename}"
    return jsonify({"success": True, "path": preview_path})

@app.route("/start-speech", methods=["POST"])
def start_speech():
    # Legacy simulated speech route
    master = get_master_agent(session.get("location"))
    ui_state["speech_state"]["active"] = True
    ui_state["speech_state"]["status"] = "Listening (Simulated)..."
    ui_state["speech_state"]["error"] = None
    simulated_transcript = "What kind of waste is this?"
    if ui_state["speech_image_path"]:
        simulated_transcript += f" {ui_state['speech_image_path']}"
        ui_state["speech_image_path"] = None
    timestamp = datetime.now().strftime("%I:%M %p")
    ui_state["chat_history"].append({
        "role": "user",
        "timestamp": timestamp,
        "mode": "speech",
        "content": f"🎤 {simulated_transcript}"
    })
    ai_response = master.process_input(simulated_transcript)
    ui_state["chat_history"].append({
        "role": "assistant",
        "timestamp": timestamp,
        "mode": "speech",
        "content": ai_response
    })
    ui_state["speech_state"]["active"] = False
    ui_state["speech_state"]["status"] = "Processing complete."
    return redirect(url_for("index"))

@app.route("/stop-speech", methods=["POST"])
def stop_speech():
    ui_state["speech_state"]["active"] = False
    ui_state["speech_state"]["status"] = "Idle"
    return redirect(url_for("index"))

# --- NEW ASYNC ENDPOINTS ---

@app.route("/process-chat", methods=["POST"])
def process_chat():
    location = session.get("location")
    if not location:
        return jsonify({"success": False, "error": "location_not_set"}), 400

    query = request.form.get("query", "").strip()
    image_file = request.files.get("image")

    if not query and not image_file:
        return jsonify({"success": False, "error": "empty_query"}), 400

    image_path = None
    if image_file and image_file.filename:
        filename = secure_filename(f"upload_{uuid.uuid4().hex}_{image_file.filename}")
        image_path = os.path.join(UPLOAD_DIR, filename)
        image_file.save(image_path)

    job_id = str(uuid.uuid4())
    jobs[job_id] = {"status": "queued", "progress": "Queued...", "created_at": time.time()}

    thread = threading.Thread(
        target=run_chat_pipeline,
        args=(job_id, query, image_path, location)
    )
    thread.daemon = True
    thread.start()

    return jsonify({"job_id": job_id})

@app.route("/process-voice-input", methods=["POST"])
def process_voice_input_async():
    if 'audio' not in request.files:
        return jsonify({"success": False, "error": "No audio file"}), 400

    location = session.get("location")
    if not location:
        return jsonify({"success": False, "error": "location_not_set"}), 400

    audio_file = request.files['audio']
    mime_type = request.form.get('mime_type', 'audio/webm')

    ext = 'wav' if 'wav' in mime_type else 'webm' if 'webm' in mime_type else 'ogg' if 'ogg' in mime_type else 'webm'
    audio_path = os.path.join(UPLOAD_DIR, f"temp_input_{uuid.uuid4().hex}.{ext}")
    audio_file.save(audio_path)

    image_path = ui_state.get("speech_image_path")
    if image_path:
        ui_state["speech_image_path"] = None

    job_id = str(uuid.uuid4())
    jobs[job_id] = {"status": "queued", "progress": "Queued...", "created_at": time.time()}

    thread = threading.Thread(
        target=run_voice_pipeline,
        args=(job_id, audio_path, mime_type, location, image_path)
    )
    thread.daemon = True
    thread.start()

    return jsonify({"job_id": job_id})

@app.route("/job/<job_id>", methods=["GET"])
def get_job_status(job_id):
    job = jobs.get(job_id)
    if not job:
        return jsonify({"status": "not_found"}), 404

    if job["status"] == "done":
        return jsonify({
            "status": "done",
            "response": job.get("response"),
            "transcript": job.get("transcript"),
            "response_text": job.get("response_text"),
            "audio_url": job.get("audio_url")
        })
    elif job["status"] == "error":
        return jsonify({
            "status": "error",
            "error": job.get("error", "Unknown error")
        })
    else:
        return jsonify({
            "status": job["status"],
            "progress": job.get("progress", "Processing...")
        })

if __name__ == "__main__":
    if not os.environ.get("WERKZEUG_RUN_MAIN"):
        print("🚀 Starting SustainAi Flask Interface...")
    app.run(debug=True, host="0.0.0.0", port=5000)