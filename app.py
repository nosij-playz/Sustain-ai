import os
import uuid
import shutil
import atexit # <--- Add this import
from datetime import datetime
from flask import Flask, render_template, request, redirect, url_for, jsonify, session, send_file
from werkzeug.utils import secure_filename
from pydub import AudioSegment

# Import your existing Master agent logic
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

# --- SHUTDOWN CLEANUP LOGIC ---
def cleanup_all_agents():
    """Performs cleanup for all active master agent sessions on exit."""
    print("\n🧹 Cleaning up all agent sessions on shutdown...")
    for location, master in master_instances.items():
        try:
            print(f"--> Cleaning up: {location}")
            master.cleanup()
        except Exception as e:
            print(f"Error cleaning up {location}: {e}")
    print("Shutdown complete.")

# Register the cleanup function to run when the script exits
atexit.register(cleanup_all_agents)
# ------------------------------

def get_master_agent(location: str = "Chittarikkal, Kerala, India") -> WasteDispoMaster:
    if location not in master_instances:
        print(f"🌟 Booting new Master Agent for {location}...")
        master_instances[location] = WasteDispoMaster(default_location=location)
    return master_instances[location]

@app.before_request
def initialize_session():
    if "mode" not in session:
        session["mode"] = "chat"
    if "location" not in session:
        session["location"] = "Chittarikkal, Kerala, India"

@app.route("/")
def index():
    system_name = os.getenv("SUSTAINAI_SYSTEM_NAME", "SustainAi")
    active_mode = session.get("mode", "chat")
    active_mode_label = "Chat Mode" if active_mode == "chat" else "Voice Mode"
    
    dashboard_ready = os.path.exists(DASHBOARD_PATH)

    return render_template(
        "index.html",
        system_name=system_name,
        active_mode=active_mode,
        active_mode_label=active_mode_label,
        speech_state=ui_state["speech_state"],
        chat_history=ui_state["chat_history"],
        dashboard_ready=dashboard_ready
    )

@app.route("/dashboard")
def dashboard():
    if os.path.exists(DASHBOARD_PATH):
        return send_file(DASHBOARD_PATH)
    return "Dashboard not generated yet.", 404

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

@app.route("/chat", methods=["POST"])
def chat():
    master = get_master_agent(session["location"])
    query = request.form.get("query", "").strip()
    image_file = request.files.get("image")

    if not query and not image_file:
        return redirect(url_for("index"))

    file_path_str = ""
    if image_file and image_file.filename:
        filename = secure_filename(f"upload_{uuid.uuid4().hex}_{image_file.filename}")
        save_path = os.path.join(UPLOAD_DIR, filename)
        image_file.save(save_path)
        file_path_str = f" {save_path}" 
        master.context.setdefault("created_files", []).append(save_path)

    full_query = f"{query}{file_path_str}".strip()

    timestamp = datetime.now().strftime("%I:%M %p")
    ui_state["chat_history"].append({
        "role": "user",
        "timestamp": timestamp,
        "mode": "chat",
        "content": full_query
    })

    if any(word in full_query.lower() for word in ["status", "processing", "working", "update"]):
        ai_response = master.get_status_update()
    else:
        ai_response = master.process_input(full_query)

    ui_state["chat_history"].append({
        "role": "assistant",
        "timestamp": datetime.now().strftime("%I:%M %p"),
        "mode": "chat",
        "content": ai_response
    })

    return redirect(url_for("index"))

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
    master = get_master_agent(session["location"])
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
        "timestamp": datetime.now().strftime("%I:%M %p"),
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

@app.route("/clear-chat", methods=["POST"])
def clear_chat():
    master = get_master_agent(session.get("location", "Chittarikkal, Kerala, India"))
    master.cleanup() 
    
    try:
        if os.path.exists(DASHBOARD_PATH):
            os.remove(DASHBOARD_PATH)
    except Exception:
        pass
    
    return redirect(url_for("index"))

@app.route("/process-voice-input", methods=["POST"])
def process_voice_input():
    if 'audio' not in request.files:
        return jsonify({"success": False, "error": "No audio file"}), 400
    
    audio_file = request.files['audio']
    
    # 1. Save the incoming file (e.g., .webm or .ogg)
    temp_path = os.path.join(UPLOAD_DIR, f"temp_input_{uuid.uuid4().hex}.webm")
    audio_file.save(temp_path)
    
    # 2. Convert the file to WAV using pydub
    wav_path = temp_path.replace(".webm", ".wav")
    try:
        audio = AudioSegment.from_file(temp_path)
        audio.export(wav_path, format="wav")
    except Exception as e:
        return jsonify({"success": False, "error": f"Conversion failed: {str(e)}"}), 500
    
    # 3. Transcribe using the new WAV file
    from backend.output.tts_and_sst import transcribe_audio_file, generate_tts_file
    user_text = transcribe_audio_file(wav_path)
    
    # Clean up temp files
    if os.path.exists(temp_path): os.remove(temp_path)
    if os.path.exists(wav_path): os.remove(wav_path)
    
    if not user_text:
        return jsonify({"success": False, "error": "Could not transcribe audio"}), 400
    
    # 4. Get AI Response
    master = get_master_agent(session.get("location", "Chittarikkal, Kerala, India"))
    ai_response = master.process_input(user_text)
    
    # 5. Generate TTS
    tts_filename = f"resp_{uuid.uuid4().hex}.mp3"
    tts_path = os.path.join(UPLOAD_DIR, tts_filename)
    
    import asyncio
    asyncio.run(generate_tts_file(ai_response, tts_path))
    
    return jsonify({
        "success": True,
        "transcript": user_text,
        "response_text": ai_response,
        "audio_url": f"/display/{tts_filename}"
    })
    
    # 1. Transcribe
    from backend.output.tts_and_sst import transcribe_audio_file, generate_tts_file
    user_text = transcribe_audio_file(save_path)
    
    if not user_text:
        return jsonify({"success": False, "error": "Could not transcribe"}), 400
    
    # 2. Get AI Response
    master = get_master_agent(session.get("location", "Chittarikkal, Kerala, India"))
    ai_response = master.process_input(user_text)
    
    # 3. Generate TTS Audio file for the client to play
    tts_filename = f"resp_{uuid.uuid4().hex}.mp3"
    tts_path = os.path.join(UPLOAD_DIR, tts_filename)
    
    import asyncio
    asyncio.run(generate_tts_file(ai_response, tts_path))
    
    # 4. Return results
    return jsonify({
        "success": True,
        "transcript": user_text,
        "response_text": ai_response,
        "audio_url": f"/display/{tts_filename}"
    })

if __name__ == "__main__":
    if not os.environ.get("WERKZEUG_RUN_MAIN"):
        print("🚀 Starting SustainAi Flask Interface...")
    app.run(host="0.0.0.0", port=5000, debug=True)