import speech_recognition as sr
import edge_tts
import asyncio
import os
import re
import unicodedata
import warnings

warnings.filterwarnings(
    "ignore",
    message="pkg_resources is deprecated as an API.*",
    category=UserWarning,
)

# ==========================================
# CONFIGURATION
# ==========================================
# Voice: AvaNeural is a natural young adult female voice
VOICE = os.getenv("SUSTAINAI_TTS_VOICE", "en-US-AvaNeural")
RATE = os.getenv("SUSTAINAI_TTS_RATE", "+0%")   # Neutral speech sounds more natural.
PITCH = os.getenv("SUSTAINAI_TTS_PITCH", "+0Hz") # Neutral pitch for a natural adult tone
OUTPUT_FILE = "response.mp3"

# ==========================================
# TEXT-TO-SPEECH (TTS) SECTION
# ==========================================
def _normalize_for_speech(text):
    if text is None:
        return ""

    text = str(text)
    text = unicodedata.normalize("NFKC", text)

    replacements = {
        "&": " and ",
        "%": " percent ",
        "$": " dollars ",
        "#": " number ",
        "@": " at ",
        "°": " degrees ",
        "→": " to ",
        "←": " from ",
        "×": " times ",
        "÷": " divided by ",
        "±": " plus or minus ",
        "≈": " approximately ",
        "≤": " less than or equal to ",
        "≥": " greater than or equal to ",
        "µ": " micro ",
        "•": " ",
        "–": " ",
        "—": " ",
        "…": " ",
    }
    for source, target in replacements.items():
        text = text.replace(source, target)

    text = re.sub(r"```.*?```", " ", text, flags=re.DOTALL)
    text = re.sub(r"`([^`]*)`", r"\1", text)
    text = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r"\1", text)
    text = re.sub(r"\*\*([^*]+)\*\*", r"\1", text)
    text = re.sub(r"\*([^*]+)\*", r"\1", text)
    text = re.sub(r"__([^_]+)__", r"\1", text)
    text = re.sub(r"_([^_]+)_", r"\1", text)
    text = re.sub(r"https?://\S+", " ", text)
    text = re.sub(r"\[[^\]]*\]", " ", text)
    text = re.sub(r"[^\w\s.,!?;:'\"()/-]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _normalize_rate(rate_value):
    rate_text = str(rate_value or "+0%").strip()
    if rate_text in {"0", "0%", "+0", "+0%"}:
        return "+0%"
    if re.fullmatch(r"\d+%", rate_text):
        return f"+{rate_text}"
    if re.fullmatch(r"[+-]\d+%", rate_text):
        return rate_text
    return "+0%"


async def _generate_audio(text):
    """Internal async function to create the mp3 file using Edge TTS"""
    clean_text = _normalize_for_speech(text)
    communicate = edge_tts.Communicate(clean_text, VOICE, rate=_normalize_rate(RATE), pitch=PITCH)
    await communicate.save(OUTPUT_FILE)

def play_audio():
    """Plays the generated mp3 file and cleans up"""
    import pygame

    pygame.mixer.init()
    pygame.mixer.music.load(OUTPUT_FILE)
    pygame.mixer.music.play()
    while pygame.mixer.music.get_busy():
        pass
    pygame.mixer.quit() 
    try:
        os.remove(OUTPUT_FILE) # Clean up the file after playing
    except OSError:
        pass

def transcribe_audio_file(file_path):
    """Transcribes an audio file (wav) into text"""
    recognizer = sr.Recognizer()
    try:
        with sr.AudioFile(file_path) as source:
            audio = recognizer.record(source)
            text = recognizer.recognize_google(audio)
            return text
    except Exception as e:
        print(f"Transcription failed: {e}")
        return None

async def generate_tts_file(text, output_path):
    """Generates an mp3 file path for the browser to play"""
    clean_text = _normalize_for_speech(text)
    if not clean_text:
        return None
    
    communicate = edge_tts.Communicate(clean_text, VOICE, rate=_normalize_rate(RATE), pitch=PITCH)
    await communicate.save(output_path)
    return output_path

def speak(text):
    """Main function to convert text to speech and play it"""
    clean_text = _normalize_for_speech(text)
    if not clean_text:
        return

    print(f"AI: {clean_text}")
    try:
        asyncio.run(_generate_audio(clean_text))
        play_audio()
    except Exception as error:
        print(f"AI: Speech playback failed: {error}")


