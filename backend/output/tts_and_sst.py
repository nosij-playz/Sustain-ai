import speech_recognition as sr
import edge_tts
import asyncio
import os
import re
import uuid
import unicodedata
import warnings
from aiohttp import ClientConnectorError

warnings.filterwarnings(
    "ignore",
    message="pkg_resources is deprecated as an API.*",
    category=UserWarning,
)

# ==========================================
# CONFIGURATION
# ==========================================
VOICE = os.getenv("SUSTAINAI_TTS_VOICE", "en-US-AvaNeural")
RATE = os.getenv("SUSTAINAI_TTS_RATE", "+0%")
PITCH = os.getenv("SUSTAINAI_TTS_PITCH", "+0Hz")
OUTPUT_DIR = os.getenv("SUSTAINAI_TTS_DIR", "./backend/display")


# ==========================================
# TEXT-TO-SPEECH (TTS) SECTION
# ==========================================
def _normalize_for_speech(text: str) -> str:
    """
    Clean and normalize text for speech synthesis.
    Removes markdown, symbols, and other artifacts that don't sound good when spoken.
    """
    if text is None:
        return ""

    text = str(text)
    text = unicodedata.normalize("NFKC", text)

    # Common symbol replacements
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

    # Remove markdown and code blocks
    text = re.sub(r"```.*?```", " ", text, flags=re.DOTALL)
    text = re.sub(r"`([^`]*)`", r"\1", text)
    text = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r"\1", text)
    text = re.sub(r"\*\*([^*]+)\*\*", r"\1", text)
    text = re.sub(r"\*([^*]+)\*", r"\1", text)
    text = re.sub(r"__([^_]+)__", r"\1", text)
    text = re.sub(r"_([^_]+)_", r"\1", text)

    # Remove URLs and brackets
    text = re.sub(r"https?://\S+", " ", text)
    text = re.sub(r"\[[^\]]*\]", " ", text)

    # Keep only speech-friendly characters
    text = re.sub(r"[^\w\s.,!?;:'\"()/-]", " ", text)
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def _normalize_rate(rate_value: str) -> str:
    """
    Normalize the speech rate to Edge TTS format.
    Edge TTS expects rates like "+0%", "-10%", "+15%"
    """
    rate_text = str(rate_value or "+0%").strip()

    if rate_text in {"0", "0%", "+0", "+0%"}:
        return "+0%"
    if re.fullmatch(r"\d+%", rate_text):
        return f"+{rate_text}"
    if re.fullmatch(r"[+-]\d+%", rate_text):
        return rate_text

    return "+0%"


async def generate_tts_file(text: str, output_path: str = None) -> str:
    """
    Generate an MP3 file from text using Edge TTS.
    Returns the path to the generated file.

    Args:
        text: The text to convert to speech
        output_path: Optional custom path. If not provided, generates a unique filename.

    Returns:
        str: Path to the generated MP3 file, or None if generation failed.
    """
    clean_text = _normalize_for_speech(text)

    if not clean_text:
        print("⚠️ No text to convert to speech")
        return None

    # Ensure output directory exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Generate unique filename if not provided
    if output_path is None:
        filename = f"tts_{uuid.uuid4().hex}.mp3"
        output_path = os.path.join(OUTPUT_DIR, filename)

    try:
        communicate = edge_tts.Communicate(
            clean_text,
            VOICE,
            rate=_normalize_rate(RATE),
            pitch=PITCH
        )
        await communicate.save(output_path)
        print(f"✅ TTS audio generated: {output_path}")
        return output_path

    except (ClientConnectorError, TimeoutError, OSError, RuntimeError) as exc:
        print(f"❌ TTS generation failed: {exc}")
        return None


def transcribe_audio_file(file_path: str) -> str:
    """
    Transcribe an audio file (WAV format) to text using Google Speech Recognition.

    Args:
        file_path: Path to the audio file (must be WAV format)

    Returns:
        str: Transcribed text, or None if transcription failed.
    """
    if not os.path.exists(file_path):
        print(f"❌ Audio file not found: {file_path}")
        return None

    recognizer = sr.Recognizer()

    try:
        with sr.AudioFile(file_path) as source:
            # Adjust for ambient noise
            recognizer.adjust_for_ambient_noise(source, duration=0.5)
            audio = recognizer.record(source)

            # Use Google's speech recognition
            text = recognizer.recognize_google(audio)
            print(f"🎤 Transcription: {text}")
            return text

    except sr.UnknownValueError:
        print("❌ Speech recognition could not understand audio")
        return None
    except sr.RequestError as e:
        print(f"❌ Speech recognition service error: {e}")
        return None
    except Exception as e:
        print(f"❌ Transcription failed: {e}")
        return None


# ==========================================
# BACKWARD COMPATIBILITY (DEPRECATED)
# ==========================================
def speak(text: str) -> str:
    """
    DEPRECATED: Legacy function that generates TTS and returns the file path.
    Use generate_tts_file() instead for new code.
    """
    import asyncio

    clean_text = _normalize_for_speech(text)
    if not clean_text:
        return None

    filename = f"tts_{uuid.uuid4().hex}.mp3"
    output_path = os.path.join(OUTPUT_DIR, filename)

    try:
        asyncio.run(generate_tts_file(clean_text, output_path))
        return output_path
    except Exception as error:
        print(f"❌ Speech generation failed: {error}")
        return None
