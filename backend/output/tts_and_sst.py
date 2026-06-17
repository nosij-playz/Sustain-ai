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

# ==========================================
# SPEECH-TO-TEXT (STT) SECTION
# ==========================================
def listen():
    """Listens to microphone and returns recognized text"""
    recognizer = sr.Recognizer()
    
    max_retries = 3
    for attempt in range(max_retries):
        if attempt > 0:
            print(f"Retrying... ({attempt + 1}/{max_retries})")

        with sr.Microphone() as source:
            print("\nListening... (Speak now)")
            recognizer.adjust_for_ambient_noise(source, duration=3)
            try:
                audio = recognizer.listen(source, timeout=5, phrase_time_limit=8)
            except sr.WaitTimeoutError:
                if attempt < max_retries - 1:
                    import time
                    time.sleep(0.5)
                continue

        try:
            text = recognizer.recognize_google(audio)
            print(f"You: {text}")
            return text
        except sr.UnknownValueError:
            print("AI: I didn't quite catch that.")
            if attempt < max_retries - 1:
                import time
                time.sleep(0.5)
            continue
        except sr.RequestError:
            print("AI: System Error: Could not connect to the speech service.")
            if attempt < max_retries - 1:
                import time
                time.sleep(0.5)
            continue

    print("AI: No input detected after retries.")
    return None

