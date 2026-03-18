# tts_indic_multi.py  ← REPLACE YOUR OLD FILE WITH THIS
import asyncio
import edge_tts
import pygame
import pyttsx3
from gtts import gTTS
import os
import threading
import uuid

pygame.mixer.init(frequency=44100, size=-16, channels=2, buffer=512)

from sign_constants import LANGUAGES, LANG_CODES, TRANSLATIONS

last_spoken = None
lock = threading.Lock()

async def _speak_async(label: str, lang_idx: int):
    global last_spoken
    lang_code = LANG_CODES[lang_idx]
    
    # Get text based on label and language
    # For english, default to label if not found. For others, default to empty or handle it.
    lang_dict = TRANSLATIONS.get(label.lower())
    if lang_dict:
        text = lang_dict.get(lang_code, label)
    else:
        text = label

    # Optimization: duplicate check
    with lock:
        if last_spoken == (label, lang_idx):
            return
        last_spoken = (label, lang_idx)

    tmp_file = f"temp_tts_{uuid.uuid4().hex}.mp3"
    
    try:
        print(f"Generating gTTS for: '{text}' in {lang_code}...")
        
        # Use gTTS (Google Translate TTS) - Sync operation wrapped in executor if needed, 
        # but here we are in async wrapper, so it blocks this thread (which is fine, it's a daemon thread).
        tts = gTTS(text=text, lang=lang_code, slow=False)
        tts.save(tmp_file)
        
        pygame.mixer.music.load(tmp_file)
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():
            await asyncio.sleep(0.1)
            
    except Exception as e:
        print(f"Warning: Online gTTS failed ({e}). Switching to Offline TTS for: '{text}'")
        try:
            # Fallback to offline TTS (pyttsx3)
            # Note: pyttsx3 generally only runs english if language packs aren't installed.
            engine = pyttsx3.init()
            engine.say(text)
            engine.runAndWait()
        except Exception as offline_e:
            print(f"Error: Offline TTS also failed: {offline_e}")
            
    finally:
        try:
            if os.path.exists(tmp_file):
                pygame.mixer.music.unload() # Ensure file is released
                os.remove(tmp_file)
        except Exception as e:
            print(f"Error cleaning up TTS file: {e}")

def speak_sign(label: str, lang_idx: int):
    global last_spoken
    if not label:
        return
        
    if last_spoken == (label, lang_idx):
        return

    # Run in a separate thread to not block GUI
    threading.Thread(
        target=lambda: asyncio.run(_speak_async(label, lang_idx)),
        daemon=True
    ).start()