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
audio_cache = {}  # Cache mapping (text, lang_code) -> audio_filepath

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

    cache_key = (text, lang_code)
    
    # ⚡ INSTANT PLAYBACK FROM CACHE ⚡
    if cache_key in audio_cache and os.path.exists(audio_cache[cache_key]):
        try:
            pygame.mixer.music.load(audio_cache[cache_key])
            pygame.mixer.music.play()
            while pygame.mixer.music.get_busy():
                await asyncio.sleep(0.05)
            # Unload after playing to prevent locking issues
            pygame.mixer.music.unload()
            return
        except Exception:
            pass # fallback to generate anew

    tmp_file = f"temp_tts_{uuid.uuid4().hex}.mp3"
    
    try:
        print(f"Generating gTTS for: '{text}' in {lang_code}...")
        
        # Use gTTS (Google Translate TTS)
        tts = gTTS(text=text, lang=lang_code, slow=False)
        tts.save(tmp_file)
        
        # Save to cache instead of deleting
        audio_cache[cache_key] = tmp_file
        
        pygame.mixer.music.load(tmp_file)
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():
            await asyncio.sleep(0.05)
        
        # Unload music so we don't lock the file
        pygame.mixer.music.unload()
            
    except Exception as e:
        print(f"Warning: Online gTTS failed ({e}). Switching to Offline TTS for: '{text}'")
        try:
            # Fallback to offline TTS (pyttsx3)
            engine = pyttsx3.init()
            engine.say(text)
            engine.runAndWait()
        except Exception as offline_e:
            print(f"Error: Offline TTS also failed: {offline_e}")

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