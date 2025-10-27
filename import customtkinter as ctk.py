import customtkinter as ctk
from kittentts import KittenTTS
import whisper
import sounddevice as sd
import soundfile as sf
import numpy as np
import pygame
import threading
import queue
import os
import sys
from scipy.io.wavfile import write

# ------------------ CONFIG FOR PI ------------------
WHISPER_MODEL = "tiny"        # Use 'tiny' or 'base' only on Pi
TTS_MODEL_NAME = "KittenML/kitten-tts-nano-0.2"
VOICE = 'expr-voice-5-m'      # Fast male voice
SAMPLE_RATE = 16000           # 16kHz for whisper
TTS_SAMPLE_RATE = 24000
RECORD_DURATION = 5           # seconds
AUDIO_FILE = "/tmp/recording.wav"
TTS_FILE = "/tmp/tts_output.wav"

# ------------------ Global Models (Loaded Once) ------------------
print("Loading models... (this may take 30-60 seconds on Pi)")

# Load Whisper (tiny is ~75MB, works on Pi 2GB)
whisper_model = whisper.load_model(WHISPER_MODEL)

# Load KittenTTS
tts = KittenTTS(TTS_MODEL_NAME)

# Initialize pygame mixer once
pygame.mixer.pre_init(frequency=TTS_SAMPLE_RATE, size=-16, channels=1, buffer=512)
pygame.mixer.init()
pygame.mixer.music.set_volume(0.8)

# ------------------ Text Redirector for GUI Console ------------------
class TextRedirector:
    def __init__(self, text_widget):
        self.text_widget = text_widget
        self.queue = queue.Queue()
        self.start_checker()

    def write(self, string):
        self.queue.put(string)

    def flush(self):
        pass

    def start_checker(self):
        def check():
            try:
                while True:
                    msg = self.queue.get_nowait()
                    self.text_widget.configure(state="normal")
                    self.text_widget.insert("end", msg)
                    self.text_widget.see("end")
                    self.text_widget.configure(state="disabled")
            except queue.Empty:
                pass
            self.text_widget.after(100, check)
        self.text_widget.after(100, check)

# ------------------ Core Translation Logic ------------------
def record_and_translate():
    def run():
        try:
            update_status("Recording... (speak now)")

            # Record audio (16kHz, mono)
            recording = sd.rec(int(RECORD_DURATION * SAMPLE_RATE),
                               samplerate=SAMPLE_RATE, channels=1, dtype='int16')
            sd.wait()
            write(AUDIO_FILE, SAMPLE_RATE, recording)

            update_status("Transcribing...")

            # Transcribe + Translate (Hindi → English)
            result = whisper_model.transcribe(
                AUDIO_FILE,
                language="hi",
                task="translate",
                fp16=False  # Pi has no CUDA
            )
            english_text = result["text"].strip()

            if not english_text:
                update_status("No speech detected.")
                return

            update_status(f"Speaking: {english_text[:50]}...")

            # Generate speech
            audio = tts.generate(english_text, voice=VOICE)
            audio = np.array(audio, dtype=np.float32)

            # Save and play
            sf.write(TTS_FILE, audio, TTS_SAMPLE_RATE)
            play_audio(TTS_FILE)

            update_status(f"Done: {english_text}")

        except Exception as e:
            update_status(f"Error: {str(e)}")

    # Run in background thread
    threading.Thread(target=run, daemon=True).start()

# ------------------ Audio Playback (Optimized) ------------------
def play_audio(file_path):
    def play():
        try:
            pygame.mixer.music.load(file_path)
            pygame.mixer.music.play()
            while pygame.mixer.music.get_busy():
                pygame.time.wait(100)
        except Exception as e:
            print(f"Playback failed: {e}")

    threading.Thread(target=play, daemon=True).start()

# ------------------ GUI Update Helper ------------------
def update_status(text):
    print(text + "\n")

# ------------------ Startup Welcome (After GUI) ------------------
def play_welcome():
    try:
        audio = tts.generate("Welcome to offline Hindi to English translator.", voice=VOICE)
        audio = np.array(audio, dtype=np.float32)
        sf.write(TTS_FILE, audio, TTS_SAMPLE_RATE)
        play_audio(TTS_FILE)
    except:
        pass  # Silent fail if startup slow

# ------------------ GUI Setup ------------------
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

app = ctk.CTk()
app.geometry("480x640")  # Pi touchscreen friendly
app.title("Pi Translator: HI → EN")

# Title
title = ctk.CTkLabel(app, text="Hindi → English", font=("Arial", 24, "bold"))
title.pack(pady=20)

# Status / Output
output_text = ctk.CTkTextbox(app, width=440, height=300, font=("Arial", 12))
output_text.pack(padx=20, pady=10)
output_text.configure(state="disabled")

# Redirect print
sys.stdout = TextRedirector(output_text)

# Button
btn = ctk.CTkButton(
    app,
    text="Press & Speak Hindi",
    font=("Arial", 18),
    height=60,
    command=record_and_translate
)
btn.pack(pady=20)

# Footer
footer = ctk.CTkLabel(app, text=f"Whisper: {WHISPER_MODEL} | TTS: nano", font=("Arial", 10), text_color="gray")
footer.pack(side="bottom", pady=10)

# ------------------ Start App ------------------
app.after(500, play_welcome)  # Play welcome after GUI loads
app.mainloop()

# ------------------ Cleanup ------------------
pygame.mixer.quit()
if os.path.exists(AUDIO_FILE):
    os.remove(AUDIO_FILE)
if os.path.exists(TTS_FILE):
    os.remove(TTS_FILE)