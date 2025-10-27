#!/usr/bin/env python3
import customtkinter as ctk
from kittentts import KittenTTS
import whisper, sounddevice as sd, soundfile as sf
import numpy as np, pygame, threading, queue, os, sys
from indic_transliteration import sanscript

# ────────────────────── CONFIG ──────────────────────
WHISPER_MODEL   = "tiny"                     # tiny = 75 MB, base = 150 MB
TTS_MODEL       = "KittenML/kitten-tts-nano-0.2"
VOICE_EN        = "expr-voice-5-m"
VOICE_HI        = "expr-voice-5-f"
REC_SR          = 16000
TTS_SR          = 24000
REC_SECONDS     = 5
REC_FILE        = "/tmp/rec.wav"
TTS_FILE        = "/tmp/tts.wav"

# ────────────────────── LOAD MODELS ONCE ──────────────────────
print("Loading models (≈30‑60 s on Pi)…")
whisper_model = whisper.load_model(WHISPER_MODEL)
tts = KittenTTS(TTS_MODEL)

# pygame (small buffer for Pi)
pygame.mixer.pre_init(frequency=TTS_SR, size=-16, channels=1, buffer=512)
pygame.mixer.init()
pygame.mixer.music.set_volume(0.9)

# ────────────────────── TEXT REDIRECTOR ──────────────────────
class TextRedirector:
    def __init__(self, widget):
        self.widget = widget
        self.q = queue.Queue()
        self._start()
    def write(self, s): self.q.put(s)
    def flush(self): pass
    def _start(self):
        def poll():
            try:
                while True:
                    txt = self.q.get_nowait()
                    self.widget.configure(state="normal")
                    self.widget.insert("end", txt)
                    self.widget.see("end")
                    self.widget.configure(state="disabled")
            except queue.Empty: pass
            self.widget.after(100, poll)
        self.widget.after(100, poll)

# ────────────────────── HELPERS ──────────────────────
def transliterate_en_to_hi(text: str) -> str:
    """ITRANS → Devanagari (offline)"""
    return sanscript.transliterate(text, sanscript.ITRANS, sanscript.DEVANAGARI)

def play_wav(path: str):
    def _play():
        try:
            pygame.mixer.music.load(path)
            pygame.mixer.music.play()
            while pygame.mixer.music.get_busy():
                pygame.time.wait(50)
        except: pass
    threading.Thread(target=_play, daemon=True).start()

def update(txt: str):
    print(txt + "\n")

# ────────────────────── CORE LOGIC ──────────────────────
mode = "hi2en"          # "hi2en" or "en2hi"

def set_mode(m):
    global mode
    mode = m
    btn.configure(text=f"Speak {'Hindi' if m=='hi2en' else 'English'}")
    update(f"Mode → {'Hindi → English' if m=='hi2en' else 'English → Hindi'}")

def record_and_process():
    def run():
        try:
            update("Recording…")
            rec = sd.rec(int(REC_SECONDS * REC_SR), samplerate=REC_SR,
                         channels=1, dtype="int16")
            sd.wait()
            sf.write(REC_FILE, rec, REC_SR)

            # ---------- Whisper ----------
            if mode == "hi2en":
                res = whisper_model.transcribe(REC_FILE, language="hi",
                                               task="translate", fp16=False)
                out_txt = res["text"].strip()
                voice = VOICE_EN
                update(f"HI: {res['text']}\nEN: {out_txt}")
            else:
                res = whisper_model.transcribe(REC_FILE, language="en",
                                               task="transcribe", fp16=False)
                en = res["text"].strip()
                hi = transliterate_en_to_hi(en)
                out_txt = hi
                voice = VOICE_HI
                update(f"EN: {en}\nHI: {hi}")

            if not out_txt:
                update("No speech detected.")
                return

            # ---------- TTS ----------
            wav = np.array(tts.generate(out_txt, voice=voice), dtype=np.float32)
            sf.write(TTS_FILE, wav, TTS_SR)
            play_wav(TTS_FILE)

        except Exception as e:
            update(f"Error: {e}")

    threading.Thread(target=run, daemon=True).start()

# ────────────────────── STARTUP WELCOME ──────────────────────
def welcome():
    txt = "Welcome to the offline bidirectional translator."
    wav = np.array(tts.generate(txt, voice=VOICE_EN), dtype=np.float32)
    sf.write(TTS_FILE, wav, TTS_SR)
    play_wav(TTS_FILE)

# ────────────────────── GUI ──────────────────────
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")
app = ctk.CTk()
app.geometry("480x720")
app.title("Pi Translator")

ctk.CTkLabel(app, text="Offline Translator", font=("Arial", 24, "bold")).pack(pady=15)

# mode selector
frm = ctk.CTkFrame(app)
frm.pack(pady=8)
rb1 = ctk.CTkRadioButton(frm, text="Hindi → English", value="hi2en",
                         command=lambda: set_mode("hi2en"))
rb2 = ctk.CTkRadioButton(frm, text="English → Hindi", value="en2hi",
                         command=lambda: set_mode("en2hi"))
rb1.pack(side="left", padx=20); rb2.pack(side="left", padx=20)
rb1.select()                     # default

# console
console = ctk.CTkTextbox(app, width=440, height=300, font=("Arial", 11))
console.pack(padx=20, pady=10)
console.configure(state="disabled")
sys.stdout = TextRedirector(console)

# big button
btn = ctk.CTkButton(app, text="Speak Hindi", font=("Arial", 18),
                    height=60, command=record_and_process)
btn.pack(pady=20)

# footer
ctk.CTkLabel(app, text=f"Whisper:{WHISPER_MODEL} • TTS:nano • Pi‑opt",
             font=("Arial", 9), text_color="gray").pack(side="bottom", pady=8)

# ────────────────────── START ──────────────────────
set_mode("hi2en")
app.after(800, welcome)          # play after GUI is shown
app.mainloop()

# ────────────────────── CLEANUP ──────────────────────
pygame.mixer.quit()
for f in (REC_FILE, TTS_FILE):
    if os.path.exists(f):
        os.remove(f)
