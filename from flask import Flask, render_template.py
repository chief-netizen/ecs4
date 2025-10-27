from flask import Flask, render_template_string, redirect, url_for
from kittentts import KittenTTS
import pygame
import whisper
import sounddevice as sd
import numpy as np
from scipy.io.wavfile import write
import soundfile as sf
import threading
import torch
import argostranslate.package
import argostranslate.translate
from transformers import VitsModel, AutoTokenizer
import scipy.io.wavfile as wavfile


# ------------------ Flask App ------------------
app = Flask(__name__)

# ------------------ HTML Template ------------------
HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>Offline Translator</title>
    <style>
        body { font-family: Arial; text-align: center; background: #121212; color: white; margin-top: 50px; }
        button {
            background-color: #0078D4; border: none; color: white;
            padding: 15px 30px; margin: 10px; font-size: 18px; border-radius: 8px;
            cursor: pointer; transition: 0.2s;
        }
        button:hover { background-color: #005EA2; }
        h1 { color: #00BFFF; }
    </style>
</head>
<body>
    <h1>Offline Translator</h1>
    <p>Choose a mode below:</p>
    <form action="/hindi_to_english" method="post">
        <button type="submit">🎙️ Hindi → English</button>
    </form>
    <form action="/english_to_hindi" method="post">
        <button type="submit">🎙️ English → Hindi</button>
    </form>
    <p>{{ message }}</p>
</body>
</html>
"""

# ------------------ Translation Functions ------------------

def play_audio(path):
    pygame.mixer.init()
    pygame.mixer.music.load(path)
    pygame.mixer.music.play()
    while pygame.mixer.music.get_busy():
        pygame.time.Clock().tick(10)
    pygame.mixer.music.stop()
    pygame.mixer.music.unload()

def hindi_to_english():
    try:
        m = KittenTTS("KittenML/kitten-tts-nano-0.2")
        fs = 16000
        duration = 10
        print("Recording Hindi speech for 10s...")
        recording = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='int16')
        sd.wait()
        write('output.wav', fs, recording)
        print("Audio saved, transcribing...")

        model = whisper.load_model("small")
        audio = whisper.load_audio("output.wav")
        audio = whisper.pad_or_trim(audio)
        mel = whisper.log_mel_spectrogram(audio).to(model.device)
        options = whisper.DecodingOptions(task="translate", language="hi")
        result = whisper.decode(model, mel, options)
        print("Translated text:", result.text)

        audio = m.generate(result.text, voice='expr-voice-5-m')
        sf.write('kittensaudio.wav', audio.astype('float32'), 24000)
        play_audio("kittensaudio.wav")
    except Exception as e:
        print("Error:", e)

def english_to_hindi():
    try:
        fs = 24000
        duration = 10
        print("Recording English speech for 10s...")
        recording = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='int16')
        sd.wait()
        write('output.wav', fs, recording)
        print("Transcribing...")

        model = whisper.load_model("medium")
        audio = whisper.load_audio("output.wav")
        audio = whisper.pad_or_trim(audio)
        mel = whisper.log_mel_spectrogram(audio).to(model.device)
        options = whisper.DecodingOptions()
        result = whisper.decode(model, mel, options)
        print("Recognized text:", result.text)

        from_code, to_code = "en", "hi"
        argostranslate.package.update_package_index()
        available_packages = argostranslate.package.get_available_packages()
        package = next(filter(lambda x: x.from_code == from_code and x.to_code == to_code, available_packages))
        argostranslate.package.install_from_path(package.download())
        translatedText = argostranslate.translate.translate(result.text, from_code, to_code)
        print("Translated text:", translatedText)

        tokenizer = AutoTokenizer.from_pretrained("facebook/mms-tts-hin")
        model_tts = VitsModel.from_pretrained("facebook/mms-tts-hin")
        inputs = tokenizer(translatedText, return_tensors="pt")
        with torch.no_grad():
            output = model_tts(**inputs).waveform
        wavfile.write("hindi_output.wav", 16000, output.squeeze().numpy())
        play_audio("hindi_output.wav")
    except Exception as e:
        print("Error:", e)


# ------------------ Routes ------------------

@app.route("/", methods=["GET"])
def home():
    return render_template_string(HTML, message="")

@app.route("/hindi_to_english", methods=["POST"])
def run_hi_en():
    threading.Thread(target=hindi_to_english, daemon=True).start()
    return render_template_string(HTML, message="🎧 Translating Hindi → English...")

@app.route("/english_to_hindi", methods=["POST"])
def run_en_hi():
    threading.Thread(target=english_to_hindi, daemon=True).start()
    return render_template_string(HTML, message="🎧 Translating English → Hindi...")

# ------------------ Startup TTS ------------------
def startup_code():
    try:
        m = KittenTTS("KittenML/kitten-tts-nano-0.2")
        audio = m.generate("Welcome to the offline translation app.", voice='expr-voice-5-m')
        sf.write('startup.wav', audio.astype('float32'), 24000)
        play_audio("startup.wav")
    except Exception as e:
        print("Startup error:", e)

# Run the welcome audio once
threading.Thread(target=startup_code, daemon=True).start()

# ------------------ Start Server ------------------
if __name__ == "__main__":
    print("🌐 Open http://localhost:8080 or http://<raspberry-pi-ip>:8080 in browser")
    app.run(host="0.0.0.0", port=8080)
