import customtkinter as ctk
from kittentts import KittenTTS
import pygame
import whisper
import sounddevice as sd
import numpy as np
from scipy.io.wavfile import write
import subprocess
from indic_transliteration import sanscript
from indic_transliteration.sanscript import transliterate
from transformers import VitsModel, AutoTokenizer
import torch
import sys
import soundfile as sf

# ------------------ Function to run on startup ------------------
def startup_code():
    m = KittenTTS("KittenML/kitten-tts-nano-0.2")

    audio = m.generate("Welcome to the offline translation app.", voice='expr-voice-5-m' )

# available_voices : [  'expr-voice-2-m', 'expr-voice-2-f', 'expr-voice-3-m', 'expr-voice-3-f',  'expr-voice-4-m', 'expr-voice-4-f', 'expr-voice-5-m', 'expr-voice-5-f' ]

# Save the audio
    sf.write('kittensaudio.wav', audio, 24000)
# Specify the path to your audio file
    audio_file_path = "kittensaudio.wav"  # Replace with your actual file path

    pygame.mixer.init()

    pygame.mixer.music.load("kittensaudio.wav")

# Play the sound
    pygame.mixer.music.play()

    pygame.mixer.music.stop()
    pygame.mixer.music.unload()

class TextRedirector:
    def __init__(self, text_widget):
        self.text_widget = text_widget

    def write(self, string):
        self.text_widget.configure(state="normal")
        self.text_widget.insert("end", string)
        self.text_widget.see("end")
        self.text_widget.configure(state="disabled")

    def flush(self):
        pass

def hindi_to_english():
    m = KittenTTS("KittenML/kitten-tts-nano-0.2")
    fs = 16000  # Common sample rate for audio

    # Recording duration in seconds
    duration = 10

    print(f"Recording audio for {duration} seconds...")

    # Record audio
    # The 'channels' parameter can be 1 for mono or 2 for stereo
    recording = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='int16')

    # Wait for the recording to complete
    sd.wait()

# Save the recording to a WAV file
    write('output.wav', fs, recording)

    print("Recording saved as output.wav")

#------------------------------------------------
    model = whisper.load_model("medium")

# Force Hindi, request translation
# result = model.transcribe("output.mp3", language="hi", task="translate")

# print(result["text"])
# # load audio and pad/trim it to fit 30 seconds
    audio = whisper.load_audio("output.wav")
    audio = whisper.pad_or_trim(audio)

# make log-Mel spectrogram and move to the same device as the model
    mel = whisper.log_mel_spectrogram(audio, n_mels=model.dims.n_mels).to(model.device)

# detect the spoken language
    _, probs = model.detect_language(mel)

    print(f"Detected language: {max(probs, key=probs.get)}")

# decode the audio
    options = whisper.DecodingOptions(task="translate", language="hi")
    result = whisper.decode(model, mel, options)

#print the recognized text
    print(result.text)

    audio = m.generate(result.text, voice='expr-voice-5-m' )

# available_voices : [  'expr-voice-2-m', 'expr-voice-2-f', 'expr-voice-3-m', 'expr-voice-3-f',  'expr-voice-4-m', 'expr-voice-4-f', 'expr-voice-5-m', 'expr-voice-5-f' ]


    pygame.mixer.music.stop()
    pygame.mixer.music.unload()

# Save the audio
    audio = audio.astype('float32')
    sf.write('kittensaudio.wav', audio, 24000)
# Specify the path to your audio file
    audio_file_path = "kittensaudio.wav"  # Replace with your actual file path

# Play the audio file

    pygame.mixer.music.load(audio_file_path)

# Play the sound
    pygame.mixer.music.play()

    while pygame.mixer.music.get_busy():
         pygame.time.Clock().tick(10)

# ------------------ Setup GUI ------------------
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

app = ctk.CTk()
app.geometry("400x200")
app.title("Offline Translator")

label = ctk.CTkLabel(app, text="Ready to translate")
label.pack(pady=50)

output_text = ctk.CTkTextbox(app, width=580, height=250)
output_text.pack(pady=20)
output_text.configure(state="disabled")

# Redirect stdout
sys.stdout = TextRedirector(output_text)

# Button
translate_button = ctk.CTkButton(app, text="Hindi to English", command=hindi_to_english)
translate_button.pack(pady=10)

# ------------------ Run startup code ------------------
# Option 1: Directly (runs before mainloop)
startup_code()

# Option 2: Schedule after mainloop starts (safer for GUI updates)
# app.after(100, startup_code)  # runs after 100 ms

# ------------------ Run the GUI ------------------
app.mainloop()
