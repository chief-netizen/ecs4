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
import threading
import argostranslate.package
import argostranslate.translate


def startup_code():
    m = KittenTTS("KittenML/kitten-tts-nano-0.2")

    audio = m.generate("Welcome to the offline translation app.", voice='expr-voice-5-m' )
    audio = audio.astype('float32')  # Fix for sf.write

    sf.write('kittensaudio.wav', audio, 24000)
    audio_file_path = "kittensaudio.wav"

    pygame.mixer.init()
    pygame.mixer.music.load(audio_file_path)

    # Play the sound and wait until it finishes
    pygame.mixer.music.play()
    while pygame.mixer.music.get_busy():
        pygame.time.Clock().tick(10)
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

    duration = 10
    print(f"Recording audio for {duration} seconds...")

    recording = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='int16')
    sd.wait()

    write('output.wav', fs, recording)
    print("Recording saved as output.wav")

    model = whisper.load_model("small")

    audio = whisper.load_audio("output.wav")
    audio = whisper.pad_or_trim(audio)

    mel = whisper.log_mel_spectrogram(audio, n_mels=model.dims.n_mels).to(model.device)

    _, probs = model.detect_language(mel)
    print(f"Detected language: {max(probs, key=probs.get)}")

    options = whisper.DecodingOptions(task="translate", language="hi")
    result = whisper.decode(model, mel, options)
    print(result.text)

    audio = m.generate(result.text, voice='expr-voice-5-m' )
    audio = audio.astype('float32')  
    sf.write('kittensaudio.wav', audio, 24000)
    audio_file_path = "kittensaudio.wav"

    pygame.mixer.music.load(audio_file_path)
    pygame.mixer.music.play()
    while pygame.mixer.music.get_busy():
        pygame.time.Clock().tick(10)
    pygame.mixer.music.stop()
    pygame.mixer.music.unload()

def english_to_hindi():
    
    fs = 24000  # Common sample rate for audio


    duration = 10

    print(f"Recording audio for {duration} seconds...")


    recording = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='int16')

    sd.wait()


    write('output.wav', fs, recording)

    print("Recording saved as output.wav")

#------------------------------------------------
    model = whisper.load_model("medium")


    audio = whisper.load_audio("output.wav")
    audio = whisper.pad_or_trim(audio)


    mel = whisper.log_mel_spectrogram(audio, n_mels=model.dims.n_mels).to(model.device)


    _, probs = model.detect_language(mel)

    print(f"Detected language: {max(probs, key=probs.get)}")


    options = whisper.DecodingOptions()
    result = whisper.decode(model, mel, options)

#print the recognized text
    print(result.text)

    from_code = "en"
    to_code = "hi"


    argostranslate.package.update_package_index()
    available_packages = argostranslate.package.get_available_packages()
    package_to_install = next(
        filter(
          lambda x: x.from_code == from_code and x.to_code == to_code, available_packages
        )
    )
    argostranslate.package.install_from_path(package_to_install.download())

# Translate
    translatedText = argostranslate.translate.translate(result.text, from_code, to_code)
    print(translatedText)


    model = VitsModel.from_pretrained("facebook/mms-tts-hin")
    tokenizer = AutoTokenizer.from_pretrained("facebook/mms-tts-hin")

    text=translatedText

    inputs = tokenizer(text, return_tensors="pt")


    with torch.no_grad():
       output = model(**inputs).waveform

    pygame.mixer.music.stop()
    pygame.mixer.music.unload()


    import scipy.io.wavfile as wavfile
    wavfile.write("hindi_output.wav", rate=16000, data=output.squeeze().numpy())

    audio_file_path = "hindi_output.wav"  # Replace with your actual file path



    pygame.mixer.music.load(audio_file_path)


    pygame.mixer.music.play()

    while pygame.mixer.music.get_busy():
         pygame.time.Clock().tick(10)

# ------------------ Setup GUI ------------------
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

app = ctk.CTk()
app.geometry("600x400")
app.title("Offline Translator")

label = ctk.CTkLabel(app, text="Ready to translate")
label.pack(pady=10)

output_text = ctk.CTkTextbox(app, width=580, height=250)
output_text.pack(pady=10)
output_text.configure(state="disabled")

sys.stdout = TextRedirector(output_text)

# ------------------ Thread-safe button ------------------
def start_hindi_to_english():
    threading.Thread(target=hindi_to_english, daemon=True).start()

translate_button = ctk.CTkButton(app, text="Hindi to English", command=start_hindi_to_english)
translate_button.pack(pady=10)

# ------------------ Thread-safe button for English to Hindi ------------------
def start_english_to_hindi():
    threading.Thread(target=english_to_hindi, daemon=True).start()

english_to_hindi_button = ctk.CTkButton(app, text="English to Hindi", command=start_english_to_hindi)
english_to_hindi_button.pack(pady=10)


# ------------------ Run startup code in a thread ------------------
threading.Thread(target=startup_code, daemon=True).start()

# ------------------ Run the GUI ------------------
app.mainloop()
