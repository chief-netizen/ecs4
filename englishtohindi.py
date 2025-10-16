import argostranslate.package
import argostranslate.translate
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


m = KittenTTS("KittenML/kitten-tts-nano-0.2")

audio = m.generate("Welcome to the offline translation app.", voice='expr-voice-5-m' )

import soundfile as sf
sf.write('kittensaudio.wav', audio, 24000)
# available_voices : [  'expr-voice-2-m', 'expr-voice-2-f', 'expr-voice-3-m', 'expr-voice-3-f',  'expr-voice-4-m', 'expr-voice-4-f', 'expr-voice-5-m', 'expr-voice-5-f' ]

pygame.mixer.init()

pygame.mixer.music.load("kittensaudio.wav")

# Play the sound
pygame.mixer.music.play()

while pygame.mixer.music.get_busy():
    pygame.time.Clock().tick(10) 
#-----------------------------------------------------
# Sampling frequency (samples per second)
fs = 24000  # Common sample rate for audio

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
options = whisper.DecodingOptions()
result = whisper.decode(model, mel, options)

#print the recognized text
print(result.text)

from_code = "en"
to_code = "hi"

# Download and install Argos Translate package
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


# 2. Load the model and tokenizer for Hindi TTS
model = VitsModel.from_pretrained("facebook/mms-tts-hin")
tokenizer = AutoTokenizer.from_pretrained("facebook/mms-tts-hin")

text=translatedText
# 4. Tokenize the text
inputs = tokenizer(text, return_tensors="pt")

# 5. Generate the waveform
with torch.no_grad():
    output = model(**inputs).waveform

pygame.mixer.music.stop()
pygame.mixer.music.unload()

# 6. You can now save or play the 'output' tensor as an audio file.
#    For example, to save as a WAV file (requires scipy):
import scipy.io.wavfile as wavfile
wavfile.write("hindi_output.wav", rate=16000, data=output.squeeze().numpy())

audio_file_path = "hindi_output.wav"  # Replace with your actual file path

# Play the audio file

pygame.mixer.music.load(audio_file_path)

# Play the sound
pygame.mixer.music.play()

while pygame.mixer.music.get_busy():
    pygame.time.Clock().tick(10) 



