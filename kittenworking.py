from kittentts import KittenTTS
import pygame
import whisper
import sounddevice as sd
from scipy.io.wavfile import write
import numpy as np

m = KittenTTS("KittenML/kitten-tts-nano-0.2")

audio = m.generate("Welcome to the offline translation app.", voice='expr-voice-5-m' )

# available_voices : [  'expr-voice-2-m', 'expr-voice-2-f', 'expr-voice-3-m', 'expr-voice-3-f',  'expr-voice-4-m', 'expr-voice-4-f', 'expr-voice-5-m', 'expr-voice-5-f' ]

# Save the audio
import soundfile as sf
sf.write('kittensaudio.wav', audio, 24000)
# Specify the path to your audio file
audio_file_path = "kittensaudio.wav"  # Replace with your actual file path

pygame.mixer.init()

pygame.mixer.music.load("kittensaudio.wav")

# Play the sound
pygame.mixer.music.play()

while pygame.mixer.music.get_busy():
    pygame.time.Clock().tick(10) 