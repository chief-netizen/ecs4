import threading
import time
import numpy as np
import pygame
import sounddevice as sd
from scipy.io.wavfile import write
import soundfile as sf
import whisper
from kittentts import KittenTTS

def hindi_to_english():
    print("Button pressed ✅", flush=True)

    try:
        print("Initializing TTS and models...", flush=True)
        m = KittenTTS("KittenML/kitten-tts-nano-0.2")
        fs = 16000
        duration = 10

        print(f"Recording audio for {duration} seconds...", flush=True)
        recording = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='int16')
        sd.wait()

        time.sleep(0.5)  # Give ALSA time to release mic
        write('output.wav', fs, recording)
        print("Recording saved as output.wav", flush=True)

        print("Loading Whisper model...", flush=True)
        model = whisper.load_model("medium")

        audio = whisper.load_audio("output.wav")
        audio = whisper.pad_or_trim(audio)
        mel = whisper.log_mel_spectrogram(audio, n_mels=model.dims.n_mels).to(model.device)

        _, probs = model.detect_language(mel)
        print(f"Detected language: {max(probs, key=probs.get)}", flush=True)

        print("Translating from Hindi to English...", flush=True)
        options = whisper.DecodingOptions(task="translate", language="hi")
        result = whisper.decode(model, mel, options)
        print(f"Translated text: {result.text}", flush=True)

        print("Generating English audio...", flush=True)
        audio = m.generate(result.text, voice='expr-voice-5-m')
        audio = np.clip(audio, -1.0, 1.0).astype('float32')
        sf.write('kittensaudio.wav', audio, 24000)

        # Initialize pygame mixer if not already
        if not pygame.mixer.get_init():
            pygame.mixer.init()

        print("Playing translation...", flush=True)
        pygame.mixer.music.load('kittensaudio.wav')
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():
            pygame.time.Clock().tick(10)

        print("All done ✅", flush=True)

    except Exception as e:
        print(f"⚠️ Error: {e}", flush=True)


def hindi_to_english_thread():
    threading.Thread(target=hindi_to_english, daemon=True).start()
