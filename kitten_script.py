def run_kitten(model):
    from kittentts import KittenTTS
    import pygame
    import whisper
    import sounddevice as sd
    from scipy.io.wavfile import write
    import numpy as np
    import soundfile as sf

    # Initialize TTS model
    m = KittenTTS("KittenML/kitten-tts-nano-0.2")

    fs = 16000  # Sample rate
    duration = 10  # seconds

    print(f"Recording audio for {duration} seconds...")
    recording = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='int16')
    sd.wait()
    write('output.wav', fs, recording)
    print("Recording saved as output.wav")

    # Load recorded audio
    audio = whisper.load_audio("output.wav")
    audio = whisper.pad_or_trim(audio)

    # Convert to mel spectrogram
    mel = whisper.log_mel_spectrogram(audio).to(model.device)

    # Force Hindi → English translation (no auto-detect)
    options = whisper.DecodingOptions(task="translate", language="hi")
    result = whisper.decode(model, mel, options)

    print("Translated text:", result.text)

    # Generate TTS output
    audio_out = m.generate(result.text, voice='expr-voice-5-m')

    # Play the translated speech
    pygame.mixer.quit()
    pygame.mixer.init(frequency=24000)
    sf.write('kittensaudio.wav', audio_out.astype('float32'), 24000)

    pygame.mixer.music.load('kittensaudio.wav')
    pygame.mixer.music.play()

    while pygame.mixer.music.get_busy():
        pygame.time.Clock().tick(10)

    pygame.mixer.music.stop()
    pygame.mixer.music.unload()
    pygame.mixer.quit()
