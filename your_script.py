def main(model):
    print("✅ your_script.main() called!")

    # Example usage
    import whisper
    import sounddevice as sd
    import numpy as np
    from scipy.io.wavfile import write
    import pygame
    import argostranslate.package, argostranslate.translate
    from transformers import VitsModel, AutoTokenizer
    import torch
    import scipy.io.wavfile as wavfile

    fs = 24000
    duration = 5
    print(f"Recording {duration}s...")
    recording = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='int16')
    sd.wait()
    write('output.wav', fs, recording)

    audio = whisper.load_audio("output.wav")
    audio = whisper.pad_or_trim(audio)
    mel = whisper.log_mel_spectrogram(audio, n_mels=model.dims.n_mels).to(model.device)

    _, probs = model.detect_language(mel)
    print(f"Detected language: {max(probs, key=probs.get)}")

    options = whisper.DecodingOptions()
    result = whisper.decode(model, mel, options)
    print(f"Recognized: {result.text}")

    # translation
    from_code, to_code = "en", "hi"
    argostranslate.package.update_package_index()
    available_packages = argostranslate.package.get_available_packages()
    package_to_install = next(
        filter(lambda x: x.from_code == from_code and x.to_code == to_code, available_packages)
    )
    argostranslate.package.install_from_path(package_to_install.download())
    translatedText = argostranslate.translate.translate(result.text, from_code, to_code)
    print("Translated:", translatedText)

    tokenizer = AutoTokenizer.from_pretrained("facebook/mms-tts-hin")
    tts_model = VitsModel.from_pretrained("facebook/mms-tts-hin")

    inputs = tokenizer(translatedText, return_tensors="pt")
    with torch.no_grad():
        output = tts_model(**inputs).waveform

    wavfile.write("hindi_output.wav", rate=16000, data=output.squeeze().numpy())

    pygame.mixer.init()
    pygame.mixer.music.load("hindi_output.wav")
    pygame.mixer.music.play()
    while pygame.mixer.music.get_busy():
        pygame.time.Clock().tick(10)
    pygame.mixer.music.stop()
    pygame.mixer.music.unload()
