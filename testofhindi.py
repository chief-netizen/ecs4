from transformers import VitsModel, AutoTokenizer
import torch

# 1. Install dependencies (if not already installed) 
# pip install transformers accelerate torch

# 2. Load the model and tokenizer for Hindi TTS
model = VitsModel.from_pretrained("facebook/mms-tts-hin")
tokenizer = AutoTokenizer.from_pretrained("facebook/mms-tts-hin")

# 3. The Hindi text you want to convert to speech
text = "पल मेरा एक परिक्षा है पर मैं उसको लिखना नहीं चाता क्यूकि मैं गर जा रहा हूं।" # Hindi for "Hello world"

# 4. Tokenize the text
inputs = tokenizer(text, return_tensors="pt")

# 5. Generate the waveform
with torch.no_grad():
    output = model(**inputs).waveform

# 6. You can now save or play the 'output' tensor as an audio file.
#    For example, to save as a WAV file (requires scipy):
import scipy.io.wavfile as wavfile
wavfile.write("hindi_output.wav", rate=16000, data=output.squeeze().numpy())