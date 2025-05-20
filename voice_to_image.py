import sounddevice as sd
import numpy as np
import whisper
import tempfile
import os
import openai
import requests
from gtts import gTTS
import pygame
from time import sleep

openai.api_key = os.getenv("OPENAI_API_KEY")

# Record audio from microphone
def record_audio(duration=5, fs=44100):
    print("🎤 Speak your image prompt (recording)...")
    recording = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype="int16")
    sd.wait()
    print("✅ Recorded")
    return recording, fs

# Save recorded audio to WAV
def save_wav(data, fs):
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    import soundfile as sf
    sf.write(tmp.name, data, fs)
    return tmp.name

# Transcribe using Whisper
def transcribe(audio_path):
    print("🧠 Transcribing...")
    model = whisper.load_model("base")
    result = model.transcribe(audio_path)
    return result["text"]

# Generate image with DALL·E
def generate_image(prompt):
    print(f"🖼️ Generating image with prompt:  {prompt}")
    response = openai.images.generate(
        model="dall-e-3",
        prompt=prompt,
        size="1024x1024",
        quality="standard",
        n=1,
    )
    image_url = response.data[0].url
    return image_url

# Download image
def download_image(url, filename="generated_image.jpg"):
    print(f"⬇️ Downloading image to {filename}")
    r = requests.get(url)
    with open(filename, "wb") as f:
        f.write(r.content)

# Speak text
def speak(text):
    print("🎤 Speaking...")
    tts = gTTS(text=text, lang='en')
    
    tmp_path = os.path.join(tempfile.gettempdir(), "temp_tts.mp3")
    tts.save(tmp_path)

    pygame.mixer.init()
    pygame.mixer.music.load(tmp_path)
    pygame.mixer.music.play()

    while pygame.mixer.music.get_busy():
        sleep(0.1)

    pygame.mixer.quit()
    os.remove(tmp_path)

# Main
audio_data, samplerate = record_audio()
wav_path = save_wav(audio_data, samplerate)
prompt = transcribe(wav_path)
print("📝 Prompt:", prompt)

speak(f"Generating image of: {prompt}")
image_url = generate_image(prompt)
download_image(image_url)
print("✅ Done! Image saved.")