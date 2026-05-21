import os
import torch
import tkinter as tk
from tkinter import filedialog
import librosa
import soundfile as sf

from models.hybrid_model import HybridModel
from models.wev2vec_model import extract_wav2vec
from models.cnn_model import SpectrogramCNN
from preprocessing.spectrogram import get_spectrogram

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Loading model...")

hybrid_model = HybridModel().to(device)
cnn_model = SpectrogramCNN().to(device)

checkpoint = torch.load("saved_model/model.pth", map_location=device)

hybrid_model.load_state_dict(checkpoint["hybrid_model"])
cnn_model.load_state_dict(checkpoint["cnn_model"])

hybrid_model.eval()
cnn_model.eval()

root = tk.Tk()
root.withdraw()

file_path = filedialog.askopenfilename(
    title="Select Audio File",
    filetypes=[("Audio Files", "*.wav *.mp3")]
)

if not file_path:
    print("No file selected.")
    exit()

print("\nSelected File:", file_path)

audio, sr = librosa.load(file_path, sr=16000, mono=True)

temp_file = "temp_audio.wav"
sf.write(temp_file, audio, 16000)

with torch.no_grad():
    wav_feat = extract_wav2vec(temp_file).to(device)
    spec = get_spectrogram(temp_file).to(device)

    cnn_feat = cnn_model(spec)
    output = hybrid_model(wav_feat, cnn_feat)

    probs = torch.softmax(output, dim=1)

    real_prob = probs[0][0].item()
    fake_prob = probs[0][1].item()

prediction = torch.argmax(probs, dim=1).item()

if prediction == 0:
    label = "REAL AUDIO"
else:
    label = "FAKE AUDIO"

print("\n==============================")
print("Prediction:", label)
print(f"Real Probability: {real_prob * 100:.2f}%")
print(f"Fake Probability: {fake_prob * 100:.2f}%")
print("==============================")

if os.path.exists(temp_file):
    os.remove(temp_file)
