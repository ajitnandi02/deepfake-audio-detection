import torch
import librosa
import numpy as np
from transformers import Wav2Vec2Model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base")
model.to(device)
model.eval()

def extract_wav2vec(audio_path):

    audio, sr = librosa.load(audio_path, sr=16000)

    audio = torch.tensor(audio).unsqueeze(0).to(device)

    with torch.no_grad():
        features = model(audio).last_hidden_state

    return features.mean(dim=1)