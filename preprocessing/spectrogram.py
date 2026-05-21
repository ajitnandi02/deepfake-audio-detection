import librosa
import numpy as np
import torch

def get_spectrogram(audio_path):
    audio, sr = librosa.load(audio_path, sr=16000, mono=True)

    if len(audio) == 0:
        audio = np.zeros(16000, dtype=np.float32)

    spec = librosa.feature.melspectrogram(
        y=audio,
        sr=sr,
        n_mels=128,
        n_fft=1024,
        hop_length=512
    )

    spec = librosa.power_to_db(spec, ref=np.max)

    # Normalize for stable CNN input
    spec = (spec - spec.mean()) / (spec.std() + 1e-6)

    return torch.tensor(spec).unsqueeze(0).unsqueeze(0).float()
