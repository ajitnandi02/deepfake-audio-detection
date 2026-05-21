import librosa
import numpy as np

def clean_audio(audio_path):

    audio, sr = librosa.load(audio_path, sr=16000)

    audio = librosa.effects.preemphasis(audio)

    return audio