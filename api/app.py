import os
import tempfile

import torch
import librosa
import soundfile as sf
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware

from models.hybrid_model import HybridModel
from models.cnn_model import SpectrogramCNN
from models.wev2vec_model import extract_wav2vec
from preprocessing.spectrogram import get_spectrogram

app = FastAPI(title="Deepfake Audio Detection API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

hybrid_model = HybridModel().to(device)
cnn_model = SpectrogramCNN().to(device)

checkpoint = torch.load("saved_model/model.pth", map_location=device)

hybrid_model.load_state_dict(checkpoint["hybrid_model"])
cnn_model.load_state_dict(checkpoint["cnn_model"])

hybrid_model.eval()
cnn_model.eval()


@app.get("/")
def home():
    return {
        "message": "Deepfake Audio Detection API is running",
        "endpoint": "/predict"
    }


def prepare_audio(input_path):
    audio, sr = librosa.load(input_path, sr=16000, mono=True)

    temp_audio = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    temp_audio.close()

    sf.write(temp_audio.name, audio, 16000)
    return temp_audio.name


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    input_path = None
    audio_path = None

    try:
        suffix = os.path.splitext(file.filename)[1].lower()

        input_temp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
        input_path = input_temp.name

        content = await file.read()
        input_temp.write(content)
        input_temp.close()

        audio_path = prepare_audio(input_path)

        with torch.no_grad():
            wav_feat = extract_wav2vec(audio_path).to(device)
            spec = get_spectrogram(audio_path).to(device)

            cnn_feat = cnn_model(spec)
            output = hybrid_model(wav_feat, cnn_feat)

            probs = torch.softmax(output, dim=1)

            real_prob = probs[0][0].item()
            fake_prob = probs[0][1].item()

        if fake_prob > real_prob:
            prediction = "FAKE AUDIO"
        else:
            prediction = "REAL AUDIO"

        return {
            "prediction": prediction,
            "real_probability": round(real_prob * 100, 2),
            "fake_probability": round(fake_prob * 100, 2)
        }

    except Exception as error:
        return {
            "error": str(error)
        }

    finally:
        for path in [input_path, audio_path]:
            if path and os.path.exists(path):
                try:
                    os.remove(path)
                except Exception:
                    pass