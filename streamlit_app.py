import os
import tempfile

import streamlit as st
import torch
import librosa
import soundfile as sf

from models.hybrid_model import HybridModel
from models.cnn_model import SpectrogramCNN
from models.wev2vec_model import extract_wav2vec
from preprocessing.spectrogram import get_spectrogram


st.set_page_config(
    page_title="Deepfake Audio Detector",
    page_icon="🎧",
    layout="centered"
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@st.cache_resource
def load_models():
    hybrid_model = HybridModel().to(device)
    cnn_model = SpectrogramCNN().to(device)

    checkpoint = torch.load("saved_model/model.pth", map_location=device)

    hybrid_model.load_state_dict(checkpoint["hybrid_model"])
    cnn_model.load_state_dict(checkpoint["cnn_model"])

    hybrid_model.eval()
    cnn_model.eval()

    return hybrid_model, cnn_model


def save_uploaded_file(uploaded_file):
    suffix = os.path.splitext(uploaded_file.name)[1].lower()

    temp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    temp.write(uploaded_file.read())
    temp.close()

    return temp.name


def prepare_audio(input_path):
    audio, sr = librosa.load(input_path, sr=16000, mono=True)

    audio_temp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    audio_temp.close()

    sf.write(audio_temp.name, audio, 16000)
    return audio_temp.name


def predict_audio(audio_path, hybrid_model, cnn_model):
    with torch.no_grad():
        wav_feat = extract_wav2vec(audio_path).to(device)
        spec = get_spectrogram(audio_path).to(device)

        cnn_feat = cnn_model(spec)
        output = hybrid_model(wav_feat, cnn_feat)

        probs = torch.softmax(output, dim=1)

        real_prob = probs[0][0].item()
        fake_prob = probs[0][1].item()

    if fake_prob > real_prob:
        label = "FAKE AUDIO"
    else:
        label = "REAL AUDIO"

    return label, real_prob, fake_prob


st.title("Deepfake Audio Detection")
st.write("Upload an audio file to check whether the speech is real or AI-generated.")

uploaded_file = st.file_uploader(
    "Upload audio file",
    type=["wav", "mp3", "ogg", "m4a"]
)

if uploaded_file is not None:
    st.audio(uploaded_file)
    st.info("File uploaded successfully.")

    if st.button("Detect"):
        input_path = None
        prepared_audio_path = None

        try:
            with st.spinner("Loading model and analyzing audio..."):
                hybrid_model, cnn_model = load_models()

                input_path = save_uploaded_file(uploaded_file)
                prepared_audio_path = prepare_audio(input_path)

                label, real_prob, fake_prob = predict_audio(
                    prepared_audio_path,
                    hybrid_model,
                    cnn_model
                )

            st.subheader("Result")

            if label == "REAL AUDIO":
                st.success("Prediction: REAL AUDIO")
            else:
                st.error("Prediction: FAKE AUDIO")

            st.write(f"Real Probability: {real_prob * 100:.2f}%")
            st.write(f"Fake Probability: {fake_prob * 100:.2f}%")

            st.progress(real_prob, text=f"Real: {real_prob * 100:.2f}%")
            st.progress(fake_prob, text=f"Fake: {fake_prob * 100:.2f}%")

        except Exception as error:
            st.error(f"Error: {error}")

        finally:
            for path in [input_path, prepared_audio_path]:
                if path and os.path.exists(path):
                    try:
                        os.remove(path)
                    except Exception:
                        pass
