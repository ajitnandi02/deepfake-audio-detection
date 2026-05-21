import os
import time
import queue
import torch
import sounddevice as sd
import numpy as np
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

samplerate = 16000
chunk_duration = 5
chunk_samples = samplerate * chunk_duration

q = queue.Queue()

real_scores = []
fake_scores = []


def audio_callback(indata, frames, time_info, status):
    if status:
        print(status)
    q.put(indata.copy())


def rms_energy(audio):
    return float(np.sqrt(np.mean(audio ** 2)))


def peak_energy(audio):
    return float(np.max(np.abs(audio)))


def normalize_live_audio(audio):
    audio = audio.astype(np.float32)

    # remove DC offset
    audio = audio - np.mean(audio)

    # normalize microphone volume
    peak = np.max(np.abs(audio))
    if peak > 0:
        audio = audio / peak * 0.95

    return audio


def predict_chunk(audio):
    temp_file = "temp_chunk.wav"
    sf.write(temp_file, audio, samplerate)

    try:
        with torch.no_grad():
            wav_feat = extract_wav2vec(temp_file).to(device)
            spec = get_spectrogram(temp_file).to(device)

            cnn_feat = cnn_model(spec)
            output = hybrid_model(wav_feat, cnn_feat)

            probs = torch.softmax(output, dim=1)
            real_prob = probs[0][0].item()
            fake_prob = probs[0][1].item()

        return real_prob, fake_prob

    finally:
        if os.path.exists(temp_file):
            os.remove(temp_file)


print("\nLive Deepfake Detection Started")
print("First 3 seconds: stay silent for noise calibration...")
print("Then speak clearly into microphone.")
print("Press CTRL+C to stop\n")

with sd.InputStream(
    samplerate=samplerate,
    channels=1,
    dtype="float32",
    blocksize=1024,
    callback=audio_callback
):
    audio_buffer = np.empty((0, 1), dtype=np.float32)

    calibration_audio = []
    calibration_start = time.time()
    noise_floor = None

    try:
        while True:
            data = q.get()
            audio_buffer = np.concatenate((audio_buffer, data))

            if noise_floor is None:
                calibration_audio.append(data.flatten())

                if time.time() - calibration_start >= 3:
                    noise_audio = np.concatenate(calibration_audio)
                    noise_floor = rms_energy(noise_audio)

                    print(f"Noise floor: {noise_floor:.6f}")
                    print("Calibration done. Speak now.\n")

                continue

            if len(audio_buffer) >= chunk_samples:
                chunk = audio_buffer[:chunk_samples]
                audio_buffer = audio_buffer[chunk_samples:]

                audio = chunk.flatten().astype(np.float32)

                energy = rms_energy(audio)
                peak = peak_energy(audio)

                speech_threshold = max(noise_floor * 4.0, 0.001)
                peak_threshold = max(noise_floor * 8.0, 0.008)

                print(f"Energy: {energy:.6f} | Peak: {peak:.6f}")

                if energy < speech_threshold or peak < peak_threshold:
                    print("SILENCE / LOW VOICE DETECTED")
                    continue

                audio = normalize_live_audio(audio)

                real_prob, fake_prob = predict_chunk(audio)

                real_scores.append(real_prob)
                fake_scores.append(fake_prob)

                if fake_prob >= 0.80:
                    label = "FAKE AUDIO"
                else:
                    label = "REAL AUDIO"

                print("---------------------------")
                print(f"Prediction: {label}")
                print(f"Real: {real_prob * 100:.2f}%")
                print(f"Fake: {fake_prob * 100:.2f}%")
                print("---------------------------")

    except KeyboardInterrupt:
        print("\nStopping detection...\n")

        if len(real_scores) == 0:
            print("No speech detected.")
            exit()

        avg_real = np.mean(real_scores)
        avg_fake = np.mean(fake_scores)

        print("====== FINAL RESULT ======")

        if avg_fake >= 0.80:
            print("FINAL: FAKE AUDIO")
        else:
            print("FINAL: REAL AUDIO")

        print(f"Real Confidence: {avg_real * 100:.2f}%")
        print(f"Fake Confidence: {avg_fake * 100:.2f}%")
