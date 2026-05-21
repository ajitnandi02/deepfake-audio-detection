import os
import torch
import torch.nn as nn
import torch.optim as optim

from models.hybrid_model import HybridModel
from models.wev2vec_model import extract_wav2vec
from models.cnn_model import SpectrogramCNN
from preprocessing.spectrogram import get_spectrogram
from utils.dataset_loader import load_dataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

hybrid_model = HybridModel().to(device)
cnn_model = SpectrogramCNN().to(device)

criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(
    list(hybrid_model.parameters()) + list(cnn_model.parameters()),
    lr=0.001
)

train_path = "dataset/training"
val_path = "dataset/validation"

train_files, train_labels = load_dataset(train_path)
val_files, val_labels = load_dataset(val_path)

print("Training samples:", len(train_files))
print("Validation samples:", len(val_files))

epochs = 5

for epoch in range(epochs):
    hybrid_model.train()
    cnn_model.train()

    total_loss = 0
    correct = 0

    for file_path, label in zip(train_files, train_labels):
        wav_feat = extract_wav2vec(file_path).to(device)
        spec = get_spectrogram(file_path).to(device)

        cnn_feat = cnn_model(spec)
        output = hybrid_model(wav_feat, cnn_feat)

        label_tensor = torch.tensor([label], dtype=torch.long).to(device)

        loss = criterion(output, label_tensor)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        pred = torch.argmax(output, dim=1).item()
        if pred == label:
            correct += 1

    train_acc = correct / len(train_files) if train_files else 0

    hybrid_model.eval()
    cnn_model.eval()

    val_correct = 0

    with torch.no_grad():
        for file_path, label in zip(val_files, val_labels):
            wav_feat = extract_wav2vec(file_path).to(device)
            spec = get_spectrogram(file_path).to(device)

            cnn_feat = cnn_model(spec)
            output = hybrid_model(wav_feat, cnn_feat)

            pred = torch.argmax(output, dim=1).item()
            if pred == label:
                val_correct += 1

    val_acc = val_correct / len(val_files) if val_files else 0

    print(
        f"Epoch {epoch + 1}/{epochs} | "
        f"Loss: {total_loss:.4f} | "
        f"Train Acc: {train_acc * 100:.2f}% | "
        f"Val Acc: {val_acc * 100:.2f}%"
    )

os.makedirs("saved_model", exist_ok=True)

torch.save(
    {
        "hybrid_model": hybrid_model.state_dict(),
        "cnn_model": cnn_model.state_dict(),
        "label_map": {
            "real": 0,
            "fake": 1
        }
    },
    "saved_model/model.pth"
)

print("Model saved successfully.")
