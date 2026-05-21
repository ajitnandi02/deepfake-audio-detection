import os
import random

def load_dataset(folder):
    files = []
    labels = []

    label_map = {
        "real": 0,
        "fake": 1
    }

    for label_name, label_value in label_map.items():
        path = os.path.join(folder, label_name)

        if not os.path.exists(path):
            continue

        for file in os.listdir(path):
            if file.lower().endswith((".wav", ".mp3")):
                files.append(os.path.join(path, file))
                labels.append(label_value)

    combined = list(zip(files, labels))
    random.shuffle(combined)

    if not combined:
        return [], []

    files, labels = zip(*combined)
    return list(files), list(labels)
