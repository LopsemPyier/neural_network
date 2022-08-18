import numpy as np


def load_images(filename):
    with open(filename, "rb") as f:
        magic_number = int.from_bytes(f.read(4), "big")
        nb_images = int.from_bytes(f.read(4), "big")
        rows = int.from_bytes(f.read(4), "big")
        columns = int.from_bytes(f.read(4), "big")

        buffer = f.read(nb_images * rows * columns)
        data = np.frombuffer(buffer, dtype=np.uint8).astype(np.float32)
        data = data.reshape(nb_images, rows*columns)
        data /= 255
        return data


def load_labels(filename):
    with open(filename, "rb") as f:
        magic_number = int.from_bytes(f.read(4), "big")
        nb_labels = int.from_bytes(f.read(4), "big")

        buffer = f.read(nb_labels)
        data = np.frombuffer(buffer, dtype=np.uint8).astype(np.int64)
        labels = [np.array([0.0 for _ in range(10)]) for _ in range(nb_labels)]
        for i, label in enumerate(data):
            labels[i][label] = 1.0
        return labels
