import numpy as np
import pickle


def load_images(filename):  # load the images from the MNIST data set
    with open(filename, "rb") as f:
        magic_number = int.from_bytes(f.read(4), "big")
        nb_images = int.from_bytes(f.read(4), "big")
        rows = int.from_bytes(f.read(4), "big")
        columns = int.from_bytes(f.read(4), "big")

        buffer = f.read(nb_images * rows * columns)
        data = np.frombuffer(buffer, dtype=np.uint8).astype(np.float32)
        data = data.reshape(nb_images, rows * columns)
        data /= 255
        return data


def load_labels(filename):  # load the labels of the images from the MNIST data set
    with open(filename, "rb") as f:
        magic_number = int.from_bytes(f.read(4), "big")
        nb_labels = int.from_bytes(f.read(4), "big")

        buffer = f.read(nb_labels)
        data = np.frombuffer(buffer, dtype=np.uint8).astype(np.int64)
        return data


def load_targets(filename):  # load the labels of the images from the MNIST data set and transform them into targets
    data = load_labels(filename)
    nb_labels = len(data)
    labels = [np.array([0.0 for _ in range(10)]) for _ in range(nb_labels)]
    for i, label in enumerate(data):
        labels[i][label] = 1.0
    return labels


def save_ai(ai, filename):
    data = {
        "layers": ai.layers,
        "weights": ai.weights,
        "bias": ai.bias
    }
    with open(filename, "wb") as f:
        pickle.dump(data, f)


def load_ai(filename):
    with open(filename, "rb") as f:
        data = pickle.load(f)
        return data.get("layers"), data.get("weights"), data.get("bias")
