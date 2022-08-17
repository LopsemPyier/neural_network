from aiaiai import AI
import constant
import numpy as np
import data


def main():
    images = data.load_images("dataset/train-images-idx3-ubyte")
    labels = data.load_labels("dataset/train-labels-idx1-ubyte")
    print(labels[2])

    ai = AI(constant.layers, constant.f)
    print(ai.run(np.random.rand(784)))


if __name__ == '__main__':
    main()
