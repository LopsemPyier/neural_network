from aiaiai import AI
import constant
import numpy as np
import data


def main():
    images = data.load_images("dataset/train-images.idx3-ubyte")
    labels = data.load_targets("dataset/train-labels.idx1-ubyte")
    n = 60000

    ai = AI(constant.layers, constant.f, constant.fprim)
    ai.train(images[:n], labels[:n])

    print(ai.run(images[2]))
    print(labels[2])


if __name__ == '__main__':
    main()
