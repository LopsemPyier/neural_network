from aiaiai import AI
import constant
import numpy as np


def main():
    ai = AI(constant.layers, constant.f)
    print(ai.run(np.random.rand(784)))


if __name__ == '__main__':
    main()
