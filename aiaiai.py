import constant
import numpy as np


class AI:
    def __init__(self, layers, f):
        self.f = f
        self.weights = [np.random.rand(layers[i], layers[i-1]) for i in range(1, len(layers))]
        self.bias = [np.random.rand(layers[i]) for i in range(1, len(layers))]

    def train(self, inputs, targets):
        pass

    def run(self, inputs):
        vect = np.array(inputs)
        for i in range(len(self.weights)):
            vect = self.f(self.weights[i] @ vect + self.bias[i])
        return vect 


        
