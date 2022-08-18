import constant
import numpy as np


class AI:
    def __init__(self, layers, f, fprim, weights=None, bias=None):
        self.f = f
        self.fprim = fprim
        self.layers = layers
        if weights:
            self.weights = weights 
        else:
            self.weights = np.array([np.random.rand(layers[i], layers[i - 1]) * 2 - 1 for i in range(1, len(layers))]) # list of the matrices containing the weights for each layer of the network
        if bias:
            self.bias = bias
        else:
            self.bias = np.array([np.random.rand(layers[i]) * 2 - 1 for i in range(1, len(layers))])  # same for the bias

    # generates the values of the neurones and the sigmaprim values (cf. more explanations) for a certain input
    def generate_nodes_and_sigmaprim(self, inputs):
        nodes = []
        sigmaprim = []
        vect = np.array(inputs)
        nodes.append(vect)
        for i in range(len(self.weights)):
            vect = self.weights[i] @ vect + self.bias[i]
            sigmaprim.append(self.fprim(vect))
            vect = self.f(vect)
            nodes.append(vect)
        return nodes, sigmaprim

    # calculates the partial derivative of the cost (cf. more explanaitions) with respect to each weight and bias,
    # for a certain input and returns them as two np.arrays of the same dimensions as the "self.weights" and
    # "self.bias"
    def partial(self, inputs, expected_result):
        nodes, sigmaprim = self.generate_nodes_and_sigmaprim(inputs)
        L = len(nodes) - 1
        nodes_partials = [2 * (nodes[L][j] - expected_result[j]) for j in range(len(nodes[L]))]
        weights_partials = np.array(
            [np.zeros((self.layers[i], self.layers[i - 1])) for i in range(1, len(self.layers))])
        bias_partials = np.array([np.zeros(self.layers[i]) for i in range(1, len(self.layers))])
        for l in range(len(nodes) - 1, 0, -1):  # for each layer
            nodes_partialsbis = [0 for _ in range(self.layers[l - 1])]
            for j in range(self.layers[l]):  # for each neurone of the l-layer
                bias_partials[l - 1][j] = nodes_partials[j] * sigmaprim[l - 1][j]
                for k in range(self.layers[l - 1]):  # for each neurone of the l-1-layer
                    weights_partials[l - 1][j][k] = nodes_partials[j] * sigmaprim[l - 1][j] * nodes[l - 1][k]
                    nodes_partialsbis[k] += self.weights[l - 1][j][k] * sigmaprim[l - 1][j] * nodes_partials[j]
            nodes_partials = nodes_partialsbis
        return weights_partials, bias_partials

    # calculates the mean of the partial derivatives (calculated by the "partial" function) for a set of inputs
    def gradient(self, minidataset, minitarget):
        weights_mean_partials = np.array(
            [np.zeros((self.layers[i], self.layers[i - 1])) for i in range(1, len(self.layers))])
        bias_mean_partials = np.array([np.zeros(self.layers[i]) for i in range(1, len(self.layers))])
        N = len(minidataset)
        for i in range(N):
            weights_partials, bias_partials = self.partial(minidataset[i], minitarget[i])
            weights_mean_partials += weights_partials
            bias_mean_partials += bias_partials
        weights_mean_partials /= N
        bias_mean_partials /= N
        return weights_mean_partials, bias_mean_partials

    def train(self, dataset, targets):  # trains the AI with the backpropagation algorithm
        minidatasets, minitargets = constant.get_minisets(dataset, targets)
        N = len(minidatasets)
        for i in range(N):
            print(i)
            weights_gradient, bias_gradient = self.gradient(minidatasets[i], minitargets[i])
            self.weights -= weights_gradient
            self.bias -= bias_gradient

    def run(self, inputs):  # gives the output of the network for a certain input
        vect = np.array(inputs)
        for i in range(len(self.weights)):
            vect = self.f(self.weights[i] @ vect + self.bias[i])
        return vect
