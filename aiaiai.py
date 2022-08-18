import constant
import numpy as np


class AI:
    def __init__(self, layers, f, fprim):
        self.f = f
        self.fprim=fprim
        self.layers=layers
        self.weights = [np.random.rand(layers[i], layers[i-1]) * 2 - 1 for i in range(1, len(layers))]
        self.bias = [np.random.rand(layers[i]) * 2 - 1 for i in range(1, len(layers))]

    def generate_nodes_and_sigmaprim(self, inputs): 
        nodes = []
        sigmaprim=[]
        vect = np.array(inputs)
        nodes.append(vect)
        for i in range(len(self.weights)):
            vect = self.weights[i] @ vect + self.bias[i]
            sigmaprim.append(self.fprim(vect))
            vect=self.f(vect)
            nodes.append(vect)
        return nodes, sigmaprim
    
    def partial(self, inputs, expected_result):
        nodes, sigmaprim=self.generate_nodes_and_sigmaprim(inputs)
        L=len(nodes)-1
        nodes_partials=[2*(nodes[L][j]-expected_result[j]) for j in range(len(nodes[L]))]
        weights_partials = [np.zeros((self.layers[i], self.layers[i-1])) for i in range(1, len(self.layers))]
        bias_partials = [np.zeros(self.layers[i]) for i in range(1, len(self.layers))]
        for l in range(len(nodes)-1, 0, -1):
            nodes_partialsbis=[0 for _ in range(self.layers[l-1])]
            for j in range(self.layers[l]):
                bias_partials[l-1][j]=nodes_partials[j]*sigmaprim[l-1][j]
                for k in range(self.layers[l-1]):
                    weights_partials[l-1][j][k]=nodes_partials[j]*sigmaprim[l-1][j]*nodes[l-1][k]
                    nodes_partialsbis[k]+=self.weights[l-1][j][k]*sigmaprim[l-1][j]*nodes_partials[j]
            nodes_partials=nodes_partialsbis
        return weights_partials, bias_partials

    def gradient(self, minidataset, minitarget):
        weights_mean_partials=np.array([np.zeros((self.layers[i], self.layers[i-1])) for i in range(1, len(self.layers))])
        bias_mean_partials=np.array([np.zeros(self.layers[i]) for i in range(1, len(self.layers))])
        N=len(minidataset)
        for i in range(N):
            weights_partials, bias_partials = self.partial(minidataset[i], minitarget[i])
            weights_mean_partials += weights_partials
            bias_mean_partials += bias_partials
        weights_mean_partials/=N
        bias_mean_partials/=N
        return weights_mean_partials, bias_mean_partials
    



    def train(self, dataset, targets):
        minidatasets,minitargets = constant.get_minisets(dataset, targets)
        N = len(minidatasets)
        for i in range(N):
            print(i)
            weights_gradient, bias_gradient = self.gradient(minidatasets[i], minitargets[i])
            self.weights -= weights_gradient
            self.bias -= bias_gradient
        

        
        


    def run(self, inputs):
        vect = np.array(inputs)
        for i in range(len(self.weights)):
            vect = self.f(self.weights[i] @ vect + self.bias[i])
        return vect 

    


        
