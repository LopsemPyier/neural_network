from math import exp
import numpy as np

layers = [784, 16, 16, 10]

def sigmoid(x):
    return 1/(1+np.exp(-x))

def f(x): 
    return sigmoid(x)

