from math import exp

layers = [784, 16, 16, 10]

def sigmoid(x):
    return 1/(1+exp(-x))

def f(x): 
    sigmoid(x)

