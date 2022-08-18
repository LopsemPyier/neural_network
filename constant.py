from math import exp
from operator import length_hint
import numpy as np
from random import shuffle

layers = [784, 16, 16, 10]
length_miniset=100

def sigmoid(x):
    return 1/(1+np.exp(-x))

def sigmoidprim(x):
    return (np.exp(-x))/((1+np.exp(-x))**2)

def f(x): 
    return sigmoid(x)

def fprim(x):
    return sigmoidprim(x)

def get_minisets(dataset, targets):
    indexes=list(range(len(dataset)))
    shuffle(indexes)
    minidatasets=[]
    minitargets=[]
    for i in range(len(dataset)//length_miniset):
        minidatasets.append([])
        minitargets.append([])
        for j in range(length_miniset):
            minidatasets[i].append(dataset[indexes[i*length_miniset+j]])
            minitargets[i].append(targets[indexes[i*length_miniset+j]])
    return minidatasets, minitargets

    
