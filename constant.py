from math import exp
from operator import length_hint
import numpy as np
from random import shuffle

layers = [784, 16, 16, 10] #number of neurones for each layer of the network
length_miniset=100 #number of images in the subdivisions of the dataset

def sigmoid(x): 
    return 1/(1+np.exp(-x))

def sigmoidprim(x): #derivative of the sigmoid
    return (np.exp(-x))/((1+np.exp(-x))**2)

def f(x): #function to always have the neurone values between 0 and 1
    return sigmoid(x)

def fprim(x): # derivative of f
    return sigmoidprim(x)

def get_minisets(dataset, targets): #shuffles the data sets and divides it into "mini-sets"
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

    
