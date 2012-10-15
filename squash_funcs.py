"""
This module implements squashing functions
and first derivatives for the squashing functions.
"""

import numpy as np

def bipolar(x):
    return (1 - np.exp(-x)) / (1 + np.exp(-x))

def bipolar_prime(x):
    return (2. * np.exp(x)) / ((1 + np.exp(x))**2)

def binary(x):
    return 1./(1+np.exp(-x))

def binary_prime(x):
    return np.exp(x) / (1 + np.exp(x)**2)

def identity(x):
    return x

def linear_thresh(x):
    if x >= 0:
        return 1
    else:
        return -1
