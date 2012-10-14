#!/usr/bin/env python2.7
"""
This exercise sets up and trains the neural network
to calculate xor on two inputs.
"""

from __future__ import print_function
import sys
import os
import neural_net as nn
import squash_funcs as sf
import numpy as np

def main():
    alpha = 0.5
    data = np.matrix("0. 0. 0.; 0. 1. 1.; 1. 1. 0.; 1. 0. 1.")
    theta = 1e-3
    iter_limit = 1000
    total_count = 0
    mynet = nn.NeuralNet((2, 2, 1), sf.binary, sf.binary_prime, alpha)
    n = data.shape[0]
    k = -1
    while True:
        total_count += 1
        k = (k+1) % n
        pattern = np.asarray(data[k,1:]).squeeze()
        teacher = [data[k,0]]
        error = mynet.train(pattern, teacher, theta)
        print('Error=>', error)
        if (total_count > iter_limit) or (error < theta):
            break

    return mynet


if __name__ == '__main__':
    main()
