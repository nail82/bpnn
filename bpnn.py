#!/usr/bin/env python2.7
"""
This module is the main entry point to exercise
the neural network code.
"""

from __future__ import print_function
import sys
import os
import neural_net as nn
import squash_funcs as sf
import numpy as np

def main():
    alpha = 0.5
    mynet = nn.NeuralNet((2,3,2), sf.binary, sf.binary_prime, alpha)
    mynet.dt.set_test_weights()
    myin = np.array((1.,2.))
    myteach = np.array((2.,1.))
    error = mynet.train(myin, myteach, 1e-3)
    mynet.dt.prettyprint()
    print("Error=>", error)

if __name__ == '__main__':
    main()
