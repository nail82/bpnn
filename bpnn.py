"""
This module is the main entry point to exercise
the neural network code.
"""

from __future__ import print_function
import sys
import os
import neural_net as nn
import squash_functs as sf

def main():
    mynet = nn.NeuralNet(sf.binary, sf.binary_prime)


if __name__ == '__main__':
    main()
