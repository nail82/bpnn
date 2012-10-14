#!/usr/bin/env python
"""
This uses a neural network to classify the non-linear
data points from assignment 2.
"""
from __future__ import print_function
import sys
import os
import neural_net as nn
import squash_funcs as sf
import make_data as md
import numpy as np
import matplotlib.pyplot as plt

def main():
    fnm = 'homework1_data.csv'
    training_data = md.read_data(fnm)
    # Go through the data and convert the omega 2 patters to have a -1 tag
    for d in training_data:
        if d[0,0] == 2:
            d[0,0] = -1

    # Make a test data set
    test_data = md.circle_data()
    for d in test_data:
        if d[0,0] == 2:
            d[0,0] = -1

    alpha = 0.5
    theta = .0001
    iter_limit = 250000
    mynet = nn.NeuralNet((2,2,1), sf.bipolar, sf.bipolar_prime, alpha)
    n = training_data.shape[0]
    total_count = 0
    k = -1
    plot_data = np.matrix("0 0")
    fh = open('nn_quad.csv', 'w')
    while True:
        total_count += 1
        pattern = np.asarray(training_data[k,1:]).squeeze()
        teacher = [training_data[k,0]]
        error = mynet.train(pattern, teacher, theta)
        if (total_count % 10) == 0:
            plot_data = np.vstack((plot_data, np.matrix((total_count, error))))
            msg = ','.join([str(total_count), str(error)])
            fh.write(msg)
            fh.write('\n')
        if (total_count > iter_limit) or (error < theta):
            print("Stopped at iter count ",
                  total_count, "with a final error of ", error)
            break

    mynet.dt.tofile('bub_quad.txt')
    fh.close()
    plt.plot(plot_data[0:,0], plot_data[0:,1])
    plt.show()

    # Now, run the test data through the network and count correct
    # classifications.


if __name__ == '__main__':
    main()
