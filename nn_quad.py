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
import data_table as dt

def main():
    training_data = md.circle_data(40)
    training_data = md.convert_data(training_data)

    # Make a test data set
    test_data = md.read_data('homework1_data.csv')
    test_data = md.convert_data(test_data)

    alpha = 0.5
    theta = .0001
    iter_limit = 50000
    mynet = nn.NeuralNet((2,2,1), sf.bipolar, sf.bipolar_prime, alpha)
    n = training_data.shape[0]
    total_count = 0
    k = -1
    plot_data = np.matrix("0 0")
    fh = open('nn_quad.csv', 'w')
    while True:
        total_count += 1
        k = (k+1) % n
        #k = np.random.randint(0, n)
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

    counter = check_net(test_data, mynet.fwd)
    print('Assigned', counter, 'patterns of', test_data.shape[0],'correctly.')



def check_net(test_data, fwd):
    # Now, run the test data through the network and count correct
    # classifications.
    counter = 0
    for d in test_data:
        expected = d[0,0]
        actual = fwd(d[0,1:])
        if np.sign(expected) == np.sign(actual):
            counter += 1
    return counter

def decision_boundary(test_data):
    """This function generates a plot of the decision boundary
    for the network."""
    space = np.linspace(-10., 10., 256)
    t = dt.DataTable((2,2,1))
    t.fromfile('bub_quad.txt')
    mynet = nn.nn_factory(t, sf.bipolar, sf.bipolar_prime, 0.5)
    plot_data = np.matrix("0 0")
    for x1 in space:
        for x2 in space:
            point = np.matrix((x1, x2))
            out = mynet.fwd(point)
            if out > 0.0:
                plot_data = np.vstack((plot_data, point))

    x1 = np.asarray(plot_data[1:,0]).squeeze()
    x2 = np.asarray(plot_data[1:,1]).squeeze()
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.scatter(x1, x2)
    ax.set_xlim(-10., 10)
    ax.set_ylim(-10., 10)

    gx1 = np.asarray(test_data[0:20,1]).squeeze()
    gx2 = np.asarray(test_data[0:20,2]).squeeze()
    rx1 = np.asarray(test_data[20:,1]).squeeze()
    rx2 = np.asarray(test_data[20:,2]).squeeze()

    ax.scatter(gx1, gx2, color='green')
    ax.scatter(rx1, rx2, color='red')
    plt.show()


if __name__ == '__main__':
    main()
