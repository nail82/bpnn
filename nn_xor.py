#!/usr/bin/env python2.7
"""
Ted Satcher
CS 640
Fall 2012

Assignment 3

File: nn_xor.py

This exercise sets up and trains the neural network
to calculate xor on two inputs.
"""

from __future__ import print_function
import sys
import os
import neural_net as nn
import squash_funcs as sf
import numpy as np
import matplotlib.pyplot as plt
import data_table as dt

def main():
    alpha = 0.5
    data = np.matrix("0. 1. 1.; 0. 0. 0.; 1. 0. 1.; 1. 1. 0.")
    theta = .01
    iter_limit = 50000
    mynet = nn.NeuralNet((2, 2, 1), sf.binary, sf.binary_prime, alpha)
    n = data.shape[0]
    total_count = 0
    k = -1
    fh = open('nn_xor.csv', 'w')
    plot_data = np.matrix("0,0")
    while True:
        total_count += 1
        k = (k+1) % n
        pattern = np.asarray(data[k,1:]).squeeze()
        teacher = [data[k,0]]
        error = mynet.train(pattern, teacher, theta)
        if (total_count % 10) == 0:
            plot_data = np.vstack((plot_data, np.matrix((total_count, error))))
            msg = ','.join([str(total_count), str(error)])
            fh.write(msg)
            fh.write('\n')
        if (total_count > iter_limit) or (error < theta):
            print("Stopped at iter count=>", total_count)
            break

    mynet.dt.tofile('bub_xor.txt')
    fh.close()
    plt.plot(plot_data[0:,0], plot_data[0:,1])
    plt.show()

def xor_boundary():
    """This function generates a plot of the xor decision
    boundary"""
    space = np.linspace(-1.,1.,256)
    t = dt.DataTable((2,2,1))
    t.fromfile('bub_xor.txt')
    mynet = nn.nn_factory(t, sf.binary, sf.binary_prime, 0.5)
    plot_data = np.matrix("0 0")
    for x1 in space:
        for x2 in space:
            point = np.matrix((x1, x2))
            out = mynet.fwd(point)
            if out < 0.02:
                plot_data = np.vstack((plot_data, point))

    x1 = np.asarray(plot_data[1:,0]).squeeze()
    x2 = np.asarray(plot_data[1:,1]).squeeze()
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.scatter(x1, x2)
    ax.axhline()
    ax.axvline()
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    plt.show()

if __name__ == '__main__':
    main()
