"""
Ted Satcher
CS 640
Fall 2012

Assignment 3

File: neural_net.py

This module and class serves as the public interface
to the neural network.
"""
from __future__ import print_function
import z_node as zn
import y_node as yn
import data_table as dt
import numpy as np

class NeuralNet(object):
    """
    This class contains the nodes of the network.
    """

    def __init__(self, dimensions, squash, squash_prime,
                 alpha):
        """Initialize the nodes of the network.
        Args:
        dimensions: A tuple, input hidden and output node counts.
        squash: A squashing function
        squash_prime: First derivative of the squashing func.
        alpha: The network learning rate.
        """
        self.dt = dt.DataTable(dimensions)
        self.input = dimensions[0]
        self.hidden = dimensions[1]
        self.output = dimensions[2]
        self.sq = squash
        self.sqp = squash_prime
        self.znodes = []
        self.ynodes = []
        for i in range(1,self.hidden+1):
            self.znodes.append(
                zn.Znode(i, squash, squash_prime, self.dt))

        for i in range(1,self.output+1):
            self.ynodes.append(
                yn.Ynode(i, squash, squash_prime, self.dt))
        self.alpha = alpha

    def train(self, input_pattern, teacher, theta):
        """Presents a pattern to the network and runs
        a forward pass and then calculates deltas for
        the input and hidden nodes in the network.

        Args:
          input_pattern: A numpy vector to present to
              the network.  Dimensionality must match the
              network input dimensionality.

          teacher: A numpy vector that is the expected
              network output for the input pattern.

          theta: If the network error is greater than theta, the
              weigth updates are calculated and applied to the network.

        Return:
          The network error from the fwd pass
        """
        output_vec = self.fwd(input_pattern)
        error = np.linalg.norm(output_vec - teacher)
        if (error > theta):
            self.dt.set_teacher(teacher)
            for y in self.ynodes:
                y.calc_delta()
            for z in self.znodes:
                z.calc_delta()
            delta_w_wts = self.calc_delta_w()
            delta_v_wts = self.calc_delta_v()
            w_update = self.dt.get_w_vector() + delta_w_wts
            v_update = self.dt.get_v_vector() + delta_v_wts
            self.dt.update_hidden_to_output(w_update)
            self.dt.update_input_to_hidden(v_update)
        return error

    def calc_delta_w(self):
        """Calculates the deltas for hidden to output weights.

        Return:
          A vector of delta weights.
        """
        delta_w_wts = np.matrix("0.")
        y_deltas = self.dt.get_y_deltas()
        z_out = self.dt.get_z_out()
        for yd in y_deltas:
            delta_w_wts = np.vstack(
                (delta_w_wts, yd[0,0] * self.alpha * z_out))
        return delta_w_wts[1:,0]

    def calc_delta_v(self):
        """Calculates the deltas for input to hidden weights.

        Return:
          A vector of delta weights.
        """
        delta_v_wts = np.matrix("0.")
        z_deltas = self.dt.get_z_deltas()
        input_vec = self.dt.get_input_vec()
        for zd in z_deltas:
            delta_v_wts = np.vstack(
                (delta_v_wts, zd[0,0] * self.alpha * input_vec))
        return delta_v_wts[1:,0]

    def fwd(self, input_pattern):
        """Run an input pattern through the forward
        pass of the network and return the network output.
        """
        self.dt.set_input_vec(input_pattern)

        for z in self.znodes:
            z.calc_output()

        for y in self.ynodes:
            y.calc_output()

        return self.dt.get_output()
