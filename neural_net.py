"""
Ted Satcher
CS 640
Fall 2012

Assignment 3

File: neural_net.py

This module and class serves as the public interface
to the neural network.
"""
import z_node as zn
import y_node as yn
import data_table as dt

class NeuralNet(object):
    """
    This class contains the nodes of the network.
    """

    def __init__(self, dimensions, squash, squash_prime,
                 learning_rate):
        """Initialize the nodes of the network.
        Args:
        dimensions: A tuple, input hidden and output node counts.
        squash: A squashing function
        squash_prime: First derivative of the squashing func.
        learning rate: The network learning rate.
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
                zn.Znode(i, squash, squash_prime, self.dt)
            )

        for i in range(1,self.output+1):
            self.ynodes.append(
                yn.Ynode(i, squash, squash_prime, self.dt)
            )


    def train(self, input_pattern, teacher):
        """Presents a pattern to the network and runs
        a forward and backward (weight adjustment) pass
        on the network.

        Args:
          input_pattern: A numpy vector to present to
              the network.  Dimensionality must match the
              network input dimensionality.

          teacher: A numpy vector that is the expected
              network output for the input pattern.

        Return:
          The error of
        """
        pass

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
