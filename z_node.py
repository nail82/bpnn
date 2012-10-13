"""
Ted Satcher
CS 640
Fall 2012

Assignment 3

File: z_node.py
"""

class Znode(object):
    """This class models a hidden node in a neural network."""

    def __init__(self, id, squash, squash_prime, dt):
        """
        Initialize the node.
        Args:
        id: The identifier of the node. The value should be
        1 <= id <= number of hidden nodes
        squash: The squashing function
        squash_prime: First derivative of the squashing function
        dt: The network data table.
        """
        self.id = id
        self.squash = squash
        self.squash_prime = squash_prime
        self.dt = dt

    def calc_output():
        """
        Calculate this node's output and store the result
        in the data table.
        """
        input_vec = dt.get_input_vec()
        v_wts = dt.get_input_to_hidden(self.id)
        net_in = input_vec.T * v_wts
        out = squash(net_in)
        dt.set_net_in_z(self.id, net_in)
        dt.set_z_out(self.id, out)
