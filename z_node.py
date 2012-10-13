"""
Ted Satcher
CS 640
Fall 2012

Assignment 3

File: z_node.py

This file implements a hidden node.
"""

class Znode(object):
    """This class models a hidden node in a neural network."""

    def __init__(self, node_id, squash, squash_prime, dt):
        """
        Initialize the node.
        Args:
        id: The identifier of the node. The value should be
        1 <= id <= number of hidden nodes
        squash: The squashing function
        squash_prime: First derivative of the squashing function
        dt: The network data table.
        """
        self.id = node_id
        self.squash = squash
        self.squash_prime = squash_prime
        self.dt = dt

    def calc_output(self):
        """
        Calculate this node's output and store the result
        in the data table.
        """
        input_vec = self.dt.get_input_vec()
        v_wts = self.dt.get_input_to_hidden(self.id)
        net_in = input_vec.T * v_wts
        out = self.squash(net_in)
        self.dt.set_net_in_z(self.id, net_in)
        self.dt.set_z_out(self.id, out)

    def calc_delta(self):
        """
        Calculate the delta for this node.  Assumes the upstream
        output deltas have been calculated."""
        w_wts = self.dt.get_w_for_z_delta(self.id)
        y_deltas = self.dt.get_y_deltas()
        net_in = self.dt.get_net_in_z(self.id)
        delta = w_wts.T * y_deltas * self.squash_prime(net_in)
        self.dt.set_z_delta(self.id, delta)
