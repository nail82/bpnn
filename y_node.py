"""
Ted Satcher
CS 640
Fall 2012

Assignment 3

File: y_node.py

This file implements an output node.
"""

class Ynode(object):
    """A model of an output node."""
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
        """Calculate this node's output."""
        input_vec = self.dt.get_z_out()
        w_wts = self.dt.get_hidden_to_output(self.id)
        net_in = input_vec.T * w_wts
        out = self.squash(net_in)
        self.dt.set_net_in_y(self.id, net_in)
        self.dt.set_y_out(self.id, out)

    def calc_delta(self):
        """Calculate the delta of this node."""
        t = self.dt.get_teacher(self.id)
        assert(t is not None)
        y = self.dt.get_y_out(self.id)
        net_in = self.dt.get_net_in_y(self.id)
        delta = (t-y) * self.squash_prime(net_in)
        self.dt.set_y_delta(self.id, delta)
