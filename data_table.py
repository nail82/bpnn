"""
Ted Satcher
CS 640
Fall 2012

Assignment 3

File: data_table.py

This module implements the data table for the neural net.
"""
from __future__ import print_function
import numpy as np

class DataTable(object):
    """
    This class is the data repository for the neural network.
    It contains all the weights and intermediate calculations
    for the network.  This object does no calculations on
    the data, rather it just provides methods to clients
    to access the data.
    """
    def __init__(self, dimensions):
        """
        Initialize the data table.
        Args:
          dimensions: A tuple of input nodes, hidden nodes
            and output nodes.
        """
        input_nodes  = dimensions[0]
        hidden_nodes = dimensions[1]
        output_nodes = dimensions[2]

        # Define the shape of the vectors in the table
        input_vec_shape = (input_nodes+1, 1)
        v_wts_shape     = (input_vec_shape[0] * hidden_nodes, 1)
        net_in_z_shape  = (hidden_nodes, 1)
        z_out_shape     = (hidden_nodes+1, 1)
        w_wts_shape     = (z_out_shape[0]*output_nodes, 1)
        net_in_y_shape  = (output_nodes, 1)
        y_out_shape     = (output_nodes, 1)
        deltas_shape    = (hidden_nodes+output_nodes, 1)
        teach_shape     = (output_nodes, 1)

        b = -0.5
        a = 0.5
        init_v_wts = (b-a) * np.random.random_sample(v_wts_shape[0])
        init_w_wts = (b-a) * np.random.random_sample(w_wts_shape[0])

        self.input_vec = np.asmatrix(np.zeros(input_vec_shape)).T
        self.v_wts     = np.asmatrix(init_v_wts).T
        self.net_in_z  = np.asmatrix(np.zeros(net_in_z_shape)).T
        self.z_out     = np.asmatrix(np.z_out(z_out_shape)).T
        self.w_wts     = np.asmatrix(init_w_wts).T
        self.net_in_y  = np.asmatrix(np.zeros(net_in_y_shape)).T
        self.y_out     = np.asmatrix(np.zeros(y_out_shape)).T
        self.deltas    = np.asmatrix(np.zeros(deltas_shape)).T
        self.teach     = np.asmatrix(np.zeros(teach_shape)).T



    def is_working(self):
        return True
