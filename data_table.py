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
          dimensions: A tuple of representing input nodes, hidden nodes
            and output nodes.
        """
        self.input  = dimensions[0]
        self.hidden = dimensions[1]
        self.output_nodes = dimensions[2]

        # Define the shape of the vectors in the table
        input_vec_shape = (self.input+1, 1)
        v_wts_shape     = (input_vec_shape[0] * self.hidden, 1)
        net_in_z_shape  = (self.hidden, 1)
        z_out_shape     = (self.hidden+1, 1)
        w_wts_shape     = (z_out_shape[0]*self.output_nodes, 1)
        net_in_y_shape  = (self.output_nodes, 1)
        y_out_shape     = (self.output_nodes, 1)
        deltas_shape    = (self.hidden+self.output_nodes, 1)
        teach_shape     = (self.output_nodes, 1)

        b = -0.5
        a = 0.5
        init_v_wts = (b-a) * np.random.random_sample(v_wts_shape[0])
        init_w_wts = (b-a) * np.random.random_sample(w_wts_shape[0])

        self.input_vec = np.asmatrix(np.zeros(input_vec_shape))
        self.v_wts     = np.asmatrix(init_v_wts).T
        self.net_in_z  = np.asmatrix(np.zeros(net_in_z_shape))
        self.z_out     = np.asmatrix(np.zeros(z_out_shape))
        self.w_wts     = np.asmatrix(init_w_wts).T
        self.net_in_y  = np.asmatrix(np.zeros(net_in_y_shape))
        self.y_out     = np.asmatrix(np.zeros(y_out_shape))
        self.deltas    = np.asmatrix(np.zeros(deltas_shape))
        self.teach     = np.asmatrix(np.zeros(teach_shape))

        # Initialize the bias units
        self.input_vec[0,0] = 1.
        self.z_out[0,0] = 1.

    def prettyprint(self):
        print()
        for i in range(0,self.input_vec.shape[0]):
            msg = ''.join(['X[',str(i),']',str(self.input_vec[i,0])])
            print(msg)
        for i in range(1,self.hidden+1):
            pass
        print("V Wts    :", self.v_wts.T)

        print("Net in Z :", self.net_in_z.T)
        print("Z out    :", self.z_out.T)
        print("W Wts    :", self.w_wts.T)
        print("Net in Y :", self.net_in_y.T)
        print("Y out    :", self.y_out.T)
        print("Deltas   :", self.deltas.T)
        print("Teachers :", self.teach.T)

    def set_input_vec(self,X):
        """Place an input vector on the input nodes of
        the network."""
        assert(len(X) == self.input)
        for i in range(0, self.input):
            self.input_vec[i+1,0] = X[i];

    def get_input_vec(self):
        """Returns the input vector."""
        return self.input_vec

    def get_input_to_hidden(self, z):
        """Returns the weights leading to a hidden node
        from all the input nodes."""
        z = z-1
        start = (self.hidden+1)*z
        stop  = start+self.hidden+1
        return self.v_wts[start:stop,0]

    def get_hidden_to_output(self, y):
        """Returns the weights leading to an output node."""
        pass

    def get_net_in_z(self, z):
        """Returns the net input to a z node."""
        pass

    def get_net_in_y(self, y):
        """Returns the net input to a y node."""
        pass
