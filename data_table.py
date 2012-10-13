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
        self.output = dimensions[2]

        # Define the shape of the vectors in the table
        input_vec_shape = (self.input+1, 1)
        v_wts_shape     = (input_vec_shape[0] * self.hidden, 1)
        net_in_z_shape  = (self.hidden, 1)
        z_out_shape     = (self.hidden+1, 1)
        w_wts_shape     = (z_out_shape[0]*self.output, 1)
        net_in_y_shape  = (self.output, 1)
        y_out_shape     = (self.output, 1)
        deltas_shape    = (self.hidden+self.output, 1)

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
        self.teacher   = [None for x in range(0,self.output)]

        # Initialize the bias units
        self.input_vec[0,0] = 1.
        self.z_out[0,0] = 1.

    def get_input_vec(self):
        """Returns the input vector."""
        return self.input_vec

    def set_input_vec(self,X):
        """Place an input vector on the input nodes of
        the network."""
        assert(len(X) == self.input)
        for i in range(0, self.input):
            self.input_vec[i+1,0] = X[i];

    def get_v_vector(self):
        """Return the entire input to hidden weight vector."""
        return self.v_wts

    def update_input_to_hidden(self, v_wts):
        """Replace the input to hidden weights."""
        assert(self.v_wts.shape == v_wts.shape)
        self.v_wts = v_wts

    def get_w_vector(self):
        """Return the hiddent to output weight vector."""
        return self.w_wts

    def update_hidden_to_output(self, w_wts):
        "Replace the hidden to output weights."""
        assert(self.w_wts.shape == w_wts.shape)
        self.w_wts = w_wts

    def get_input_to_hidden(self, z):
        """Returns the weights leading to a hidden node
        from all the input nodes."""
        assert(0 < z <= self.hidden)
        z = z-1
        start = (self.hidden)*z
        stop  = start+self.hidden
        return self.v_wts[start:stop,0]

    def get_hidden_to_output(self, y):
        """Returns the weights leading to an output node."""
        assert(0 < y <= self.output)
        y = y-1
        start = (self.hidden+1)*y
        stop = start+self.hidden+1
        return self.w_wts[start:stop,0]

    def get_net_in_z(self, z):
        """Returns the net input to a z node."""
        assert(0 < z <= self.hidden)
        return self.net_in_z[z-1, 0]

    def set_net_in_z(self, z, net_in):
        """Set the net input of a z node."""
        assert(0 < z <= self.hidden)
        self.net_in_z[z-1,0] = net_in

    def get_z_out(self):
        """Return the outputs of the hidden nodes
        and the hidden bias."""
        return self.z_out

    def set_z_out(self, z, out):
        """Set the output of a z node."""
        assert(0 < z <= self.hidden)
        self.z_out[z,0] = out

    def get_net_in_y(self, y):
        """Returns the net input to a y node."""
        assert(0 < y <= self.output)
        return self.net_in_y[y-1, 0]

    def set_net_in_y(self, y, net_in):
        """Set the net input of a y node."""
        assert(0 < y <= self.output)
        self.net_in_y[y-1,0] = net_in

    def get_y_deltas(self):
        return self.deltas[0:self.output,0]

    def set_y_delta(self, y, delta):
        assert(0 < y <= self.output)
        self.deltas[y-1,0] = delta

    def get_z_deltas(self):
        return self.deltas[self.output:,0]

    def set_z_delta(self, z, delta):
        assert(0 < z <= self.hidden)
        index = self.output + z - 1
        self.deltas[index,0] = delta

    def get_teacher(self, y):
        """Get the teacher value for y node."""
        assert(0 < y <= self.output)
        v = self.teacher[y-1]
        # Set this teacher to None in prep for the next round
        self.teacher[y-1] = None
        return v

    def get_output(self):
        return self.y_out

    def get_y_out(self, y):
        """Return the output for a particular y node."""
        assert(0 < y <= self.hidden)
        return self.y_out[y-1,0]

    def set_y_out(self, y, out):
        """Set the output of an output node."""
        assert(0 < y <= self.hidden)
        self.y_out[y-1,0] = out

    def set_teacher(self, t):
        assert(len(t) == len(self.teacher))
        for i in range(len(t)):
            self.teacher[i] = t[i]


    def set_test_weights(self):
        """Set the weight vectors and deltas to deterministic
        values for testing access functions"""
        for i in range(0,self.v_wts.shape[0]):
            self.v_wts[i,0] = (i+1)/10.

        for i in range(0,self.w_wts.shape[0]):
            self.w_wts[i,0] = (i+1)/10.

        for i in range(0,self.deltas.shape[0]):
            self.deltas[i,0] = (i+1)/10.

    def get_w_for_z_delta(self,z):
        """Returns the hidden to output weights associated
        with a delta z calculation."""
        assert(0 < z <= self.hidden)
        step = self.w_wts.shape[0] / self.output
        return self.w_wts[z::step,0]

    def prettyprint(self):
        float_fmt = '{value:0.4f}'
        print()
        # Inputs
        msglines = []
        for i in range(0,self.input_vec.shape[0]):
            fmt = ''.join(['X{index} = ',float_fmt])
            msg = fmt.format(
                index=i, value=self.input_vec[i,0])
            msglines.append(msg)
        print('\n'.join(msglines))
        print()
        msglines = []
        for i in range(1,self.hidden+1):
             v_vec = self.get_input_to_hidden(i)
             for j in range(0,self.hidden):
                 fmt = ''.join(['V{in_node}{hid} = ',float_fmt])
                 msg = fmt.format(
                     in_node=j, hid=i, value=v_vec[j,0])
                 msglines.append(msg)
        print('\n'.join(msglines))
        print()

        # Hidden to output
        msglines = []
        for i in range(1,self.output+1):
            w_vec = self.get_hidden_to_output(i)
            for j in range(0,self.hidden+1):
                fmt = ''.join(['W{hid}{out_node} = ',float_fmt])
                msg =  fmt.format(
                    hid=j, out_node=i, value=w_vec[j,0])
                msglines.append(msg)
        print('\n'.join(msglines))
        print()

        # net_in_z
        msglines = []
        for i in range(0,self.hidden):
            fmt = ''.join(['net_in_Z{hid} = ', float_fmt])
            msg = fmt.format(
                hid=i+1, value=self.net_in_z[i,0])
            msglines.append(msg)
        print('\n'.join(msglines))
        print()

        # z out
        msglines = []
        for i in range(0,self.hidden+1):
            fmt = ''.join(['z{hid} = ', float_fmt])
            msg = fmt.format(
                hid=i, value=self.z_out[i,0])
            msglines.append(msg)
        print('\n'.join(msglines))
        print()

        # net_in_y
        msglines = []
        for i in range(0,self.output):
            fmt = ''.join(['net_in_Y{out} = ', float_fmt])
            msg = fmt.format(
                out=i+1, value=self.net_in_y[i,0])
            msglines.append(msg)
        print('\n'.join(msglines))
        print()

        # y out
        msglines = []
        for i in range(0,self.output):
            fmt = ''.join(['y{out} = ', float_fmt])
            msg = fmt.format(
                out=i+1, value=self.y_out[i,0])
            msglines.append(msg)
        print('\n'.join(msglines))
        print()

        # output node deltas
        msglines = []
        y_deltas = self.get_y_deltas()
        for i in range(0,self.output):
            fmt = ''.join(['y{out}_delta = ',float_fmt])
            msg = fmt.format(
                out=i+1, value=y_deltas[i,0])
            msglines.append(msg)
        print('\n'.join(msglines))
        print()

        # hidden node deltas
        msglines = []
        z_deltas = self.get_z_deltas()
        for i in range(0, self.hidden):
            fmt = ''.join(['z{out}_delta = ',float_fmt])
            msg = fmt.format(
                out=i+1, value=z_deltas[i,0])
            msglines.append(msg)
        print('\n'.join(msglines))
        print()
