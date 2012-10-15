"""
Ted Satcher
CS 640
Fall 2012

Assignment 3

File: y_node.py

This file implements an output node.
"""


def calc_output(node_id, data_table, f):
    """Calculate this node's output."""
    input_vec = data_table.get_z_out()
    w_wts = data_table.get_hidden_to_output(node_id)
    net_in = input_vec.T * w_wts
    out = f(net_in)
    return (net_in, out)

def calc_delta(node_id, data_table, fp):
    """Calculate the delta of this node."""
    t = data_table.get_teacher(node_id)
    assert(t is not None)
    y = data_table.get_y_out(node_id)
    net_in = data_table.get_net_in_y(node_id)
    delta = (t-y) * fp(net_in)
    return delta
