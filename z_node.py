"""
Ted Satcher
CS 640
Fall 2012

Assignment 3

File: z_node.py

This file implements a hidden node.
"""


def calc_output(node_id, data_table, f):
    """
    Calculates net in, output for this node and returns both as a tuple.
    """
    input_vec = data_table.get_input_vec()
    v_wts = data_table.get_input_to_hidden(node_id)
    net_in = input_vec.T * v_wts
    out = f(net_in)
    return (net_in, out)

def calc_delta(node_id, data_table, fp):
    """
    Calculate the delta for this node.  Assumes the upstream
    output deltas have been calculated."""
    w_wts = data_table.get_w_for_z_delta(node_id)
    y_deltas = data_table.get_y_deltas()
    net_in = data_table.get_net_in_z(node_id)
    delta = w_wts.T * y_deltas * fp(net_in)
    return delta
