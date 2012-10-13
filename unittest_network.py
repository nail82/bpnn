"""
Unit tests for the network.
"""
from __future__ import print_function
import sys
import unittest
import data_table as dt
import numpy as np
import z_node as zn
import y_node as yn
import squash_funcs as sf


class DataTableTest(unittest.TestCase):
    def setUp(self):
        self.in_nd = 2
        self.hid_nd = 3
        self.out_nd = 2
        self.t = dt.DataTable((self.in_nd, self.hid_nd, self.out_nd))
        self.t.set_test_weights()

    def tearDown(self):
        pass

    def testSetInput(self):
        X = np.array((2.,3.))
        self.t.set_input_vec(X)
        vec = self.t.get_input_vec()
        self.assertEqual(1., vec[0,0])
        self.assertEqual(2., vec[1,0])
        self.assertEqual(3., vec[2,0])

    def testGetInputToHidden(self):
        v_vec = self.t.get_input_to_hidden(1)
        start = 0
        for i in range(0,3):
            self.assertEqual(start, v_vec[i])
            start += 1

        v_vec = self.t.get_input_to_hidden(2)
        for i in range(0,3):
            self.assertEqual(start, v_vec[i])
            start += 1

        v_vec = self.t.get_input_to_hidden(3)
        for i in range(0,3):
            self.assertEqual(start, v_vec[i])
            start += 1

    def testGetHiddenToOutput(self):
        w_vec = self.t.get_hidden_to_output(1)
        start = 0
        for i in range(0,4):
            self.assertEqual(start, w_vec[i])
            start += 1

        w_vec = self.t.get_hidden_to_output(2)
        for i in range(0,4):
            self.assertEqual(start, w_vec[i])
            start += 1

    def testGetYdeltas(self):
        y_deltas = self.t.get_y_deltas()
        for i in range(1, self.out_nd+1):
            self.assertEqual(i-1, y_deltas[i-1])

    def testGetZdeltas(self):
        z_deltas = self.t.get_z_deltas()
        for i in range(1, self.hid_nd+1):
             self.assertEqual(i+1, z_deltas[i-1])

    def testSetYdelta(self):
        mytable = dt.DataTable((2,3,2))
        mytable.set_y_delta(1, 1.)
        mytable.set_y_delta(2, 2.)
        y_deltas = mytable.get_y_deltas()
        for i in range(1,3):
            self.assertEqual(y_deltas[i-1], i)

    def testSetZdelta(self):
        mytable = dt.DataTable((2,3,2))
        mytable.set_z_delta(1, 1.)
        mytable.set_z_delta(2, 2.)
        mytable.set_z_delta(3, 3.)
        z_deltas = mytable.get_z_deltas()
        for i in range(1,4):
            self.assertEqual(z_deltas[i-1], i)



class ZnodeTest(unittest.TestCase):
    def setUp(self):
        self.dt = dt.DataTable((2,3,2))
        self.z_node = zn.Znode(1, sf.binary, sf.binary_prime, self.dt)

    def tearDown(self):
        pass

    def testCalcOutput(self):
        self.dt.set_input_vec(np.array((1.,2.)))
        print("\nBEFORE:")
        self.dt.prettyprint()
        self.z_node.calc_output()
        print("\nAFTER:")
        self.dt.prettyprint()

class YnodeTest(unittest.TestCase):
    def setUp(self):
        self.dt = dt.DataTable((2,3,2))
        self.y_node = yn.Ynode(1, sf.binary, sf.binary_prime, self.dt)

    def tearDown(self):
        pass

    def testCalcOutput(self):
        self.dt.set_z_out(1, 1.)
        self.dt.set_z_out(2, 2.)
        self.dt.set_z_out(3, 3.)
        print("\nBEFORE:")
        self.dt.prettyprint()
        self.y_node.calc_output()
        print("\nAFTER:")
        self.dt.prettyprint()



def suite():
    suite = unittest.TestLoader().loadTestsFromTestCase(DataTableTest)
    suite.addTests(
        unittest.TestLoader().loadTestsFromTestCase(ZnodeTest))
    return suite

if __name__ == '__main__':
    suite = suite()
    unittest.TextTestRunner(verbosity=2).run(suite)
