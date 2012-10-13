from __future__ import print_function
import sys
import unittest
import data_table as dt
import numpy as np


class DataTableTest(unittest.TestCase):
    def setUp(self):
        self.in_nd = 2
        self.hid_nd = 3
        self.out_nd = 2
        self.t = dt.DataTable((self.in_nd, self.hid_nd, self.out_nd))
        self.t.set_test_weights()

    def tearDown(self):
        pass

    def testPrettyprint(self):
        self.t.prettyprint()

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


class ZnodeTest(unittest.TestCase):
    def setUp(self):
        pass

    def tearDown(self):
        pass

def suite():
    suite = unittest.TestLoader().loadTestsFromTestCase(DataTableTest)
    suite.addTests(
        unittest.TestLoader().loadTestsFromTestCase(ZnodeTest))
    return suite

if __name__ == '__main__':
    suite = suite()
    unittest.TextTestRunner(verbosity=2).run(suite)
