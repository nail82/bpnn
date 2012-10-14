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

    def testAAinputToHidden(self):
        t = dt.DataTable((2,2,1))
        v = t.get_input_to_hidden(1)
        t = dt.DataTable((2,3,2))
        v = t.get_input_to_hidden(1)

    def testSetInput(self):
        X = np.array((2.,3.))
        self.t.set_input_vec(X)
        vec = self.t.get_input_vec()
        self.assertEqual(1., vec[0,0])
        self.assertEqual(2., vec[1,0])
        self.assertEqual(3., vec[2,0])

    def testGetInputToHidden(self):
        v_vec = self.t.get_input_to_hidden(1)
        start = .1
        for i in range(0,3):
            self.assertAlmostEqual(start, v_vec[i])
            start += .1

        v_vec = self.t.get_input_to_hidden(2)
        for i in range(0,3):
            self.assertAlmostEqual(start, v_vec[i])
            start += .1

        v_vec = self.t.get_input_to_hidden(3)
        for i in range(0,3):
            self.assertAlmostEqual(start, v_vec[i])
            start += .1

    def testGetInputToHidden2(self):
        t = dt.DataTable((2,2,1))
        v_wts = t.get_input_to_hidden(1)
        self.assertEqual(v_wts.shape, (3,1))

    def testGetHiddenToOutput(self):
        w_vec = self.t.get_hidden_to_output(1)
        start = .1
        for i in range(0,4):
            self.assertAlmostEqual(start, w_vec[i], 3)
            start += .1

        w_vec = self.t.get_hidden_to_output(2)
        for i in range(0,4):
            self.assertAlmostEqual(start, w_vec[i], 3)
            start += .1

    def testNetInZ(self):
        for i in range(1, self.hid_nd+1):
            self.t.set_net_in_z(i, i)
        self.assertEqual(1, self.t.get_net_in_z(1))
        self.assertEqual(2, self.t.get_net_in_z(2))
        self.assertEqual(3, self.t.get_net_in_z(3))

    def testZout(self):
        for i in range(1, self.hid_nd+1):
            self.t.set_z_out(i, i)
        z_out = self.t.get_z_out()
        self.assertEqual(1, z_out[0])
        self.assertEqual(1, z_out[1])
        self.assertEqual(2, z_out[2])
        self.assertEqual(3, z_out[3])

    def testNetInY(self):
        for i in range(1, self.out_nd+1):
            self.t.set_net_in_y(i, i)
        self.assertEqual(1, self.t.get_net_in_y(1))
        self.assertEqual(2, self.t.get_net_in_y(2))

    def testYdeltas(self):
        self.t.set_y_delta(1, .1)
        self.t.set_y_delta(2, .2)
        y_deltas = self.t.get_y_deltas()
        for i in range(1, self.out_nd+1):
            self.assertAlmostEqual((i)/10., y_deltas[i-1,0], 3)

    def testZdeltas(self):
        self.t.set_z_delta(1, .1)
        self.t.set_z_delta(2, .2)
        self.t.set_z_delta(3, .3)
        z_deltas = self.t.get_z_deltas()
        for i in range(1, self.hid_nd+1):
             self.assertEqual(i/10., z_deltas[i-1])

    def testTeacher(self):
        self.t.set_teacher(np.array((1.,2.)))
        for i in range(1,3):
            tch = self.t.get_teacher(i)
            self.assertEqual(i, tch)
            tch = self.t.get_teacher(i)
            self.assertTrue(tch is None)

    def testGetWforZdelta(self):
        ldt = dt.DataTable((2,3,3))
        ldt.set_test_weights()
        for i in range(1,4):
            w = ldt.get_w_for_z_delta(i)
            print(i,"=>",w)
            print()



class ZnodeTest(unittest.TestCase):
    def setUp(self):
        self.dt = dt.DataTable((2,3,2))
        self.dt.set_test_weights()
        self.z_node = zn.Znode(1, sf.binary, sf.binary_prime, self.dt)

    def tearDown(self):
        pass

    def testCalcOutput(self):
        self.dt.set_input_vec(np.array((1.,2.)))
        self.z_node.calc_output()
        self.assertAlmostEqual(0.9, self.dt.get_net_in_z(1))
        z_out = self.dt.get_z_out()
        self.assertAlmostEqual(.711, z_out[1], 3)

    def testCalcDelta(self):
        self.dt.set_net_in_z(1, .3)
        self.dt.set_y_delta(1, 4.)
        self.dt.set_y_delta(2, 5.)
        self.z_node.calc_delta()
        deltas = self.dt.get_z_deltas()
        self.assertAlmostEqual(1.8176, deltas[0],4)

class YnodeTest(unittest.TestCase):
    def setUp(self):
        self.dt = dt.DataTable((2,3,2))
        self.dt.set_test_weights()
        self.y_node = yn.Ynode(1, sf.binary, sf.binary_prime, self.dt)

    def tearDown(self):
        pass

    def testCalcOutput(self):
        self.dt.set_z_out(1, 1.)
        self.dt.set_z_out(2, 2.)
        self.dt.set_z_out(3, 3.)
        self.y_node.calc_output()
        self.assertAlmostEqual(2.1, self.dt.get_net_in_y(1), 3)
        self.assertAlmostEqual(0.8909, self.dt.get_y_out(1), 4)

    def testCalcDelta(self):
        self.dt.set_z_out(1, 1.)
        self.dt.set_z_out(2, 2.)
        self.dt.set_z_out(3, 3.)
        self.y_node.calc_output()
        self.assertAlmostEqual(0.8909, self.dt.get_y_out(1), 3)
        self.dt.set_teacher(np.array((1,2)))
        self.y_node.calc_delta()
        deltas = self.dt.get_y_deltas()
        self.assertAlmostEqual(0.0132, deltas[0], 3)



def suite():
    suite = unittest.TestLoader().loadTestsFromTestCase(DataTableTest)
    suite.addTests(
        unittest.TestLoader().loadTestsFromTestCase(ZnodeTest))
    suite.addTests(
        unittest.TestLoader().loadTestsFromTestCase(YnodeTest))
    return suite

if __name__ == '__main__':
    suite = suite()
    unittest.TextTestRunner(verbosity=2).run(suite)
