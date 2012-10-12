from __future__ import print_function
import sys
import unittest
import data_table as dt
import numpy as np


class DataTableTest(unittest.TestCase):
    def setUp(self):
        self.t = dt.DataTable((2,3,2))
        pass

    def tearDown(self):
        pass

    def testPrettyprint(self):
        #self.t.prettyprint()
        pass

    def testSetInput(self):
        X = np.array((2.,3.))
        self.t.set_input_vec(X)
        vec = self.t.get_input_vec()
        self.assertEqual(1., vec[0,0])
        self.assertEqual(2., vec[1,0])
        self.assertEqual(3., vec[2,0])

    def testGetInputToHidden(self):
        pass

def suite():
    suite = unittest.TestLoader().loadTestsFromTestCase(DataTableTest)
    return suite

if __name__ == '__main__':
    suite = suite()
    unittest.TextTestRunner(verbosity=2).run(suite)
