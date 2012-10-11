from __future__ import print_function
import sys
import unittest
import data_table as dt


class DataTableTest(unittest.TestCase):
    def setUp(self):
        pass

    def tearDown(self):
        pass

    def testIsWorking(self):
        mytable = dt.DataTable()
        self.assertTrue(mytable.is_working())


def suite():
    suite = unittest.TestLoader().loadTestsFromTestCase(DataTableTest)
    return suite

if __name__ == '__main__':
    suite = suite()
    unittest.TextTestRunner(verbosity=2).run(suite)
