'''
Created on Dec 13, 2013

@author: mkiyer
'''
import unittest

# third party packages
import numpy as np

# local imports
from ssea.lib.base import Result

class TestAlgo(unittest.TestCase):

    def test_default_result(self):
        result = Result.default()
        self.assertTrue(result.t_id is None)
        self.assertTrue(result.ss_rank is None)
      


if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()