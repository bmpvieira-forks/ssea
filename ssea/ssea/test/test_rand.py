'''
Created on Nov 7, 2013

@author: mkiyer
'''
import unittest

# third party packages
import numpy as np

# local imports
import ssea.lib.kernel as kernel

class TestLCG(unittest.TestCase):

    def test_lcg_range(self):
        # magic seed will cause random number generator
        # to yield a double value of 1.0 and lead to 
        # array out of bounds exceptions 
        rng = kernel.RandomState(230538014)
        try:
            kernel.shufflei(np.arange(10), rng)
        except Exception:
            self.assertFalse(True)

    def test_poisson(self):
        x = np.empty(100000, dtype=np.float)
        x[:] = 4
        rng = kernel.RandomState(1383872316)
        kernel.resample_poisson(x, rng)
        h = np.histogram(x, bins=np.arange(20))[0]
        correct = [1777, 7263, 14648, 19577, 19792, 15433, 10468, 5985, 
                   2966, 1271, 523, 195, 76, 20, 5, 0, 0, 1, 0]
        self.assertEqual(list(h), correct)

    def test_gaussian(self):
        x = np.zeros(100000, dtype=np.float)
        #seed = kernel.lcg_seed()
        rng = kernel.RandomState(1383870569)
        kernel.add_gaussian_noise(x, rng, 10.0, 1.0)
        h = np.histogram(x, bins=np.linspace(0, 20, 100))[0]
        correct = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
                   0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 4, 3, 16, 22, 35, 83, 
                   139, 263, 413, 701, 1007, 1573, 2182, 2932, 3835, 4926, 
                   5991, 6585, 7274, 7888, 8100, 7811, 7505, 6808, 5782, 
                   4887, 3859, 2954, 2189, 1546, 1024, 663, 428, 239, 149, 
                   84, 54, 25, 10, 7, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
                   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        self.assertTrue(np.array_equal(h, correct))

    def test_shuffle(self):
        a = np.zeros(100, dtype=np.float)
        a[0] = 1
        b = np.zeros(a.shape[0], dtype=np.float)        
        rng = kernel.RandomState(1383867927)
        for i in xrange(100000):
            kernel.shufflef(a, rng)
            b += a
        # ensure values are reproducible
        correct = [991.0, 1041.0, 989.0, 979.0, 985.0, 1027.0, 1053.0, 
                   1033.0, 991.0, 946.0, 1013.0, 979.0, 978.0, 1023.0, 
                   994.0, 1033.0, 952.0, 1048.0, 1002.0, 1061.0, 1025.0, 
                   1019.0, 980.0, 1018.0, 986.0, 968.0, 1040.0, 981.0, 
                   1015.0, 1028.0, 1014.0, 1031.0, 1050.0, 985.0, 1001.0, 
                   971.0, 1000.0, 1004.0, 935.0, 1067.0, 976.0, 953.0, 
                   947.0, 989.0, 981.0, 1016.0, 956.0, 1003.0, 924.0, 
                   1014.0, 1016.0, 929.0, 966.0, 999.0, 984.0, 1033.0, 
                   957.0, 1028.0, 1026.0, 976.0, 1043.0, 973.0, 1047.0, 
                   967.0, 983.0, 993.0, 1005.0, 1009.0, 998.0, 966.0, 
                   1014.0, 1020.0, 1008.0, 1041.0, 1044.0, 1001.0, 932.0, 
                   1034.0, 972.0, 1048.0, 1007.0, 1007.0, 1011.0, 1049.0, 
                   1027.0, 1012.0, 1002.0, 953.0, 975.0, 1021.0, 1019.0, 
                   947.0, 1025.0, 1033.0, 1013.0, 997.0, 980.0, 1001.0, 
                   971.0, 943.0]
        self.assertTrue(np.array_equal(b, correct))


if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()