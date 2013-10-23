'''
Created on Oct 23, 2013

@author: mkiyer
'''
import unittest

import random
import numpy as np
import matplotlib.pyplot as plt

from ssea.base import SampleSet
from ssea.algo import ssea_run

def generate_random_sample_sets(N, minsize, maxsize, population):
    names = []
    descs = []
    sample_sets = []
    for i in xrange(N):
        size = random.randint(minsize, maxsize)
        sample_set = map(str,random.sample(population, size))
        sample_sets.append(sample_set)
        names.append("SS%d" % (i))
        descs.append("Sample Set %d" % (i))
    return names, descs, sample_sets

class TestKernel(unittest.TestCase):

    def test_sign_weights(self):
        # setup samples
        N = 1000
        samples = ['S%d' % x for x in xrange(N)]        
        # make sample sets
        sample_sets = []
        for size in xrange(1, 1000, 20):
            name = 'SS%d' % (size)
            desc = 'sample set with size %d' % (size)
            value = set(samples[(N-size):N])
            sample_sets.append(SampleSet(name, desc, value))
        # test negative weights        
        weights = np.linspace(-1000, -1, num=N)
        results = ssea_run(samples, weights, sample_sets, 
                           weight_method_miss='weighted',
                           weight_method_hit='weighted',
                           perms=1000)
        for res in results:
            self.assertAlmostEqual(res.es, 1.0)
        # test positive weights
        weights = np.linspace(0, 1000, num=N)
        results = ssea_run(samples, weights, sample_sets, 
                           weight_method_miss='weighted',
                           weight_method_hit='weighted',
                           perms=1000)
        for res in results:
            self.assertAlmostEqual(res.es, 1.0)
        # test positive and negative weights
        weights = np.linspace(-800, 200, num=N)
        results = ssea_run(samples, weights, sample_sets, 
                           weight_method_miss='weighted',
                           weight_method_hit='weighted',
                           perms=1000)
        for res in results:
            self.assertAlmostEqual(res.es, 1.0)

        


if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.test_pos_weights']
    unittest.main()