'''
Created on Oct 23, 2013

@author: mkiyer
'''
import unittest

import random
import numpy as np
import matplotlib.pyplot as plt

from ssea.base import SampleSet
from ssea.algo import ssea_run, transform_weights
from ssea.kernel import ssea_kernel, run_lengths, rld2d

class TestAlgorithm(unittest.TestCase):

    def test_run_lengths(self):
        # empty list
        a = np.array([])
        self.assertEqual(run_lengths(a), [])
        # all different values
        a = np.arange(10, dtype=np.float)
        self.assertTrue(all(x == 1 for x in run_lengths(a)))
        # various lengths
        a = np.array([1,10,1,1,10,1,1,1,10,1,1,1,1,10], dtype=np.float)
        self.assertEqual(run_lengths(a), [1,1,2,1,3,1,4,1])

    def test_rld2d(self):
        # test lengths of one
        lengths = [1,1,1,1,1]
        nsets = 100
        shape = (len(lengths),nsets)
        arr = np.empty(shape, dtype=np.float)
        arr[:] = np.random.random_integers(-10, 10, shape)
        out = rld2d(lengths, arr)
        self.assertTrue(np.array_equal(out,arr))
        self.assertEqual(out.shape, (sum(lengths),nsets))
        # test various lengths
        lengths = [1,2,3,4,5]
        for i in xrange(len(lengths)):
            arr[i,:] = i+1
        out = rld2d(lengths, arr)
        correct = np.empty((sum(lengths),nsets))
        correct[0,:] = 1
        correct[1:3,:] = 2
        correct[3:6,:] = 3
        correct[6:10,:] = 4
        correct[10:15,:] = 5
        self.assertTrue(np.array_equal(out,correct))                

    def test_weight_methods(self):
        # setup samples
        N = 100
        samples = ['S%d' % x for x in xrange(N)] 
        # create membership arrays (sample sets)
        membership = np.zeros((len(samples),2), dtype=np.uint8)
        membership[:10,0] = 1
        membership[10:20,1] = 1
        # create weights
        weights = np.zeros(N, dtype=np.float)
        weights[:] = np.sort(np.arange(N))[::-1]
        # transform weights based on weight method
        weight_method = 'weighted'
        weights_miss = np.fabs(transform_weights(weights, weight_method))
        weights_hit = np.fabs(transform_weights(weights, weight_method))
        # run kernel
        es_vals, es_run_inds, es_runs = \
            ssea_kernel(weights, 
                        weights_miss,
                        weights_hit,
                        membership,
                        perm=np.arange(N))
        self.assertAlmostEqual(es_vals[0], 1.0)
        self.assertAlmostEqual(es_vals[1], 0.76979294)
        self.assertEqual(es_run_inds[0], 9)
        self.assertEqual(es_run_inds[1], 19)
#         print es_vals
#         print es_run_inds
#         print 'runs'
#         print es_runs

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