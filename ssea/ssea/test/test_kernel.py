'''
Created on Nov 7, 2013

@author: mkiyer
'''
import unittest

# third party packages
import numpy as np

# local imports
from ssea.lib.base import INT_DTYPE, WeightMethod
import ssea.lib.kernel as kernel

class TestKernel(unittest.TestCase):

    def test_power_transform(self):
        size = 100
        # unweighted
        a = np.linspace(0, 100, size)
        b = kernel.power_transform(a, WeightMethod.UNWEIGHTED)
        self.assertTrue(np.array_equal(b, np.ones(size,dtype=np.float)))
        # weighted
        b = kernel.power_transform(a, WeightMethod.WEIGHTED)
        self.assertTrue(np.array_equal(a,b))
        # exponent
        b = kernel.power_transform(a, WeightMethod.EXP, 0.5)
        correct = a ** 0.5
        self.assertTrue(np.array_equal(b,correct))        
        # log
        b = kernel.power_transform(a, WeightMethod.LOG, 1.0)
        correct = np.log2(a+1.0)
        self.assertTrue(np.array_equal(b,correct))

    def test_random_walk_empty(self):
        '''
        test trying to run random walk with an empty array
        '''
        weights_hit = np.array([], dtype=np.float)
        weights_miss = np.array([], dtype=np.float)
        membership = np.empty((0,), dtype=INT_DTYPE)
        ranks = np.argsort(weights_hit)[::-1]
        perm = np.arange(0)
        es_val, es_rank, es_run = \
            kernel.random_walk(weights_miss, weights_hit, membership, ranks, perm)
        self.assertTrue(es_val == 0)
        self.assertTrue(es_rank == 0)
        self.assertTrue(np.all(es_run == 0.0))

    def test_kernel_empty(self):        
        '''
        test trying to run kernel with an empty array
        '''
        counts = np.array([], dtype=np.float)
        size_factors = np.array([], dtype=np.float)
        membership = np.empty((0,), dtype=INT_DTYPE)
        rng = kernel.RandomState()
        k = kernel.ssea_kernel(counts, size_factors, membership, rng,
                               resample_counts=True,
                               permute_samples=True,
                               add_noise=True,
                               noise_loc=1.0,
                               noise_scale=1.0,
                               method_miss=3,
                               method_hit=3,
                               method_param=1.0)
        (ranks, norm_counts, norm_counts_miss, norm_counts_hit, 
        es_val, es_rank, es_run) = k
        self.assertTrue(len(ranks) == 0)
        self.assertTrue(len(norm_counts) == 0)
        self.assertTrue(len(norm_counts_miss) == 0)
        self.assertTrue(len(norm_counts_hit) == 0)
        self.assertTrue(es_val == 0)
        self.assertTrue(es_rank == 0)
        self.assertTrue(es_run.shape == (0,))
         
    def test_random_walk_boundary_cases(self):
        nsamples = 100
        # test all zeros
        weights_hit = np.zeros(nsamples, dtype=np.float)
        weights_miss = np.zeros(nsamples, dtype=np.float)
        membership = np.zeros(nsamples, dtype=INT_DTYPE)
        ranks = np.argsort(weights_hit)[::-1]
        perm = np.arange(nsamples, dtype=np.int)        
        es_val, es_rank, es_run = \
            kernel.random_walk(weights_miss, weights_hit, membership, 
                               ranks, perm)        
        self.assertTrue(es_val == 0.0)
        self.assertTrue(es_rank == 0.0)
        self.assertTrue(es_run.shape[0] == nsamples)
        # nonzero weights but empty membership
        weights_hit = np.linspace(1, 10, nsamples)
        weights_miss = weights_hit.copy()
        membership = np.zeros(nsamples, dtype=INT_DTYPE)
        ranks = np.argsort(weights_hit)[::-1]
        perm = np.arange(nsamples, dtype=np.int)        
        es_val, es_rank, es_run = \
            kernel.random_walk(weights_miss, weights_hit, membership, 
                               ranks, perm)
        self.assertTrue(es_val == -1.0)
        self.assertTrue(es_rank == (nsamples-1))
        self.assertTrue(es_run.shape[0] == nsamples)
        self.assertTrue(np.all(es_run == -1.0))
        # nonzero weights but full membership
        weights_hit = np.linspace(1, 10, nsamples)
        weights_miss = weights_hit.copy()
        membership = np.ones(nsamples, dtype=INT_DTYPE)
        ranks = np.argsort(weights_hit)[::-1]
        perm = np.arange(nsamples, dtype=np.int)        
        es_val, es_rank, es_run = \
            kernel.random_walk(weights_miss, weights_hit, membership, 
                               ranks, perm)
        self.assertTrue(es_val == 1.0)
        self.assertTrue(es_rank == 0)
        self.assertTrue(es_run.shape[0] == nsamples)
        self.assertTrue(np.all(es_run == 1.0))

    def test_random_walk(self):
        nsamples = 10
        weights_hit = np.linspace(10, 1, nsamples)
        weights_miss = weights_hit.copy()
        ranks = np.argsort(weights_hit)[::-1]
        perm = np.arange(nsamples, dtype=np.int)
        membership = np.ones(nsamples, dtype=INT_DTYPE)
        # test all combinations of membership
        # with perfect enrichment (all hits before misses)
        for i in xrange(1,nsamples):
            membership[:] = 1
            membership[i:] = 0
            es_val, es_rank, es_run = \
                kernel.random_walk(weights_miss, weights_hit, membership, 
                                   ranks, perm)
            self.assertTrue(es_val == 1.0)
            self.assertTrue(es_rank == (i-1))
            self.assertTrue(es_run[es_rank] == 1.0)
        # test all combinations of membership
        # with perfect negative enrichment (all misses before hits)
        for i in xrange(nsamples-1,0,-1):
            membership[:] = 1
            membership[:i] = 0
            es_val, es_rank, es_run = \
                kernel.random_walk(weights_miss, weights_hit, membership, 
                                   ranks, perm)
            self.assertTrue(es_val == -1.0)
            self.assertTrue(es_rank == (i-1))
            self.assertTrue(es_run[es_rank] == -1.0)

if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.test_power_transform']
    unittest.main()