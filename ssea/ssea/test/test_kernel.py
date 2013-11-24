'''
Created on Nov 7, 2013

@author: mkiyer
'''
import unittest

# third party packages
import numpy as np

# local imports
from ssea.base import BOOL_DTYPE
import ssea.kernel as kernel

class TestKernel(unittest.TestCase):

    def test_power_transform(self):
        # power transform methods
        UNWEIGHTED = 0
        WEIGHTED = 1
        EXP = 2
        LOG = 3
        size = 100
        # unweighted
        a = np.linspace(0, 100, size)
        b = kernel.power_transform(a, UNWEIGHTED)
        self.assertTrue(np.array_equal(b, np.ones(size,dtype=np.float)))
        # weighted
        b = kernel.power_transform(a, WEIGHTED)
        self.assertTrue(np.array_equal(a,b))
        # exponent
        b = kernel.power_transform(a, EXP, 0.5)
        correct = a ** 0.5
        self.assertTrue(np.array_equal(b,correct))        
        # log
        b = kernel.power_transform(a, LOG, 1.0)
        correct = np.log2(a+1.0)
        self.assertTrue(np.array_equal(b,correct))

    def test_random_walk_empty(self):
        '''
        test trying to run random walk with an empty array
        '''
        nsets = 100
        weights_hit = np.array([], dtype=np.float)
        weights_miss = np.array([], dtype=np.float)
        membership = np.empty((0, nsets), dtype=BOOL_DTYPE)
        ranks = np.argsort(weights_hit)[::-1]
        perm = np.arange(0)
        es_vals, es_ranks, es_runs = \
            kernel.random_walk(weights_miss, weights_hit, membership, ranks, perm)
        self.assertTrue(np.all(es_vals == 0))
        self.assertTrue(np.all(es_ranks == 0))
        self.assertTrue(es_runs.shape == (0,100))

    def test_kernel_empty(self):
        '''
        test trying to run kernel with an empty array
        '''
        nsets = 100
        counts = np.array([], dtype=np.float)
        size_factors = np.array([], dtype=np.float)
        membership = np.empty((0, nsets), dtype=BOOL_DTYPE)
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
        es_vals, es_ranks, es_runs) = k
        self.assertTrue(len(ranks) == 0)
        self.assertTrue(len(norm_counts) == 0)
        self.assertTrue(len(norm_counts_miss) == 0)
        self.assertTrue(len(norm_counts_hit) == 0)
        self.assertTrue(np.all(es_vals == 0))
        self.assertTrue(np.all(es_ranks == 0))
        self.assertTrue(es_runs.shape == (0,100))         
         
    def test_random_walk_boundary_cases(self):
        nsamples = 10
        nsets = 1
        # test all zeros
        weights_hit = np.zeros(nsamples, dtype=np.float)
        weights_miss = np.zeros(nsamples, dtype=np.float)
        membership = np.zeros((nsamples,nsets), dtype=BOOL_DTYPE)
        ranks = np.argsort(weights_hit)[::-1]
        perm = np.arange(nsamples, dtype=np.int)        
        es_vals, es_run_inds, es_runs = \
            kernel.random_walk(weights_miss, weights_hit, membership, 
                               ranks, perm)
        self.assertTrue(es_vals.shape[0] == nsets)
        self.assertTrue(es_vals[0] == 0.0)
        self.assertTrue(es_run_inds.shape[0] == nsets)
        self.assertTrue(es_run_inds[0] == 0.0)
        self.assertTrue(es_runs.shape == (nsamples,nsets))
        self.assertTrue(np.all(es_run_inds == 0.0))
        # nonzero weights but empty membership
        nsets = 5        
        weights_hit = np.linspace(1, 10, nsamples)
        weights_miss = weights_hit.copy()
        membership = np.zeros((nsamples,nsets), dtype=BOOL_DTYPE)
        ranks = np.argsort(weights_hit)[::-1]
        perm = np.arange(nsamples, dtype=np.int)        
        es_vals, es_run_inds, es_runs = \
            kernel.random_walk(weights_miss, weights_hit, membership, 
                               ranks, perm)
        self.assertTrue(es_vals.shape[0] == nsets)
        self.assertTrue(np.all(es_vals == -1.0))
        self.assertTrue(es_run_inds.shape[0] == nsets)
        self.assertTrue(np.all(es_run_inds == nsamples-1))
        self.assertTrue(es_runs.shape == (nsamples,nsets))
        self.assertTrue(np.all(es_runs == -1.0))
        # nonzero weights but full membership
        nsets = 5        
        weights_hit = np.linspace(1, 10, nsamples)
        weights_miss = weights_hit.copy()
        membership = np.ones((nsamples,nsets), dtype=BOOL_DTYPE)
        ranks = np.argsort(weights_hit)[::-1]
        perm = np.arange(nsamples, dtype=np.int)        
        es_vals, es_run_inds, es_runs = \
            kernel.random_walk(weights_miss, weights_hit, membership, 
                               ranks, perm)
        self.assertTrue(es_vals.shape[0] == nsets)
        self.assertTrue(np.all(es_vals == 1.0))
        self.assertTrue(es_run_inds.shape[0] == nsets)
        self.assertTrue(np.all(es_run_inds == 0))
        self.assertTrue(es_runs.shape == (nsamples,nsets))
        self.assertTrue(np.all(es_runs == 1.0))

    def test_random_walk(self):
        nsamples = 10
        nsets = 11
        weights_hit = np.linspace(10, 1, nsamples)
        weights_miss = weights_hit.copy()
        membership = np.ones((nsamples,nsets), dtype=BOOL_DTYPE)
        # test all combinations of membership
        # with perfect enrichment (all hits before misses)
        for i in xrange(nsets):
            membership[i:,i] = 0
        ranks = np.argsort(weights_hit)[::-1]
        perm = np.arange(nsamples, dtype=np.int)
        es_vals, es_run_inds, es_runs = \
            kernel.random_walk(weights_miss, weights_hit, membership, 
                               ranks, perm)
        self.assertTrue(es_vals[0] == -1.0)
        self.assertTrue(es_run_inds[0] == nsamples-1)
        self.assertTrue(np.all(es_vals[1:] == 1.0))
        self.assertTrue(np.array_equal(es_run_inds[1:(nsets-1)], range(9)))
        self.assertTrue(es_run_inds[-1] == 0)
        # test all combinations of membership
        # with perfect enrichment (all hits before misses)
        membership = np.logical_not(membership).astype(BOOL_DTYPE)
        es_vals, es_run_inds, es_runs = \
            kernel.random_walk(weights_miss, weights_hit, membership, 
                               ranks, perm)
        self.assertTrue(es_vals[0] == 1.0)
        self.assertTrue(es_run_inds[0] == 0)        
        self.assertTrue(np.all(es_vals[1:] == -1.0))
        self.assertTrue(np.array_equal(es_run_inds[1:], range(10)))



if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.test_power_transform']
    unittest.main()