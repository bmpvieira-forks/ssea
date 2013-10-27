'''
Created on Oct 24, 2013

@author: mkiyer
'''
import unittest

import numpy as np
import sys

from ssea.kernel import ssea_kernel
from ssea.algo import transform_weights

class TestPerformance(unittest.TestCase):

    def test1(self):
        # setup samples
        nsamples = 10000
        nsets = 100
        iterations = 2000
        # create membership array and fill with random values
        membership = np.empty((nsamples,nsets), dtype=np.uint8)
        membership[:] = np.random.random_integers(0, 1, (nsamples,nsets))
        # create random weights (use integers to create ties)
        weights = np.empty(nsamples, dtype=np.float)
        weights[:] = np.sort(np.random.random_integers(-10, 10, nsamples))[::-1]
        # transform weights based on weight method
        weight_method = 'weighted'
        weights_miss = np.fabs(transform_weights(weights, weight_method))
        weights_hit = np.fabs(transform_weights(weights, weight_method))
        # run kernel
        perm = np.arange(nsamples)
        for i in xrange(iterations):
            np.random.shuffle(perm)
            print i
            es_vals, es_run_inds, es_runs = \
                ssea_kernel(weights, weights_miss, weights_hit, membership, perm)

if __name__ == "__main__":
    import cProfile
    import pstats
    profile_filename = '_profile.bin'
    cProfile.run('unittest.main()', profile_filename)
    statsfile = open("profile_stats.txt", "wb")
    p = pstats.Stats(profile_filename, stream=statsfile)
    stats = p.strip_dirs().sort_stats('cumulative')
    stats.print_stats()
    statsfile.close()
    sys.exit(0)
