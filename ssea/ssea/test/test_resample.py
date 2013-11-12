'''
Created on Nov 12, 2013

@author: mkiyer
'''
import unittest
import logging
import os

import numpy as np
import matplotlib.pyplot as plt

from ssea.countdata import BigCountMatrix
from ssea.kernel import normalize_counts, RandomState

def rpkm_versus_count(input_dir):
    bm = BigCountMatrix.open(input_dir)
    for j in xrange(bm.shape[1]):
        a = bm.counts_t[j,:]
        a = a[np.isfinite(a)]
        print bm.colnames[j], a.sum()

def dump_sample(input_dir):
    bm = BigCountMatrix.open(input_dir)
    for i in xrange(bm.shape[0]):
        a = bm.counts[i,0]
        if np.isnan(a):
            a = 'NA'
        print bm.rownames[i], str(a)

def normalize_count_data(input_dir):
    # setup matrix
    logging.debug("Opening matrix memmap files")
    bm = BigCountMatrix.open(input_dir)
    r = RandomState()

    for i in xrange(bm.shape[0]):
        a = bm.counts[i,:]
        a = np.array(a, dtype=np.float)
        b = a.copy()
        normalize_counts(b, bm.size_factors, r,
                         resample=True,
                         add_noise=True,
                         noise_loc=1.0,
                         noise_scale=1.0)        
        logging.debug(str(i))

        y1 = np.log2((a / bm.size_factors) + 1)
        y2 = np.log2(b + 1)
        order = y1.argsort()[::-1]
        y1 = y1[order]
        y2 = y2[order]
        x = np.arange(len(y1))
        f = plt.figure()
        plt.plot(x, y2, 'ro', ms=1, mew=0, label='Resampled counts')
        plt.plot(x, y1, 'b-', lw=3, label='Original counts')
        plt.legend(markerscale=5, numpoints=3)
        plt.xlabel('Samples sorted by counts')
        plt.ylabel('log2(counts + 1)')
        plt.savefig('/home/mkiyer/Documents/t%d.png' % (i))
        plt.show()
        print 'y1', y1.min(), y1.max(), y1.mean(), np.median(y1)
        print 'y2', y2.min(), y2.max(), y2.mean(), np.median(y2)
        
        if i == 100:
            break

class Test(unittest.TestCase):

    def testName(self):
        dump_sample('/mctp/users/mkiyer/projects/ssea/isoform_count_matrix_v4')
        return
        rpkm_versus_count('/mctp/users/mkiyer/projects/ssea/isoform_count_matrix_v4')
        return        
        normalize_count_data('/mctp/users/mkiyer/projects/ssea/isoform_count_matrix_v4')        


if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()