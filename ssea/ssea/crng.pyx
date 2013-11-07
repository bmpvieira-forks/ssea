# encoding: utf-8
# cython: profile=True
# filename: kernel.pyx
'''
Created on Nov 6, 2013

@author: mkiyer
'''
import numpy as np
cimport numpy as np
cimport cython

# local imports
cimport rng

# Define array types
FLOAT_DTYPE = np.float
ctypedef np.float_t FLOAT_t

@cython.boundscheck(False)
@cython.wraparound(False)
def resample_counts(np.ndarray[FLOAT_t, ndim=1] counts not None,
                    int seed): 
    cdef int i
    cdef np.ndarray[FLOAT_t, ndim=1] newcounts    
    newcounts = np.empty(counts.shape[0], dtype=FLOAT_DTYPE)
    
    for i in xrange(counts.shape[0]):
        newcounts[i] = rng.lcg_poisson(&seed, counts[i])
    return newcounts, seed