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

def lcg_seed():
    return rng.lcg_init_state()

@cython.boundscheck(False)
@cython.wraparound(False)
def shuffle(np.ndarray[FLOAT_t, ndim=1] x):
    cdef int i, j
    i = len(x) - 1
    while i > 0:
        j = rng.lcg_range(0, i)
        x[i], x[j] = x[j], x[i]
        i = i - 1

@cython.boundscheck(False)
@cython.wraparound(False)
def resample_poisson(np.ndarray[FLOAT_t, ndim=1] counts not None,
                     int seed): 
    cdef int i    
    for i in xrange(counts.shape[0]):
        counts[i] = rng.lcg_poisson(&seed, counts[i])
    return seed

@cython.boundscheck(False)
@cython.wraparound(False)
def add_gaussian_noise(np.ndarray[FLOAT_t, ndim=1] counts not None,
                       FLOAT_t loc, FLOAT_t scale, int seed):
    cdef int i
    cdef double n
    for i in xrange(counts.shape[0]):
        n = rng.lcg_normal(&seed, loc, scale)
        counts[i] += n
        if counts[i] < 0:
            counts[i] = 0.0
    return seed

@cython.boundscheck(False)
@cython.wraparound(False)
def normalize_counts(np.ndarray[FLOAT_t, ndim=1] counts not None,
                     np.ndarray[FLOAT_t, ndim=1] size_factors not None,
                     int seed, 
                     float noise_loc, 
                     float noise_scale,
                     bint resample):
    cdef int i
    if resample:
        seed = resample_poisson(counts, seed)
    for i in xrange(counts.shape[0]):
        counts[i] = counts[i] / size_factors[i]
    seed = add_gaussian_noise(counts, noise_loc, noise_scale, seed)
    return seed
