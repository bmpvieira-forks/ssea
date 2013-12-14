# encoding: utf-8
# cython: profile=True
# filename: kernel.pyx
'''
Created on Oct 16, 2013

@author: mkiyer
'''
import numpy as np
cimport numpy as np
cimport cython

# local imports
cimport rng
from libc.math cimport log2, fabs

# define power transform methods
DEF UNWEIGHTED = 0
DEF WEIGHTED = 1
DEF EXP = 2
DEF LOG = 3

# define array types
FLOAT_DTYPE = np.float
INT_DTYPE = np.int
ctypedef np.float_t FLOAT_t
ctypedef np.int_t INT_t

cdef class RandomState:
    cdef public int seed    
    def __cinit__(self, seed=0):
        if seed == 0:
            self.seed = rng.lcg_init_state()
        else:
            self.seed = seed

@cython.boundscheck(False)
@cython.wraparound(False)
def shufflef(np.ndarray[FLOAT_t, ndim=1] x, RandomState r):
    cdef int i, j
    i = len(x) - 1
    while i > 0:
        j = rng.lcg_range(&r.seed, 0, i)
        x[i], x[j] = x[j], x[i]
        i = i - 1

@cython.boundscheck(False)
@cython.wraparound(False)
def shufflei(np.ndarray[INT_t, ndim=1] x, RandomState r):
    cdef int i, j
    i = len(x) - 1
    while i > 0:
        j = rng.lcg_range(&r.seed, 0, i)
        x[i], x[j] = x[j], x[i]
        i = i - 1

@cython.boundscheck(False)
@cython.wraparound(False)
def resample_poisson(np.ndarray[FLOAT_t, ndim=1] x not None,
                     RandomState r):
    cdef int i    
    for i in xrange(x.shape[0]):
        x[i] = rng.lcg_poisson(&r.seed, x[i])

@cython.boundscheck(False)
@cython.wraparound(False)
def add_gaussian_noise(np.ndarray[FLOAT_t, ndim=1] x not None, 
                       RandomState r,
                       double loc=0.0,
                       double scale=1.0):
    cdef int i
    cdef double n
    for i in xrange(x.shape[0]):
        n = rng.lcg_normal(&r.seed, loc, scale)
        x[i] += n
        if x[i] < 0:
            x[i] = 0

@cython.boundscheck(False)
@cython.wraparound(False)
def normalize_counts(np.ndarray[FLOAT_t, ndim=1] counts not None,
                     np.ndarray[FLOAT_t, ndim=1] size_factors not None,
                     RandomState r,
                     bint resample,
                     bint add_noise,
                     double noise_loc, 
                     double noise_scale):
    cdef int i
    for i in xrange(counts.shape[0]):
        # resample the count using the original count as the mean of a 
        # poisson distribution
        if resample:
            counts[i] = rng.lcg_poisson(&r.seed, counts[i])
        # normalize for library size
        counts[i] = counts[i] / size_factors[i]
        # add noise
        if add_noise:
            counts[i] += rng.lcg_poisson(&r.seed, noise_loc)
            counts[i] += rng.lcg_double(&r.seed) * noise_scale
            #counts[i] += rng.lcg_normal(&r.seed, noise_loc, noise_scale)
            #if counts[i] < 0:
            #    counts[i] = 0

@cython.boundscheck(False)
@cython.wraparound(False)
def power_transform(np.ndarray[FLOAT_t, ndim=1] x not None,
                    int method, double param=1.0):
    cdef np.ndarray[FLOAT_t, ndim=1] newx
    cdef int i, size
    cdef double v
    size = x.shape[0]
    newx = np.empty_like(x)    
    if method == UNWEIGHTED:
        for i in xrange(size):
            newx[i] = 1.0
    elif method == WEIGHTED:
        for i in xrange(size):
            newx[i] = x[i]
    elif method == EXP:
        for i in xrange(size):
            newx[i] = x[i] ** param
    elif method == LOG:
        for i in xrange(size):
            if x[i] < 0:
                newx[i] = -log2(fabs(x[i]) + param)
            else:
                newx[i] = log2(fabs(x[i]) + param)
    else:
        assert False
    return newx

@cython.boundscheck(False)
@cython.wraparound(False)
def random_walk(np.ndarray[FLOAT_t, ndim=1] weights_miss not None,
                np.ndarray[FLOAT_t, ndim=1] weights_hit not None,
                np.ndarray[INT_t, ndim=1] membership not None,
                np.ndarray[INT_t, ndim=1] ranks not None,
                np.ndarray[INT_t, ndim=1] perm not None):
    # check types
    assert weights_miss.dtype == FLOAT_DTYPE
    assert weights_hit.dtype == FLOAT_DTYPE
    assert membership.dtype == INT_DTYPE
    assert ranks.dtype == INT_DTYPE
    assert perm.dtype == INT_DTYPE
    # declare variables
    cdef int i, r, p, last
    cdef int nsamples = membership.shape[0]
    cdef int es_rank = 0
    cdef float es_val = 0.0
    cdef np.ndarray[FLOAT_t, ndim=1] phit = np.empty(nsamples, dtype=FLOAT_DTYPE)
    cdef np.ndarray[FLOAT_t, ndim=1] pmiss = np.empty(nsamples, dtype=FLOAT_DTYPE)
    cdef np.ndarray[FLOAT_t, ndim=1] es_run = np.zeros(nsamples, dtype=FLOAT_DTYPE)
    # if the number of samples is zero return immediately
    if nsamples == 0:
        return es_val, es_rank, es_run
    # evaluate the fraction of samples in S "hits" and the fraction of 
    # samples not in S "misses" present up to a given position i in L
    for i in xrange(nsamples):
        # ranks contains indexes into the weight vectors that allow 
        # iteration in sorted order
        r = ranks[i]
        wt_miss = weights_miss[r]
        wt_hit = weights_hit[r]
        # perm contains shuffled indexes into the membership array
        p = perm[r]
        # calculate cumulative sum of hits and misses at this index        
        # include weight from previous index
        if i == 0:
            phit[i] = 0
            pmiss[i] = 0
        else:
            phit[i] = phit[i-1]
            pmiss[i] = pmiss[i-1]
        if membership[p]:
            phit[i] += wt_hit
        else:
            pmiss[i] += wt_miss
    # the enrichment score (ES) is the maximum deviation from zero of
    # phit - pmiss. for a randomly distributed S, ES(S) will be relatively
    # small, but if it is concentrated at the top or bottom of the list,
    # or otherwise nonrandomly distributed, then ES(S) will be 
    # correspondingly high   
    last = nsamples - 1 # last index
    # if all weights equal to zero skip
    if (phit[last] > 0) or (pmiss[last] > 0):
        if phit[last] == 0:
            # empty sample set
            es_val = -1.0
            es_rank = last
            es_run[:] = -1.0
        elif pmiss[last] == 0:
            # full sample set (only hits)
            es_val = 1.0
            es_rank = 0
            es_run[:] = 1.0
        else:
            for i in xrange(nsamples):
                es_run[i] = (phit[i] / phit[last]) - (pmiss[i] / pmiss[last])
                if fabs(es_run[i]) >= fabs(es_val):
                    es_val = es_run[i]
                    es_rank = i
    return es_val, es_rank, es_run
      
@cython.boundscheck(False)
@cython.wraparound(False)
def ssea_kernel(np.ndarray[FLOAT_t, ndim=1] counts not None,
                np.ndarray[FLOAT_t, ndim=1] size_factors not None,
                np.ndarray[INT_t, ndim=1] membership not None,
                RandomState r,
                bint resample_counts,
                bint permute_samples,
                bint add_noise,
                double noise_loc, 
                double noise_scale,
                int method_miss,
                int method_hit,
                double method_param):
    cdef KernelResult
    cdef np.ndarray[INT_t, ndim=1] perm
    cdef np.ndarray[INT_t, ndim=1] ranks
    cdef np.ndarray[FLOAT_t, ndim=1] norm_counts
    cdef np.ndarray[FLOAT_t, ndim=1] norm_counts_hit
    cdef np.ndarray[FLOAT_t, ndim=1] norm_counts_miss    
    cdef float es_val
    cdef int es_rank
    cdef np.ndarray[FLOAT_t, ndim=1] es_run 
    # define constant values
    cdef int nsamples = counts.shape[0]
    # normalize the counts 
    norm_counts = np.copy(counts)
    normalize_counts(norm_counts, size_factors, r,
                     resample=resample_counts,
                     add_noise=add_noise,
                     noise_loc=noise_loc,
                     noise_scale=noise_scale) 
    # perform power transform and adjust by constant
    norm_counts_miss = power_transform(norm_counts, method_miss, method_param)
    norm_counts_hit = power_transform(norm_counts, method_hit, method_param)
    # rank order the N samples in D to form L={s1...sn} 
    ranks = np.argsort(norm_counts)[::-1]
    perm = np.arange(nsamples)
    if permute_samples:
        # randomize order to walk through samples
        shufflei(perm, r)
    # perform random walk iteration
    es_val, es_rank, es_run = \
        random_walk(weights_miss=norm_counts_miss,
                    weights_hit=norm_counts_hit,
                    membership=membership,
                    ranks=ranks,
                    perm=perm)
    return (ranks, norm_counts, norm_counts_miss, norm_counts_hit, 
            es_val, es_rank, es_run)
