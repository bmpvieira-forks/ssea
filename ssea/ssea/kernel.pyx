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
UINT8_DTYPE = np.uint8
INT_DTYPE = np.int
ctypedef np.float_t FLOAT_t
ctypedef np.uint8_t UINT8_t
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
        # add gaussian noise
        if add_noise:
            counts[i] += rng.lcg_normal(&r.seed, noise_loc, noise_scale)
            if counts[i] < 0:
                counts[i] = 0

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
                np.ndarray[UINT8_t, ndim=2] membership not None,
                np.ndarray[INT_t, ndim=1] ranks not None,
                np.ndarray[INT_t, ndim=1] perm not None):
    # check types
    assert weights_miss.dtype == FLOAT_DTYPE
    assert weights_hit.dtype == FLOAT_DTYPE
    assert membership.dtype == UINT8_DTYPE
    assert ranks.dtype == INT_DTYPE
    assert perm.dtype == INT_DTYPE
    # declare variables
    cdef np.ndarray[FLOAT_t, ndim=2] phit
    cdef np.ndarray[FLOAT_t, ndim=2] pmiss
    cdef np.ndarray[FLOAT_t, ndim=2] es_runs
    cdef np.ndarray[INT_t, ndim=1] es_ranks
    cdef np.ndarray[FLOAT_t, ndim=1] es_vals
    cdef int i,j,r,p,last
    cdef int nsamples, nsets
    # init variables
    nsamples = membership.shape[0]
    nsets = membership.shape[1]
    phit = np.empty((nsamples,nsets), dtype=FLOAT_DTYPE)
    pmiss = np.empty((nsamples,nsets), dtype=FLOAT_DTYPE)
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
        for j in xrange(nsets):
            # include weight from previous index
            if i == 0:
                phit[i,j] = 0
                pmiss[i,j] = 0
            else:
                phit[i,j] = phit[i-1,j]
                pmiss[i,j] = pmiss[i-1,j]
            if membership[p,j]:
                phit[i,j] += wt_hit
            else:
                pmiss[i,j] += wt_miss
    # the enrichment score (ES) is the maximum deviation from zero of
    # phit - pmiss. for a randomly distributed S, ES(S) will be relatively
    # small, but if it is concentrated at the top or bottom of the list,
    # or otherwise nonrandomly distributed, then ES(S) will be 
    # correspondingly high
    es_runs = np.zeros((nsamples,nsets), dtype=FLOAT_DTYPE)
    es_ranks = np.zeros(nsets, dtype=INT_DTYPE)
    es_vals = np.zeros(nsets, dtype=FLOAT_DTYPE)
    last = nsamples - 1 # last index
    for j in xrange(nsets):
        # handle cases where cumulative sum of hits and/or misses is zero
        if (phit[last,j] == 0) and (pmiss[last,j] == 0):
            # all weights equal to zero
            continue
        elif phit[last,j] == 0:
            # empty sample set
            es_vals[j] = -1.0
            es_ranks[j] = last
            es_runs[:,j] = -1.0
        elif pmiss[last,j] == 0:
            # full sample set (only hits)
            es_vals[j] = 1.0
            es_ranks[j] = 0
            es_runs[:,j] = 1.0
        else:
            for i in xrange(nsamples):
                es_runs[i,j] = ((phit[i,j] / phit[last,j]) - 
                                (pmiss[i,j] / pmiss[last,j]))
                if fabs(es_runs[i,j]) >= fabs(es_vals[j]):
                    es_vals[j] = es_runs[i,j]
                    es_ranks[j] = i
    return es_vals, es_ranks, es_runs
 
@cython.boundscheck(False)
@cython.wraparound(False)
def ssea_kernel2(np.ndarray[FLOAT_t, ndim=1] counts not None,
                 np.ndarray[FLOAT_t, ndim=1] size_factors not None,
                 np.ndarray[UINT8_t, ndim=2] membership not None,
                 RandomState r,
                 bint resample_counts,
                 bint permute_samples,
                 bint add_noise,
                 double noise_loc, 
                 double noise_scale,
                 int method_miss,
                 int method_hit,
                 double method_param):
    cdef np.ndarray[FLOAT_t, ndim=1] norm_counts
    cdef np.ndarray[FLOAT_t, ndim=1] norm_counts_hit
    cdef np.ndarray[FLOAT_t, ndim=1] norm_counts_miss    
    cdef np.ndarray[INT_t, ndim=1] ranks
    cdef np.ndarray[INT_t, ndim=1] perm
    cdef np.ndarray[FLOAT_t, ndim=1] es_vals
    cdef np.ndarray[INT_t, ndim=1] es_ranks
    cdef np.ndarray[FLOAT_t, ndim=2] es_runs
    cdef int nsamples
    # define constant values
    nsamples = counts.shape[0]
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
    es_vals, es_ranks, es_runs = \
        random_walk(weights_miss=norm_counts_miss,
                    weights_hit=norm_counts_hit,
                    membership=membership,
                    ranks=ranks,
                    perm=perm)
    return norm_counts, ranks, es_vals, es_ranks, es_runs


@cython.boundscheck(False)
@cython.wraparound(False)
def run_lengths(np.ndarray[FLOAT_t, ndim=1] arr):
    '''
    counts runs of consecutive values in a sequence
    a: sequence or iterable
    returns list
    '''
    assert arr.dtype == FLOAT_DTYPE
    cdef float prev, cur
    cdef list lengths = []
    cdef int count, i, length
    # ignore zero length array
    length = len(arr)
    if length == 0:
        return lengths    
    prev = arr[0]
    count = 1
    for i in xrange(1,length):
        cur = arr[i]
        if cur != prev:
            lengths.append(count)
            count = 1
            prev = cur
        else:
            count += 1
    lengths.append(count)
    return lengths

@cython.boundscheck(False)
@cython.wraparound(False)
def rld2d(list lengths, 
          np.ndarray[FLOAT_t, ndim=2] arr):
    '''run length decode for 2d array'''
    assert arr.dtype == FLOAT_DTYPE
    assert len(lengths) == arr.shape[0]
    cdef np.ndarray[FLOAT_t, ndim=2] out
    cdef int length, offset, i, j, ix
    cdef int nsamples, nsets, nlengths    
    nlengths = len(lengths)
    nsamples = sum(lengths)
    nsets = arr.shape[1]
    out = np.empty((nsamples,nsets), dtype=FLOAT_DTYPE)
    offset = 0
    for i in xrange(nlengths):
        length = lengths[i]
        for ix in xrange(offset, offset+length):
            for j in xrange(nsets):
                out[ix,j] = arr[i,j]
        offset += length
    return out

@cython.boundscheck(False)
@cython.wraparound(False)
def ssea_kernel(np.ndarray[FLOAT_t, ndim=1] weights not None,
                np.ndarray[FLOAT_t, ndim=1] weights_miss not None,
                np.ndarray[FLOAT_t, ndim=1] weights_hit not None,
                np.ndarray[UINT8_t, ndim=2] membership not None,
                np.ndarray[INT_t, ndim=1] perm not None):
    assert weights_miss.dtype == FLOAT_DTYPE
    assert weights_hit.dtype == FLOAT_DTYPE
    assert membership.dtype == UINT8_DTYPE
    # declare variables
    cdef np.ndarray[FLOAT_t, ndim=2] phit
    cdef np.ndarray[FLOAT_t, ndim=2] pmiss
    cdef np.ndarray[FLOAT_t, ndim=1] norm_hit
    cdef np.ndarray[FLOAT_t, ndim=1] norm_miss
    cdef np.ndarray[FLOAT_t, ndim=2] es_runs
    cdef np.ndarray[FLOAT_t, ndim=1] es_vals
    cdef np.ndarray[INT_t, ndim=1] es_run_inds
    cdef np.ndarray[INT_t, ndim=1] cumlengths
    cdef int offset, i, ix, j, p, length
    cdef int nlengths, nsets
    cdef float wt_hit, wt_miss
    cdef list lengths
    # init variables
    lengths = run_lengths(weights)    
    nlengths = len(lengths)
    nsets = membership.shape[1]
    phit = np.empty((nlengths,nsets), dtype=FLOAT_DTYPE)
    pmiss = np.empty((nlengths,nsets), dtype=FLOAT_DTYPE)
    cumlengths = np.empty(nlengths, dtype=INT_DTYPE)
    # evaluate the fraction of samples in S "hits" and the fraction of 
    # samples not in S "misses" present up to a given position i in L
    offset = 0
    for i in xrange(nlengths):
        # run length encoding ensures that tied weights get added to same 
        # index of the hit/miss array, so all samples have identical weight
        # each each index
        length = lengths[i]
        wt_miss = weights_miss[offset]
        wt_hit = weights_hit[offset]        
        # calculate cumulative sum of hits and misses at this index
        if i == 0:
            for j in xrange(nsets):
                phit[i,j] = 0
                pmiss[i,j] = 0
        else:
            for j in xrange(nsets):
                phit[i,j] = phit[i-1,j]
                pmiss[i,j] = pmiss[i-1,j]
        for ix in xrange(offset, offset+length):
            p = perm[ix]
            for j in xrange(nsets):
                if membership[p,j]:
                    phit[i,j] += wt_hit
                else:
                    pmiss[i,j] += wt_miss
        offset += length
        cumlengths[i] = offset
    # normalize cumulative sums and handle cases where a sample set has a 
    # cumulative sum of zero
    norm_miss = np.empty(nsets, dtype=FLOAT_DTYPE)
    norm_hit = np.empty(nsets, dtype=FLOAT_DTYPE)
    i = nlengths - 1 # last index
    for j in xrange(nsets):
        norm_miss[j] = pmiss[i,j] if pmiss[i,j] > 0 else 1.0
        norm_hit[j] = phit[i,j] if phit[i,j] > 0 else 1.0
    # the enrichment score (ES) is the maximum deviation from zero of
    # phit - pmiss. for a randomly distributed S, ES(S) will be relatively
    # small, but if it is concentrated at the top of bottom of the list,
    # or otherwise nonrandomly distributed, then ES(S) will be 
    # correspondingly high
    es_runs = (phit / norm_hit) - (pmiss / norm_miss)
    es_run_inds = np.fabs(es_runs).argmax(axis=0)
    es_vals = np.empty(nsets, dtype=FLOAT_DTYPE)
    for j,ix in enumerate(es_run_inds):
        es_vals[j] = es_runs[ix,j]
    # decode run length encoded results
    for j in xrange(nsets):
        es_run_inds[j] = cumlengths[es_run_inds[j]]    
    #es_run_inds = np.array([sum(lengths[:i]) for i in es_run_inds])
    es_runs = rld2d(lengths, es_runs)
    return es_vals, es_run_inds, es_runs

