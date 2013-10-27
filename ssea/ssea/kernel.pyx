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

# Define array types
FLOAT_DTYPE = np.float
UINT8_DTYPE = np.uint8
INT_DTYPE = np.int
ctypedef np.float_t FLOAT_t
ctypedef np.uint8_t UINT8_t
ctypedef np.int_t INT_t

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
def ssea_kernel(np.ndarray[FLOAT_t, ndim=1] weights,
                np.ndarray[FLOAT_t, ndim=1] weights_miss,
                np.ndarray[FLOAT_t, ndim=1] weights_hit,
                np.ndarray[UINT8_t, ndim=2] membership,
                np.ndarray[INT_t, ndim=1] perm):
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
    cdef int offset, i, ix, j, p, length
    cdef int nsamples, nsets
    cdef float wt_hit, wt_miss
    cdef list lengths
    # init variables
    lengths = run_lengths(weights)    
    nsamples = len(lengths)
    nsets = membership.shape[1]
    phit = np.empty((nsamples,nsets), dtype=FLOAT_DTYPE)
    pmiss = np.empty((nsamples,nsets), dtype=FLOAT_DTYPE)
    # evaluate the fraction of samples in S "hits" and the fraction of 
    # samples not in S "misses" present up to a given position i in L
    offset = 0
    for i in xrange(nsamples):
        # run length encoding ensures that tied weights get added to same 
        # index of the hit/miss array, so all samples have identical weight
        # each each index
        length = lengths[i]
        wt_miss = weights_miss[offset]
        wt_hit = weights_hit[offset]        
        # calculate cumulative sum of hits and misses at this index
        for j in xrange(nsets):
            if i == 0:
                phit[i,j] = 0
                pmiss[i,j] = 0
            else:
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
    # normalize cumulative sums and handle cases where a sample set has a 
    # cumulative sum of zero
    norm_miss = np.empty(nsets, dtype=FLOAT_DTYPE)
    norm_hit = np.empty(nsets, dtype=FLOAT_DTYPE)
    i = nsamples - 1 # last index
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
    es_run_inds = np.array([sum(lengths[:x]) for x in es_run_inds])
    es_runs = rld2d(lengths, es_runs)
    return es_vals, es_run_inds, es_runs

