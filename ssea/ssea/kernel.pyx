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
def ssea_kernel(tuple rle_lengths, 
                np.ndarray[FLOAT_t, ndim=1] rle_weights_miss,
                np.ndarray[FLOAT_t, ndim=1] rle_weights_hit,
                np.ndarray[UINT8_t, ndim=2] membership,
                np.ndarray[INT_t, ndim=1] perm):
    assert rle_weights_miss.dtype == FLOAT_DTYPE
    assert rle_weights_hit.dtype == FLOAT_DTYPE
    assert membership.dtype == UINT8_DTYPE
    # declare variables
    cdef np.ndarray[FLOAT_t, ndim=2] phit
    cdef np.ndarray[FLOAT_t, ndim=2] pmiss
    cdef np.ndarray[FLOAT_t, ndim=1] norm_hit
    cdef np.ndarray[FLOAT_t, ndim=1] norm_miss
    cdef np.ndarray[FLOAT_t, ndim=2] es_runs
    cdef np.ndarray[FLOAT_t, ndim=1] es_vals
    cdef np.ndarray[INT_t, ndim=1] es_run_inds
    cdef int offset, i, j, ix, length
    cdef float wt_hit, wt_miss
    # init variables
    cdef int nsamples = len(rle_lengths)
    cdef int nsets = membership.shape[1]
    phit = np.zeros((nsamples,nsets), dtype=FLOAT_DTYPE)
    pmiss = np.zeros((nsamples,nsets), dtype=FLOAT_DTYPE)
    # evaluate the fraction of samples in S "hits" and the fraction of 
    # samples not in S "misses" present up to a given position i in L
    offset = 0
    for i in xrange(nsamples):
        # run length encoding ensures that tied weights get added to same 
        # index of the hit/miss array, so all samples have identical weight
        # each each index
        length = rle_lengths[i]
        wt_miss = rle_weights_miss[i]
        wt_hit = rle_weights_hit[i]
        # count hits and misses at this index
        for ix in xrange(offset, offset+length):
            for j in xrange(nsets):
                if membership[perm[ix],j]:
                    phit[i,j] += wt_hit
                else:
                    pmiss[i,j] += wt_miss    
        offset += length
    pmiss = pmiss.cumsum(axis=0)
    phit = phit.cumsum(axis=0)
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
    es_run_inds = np.abs(es_runs).argmax(axis=0)
    es_vals = np.empty(nsets, dtype=FLOAT_DTYPE)
    for j,ix in enumerate(es_run_inds):
        es_vals[j] = es_runs[ix,j]
    return es_vals, es_run_inds, es_runs
