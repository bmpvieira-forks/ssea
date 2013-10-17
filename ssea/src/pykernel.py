'''
Created on Oct 17, 2013

@author: mkiyer
'''
import numpy as np

FLOAT_DTYPE = np.float

def ssea_kernel_py(rle_lengths, rle_weights_miss, rle_weights_hit, 
                   membership, perm):
    nsamples = len(rle_lengths)
    nsets = membership.shape[1]    
    # evaluate the fraction of samples in S "hits" and the fraction of 
    # samples not in S "misses" present up to a given position i in L
    phit = np.zeros((nsamples,nsets), dtype=FLOAT_DTYPE)
    pmiss = np.zeros((nsamples,nsets), dtype=FLOAT_DTYPE)    
    offset = 0
    for i in xrange(nsamples):
        # run length encoding ensures that tied weights get added to same 
        # index of the hit/miss array
        length = rle_lengths[i]
        # count hits and misses at this index (all samples have identical 
        # weight)
        inds = perm[offset:offset+length]
        counts_hit = membership[inds,:].sum(axis=0)
        counts_miss = length - counts_hit
        # update pmiss/phit vectors
        pmiss[i] = rle_weights_miss[i] * counts_miss
        phit[i] = rle_weights_hit[i] * counts_hit        
        offset += length
    pmiss = pmiss.cumsum(axis=0)
    phit = phit.cumsum(axis=0)
    # handle cases where a sample set has a cumulative sum of zero
    norm_miss = np.where(pmiss[-1,:] > 0, pmiss[-1,:], 1.0)
    norm_hit = np.where(phit[-1,:] > 0, phit[-1,:], 1.0)
    # normalize cumulative sums
    pmiss /= norm_miss
    phit /= norm_hit
    # the enrichment score (ES) is the maximum deviation from zero of
    # phit - pmiss. for a randomly distributed S, ES(S) will be relatively
    # small, but if it is concentrated at the top of bottom of the list,
    # or otherwise nonrandomly distributed, then ES(S) will be 
    # correspondingly high
    es_runs = phit - pmiss
    es_run_inds = np.abs(es_runs).argmax(axis=0)
    es_vals = np.array([es_runs[x,j] for j,x in enumerate(es_run_inds)])
    return es_vals, es_run_inds, es_runs