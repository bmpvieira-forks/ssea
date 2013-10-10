'''
Created on Oct 9, 2013

@author: mkiyer
'''
import numpy as np

class SampleSet(object):
    pass


def _ssea_kernel(samples, weights, sample_set):
    # rank order the N samples in D to form L={s1...sn} 
    order = np.argsort(weights)[::-1]
    weights = weights[order]
    samples = samples[order]
    # evaluate the fraction of samples in S "hits" and the fraction of 
    # samples not in S "misses" present up to a given position i in L
    sum_weights = 0
    sum_weights_hit = 0
    phit = np.zeros(len(samples), dtype=np.float)
    pmiss = np.zeros(len(samples), dtype=np.float)
    for i in xrange(len(samples)):
        w = weights[i]
        sum_weights += w
        if samples[i] in sample_set:
            sum_weights_hit += w
            phit[i] = w
        else:
            pmiss[i] = w
    sum_weights_miss = sum_weights - sum_weights_hit
    phit = np.cumsum(phit / sum_weights_hit)
    pmiss = np.cumsum(pmiss / sum_weights_miss)
    # the enrichment score (ES) is the maximum deviation from zero of
    # phit - pmiss. for a randomly distributed S, ES(S) will be relatively
    # small, but if it is concentrated at the top of bottom of the list,
    # or otherwise nonrandomly distributed, then ES(S) will be 
    # correspondingly high
    diff = phit - pmiss
    es_ind = np.abs(diff).argmax()
    es_score = diff[es_ind]    
    # print    
#     print samples
#     print weights
#     print phit
#     print pmiss
#     print diff
#     print es_ind, es_score    
#     print ['%.2f' % (x) for x in (diff)]

    return es_score

def ssea(samples, weights, sample_set, 
         weight_method=0, perms=100):
    # define data set with N samples
    samples = np.array(samples, dtype=np.string_)    
    weights = np.array(weights, dtype=np.float)
    # determine enrichment score (ES)
    es_score = _ssea_kernel(samples, weights, sample_set)
    # permute samples and determine ES_null
    es_null = np.zeros(perms, dtype=np.float)
    null_weights = weights.copy()
    null_samples = samples.copy()
    for i in xrange(perms):
        np.random.shuffle(null_weights)
        order = np.argsort(null_weights)[::-1]
        null_weights = null_weights[order]
        null_samples = null_samples[order]
        es_null[i] = _ssea_kernel(null_samples, null_weights, sample_set)
    
    print es_score
    print es_null


