'''
Created on Oct 9, 2013

@author: mkiyer
'''
from itertools import groupby
import numpy as np
import matplotlib.pyplot as plt 
import matplotlib.gridspec as gridspec

WEIGHT_METHODS = ['unweighted', 'weighted', 'log']
MIN_CONSTANT = 1e-8
DEFAULT_LOG_BASE = 2.0

def transform_weights(weights, method, params):
    if method == 'unweighted':
        return np.ones(len(weights), dtype=np.float)
    elif method == 'weighted':
        return np.array(weights, dtype=np.float)
    elif method == 'log':
        weights = np.array(weights, dtype=np.float)
        nzweights = weights[weights > 0]
        if len(nzweights) == 0:
            const = MIN_CONSTANT
        else:
            const = nzweights.min()
        if params is None:
            base = DEFAULT_LOG_BASE
        else:
            base = float(params)
        n = np.log(const + weights)
        d = np.log(base * np.ones(len(weights),dtype=np.float))
        return n/d

def rle(vals):
    '''run length encoder'''
    lengths, vals = zip(*[(len(list(v)),k) for k,v in groupby(vals)])
    return lengths, vals

def rld(lengths, vals):
    '''run length decode'''
    out = np.zeros(np.sum(lengths), dtype=np.float)    
    offset = 0
    for i in xrange(len(lengths)):
        length = lengths[i]
        val = vals[i]
        out[offset:offset+length] = val
        offset += length
    return out

def rle_group(lengths, origlist):
    newlist = []
    i = 0
    for length in lengths:
        newlist.append(origlist[i:i+length])
        i += length
    return newlist


class SampleSet(object):    
    def __init__(self, name=None, desc=None, value=None):
        self.name = name
        self.desc = desc
        self.value = value
        
    def get_array(self, samples):
        return np.array([x in self.value for x in samples])

class SampleSetResult(object):
    pass

class SSEA(object):
    pass

def plot(diff, membership, weights):
    fig = plt.figure()
    #fig = plt.figure(figsize=(8, 6)) 
    gs = gridspec.GridSpec(3, 1, height_ratios=[2,1,1])
    # running enrichment score
    ax0 = plt.subplot(gs[0])
    newdiff = [0]
    newdiff.extend(diff)
    ax0.plot(np.arange(len(newdiff)),newdiff)
    ax0.axhline(y=0, color='gray')
    #ax0.axvline(x=res.es_ind+1, linestyle='--', color='black')
    ax0.grid(True)
    ax0.set_xticklabels([])
    ax0.set_ylabel('Enrichment score (ES)')
    # membership in sample set
    ax1 = plt.subplot(gs[1])
    ax1.bar(np.arange(len(diff)), membership, 1, color='black', edgecolor='none')
    ax1.set_xticks([])
    ax1.set_yticks([])
    ax1.set_xticklabels([])
    ax1.set_yticklabels([])
    #ax1.set_yticklabels([])
    ax1.set_ylabel('Set')
    # weights
    ax2 = plt.subplot(gs[2])
    ax2.plot(weights[:,0], color='blue')
    ax2.plot(weights[:,1], color='red')
    ax2.set_xlabel('Samples')
    ax2.set_ylabel('Weights')
    # draw
    plt.tight_layout()
    plt.show()
    #plt.savefig('grid_figure.pdf')


def _ssea_kernel(rle_lengths, rle_weights_arr, membership):
    # evaluate the fraction of samples in S "hits" and the fraction of 
    # samples not in S "misses" present up to a given position i in L
    hitmiss = np.zeros((len(rle_lengths),2), dtype=np.float)
    offset = 0
    for i in xrange(len(rle_lengths)):
        # run length encoding and second level of looping ensures that
        # tied weights get added to same index of the hit/miss array
        length = rle_lengths[i]
        for j in xrange(length):
            ishit = membership[offset + j]
            hitmiss[i,ishit] += rle_weights_arr[i,ishit]
        offset += length
    hitmiss = hitmiss.cumsum(axis=0)
    hitmiss /= hitmiss[-1,:]
    # TODO: handle case where one set has all zeros
    # and cumsum is zero    
    # the enrichment score (ES) is the maximum deviation from zero of
    # phit - pmiss. for a randomly distributed S, ES(S) will be relatively
    # small, but if it is concentrated at the top of bottom of the list,
    # or otherwise nonrandomly distributed, then ES(S) will be 
    # correspondingly high
    diff = hitmiss[:,1] - hitmiss[:,0]
    es_ind = np.abs(diff).argmax()
    es = diff[es_ind]
    return es, es_ind, diff

def _ssea_sample_set(rle_lengths, rle_weights_arr, membership, perms):
    # determine enrichment score (ES)
    es, es_ind, es_arr = _ssea_kernel(rle_lengths, rle_weights_arr, 
                                      membership)
    # permute samples and determine ES_null
    es_null = np.zeros(perms, dtype=np.float)
    null_membership = membership.copy()
    for i in xrange(perms):
        np.random.shuffle(null_membership)
        es_i = _ssea_kernel(rle_lengths, rle_weights_arr, 
                            null_membership)[0]
        es_null[i] = es_i
        
    plt.hist(es_null, bins=100, range=(-10,10))
    plt.show()
    # estimate nominal p value for S from ES_null by using the
    # positive or negative portion of the distribution corresponding
    # to the sign of the observed ES(S)
    es_null_sign = np.abs(es_null[np.sign(es_null) == np.sign(es)])
    es_null_sign.sort()
    nominal_p = 1.0 - (float(es_null_sign.searchsorted(abs(es))) / 
                       len(es_null_sign))
    # adjust for variation in gene set size. Normalize the ES_null
    # and the observed ES(S), separately rescaling the positive and
    # negative scores by dividing by the mean of the ES_null to
    # yield the normalized scores nes_null and nes_score
    es_null_sign_mean = es_null_sign.mean()
    nes = es / es_null_sign_mean
    # now normalized the null scores
    nes_null_neg = es_null[np.sign(es_null) < 0]
    nes_null_pos = es_null[np.sign(es_null) > 0]
    nes_null_neg /= abs(nes_null_neg.mean())
    nes_null_pos /= nes_null_pos.mean()
    # save the min/max of the NES and NES_null scores 
    # for computing the FWER
    nes_min = min(nes, nes_null_neg.min())
    nes_max = max(nes, nes_null_pos.max())
    # undo run length encoding to save final result
    es_arr = rld(rle_lengths, es_arr)
    es_ind = sum(rle_lengths[:es_ind+1])
    # save result
    res = SampleSetResult()
    res.es = es
    res.es_ind = es_ind
    res.diff = es_arr
    res.nominal_p = nominal_p
    res.nes = nes
    res.nes_min = nes_min
    res.nes_max = nes_max
    return res

def ssea(samples, weights, sample_sets, 
         weight_methods=('unweighted', 'unweighted'), 
         weight_params=None, 
         perms=10000):
    # rank order the N samples in D to form L={s1...sn} 
    ranks = np.argsort(weights)[::-1]
    samples = [samples[i] for i in ranks]
    weights = [weights[i] for i in ranks]
    # transform weights based on weight method
    weights_miss = transform_weights(weights, weight_methods[0], 
                                     weight_params)
    weights_hit = transform_weights(weights, weight_methods[1], 
                                    weight_params)    
    weights_arr = np.transpose((weights_miss, weights_hit))
    # perform run length encoding to keep track of ties in weights
    # and transform weights (same as above)
    rle_lengths, rle_weights = rle(weights)
    rle_weights_miss = transform_weights(rle_weights, weight_methods[0], 
                                         weight_params)
    rle_weights_hit = transform_weights(rle_weights, weight_methods[1], 
                                        weight_params)
    rle_weights_arr = np.transpose((rle_weights_miss, rle_weights_hit))
    # process each sample set
    for sample_set in sample_sets:
        print sample_set.name, sample_set.desc
        # convert sample set to membership vector
        membership = sample_set.get_array(samples)
        # analyze sample set
        res = _ssea_sample_set(rle_lengths, rle_weights_arr, membership, perms)
        print res.es, res.nes, 'p', res.nominal_p, 'nes min', res.nes_min, 'nes max', res.nes_max        
        plot(res.diff, membership, weights_arr)

    
    return

