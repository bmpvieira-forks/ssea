'''
Created on Oct 9, 2013

@author: mkiyer
'''
import numpy as np

# local imports
from kernel import ssea_kernel
from base import BOOL_DTYPE, Result

def transform_weights(weights, method):
    if method == 'unweighted':
        return np.ones(len(weights), dtype=np.float)
    elif method == 'weighted':
        return weights
    elif method == 'log':
        signs = np.array([-1.0 if w < 0 else 1.0 for w in weights])
        absarr = np.fabs(weights)
        assert np.all(absarr >= 1.0)
        return signs * np.log2(absarr)
    else:
        assert False

def ssea_run(samples, weights, sample_sets, 
             weight_method_miss='unweighted', 
             weight_method_hit='unweighted',
             weight_const=0.0, 
             weight_noise=0.0,
             perms=10000):
    '''
    '''
    weights = np.array(weights, dtype=np.float)
    # noise does not preserve rank so need to add first
    tweights = weights.copy()
    if weight_noise > 0.0:
        tweights += weight_noise * np.random.random(len(weights))
    # rank order the N samples in D to form L={s1...sn} 
    ranks = np.argsort(tweights)[::-1]
    samples = [samples[i] for i in ranks]
    weights = weights[ranks]
    tweights = tweights[ranks]
    # perform power transform and adjust by constant
    tweights += weight_const
    tweights_miss = np.fabs(transform_weights(tweights, weight_method_miss)) 
    tweights_hit = np.fabs(transform_weights(tweights, weight_method_hit))    
    # convert sample sets to membership vectors
    membership = np.zeros((len(samples),len(sample_sets)), 
                          dtype=BOOL_DTYPE)
    for j,sample_set in enumerate(sample_sets):
        membership[:,j] = sample_set.get_array(samples)
    # determine enrichment score (ES)
    perm = np.arange(len(samples))
    es_vals, es_run_inds, es_runs = \
        ssea_kernel(tweights, tweights_miss, tweights_hit, 
                    membership, perm)
    # permute samples and determine ES null distribution
    es_null = np.zeros((perms, len(sample_sets)), dtype=np.float)
    for i in xrange(perms):
        np.random.shuffle(perm)
        es_null[i] = ssea_kernel(tweights, tweights_miss, 
                                 tweights_hit, membership, perm)[0]
    # default containers for results
    nes_vals = np.zeros(membership.shape[1], dtype=np.float)
    pvals = np.ones(membership.shape[1], dtype=np.float)
    # separate the positive and negative sides of the null distribution
    # based on the observed enrichment scores
    es_neg_inds = (es_vals < 0).nonzero()[0]
    if len(es_neg_inds) > 0:
        # mask positive scores 
        es_null_neg = np.ma.masked_greater(es_null[:,es_neg_inds], 0)
        # Adjust for variation in gene set size. Normalize ES(S,null)
        # and the observed ES(S), separately rescaling the positive and
        # negative scores by dividing by the mean of the ES(S,null) to
        # yield normalized scores NES(S,null)
        es_null_neg_means = es_null_neg.mean(axis=0)
        nes_null_neg = es_null_neg / np.fabs(es_null_neg_means)
        nes_null_neg_count = nes_null_neg.count()
        # To compute FWER create a histogram of the maximum NES(S,null) 
        # over all S for each of the permutations by using the positive 
        # or negative values corresponding to the sign of the observed NES(S). 
        nes_null_min = nes_null_neg.min(axis=1).compressed()
        # Normalize the observed ES(S) by rescaling by the mean of
        # the ES(S,null) separately for positive and negative ES(S)
        nes_obs_neg = (np.ma.MaskedArray(es_vals[es_neg_inds]) / 
                       np.fabs(es_null_neg_means))
        nes_obs_neg_count = nes_obs_neg.count()
        nes_vals[es_neg_inds] = nes_obs_neg
        # estimate nominal p value for S from ES(S,null) by using the
        # positive or negative portion of the distribution corresponding
        # to the sign of the observed ES(S)
        pneg = 1.0 + (es_null_neg <= es_vals[es_neg_inds]).sum(axis=0)
        pneg = pneg / (1.0 + es_null_neg.count(axis=0).astype(float))    
        pvals[es_neg_inds] = 2.0 * pneg
    # do the same for the positive enrichment scores (see above for
    # detailed comments
    es_pos_inds = (es_vals >= 0).nonzero()[0]
    if len(es_pos_inds) > 0:
        # mask negative scores 
        es_null_pos = np.ma.masked_less(es_null[:,es_pos_inds], 0)
        # normalize
        es_null_pos_means = es_null_pos.mean(axis=0)    
        nes_null_pos = es_null_pos / np.fabs(es_null_pos_means)
        nes_null_pos_count = nes_null_pos.count()
        # store max NES for FWER calculation
        nes_null_max = nes_null_pos.max(axis=1).compressed()
        nes_obs_pos = (np.ma.MaskedArray(es_vals[es_pos_inds]) / 
                       np.fabs(es_null_pos_means))
        nes_obs_pos_count = nes_obs_pos.count()
        nes_vals[es_pos_inds] = nes_obs_pos
        # estimate p values
        ppos = 1.0 + (es_null_pos >= es_vals[es_pos_inds]).sum(axis=0)
        ppos = ppos / (1.0 + es_null_pos.count(axis=0).astype(np.float))
        pvals[es_pos_inds] = 2.0 * ppos
    # Control for multiple hypothesis testing and summarize results
    results = []
    for j in xrange(membership.shape[1]):
        # For a given NES(S) = NES* >= 0, the FDR is the ratio of the 
        # percentage of all permutations NES(S,null) >= 0, whose 
        # NES(S,null) >= NES*, divided by the percentage of observed S with 
        # NES(S) >= 0, whose NES(S) >= NES*, and similarly for 
        # NES(S) = NES* <= 0.
        # Also, compute FWER p values by finding the percentage 
        # of NES_max(S,null) >= NES*, and similarly for 
        # NES_min(S,null) <= NES* for positive and negative NES*, 
        # respectively 
        nes = nes_vals[j]
        if np.sign(es_vals[j]) < 0:
            n = (nes_null_neg <= nes).sum() / float(nes_null_neg_count)
            d = (nes_obs_neg <= nes).sum() / float(nes_obs_neg_count)
            fwerp = (nes_null_min <= nes).sum() / float(len(nes_null_min))
        else:
            n = (nes_null_pos >= nes).sum() / float(nes_null_pos_count)
            d = (nes_obs_pos >= nes).sum() / float(nes_obs_pos_count)
            fwerp = (nes_null_max >= nes).sum() / float(len(nes_null_max))
        qval = n / d
        # create result object
        res = Result()
        res.sample_set = sample_sets[j]
        res.samples = samples
        res.weights = weights
        res.membership = membership[:,j]
        res.weights_miss = tweights_miss
        res.weights_hit = tweights_hit
        res.es = es_vals[j]
        res.es_run_ind = es_run_inds[j]
        res.es_run = es_runs[:,j]
        res.es_null = es_null[:,j]
        res.pval = pvals[j]
        res.nes = nes_vals[j]
        res.qval = qval
        res.fwerp = fwerp
        results.append(res)
    return results
