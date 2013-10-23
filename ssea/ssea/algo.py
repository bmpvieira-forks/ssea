'''
Created on Oct 9, 2013

@author: mkiyer
'''
from itertools import groupby
import numpy as np
import matplotlib.gridspec as gridspec
from matplotlib import figure

# local imports
from kernel import ssea_kernel
from base import BOOL_DTYPE

# constant to saturate logarithms of small numbers and prevent log of zero 
LOG_CONSTANT = 0.1

def quantile(a, frac, limit=(), interpolation_method='fraction'):
    '''copied verbatim from scipy code (scipy.org)'''
    def _interpolate(a, b, fraction):
        return a + (b - a)*fraction;
    values = np.sort(a, axis=0)
    if limit:
        values = values[(limit[0] <= values) & (values <= limit[1])]

    idx = frac * (values.shape[0] - 1)
    if (idx % 1 == 0):
        score = values[idx]
    else:
        if interpolation_method == 'fraction':
            score = _interpolate(values[int(idx)], values[int(idx) + 1],
                                 idx % 1)
        elif interpolation_method == 'lower':
            score = values[np.floor(idx)]
        elif interpolation_method == 'higher':
            score = values[np.ceil(idx)]
        else:
            raise ValueError("interpolation_method can only be 'fraction', " \
                             "'lower' or 'higher'")
    return score

def transform_weights(weights, method):
    if method == 'unweighted':
        return np.ones(len(weights), dtype=np.float)
    elif method == 'weighted':
        return np.array(weights, dtype=np.float)
    elif method == 'log':
        weights = np.array(weights, dtype=np.float)
        absweights = np.fabs(weights)
        const = max(LOG_CONSTANT, absweights.min())
        return np.sign(weights) * (np.log2(const + absweights) - 
                                   np.log2(const))

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

def rld2d(lengths, arr2d):
    '''run length decode for 2d array'''
    assert len(lengths) == arr2d.shape[0]
    out = np.zeros((np.sum(lengths),arr2d.shape[1]), dtype=np.float)    
    offset = 0
    for i in xrange(len(lengths)):
        length = lengths[i]
        vals = arr2d[i,:]
        for j in xrange(length):
            out[offset + j,:] = vals
        offset += length
    return out

class SampleSetResult(object):
    def __init__(self):
        self.sample_set = None
        self.es = 0.0
        self.nes = 0.0
        self.es_run_ind = 0
        self.es_run = None
        self.pval = 1.0
        self.qval = 1.0
        self.fwerp = 1.0
        self.es_null = None
        self.membership = None
        self.samples = None
        self.weights = None
        self.membership = None
        self.weights_miss = None
        self.weights_hit = None

    def plot_null_distribution(self, fig=None):
        if fig is None:
            fig = figure.Figure()
        fig.clf()
        percent_neg = (100. * (self.es_null < 0).sum() / 
                       self.es_null.shape[0])
        num_bins = int(round(float(self.es_null.shape[0]) ** (1./2.)))
        #n, bins, patches = ax.hist(es_null, bins=num_bins, histtype='stepfilled')
        ax = fig.add_subplot(1,1,1)
        ax.hist(self.es_null, bins=num_bins, histtype='bar')
        ax.axvline(x=self.es, linestyle='--', color='black')
        ax.set_title('Random ES distribution')
        ax.set_ylabel('P(ES)')
        ax.set_xlabel('ES (Sets with neg scores: %.0f%%)' % (percent_neg))
        return fig

    def plot(self, plot_conf_int=True, conf_int=0.95, fig=None):
        if fig is None:
            fig = figure.Figure()
        fig.clf()
        gs = gridspec.GridSpec(3, 1, height_ratios=[2,1,1])
        # running enrichment score
        ax0 = fig.add_subplot(gs[0])
        x = np.arange(len(self.es_run))
        y = self.es_run
        ax0.plot(x, y, lw=2, color='blue', label='Enrichment profile')
        ax0.axhline(y=0, color='gray')
        ax0.axvline(x=self.es_run_ind, lw=1, linestyle='--', color='black')
        # confidence interval
        if plot_conf_int:
            if np.sign(self.es) < 0:
                es_null_sign = self.es_null[self.es_null < 0]                
            else:
                es_null_sign = self.es_null[self.es_null >= 0]                
            # plot confidence interval band
            es_null_mean = es_null_sign.mean()
            es_null_low = quantile(es_null_sign, 1.0-conf_int)
            es_null_hi = quantile(es_null_sign, conf_int)
            lower_bound = np.repeat(es_null_low, len(x))
            upper_bound = np.repeat(es_null_hi, len(x))
            ax0.axhline(y=es_null_mean, lw=2, color='red', ls=':')
            ax0.fill_between(x, lower_bound, upper_bound,
                             lw=0, facecolor='yellow', alpha=0.5,
                             label='%.2f CI' % (100. * conf_int))
            # here we use the where argument to only fill the region 
            # where the ES is above the confidence interval boundary
            if np.sign(self.es) < 0:
                ax0.fill_between(x, y, lower_bound, where=y<lower_bound, 
                                 lw=0, facecolor='blue', alpha=0.5)
            else:
                ax0.fill_between(x, upper_bound, y, where=y>upper_bound, 
                                 lw=0, facecolor='blue', alpha=0.5)
        ax0.set_xlim((0,len(self.es_run)))
        ax0.grid(True)
        ax0.set_xticklabels([])
        ax0.set_ylabel('Enrichment score (ES)')
        ax0.set_title('Enrichment plot: %s' % (self.sample_set.name))
        # membership in sample set
        ax1 = fig.add_subplot(gs[1])
        ax1.vlines(self.membership.nonzero()[0], ymin=0, ymax=1, lw=0.5, 
                   color='black', label='Hits')
        ax1.set_xlim((0,len(self.es_run)))
        ax1.set_ylim((0,1))
        ax1.set_xticks([])
        ax1.set_yticks([])
        ax1.set_xticklabels([])
        ax1.set_yticklabels([])
        ax1.set_ylabel('Set')
        # weights
        ax2 = fig.add_subplot(gs[2])
        ax2.plot(self.weights_miss, color='blue')
        ax2.plot(self.weights_hit, color='red')
        ax2.set_xlim((0,len(self.es_run)))
        ax2.set_xlabel('Samples')
        ax2.set_ylabel('Weights')
        # draw
        fig.tight_layout()
        return fig
       
    def get_details_table(self):
        rows = [['index', 'sample', 'rank', 'weight', 'running_es', 
                 'core_enrichment']]
        member_inds = (self.membership > 0).nonzero()[0]
        for i,ind in enumerate(member_inds):
            is_enriched = int(ind <= self.es_run_ind)
            rows.append([i, self.samples[ind], ind+1, self.weights[ind], 
                         self.es_run[ind], is_enriched])
        return rows

def ssea_run(samples, weights, sample_sets, 
             weight_method_miss='unweighted',
             weight_method_hit='unweighted',
             perms=10000):
    # rank order the N samples in D to form L={s1...sn} 
    ranks = np.argsort(weights)[::-1]
    samples = [samples[i] for i in ranks]
    weights = [weights[i] for i in ranks]
    # transform weights based on weight method
    weights_miss = transform_weights(weights, weight_method_miss)
    weights_hit = transform_weights(weights, weight_method_hit) 
    # perform run length encoding to handle ties in weights
    rle_lengths, rle_weights = rle(weights)
    # use the absolute value of the run length encoded weights for the
    # main SSEA calculation
    rle_weights = np.fabs(rle_weights)
    # transform weights (same as above)
    rle_weights_miss = transform_weights(rle_weights, weight_method_miss)
    rle_weights_hit = transform_weights(rle_weights, weight_method_hit)
    # convert sample sets to membership vectors
    membership = np.zeros((len(samples),len(sample_sets)), 
                            dtype=BOOL_DTYPE)
    for j,sample_set in enumerate(sample_sets):
        membership[:,j] = sample_set.get_array(samples)
    # determine enrichment score (ES)
    perm = np.arange(len(samples))
    es_vals, rle_es_inds, rle_es_runs = \
        ssea_kernel(rle_lengths, rle_weights_miss, rle_weights_hit, 
                    membership, perm)
    # permute samples and determine ES null distribution
    es_null = np.zeros((perms, len(sample_sets)), dtype=np.float)
    for i in xrange(perms):
        np.random.shuffle(perm)
        es_null[i] = ssea_kernel(rle_lengths, rle_weights_miss, 
                                 rle_weights_hit, membership, perm)[0]      
    # decode run length encoding
    es_run_inds = [sum(rle_lengths[:x]) for x in rle_es_inds]
    es_runs = rld2d(rle_lengths, rle_es_runs)
    # default containers for results
    nes_vals = np.zeros(membership.shape[1], dtype=np.float)
    pvals = np.ones(membership.shape[1], dtype=np.float)
    # separate the positive and negative sides of the null distribution
    # based on the observed enrichment scores
    es_neg_inds = (es_vals < 0).nonzero()[0]
    if len(es_neg_inds) > 0:
        # mask positive scores 
        es_null_neg = np.ma.masked_greater_equal(es_null[:,es_neg_inds], 0)
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
        pneg = (es_null_neg <= es_vals[es_neg_inds]).sum(axis=0)
        pneg = pneg / (es_null_neg.count(axis=0).astype(float))    
        pvals[es_neg_inds] = pneg
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
        ppos = (es_null_pos >= es_vals[es_pos_inds]).sum(axis=0)
        ppos = ppos / (es_null_pos.count(axis=0).astype(np.float))
        pvals[es_pos_inds] = ppos
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
        res = SampleSetResult()
        res.sample_set = sample_sets[j]
        res.es = es_vals[j]
        res.nes = nes_vals[j]
        res.es_run_ind = es_run_inds[j]
        res.es_run = es_runs[:,j]
        res.pval = pvals[j]
        res.qval = qval
        res.fwerp = fwerp
        res.es_null = es_null[:,j]
        res.samples = samples
        res.weights = weights
        res.membership = membership[:,j]
        res.weights_miss = weights_miss
        res.weights_hit = weights_hit
        results.append(res)
    return results
