'''
Created on Oct 9, 2013

@author: mkiyer
'''
from itertools import groupby
import numpy as np
import matplotlib.pyplot as plt 
import matplotlib.gridspec as gridspec

WEIGHT_METHODS = ['unweighted', 'weighted', 'log']
LOG_TRANSFORM_CONSTANT = 1e-3
LOG_TRANSFORM_BASE = 2.0

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

def transform_weights(weights, method, params):
    if method == 'unweighted':
        return np.ones(len(weights), dtype=np.float)
    elif method == 'weighted':
        return np.array(weights, dtype=np.float)
    elif method == 'log':
        weights = np.array(weights, dtype=np.float)
        nzweights = weights[weights > 0]
        if len(nzweights) == 0:
            const = LOG_TRANSFORM_CONSTANT
        else:
            const = max(LOG_TRANSFORM_CONSTANT, nzweights.min())
        return np.log2(const + weights) - np.log2(const)

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

class SampleSet(object):    
    def __init__(self, name=None, desc=None, value=None):
        self.name = name
        self.desc = desc
        self.value = value
        
    def get_array(self, samples):
        return np.array([x in self.value for x in samples])

class SampleSetResult(object):
    def __init__(self):
        self.sample_set = None
        self.es = 0.0
        self.es_ind = 0
        self.es_arr = None
        self.nominal_p = 1.0
        self.nes = 0.0
        self.es_null_neg = None
        self.es_null_pos = None

    def get_nes_null(self, sign):
        if sign < 0:
            return -self.es_null_neg / self.es_null_neg.mean()
        else:
            return self.es_null_pos / self.es_null_pos.mean()

    def plot_null_distribution(self):
        num_neg = len(self.es_null_neg)
        num_pos = len(self.es_null_pos)
        percent_pos = 100. * float(num_pos) / (num_neg + num_pos)
        es_null = np.concatenate((-self.es_null_neg[::-1], self.es_null_pos))
        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)
        num_bins = int(round(float(num_neg + num_pos) ** (1./2.)))
        #n, bins, patches = ax.hist(es_null, bins=num_bins, histtype='stepfilled')
        n, bins, patches = ax.hist(es_null, bins=num_bins, histtype='bar')
        ax.axvline(x=self.es, linestyle='--', color='black')
        ax.set_title('Random ES distribution')
        ax.set_ylabel('P(ES)')
        ax.set_xlabel('ES (Sets with pos scores: %.0f%%)' % (percent_pos))
        return fig

    def plot(self, membership, weights,
             title='Enrichment plot',
             plot_conf_int=True, conf_int=0.95):
        fig = plt.figure()
        #fig = plt.figure(figsize=(8, 6)) 
        gs = gridspec.GridSpec(3, 1, height_ratios=[2,1,1])
        # running enrichment score
        ax0 = plt.subplot(gs[0])
        y = [0]
        y.extend(self.es_arr)
        x = np.arange(len(y))
        ax0.plot(x, y, lw=2, color='blue', label='Enrichment profile')
        ax0.axhline(y=0, color='gray')
        ax0.axvline(x=self.es_ind, linestyle='--', color='black')
        # confidence interval
        if plot_conf_int:
            if np.sign(self.es) < 0:                
                es_null_sign = -self.es_null_neg[::-1]
            else:
                es_null_sign = self.es_null_pos
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
        ax0.grid(True)
        ax0.set_xticklabels([])
        ax0.set_ylabel('Enrichment score (ES)')
        ax0.set_title(title)
        # membership in sample set
        ax1 = plt.subplot(gs[1])
        ax1.bar(np.arange(len(self.es_arr)), membership, 1, color='black', 
                edgecolor='none', label='Hits')
        ax1.set_xticks([])
        ax1.set_yticks([])
        ax1.set_xticklabels([])
        ax1.set_yticklabels([])
        ax1.set_ylabel('Set')
        # weights
        ax2 = plt.subplot(gs[2])
        ax2.plot(weights[:,0], color='blue')
        ax2.plot(weights[:,1], color='red')
        ax2.set_xlabel('Samples')
        ax2.set_ylabel('Weights')
        # draw
        fig.tight_layout()
        return fig
    
    def report(self, samples, weights, membership):
        lines = []
        lines.append(['# INDEX', 'SAMPLE', 'RANK', 'WEIGHT', 'RUNNING_ES', 
                      'CORE_ENRICHMENT'])
        member_inds = (membership > 0).nonzero()[0]
        for i,ind in enumerate(member_inds):
            is_enriched = 'Yes' if (ind < self.es_ind) else 'No'
            fields = [i, samples[ind], ind+1, weights[ind], 
                      self.es_arr[ind], is_enriched]
            lines.append(map(str, fields))
        return lines
    
    def report_html(self):
        pass

class SSEA(object):
    pass

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
    # TODO: handle case where one set has all zeros and cumsum is zero
    hitmiss /= hitmiss[-1,:]
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
        es_null[i] = _ssea_kernel(rle_lengths, rle_weights_arr, 
                                  null_membership)[0]
    # separate the positive and negative null distribution values
    es_null_neg = np.sort(np.abs(es_null[np.sign(es_null) < 0]))
    es_null_pos = np.sort(es_null[np.sign(es_null) > 0])
    # estimate nominal p value for S from ES_null by using the
    # positive or negative portion of the distribution corresponding
    # to the sign of the observed ES(S)
    es_null_sign = es_null_neg if np.sign(es) < 0 else es_null_pos  
    nominal_p = 1.0 - (float(es_null_sign.searchsorted(abs(es))) / 
                       len(es_null_sign))
    # adjust for variation in gene set size. Normalize the ES_null
    # and the observed ES(S), separately rescaling the positive and
    # negative scores by dividing by the mean of the ES_null to
    # yield the normalized scores nes_null and nes_score
    nes = es / es_null_sign.mean()
    # undo run length encoding to save final result
    es_arr = rld(rle_lengths, es_arr)
    es_ind = sum(rle_lengths[:es_ind+1])
    # save result
    res = SampleSetResult()
    res.es = es
    res.es_ind = es_ind
    res.es_arr = es_arr
    res.nominal_p = nominal_p
    res.nes = nes
    res.es_null_sign = es_null_sign
    res.es_null_neg = es_null_neg
    res.es_null_pos = es_null_pos
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
        # convert sample set to membership vector
        membership = sample_set.get_array(samples)
        # analyze sample set
        res = _ssea_sample_set(rle_lengths, rle_weights_arr, membership, perms)
        # create report
        fig = res.plot_null_distribution()    
        fig.savefig('null_distribution_plot.png')
        fig = res.plot(membership, weights_arr,
                       title='Enrichment plot: %s' % (sample_set.name))
        fig.savefig('enrichment_plot.png')
        lines = res.report(samples, weights_hit, membership)
        print sample_set.name, sample_set.desc
        print res.es, res.nes, 'p', res.nominal_p
        for line in lines:
            print '\t'.join(line)
        
        # save the min/max of the NES and NES_null scores 
        # for computing the FWER
        #nes_min = min(nes, nes_null_neg.min())
        #nes_max = max(nes, nes_null_pos.max())
        #print 'nes min', res.nes_min, 'nes max', res.nes_max        
        #nes_null_neg = es_null_neg / es_null_neg.mean()
        #nes_null_pos = es_null_pos / es_null_pos.mean()

    
    return

