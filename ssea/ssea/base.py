'''
Created on Oct 18, 2013

@author: mkiyer
'''
import numpy as np
import matplotlib.gridspec as gridspec
from matplotlib import figure

BOOL_DTYPE = np.uint8
FLOAT_DTYPE = np.float
WEIGHT_METHODS = ['unweighted', 'weighted', 'log']

class ParserError(Exception):
    '''Error parsing a file.'''
    def __init__(self, msg):
        super(ParserError).__init__(type(self))
        self.msg = "ERROR: %s" % msg
    def __str__(self):
        return self.msg
    def __unicode__(self):
        return self.msg

class SampleSet(object): 
    def __init__(self, name=None, desc=None, value=None):
        self.name = name
        self.desc = desc
        self.value = value
    
    def __len__(self):
        return 0 if self.value is None else len(self.value)

    def get_array(self, samples):
        return np.array([x in self.value for x in samples], 
                        dtype=BOOL_DTYPE)

    @staticmethod
    def parse_smx(filename):
        fileh = open(filename)
        names = fileh.next().rstrip('\n').split('\t')
        descs = fileh.next().rstrip('\n').split('\t')
        if len(names) != len(descs):
            raise ParserError("Number of fields in differ in columns 1 and 2 of sample set file")
        sample_sets = [SampleSet(name=n,desc=d,value=set()) for n,d in zip(names,descs)]
        lineno = 3
        for line in fileh:
            if not line:
                continue
            line = line.rstrip('\n')
            if not line:
                continue
            fields = line.split('\t')
            for i,f in enumerate(fields):
                if not f:
                    continue
                sample_sets[i].value.add(f)
            lineno += 1
        fileh.close()
        return sample_sets

    @staticmethod
    def parse_smt(filename):
        sample_sets = []
        fileh = open(filename)    
        for line in fileh:
            fields = line.strip().split('\t')
            name = fields[0]
            desc = fields[1]
            values = set(fields[2:])
            sample_sets.append(SampleSet(name, desc, values))
        fileh.close()
        return sample_sets

class WeightVector(object):    
    def __init__(self, name=None, metadata=None, samples=None, weights=None):
        self.name = name
        self.metadata = metadata
        self.samples = samples
        self.weights = weights

    @staticmethod
    def parse_wmt(filename, na_val='NA', metadata_cols=2):
        '''
        generator function to parse a weight matrix and return WeightVector 
        objects
        
        filename: string path to file
        na_val: value corresponding to missing data 
        '''
        if metadata_cols < 1:
            raise ParserError("metadata_cols param must be >=1")
        fileh = open(filename)
        header_fields = fileh.next().strip().split('\t')
        lineno = 2
        for line in fileh:
            fields = line.strip().split('\t')
            if len(fields) != len(header_fields):
                raise ParserError("Number of fields in line %d of weight " 
                                  "matrix file %s does not header" %
                                  (lineno, filename))
            name = fields[0]
            metadata = fields[1:metadata_cols]
            weights = []
            samples = []            
            for i in xrange(metadata_cols, len(fields)):
                val = fields[i]
                if val == na_val:
                    continue
                try:
                    weights.append(float(val))
                except ValueError:
                    raise ParserError("Value %s at line number %d cannot be "
                                      "converted to a floating point number" 
                                      % (val, lineno))                                    
                samples.append(header_fields[i])
            yield WeightVector(name, metadata, samples, weights)
            lineno += 1
        fileh.close()

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

class Result(object):
    '''    
    '''
    def __init__(self):
        self.weights = None
        self.samples = None
        self.sample_set = None
        self.membership = None
        self.weights_miss = None
        self.weights_hit = None
        self.es = 0.0
        self.nes = 0.0
        self.es_run_ind = 0
        self.es_run = None
        self.pval = 1.0
        self.qval = 1.0
        self.fwerp = 1.0
        self.es_null = None

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
    