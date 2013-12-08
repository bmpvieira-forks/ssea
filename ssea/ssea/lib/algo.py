'''
Created on Oct 9, 2013

@author: mkiyer
'''
import os
import logging
import shutil
from collections import namedtuple
from multiprocessing import Process
import numpy as np

# local imports
import ssea.lib.cfisher as fisher
from ssea.lib.batch_sort import batch_sort, batch_merge
from ssea.lib.kernel import ssea_kernel, RandomState
from ssea.lib.base import BOOL_DTYPE, Result, interp
from ssea.lib.config import Config
from ssea.lib.countdata import BigCountMatrix

# results directories
TMP_DIR = 'tmp'

# temporary files
JSON_UNSORTED_SUFFIX = '.unsorted.json'
JSON_SORTED_SUFFIX = '.sorted.json'
NPY_HISTS_SUFFIX = '.hists.npz'

# improves readability of code
KernelResult = namedtuple('KernelResult', ('ranks',
                                           'norm_counts', 
                                           'norm_counts_miss',
                                           'norm_counts_hit',
                                           'es_vals', 
                                           'es_ranks', 
                                           'es_runs'))

RunResult = namedtuple('RunResult', ('sample_set_index',
                                     'result',
                                     'null_nes',
                                     'resample_nes'))

# batch sort configuration
SORT_BUFFER_SIZE = 32000

# floating point precision to output in report
FLOAT_PRECISION = 10
P_VALUE_PRECISION = 200

# to compute FDR q values we need to store normalized enrichment 
# scores (NES). we accomplish this storing a histogram of NES values during
# runtime and then merging histograms from parallel processes prior to 
# computing global statistics. the NES can range from [-inf, +inf] making
# it difficult for a histogram to store enough granularity in few bins.
# we accomplish this here by using log-space histogram bins that vary from
# 0.1 to 100. values outside this range are clipped and stored so that no
# values are lost
NUM_NES_BINS = 10001
NES_BINS = np.logspace(-1,2,num=NUM_NES_BINS,base=10)
LOG_NES_BINS = np.log10(NES_BINS)
NES_MIN = NES_BINS[0]
NES_MAX = NES_BINS[-1]
#LOG_NES_BIN_CENTERS = (np.log10(NES_BINS[:-1]) + np.log10(NES_BINS[1:])) / 2
#NES_BIN_CENTERS = 10.0 ** LOG_NES_BIN_CENTERS

def _init_hists(nsets):
    '''returns a dictionary with histogram arrays initialized to zero'''
    return {'null_nes_pos': np.zeros((nsets,NUM_NES_BINS-1), dtype=np.float),
            'null_nes_neg': np.zeros((nsets,NUM_NES_BINS-1), dtype=np.float),
            'obs_nes_pos': np.zeros((nsets,NUM_NES_BINS-1), dtype=np.float),
            'obs_nes_neg': np.zeros((nsets,NUM_NES_BINS-1), dtype=np.float)}

def _cmp_json_nes(line):
    '''comparison function for batch_sort'''
    res = Result.from_json(line.strip())
    return abs(res.nes)

def ssea_run(counts, size_factors, membership, rng, config):
    '''
    counts: numpy array of float values
    size_factors: normalization factors for counts
    membership: boolean 2d array (samples, sets) with set membership
    rng: RandomState object
    config: Config object
    '''
    # first run without resampling count data and save seed
    rand_seed = rng.seed
    k = ssea_kernel(counts, size_factors, membership, rng,
                     resample_counts=False,
                     permute_samples=False,
                     add_noise=True,
                     noise_loc=config.noise_loc, 
                     noise_scale=config.noise_scale,
                     method_miss=config.weight_miss,
                     method_hit=config.weight_hit,
                     method_param=config.weight_param)
    k = KernelResult._make(k)
    ranks = k.ranks
    es_vals = k.es_vals
    es_ranks = k.es_ranks
    # next run to generate a range of observed enrichment scores
    shape = (config.resampling_iterations, membership.shape[1])    
    resample_es_vals = np.zeros(shape, dtype=np.float) 
    resample_es_ranks = np.zeros(shape, dtype=np.int)
    resample_nes_vals = np.ma.empty(shape, dtype=np.float)
    for i in xrange(config.resampling_iterations):
        k = ssea_kernel(counts, size_factors, membership, rng,
                         resample_counts=True,
                         permute_samples=False,
                         add_noise=True,
                         noise_loc=config.noise_loc, 
                         noise_scale=config.noise_scale,
                         method_miss=config.weight_miss,
                         method_hit=config.weight_hit,
                         method_param=config.weight_param)
        k = KernelResult._make(k)
        resample_es_vals[i,:] = k.es_vals
        resample_es_ranks[i,:] = k.es_ranks
    # permute samples and determine ES null distribution
    shape = (config.perms, membership.shape[1])
    null_es_vals = np.zeros(shape, dtype=np.float) 
    null_es_ranks = np.zeros(shape, dtype=np.float)
    for i in xrange(config.perms):
        k = ssea_kernel(counts, size_factors, membership, rng,
                         resample_counts=True,
                         permute_samples=True,
                         add_noise=True,
                         noise_loc=config.noise_loc, 
                         noise_scale=config.noise_scale,
                         method_miss=config.weight_miss,
                         method_hit=config.weight_hit,
                         method_param=config.weight_param)
        k = KernelResult._make(k)
        null_es_vals[i,:] = k.es_vals
        null_es_ranks[i,:] = k.es_ranks
    # default containers for results
    nes_vals = np.empty(membership.shape[1], dtype=np.float)
    null_nes_vals = np.ma.empty(shape, dtype=np.float)
    pvals = np.empty(membership.shape[1], dtype=np.float)
    fdr_q_values = np.empty(membership.shape[1], dtype=np.float)
    # separate the positive and negative sides of the null distribution
    # based on the observed enrichment scores
    es_inds_neg = (es_vals < 0).nonzero()[0]
    es_inds_pos = (es_vals >= 0).nonzero()[0]
    for es_inds_sign, mask_func in ((es_inds_neg, np.ma.masked_greater_equal), 
                                    (es_inds_pos, np.ma.masked_less_equal)):
        if len(es_inds_sign) == 0:
            continue
        # mask scores 
        null_es_sign = mask_func(null_es_vals[:,es_inds_sign], 0)
        # Adjust for variation in gene set size. Normalize ES(S,null)
        # and the observed ES(S), separately rescaling the positive and
        # negative scores by dividing by the mean of the ES(S,null) to
        # yield normalized scores NES(S,null)
        null_es_means_sign = null_es_sign.mean(axis=0)
        null_nes_sign = null_es_sign / np.fabs(null_es_means_sign)
        null_nes_vals[:,es_inds_sign] = null_nes_sign
        # Normalize the observed ES(S) by rescaling by the mean of
        # the ES(S,null) separately for positive and negative ES(S)
        obs_nes_sign = (np.ma.MaskedArray(es_vals[es_inds_sign]) / 
                        np.fabs(null_es_means_sign))
        nes_vals[es_inds_sign] = obs_nes_sign
        # Normalize the resampled ES(S) by rescaling in a similar manner
        resample_es_sign = mask_func(resample_es_vals[:,es_inds_sign], 0)
        resample_nes_sign = resample_es_sign / np.fabs(null_es_means_sign)
        resample_nes_vals[:,es_inds_sign] = resample_nes_sign
        # estimate nominal p value for S from ES(S,null) by using the
        # positive or negative portion of the distribution corresponding
        # to the sign of the observed ES(S)
        p = (np.fabs(null_es_sign) >= np.fabs(es_vals[es_inds_sign])).sum(axis=0).astype(np.float)
        p /= null_es_sign.count(axis=0).astype(np.float)
        pvals[es_inds_sign] = p
        # For a given NES(S) = NES* >= 0, the FDR is the ratio of the 
        # percentage of all permutations NES(S,null) >= 0, whose 
        # NES(S,null) >= NES*, divided by the percentage of observed S with 
        # NES(S) >= 0, whose NES(S) >= NES*, and similarly for 
        # NES(S) = NES* <= 0.
        # Also, compute FWER p values by finding the percentage 
        # of NES_max(S,null) >= NES*, and similarly for 
        # NES_min(S,null) <= NES* for positive and negative NES*, 
        # respectively
        null_nes_count_sign = float(null_nes_sign.count())
        obs_nes_count_sign = float(obs_nes_sign.count())
        #resample_nes_count_sign = float(resample_nes_sign.count())
        null_nes_sign_abs = np.fabs(null_nes_sign)
        obs_nes_sign_abs = np.fabs(obs_nes_sign)
        #resample_nes_sign_abs = np.fabs(resample_nes_sign)
        for j in es_inds_sign:
            nes = np.fabs(nes_vals[j])
            if null_nes_count_sign == 0:
                n = 0.0
            else:
                n = (null_nes_sign_abs >= nes).sum() / null_nes_count_sign
            if obs_nes_count_sign == 0:
                d = 0.0
            else:
                d = (obs_nes_sign_abs >= nes).sum() / obs_nes_count_sign
                #d = (resample_nes_sign_abs >= nes).sum() / resample_nes_count_sign
            if (n == 0) or (d == 0):
                fdr_q_values[j] = 0.0
            else:
                fdr_q_values[j] = n / d
    # q value is defined as the minimum FDR for which a test can be called
    # significant. to compute q values iterate over sample sets sorted by
    # NES and assign q values to either the minimum FDR previous seen or 
    # the current FDR, whichever is lesser.
    min_sign_fdr = [1.0, 1.0]
    for j in np.argsort(np.fabs(nes_vals)):
        ispos = 0 if es_vals[j] < 0 else 1
        if fdr_q_values[j] < min_sign_fdr[ispos]:
            min_sign_fdr[ispos] = fdr_q_values[j]
        else:
            fdr_q_values[j] = min_sign_fdr[ispos]
    # Create result objects for this SSEA test
    for j in xrange(membership.shape[1]):            
        res = Result()
        res.rand_seed = rand_seed
        res.es = round(es_vals[j], FLOAT_PRECISION)
        res.es_rank = int(es_ranks[j])
        res.nominal_p_value = round(pvals[j], P_VALUE_PRECISION)
        res.nes = round(nes_vals[j], FLOAT_PRECISION)
        res.fdr_q_value = round(fdr_q_values[j], P_VALUE_PRECISION)
        # save some of the resampled es points 
        res.resample_es_vals = np.around(resample_es_vals[:Config.MAX_ES_POINTS,j], FLOAT_PRECISION)
        res.resample_es_ranks = resample_es_ranks[:Config.MAX_ES_POINTS,j]
        res.null_es_vals = np.around(null_es_vals[:Config.MAX_ES_POINTS,j], FLOAT_PRECISION)
        res.null_es_ranks = null_es_ranks[:Config.MAX_ES_POINTS,j]
        res.null_es_hist = np.histogram(null_es_vals[:,j], 
                                        bins=Config.NULL_ES_BINS)[0]
        # get indexes of hits in this set
        m = membership[ranks,j]
        hit_inds = (m > 0).nonzero()[0]       
        num_hits = hit_inds.shape[0]
        num_misses = m.shape[0] - num_hits
        # calculate leading edge stats
        if es_vals[j] < 0:
            core_hits = sum(i >= es_ranks[j] for i in hit_inds)
            core_misses = (membership.shape[0] - es_ranks[j]) - core_hits
        else:
            core_hits = sum(i <= es_ranks[j] for i in hit_inds)
            core_misses = 1 + es_ranks[j] - core_hits
        null_hits = num_hits - core_hits
        null_misses = num_misses - core_misses
        # fisher exact test (one-sided hypothesis that LE is enricheD)
        fisher_p_value = fisher.pvalue(core_hits, core_misses, null_hits, null_misses).right_tail
        # odds ratio
        n = np.inf if null_hits == 0 else float(core_hits) / null_hits
        d = np.inf if null_misses == 0 else float(core_misses) / null_misses
        if np.isfinite(n) and np.isfinite(d):
            if n == 0 and d == 0:
                odds_ratio = np.nan
            else:
                odds_ratio = np.inf if d == 0 else (n / d)
        elif np.isfinite(d):
            odds_ratio = np.inf
        else:
            odds_ratio = np.nan if n == 0 else 0.0
        # create dictionary result
        res.core_hits = int(core_hits)
        res.core_misses = int(core_misses)
        res.null_hits = int(null_hits)
        res.null_misses = int(null_misses)
        res.fisher_p_value = np.round(fisher_p_value, P_VALUE_PRECISION)
        res.odds_ratio = odds_ratio
        # return null distributions for subsequent fdr calculations
        yield RunResult(j, res, null_nes_vals[:,j].compressed(), 
                        resample_nes_vals[:,j].compressed())

def ssea_serial(matrix_dir, shape, sample_sets, config, output_basename, 
                startrow=None, endrow=None):
    '''
    main SSEA loop (single processor)
    
    matrix_dir: numpy memmap matrix containing numeric data 
    shape: tuple with shape of weight matrix (nrows,ncols)
    sample_sets: list of SampleSet objects
    config: Config object
    output_basename: prefix for writing result files
    '''
    # initialize random number generator
    rng = RandomState()
    # open data matrix
    bm = BigCountMatrix.open(matrix_dir)
    # determine range of matrix to process
    if startrow is None:
        startrow = 0
    if endrow is None:
        endrow = bm.shape[0]
    assert startrow < endrow
    # setup global ES histograms
    hists = _init_hists(len(sample_sets))
    # setup report file
    unsorted_json_file = output_basename + JSON_UNSORTED_SUFFIX
    outfileh = open(unsorted_json_file, 'wb')    
    for i in xrange(startrow, endrow):
        logging.debug("\tRow: %d (%d-%d)" % (i, startrow, endrow))
        # read from memmap
        counts = np.array(bm.counts[i,:], dtype=np.float)
        # remove 'nan' values        
        sample_ids = np.isfinite(counts).nonzero()[0]
        counts = counts[sample_ids]
        size_factors = bm.size_factors[sample_ids]
        # convert sample sets to membership vectors
        membership = np.empty((len(sample_ids),len(sample_sets)), 
                              dtype=BOOL_DTYPE)
        for j,sample_set in enumerate(sample_sets):
            membership[:,j] = sample_set.get_array(sample_ids)
        # run ssea
        for tup in ssea_run(counts, size_factors, membership, rng, config):
            j = tup.sample_set_index
            # save row and column id
            result = tup.result
            result.t_id = i
            result.ss_id = j
            # convert to json
            print >>outfileh, result.to_json()
            # update NES histograms
            null_keys = []
            obs_keys = []
            if result.es <= 0:
                null_keys.append('null_nes_neg')
                obs_keys.append('obs_nes_neg')
            if result.es >= 0:
                null_keys.append('null_nes_pos')
                obs_keys.append('obs_nes_pos')
            for k in xrange(len(null_keys)):
                null_nes = np.clip(np.fabs(tup.null_nes), NES_MIN, NES_MAX)
                obs_nes = np.clip(np.fabs(result.nes), NES_MIN, NES_MAX)
                hists[null_keys[k]][j] += np.histogram(null_nes, NES_BINS)[0]
                hists[obs_keys[k]][j] += np.histogram(obs_nes, NES_BINS)[0]
                # TODO: how to use resampled NES? 
                #resample_nes = np.clip(np.fabs(tup.resample_nes), NES_MIN, NES_MAX)
                #hists[obs_keys[k]][j] += np.histogram(resample_nes, NES_BINS)[0]
    # close report file
    outfileh.close()
    # save histograms to a file
    output_hist_file = output_basename + NPY_HISTS_SUFFIX
    np.savez(output_hist_file, **hists)
    # cleanup
    bm.close()
    # sort output json file by abs(NES)
    logging.debug("Worker %s: sorting results" % (output_basename))
    # make tmp dir for sorting
    if os.path.exists(output_basename):
        shutil.rmtree(output_basename)
    os.makedirs(output_basename) 
    # call batch sort python function
    sorted_json_file = output_basename + JSON_SORTED_SUFFIX
    batch_sort(input=unsorted_json_file,
               output=sorted_json_file,
               key=_cmp_json_nes,
               buffer_size=SORT_BUFFER_SIZE,
               tempdirs=[output_basename])
    # remove tmp dir
    shutil.rmtree(output_basename)
    # remove unsorted json file
    os.remove(unsorted_json_file)    
    logging.debug("Worker %s: done" % (output_basename))
    return 0

def ssea_map(matrix_dir, shape, sample_sets, config,
             worker_basenames, worker_chunks):
    '''
    parallel map step of SSEA run
    
    processes chunks of input matrix in parallel using multiprocessing
    '''
    # start worker processes
    procs = []
    for i in xrange(len(worker_basenames)):
        basename = worker_basenames[i]
        startrow, endrow = worker_chunks[i]
        args = (matrix_dir, shape, sample_sets, config, 
                basename, startrow, endrow)        
        p = Process(target=ssea_serial, args=args)
        p.start()
        procs.append(p)
    # join worker processes (wait for processes to finish)
    for p in procs:
        p.join()
    return 0

def compute_qvalues(json_iterator, hists_file, nrows, nsets):
    '''
    computes fdr q values from json Result objects sorted
    by abs(NES) (low to high)
    
    json_iterator: iterator that yields json objects in sorted order
    hists_file: contains histogram data from null distribution
    nrows: number of rows (transcripts) in analysis
    nsets: number of sample sets in analysis
    '''
    # load histogram data
    hists = np.load(hists_file)
    # compute cumulative sums for fdr interpolation
    cdfs = {}
    for k in ('null_nes_neg', 'null_nes_pos', 'obs_nes_neg', 'obs_nes_pos'):
        h = hists[k]
        cdf2d = np.zeros((h.shape[0],h.shape[1]+1), dtype=np.float)
        for j in xrange(h.shape[0]):
            cdf2d[j,1:] = h[j,:].cumsum()
        cdfs[k] = cdf2d
    # keep track of minimum FDR for each sample set for positive
    # and negative NES separately
    min_fdrs_pos = np.ones(nsets, dtype=np.float)
    min_fdrs_neg = np.ones(nsets, dtype=np.float)
    # keep track of the rank of each transcript within the sample set
    ss_ranks_pos = np.repeat(1, nsets)
    ss_ranks_neg = np.repeat(-1, nsets)
    # perform merge of sorted json files 
    for line in json_iterator:
        # load json document (one per line)
        res = Result.from_json(line.strip())
        # get relevant columns
        i = res.ss_id
        es = res.es
        log_nes_clip = np.log10(np.clip(abs(res.nes), NES_MIN, NES_MAX))
        # compute sample set FDR q-values
        if es < 0:
            null_key = 'null_nes_neg'
            obs_key = 'obs_nes_neg'
            min_fdrs = min_fdrs_neg
            res.ss_rank = int(ss_ranks_neg[i])
            ss_ranks_neg[i] -= 1
        else:
            null_key = 'null_nes_pos'
            obs_key = 'obs_nes_pos'
            min_fdrs = min_fdrs_pos
            res.ss_rank = int(ss_ranks_pos[i])
            ss_ranks_pos[i] += 1
        # to compute a sample set specific FDR q value we look at the
        # aggregated enrichment scores for all tests of that sample set
        # compute the cumulative sums of NES histograms
        # use interpolation to find fraction NES(null) >= NES* and account for
        # the observed permutation in the null set
        # interpolate NES in log space
        null_nes_cumsum = cdfs[null_key][i]
        null_n = interp(log_nes_clip, LOG_NES_BINS, null_nes_cumsum)
        obs_nes_cumsum = cdfs[obs_key][i]
        obs_n = interp(log_nes_clip, LOG_NES_BINS, obs_nes_cumsum)
        n = 1.0 - (null_n / null_nes_cumsum[-1])
        d = 1.0 - (obs_n / obs_nes_cumsum[-1])
        #print 'SS_ID=%d ES=%f NES=%f n=%f (%f / %f) d=%f (%f / %f)' % (i, res.es, res.nes, n, null_n, null_nes_cumsum[-1], d, obs_n, obs_nes_cumsum[-1])
        # update json dict
        if (n <= 0.0) or (d <= 0.0):
            res.ss_fdr_q_value = 0.0
        else:
            res.ss_fdr_q_value = n / d
        #print 'SS_ID=%d ES=%f NES=%f fdr=%f minfdr=%f' % (i, res.es, res.nes, res.ss_fdr_q_value, min_fdrs[i]) 
        # compare with minimum FDR and adjust minimum FDR if necessary
        if res.ss_fdr_q_value < min_fdrs[i]:
            min_fdrs[i] = res.ss_fdr_q_value
        else:
            res.ss_fdr_q_value = min_fdrs[i]
        # convert back to json
        yield res.to_json()
        yield os.linesep
    # cleanup
    hists.close()

def ssea_reduce(input_basenames, nrows, nsets, output_json_file, 
                output_hist_file):
    '''
    reduce step of SSEA run
    
    merges the null distribution histograms from individual worker
    processes from the map step, and then merges the sorted results
    json files from individual worker processes and computes global
    fdr q-value statistics
    '''
    # merge NES histograms
    logging.debug("Merging %d worker histograms" % (len(input_basenames)))
    hists = _init_hists(nsets)
    json_iterables = []
    for i in xrange(len(input_basenames)):        
        # create sorted json file streams
        json_file = input_basenames[i] + JSON_SORTED_SUFFIX
        json_fileh = open(json_file, 'rb', 64*1024)
        json_iterables.append(json_fileh)
        # aggregate numpy arrays
        hist_file = input_basenames[i] + NPY_HISTS_SUFFIX
        npzfile = np.load(hist_file)
        for k in hists.iterkeys():
            hists[k] += npzfile[k]
        npzfile.close()
    np.savez(output_hist_file, **hists)
    # perform merge of sorted json files
    try:
        with open(output_json_file, 'wb', 64*1024) as output:
            iterator = batch_merge(_cmp_json_nes, *json_iterables)
            output.writelines(compute_qvalues(iterator, output_hist_file, nrows, nsets))
    finally:
        for iterable in json_iterables:
            try:
                iterable.close()
            except Exception:
                pass
    # remove worker files
    logging.debug("Removing temporary files")
    for i in xrange(len(input_basenames)):        
        hist_file = input_basenames[i] + NPY_HISTS_SUFFIX
        os.remove(hist_file)
        json_file = input_basenames[i] + JSON_SORTED_SUFFIX
        os.remove(json_file)
