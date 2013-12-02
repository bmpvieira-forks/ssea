'''
Created on Oct 9, 2013

@author: mkiyer
'''
import os
import logging
import subprocess
import shutil
from collections import namedtuple
from multiprocessing import Process
import numpy as np

# local imports
import ssea.cfisher as fisher

from ssea.lib.batch_sort import batch_sort, batch_merge
from ssea.kernel import ssea_kernel, RandomState
from base import BOOL_DTYPE, Config, Result, chunk, interp
from countdata import BigCountMatrix

# results directories
TMP_DIR = 'tmp'

# temporary files
JSON_RAW_SUFFIX = '.raw.json'
JSON_STATS_SUFFIX = '.stats.json'
JSON_SORTED_SUFFIX = '.stats.sorted.json'

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
    resample_nes_vals = np.empty(shape, dtype=np.float)
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
    null_nes_vals = np.empty(shape, dtype=np.float)
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
        yield RunResult(j, res, 
                        null_nes_vals[:,j],
                        resample_nes_vals[:,j])

def ssea_serial(matrix_dir, shape, sample_sets, config, 
                output_basename, output_hist_file, 
                startrow=None, endrow=None):
    '''
    main SSEA loop (single processor)
    
    matrix_dir: numpy memmap matrix containing numeric data 
    shape: tuple with shape of weight matrix (nrows,ncols)
    sample_sets: list of SampleSet objects
    config: Config object
    output_json_file: filename for writing results (JSON format)
    output_hist_file: filename for writing ES histograms
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
    output_json_file = output_basename + JSON_RAW_SUFFIX
    outfileh = open(output_json_file, 'wb')    
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
    np.savez(output_hist_file, **hists)
    # cleanup
    bm.close()

def ssea_parallel(matrix_dir, shape, sample_sets, config, 
                  output_hist_file, tmp_dir): 
    '''
    main SSEA loop (multiprocessing implementation)
    
    See ssea_serial function for documentation
    '''
    # start worker processes
    procs = []
    chunks = []
    worker_basenames = []
    worker_hist_files = []
    for startrow,endrow in chunk(shape[0], config.num_processes):
        i = len(procs)
        logging.debug("Worker process %d range %d-%d (%d total rows)" % 
                      (i, startrow, endrow, (endrow-startrow)))
        # worker output files
        basename = os.path.join(tmp_dir, "w%03d" % (i))
        hist_file = basename + '.hists.npz'
        worker_basenames.append(basename)
        worker_hist_files.append(hist_file)
        args = (matrix_dir, shape, sample_sets, config, 
                basename, hist_file, startrow, endrow)        
        p = Process(target=ssea_serial, args=args)
        p.start()
        procs.append(p)
        chunks.append((startrow,endrow))
    # join worker processes (wait for processes to finish)
    for p in procs:
        p.join()
    # merge workers
    logging.info("Merging %d worker histograms" % (len(procs)))
    # setup global ES histograms
    hists = _init_hists(len(sample_sets))
    for i in xrange(len(procs)):        
        # aggregate numpy arrays
        npzfile = np.load(worker_hist_files[i])
        for k in hists.iterkeys():
            hists[k] += npzfile[k]
        npzfile.close()
    np.savez(output_hist_file, **hists)
    # remove worker histograms
    for filename in worker_hist_files:
        os.remove(filename)
    return worker_basenames

def _compute_fdr_worker(hists_file, input_json_file):
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
    # parse report json and write new values
    fin = open(input_json_file, 'r')
    for line in fin:
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
        else:
            null_key = 'null_nes_pos'
            obs_key = 'obs_nes_pos'
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
        if (n <= 0.0) or (d <= 0.0):
            ss_fdr_q_value = 0.0
        else:
            ss_fdr_q_value = n / d
        # update json dict
        res.ss_fdr_q_value = ss_fdr_q_value
        # convert back to json
        yield res.to_json()
    # cleanup
    fin.close()
    hists.close()
    
def _compute_qvalues(nsets, input_json_file, output_json_file):
    min_pos_fdrs = np.ones(nsets, dtype=np.float)
    min_neg_fdrs = np.ones(nsets, dtype=np.float)
    # parse report json and write new values
    with open(output_json_file, 'w') as fout:
        with open(input_json_file, 'r') as fin:
            for line in fin:
                # load json document (one per line)
                res = Result.from_json(line.strip())
                i = res.ss_id
                min_fdrs = min_neg_fdrs if res.es < 0 else min_pos_fdrs
                #print 'SS_ID=%d ES=%f NES=%f fdr=%f minfdr=%f' % (i, res.es, res.nes, res.ss_fdr_q_value, min_fdrs[i]) 
                if res.ss_fdr_q_value < min_fdrs[i]:
                    min_fdrs[i] = res.ss_fdr_q_value
                else:
                    res.ss_fdr_q_value = min_fdrs[i]
                print >>fout, res.to_json()
   
def compute_qvalues_parallel(num_sample_sets, hists_file, input_basenames, 
                             output_json_file):
    '''
    '''
    def _sort_result_json(line):
        '''comparison function for batch_sort'''
        res = Result.from_json(line.strip())
        return abs(res.nes)

    def _worker(input_basename):
        '''
        parallel process to compute fdr values for single json result file
        '''
        # unsorted json file
        input_json_file = input_basename + JSON_RAW_SUFFIX
        stats_json_file = input_basename + JSON_STATS_SUFFIX
        with open(stats_json_file, 'w') as fout:
            for json_string in _compute_fdr_worker(hists_file, input_json_file):
                print >>fout, json_string
        # remove raw input file
        os.remove(input_json_file)
        # make tmp dir for sorting
        if os.path.exists(input_basename):
            shutil.rmtree(input_basename)
        os.makedirs(input_basename) 
        # sort by nes
        sorted_json_file = input_basename + JSON_SORTED_SUFFIX
        batch_sort(input=stats_json_file,
                   output=sorted_json_file,
                   key=_sort_result_json,
                   buffer_size=SORT_BUFFER_SIZE,
                   tempdirs=[input_basename])
        # remove tmp dir
        shutil.rmtree(input_basename)
        # remove stats json file
        os.remove(stats_json_file)
    # start worker processes
    procs = []
    sorted_json_files = []
    for i in xrange(len(input_basenames)):
        p = Process(target=_worker, args=(input_basenames[i],))
        p.start()
        procs.append(p)
        sorted_json_files.append(input_basenames[i] + JSON_SORTED_SUFFIX)
    # wait for consumers to finish
    for p in procs:
        p.join()
    # perform merge of sorted json files
    merged_json_file = output_json_file + '.raw'
    batch_merge(sorted_json_files, merged_json_file, key=_sort_result_json)
    # remove individual json files
    for f in sorted_json_files:
        os.remove(f)
    # perform final iteration to compute q values
    _compute_qvalues(num_sample_sets, merged_json_file, output_json_file)
    os.remove(merged_json_file)

def ssea_main(config, sample_sets, row_metadata, col_metadata):
    '''
    config: Config object
    sample_sets: list of SampleSet objects
    row_metadata: list of Metadata objects corresponding to rows
    col_metadata: list of Metadata objects corresponding to columns
    '''
    # setup output directory
    if not os.path.exists(config.output_dir):
        logging.debug("Creating output directory '%s'" % 
                      (config.output_dir))
        os.makedirs(config.output_dir)
    # create temp directory
    tmp_dir = os.path.join(config.output_dir, TMP_DIR)
    if not os.path.exists(tmp_dir):
        logging.debug("Creating tmp directory '%s'" % (tmp_dir))
        os.makedirs(tmp_dir)
    # write row metadata output
    metadata_file = os.path.join(config.output_dir, 
                                 Config.METADATA_JSON_FILE)
    logging.debug("Writing row metadata '%s'" % (metadata_file))
    with open(metadata_file, 'w') as fp:
        for m in row_metadata:
            print >>fp, m.to_json()
    # write column metadata output
    samples_file = os.path.join(config.output_dir, 
                                Config.SAMPLES_JSON_FILE) 
    logging.debug("Writing column metadata '%s'" % (samples_file))
    with open(samples_file, 'w') as fp:
        for m in col_metadata:
            print >>fp, m.to_json()
    # write sample sets
    sample_sets_file = os.path.join(config.output_dir, 
                                    Config.SAMPLE_SETS_JSON_FILE)
    with open(sample_sets_file, 'w') as fp:
        for ss in sample_sets:
            print >>fp, ss.to_json()
    # write configuration
    config_file = os.path.join(config.output_dir, 
                               Config.CONFIG_JSON_FILE)
    logging.debug("Writing configuration '%s'" % (config_file))
    with open(config_file, 'w') as fp:
        print >>fp, config.to_json()
    # output files
    es_hists_file = os.path.join(config.output_dir, Config.OUTPUT_HISTS_FILE)
    shape = (len(row_metadata),len(col_metadata))
    logging.info("Running SSEA in parallel with %d processes" % 
                 (config.num_processes))
    worker_basenames = ssea_parallel(config.matrix_dir, shape, 
                                     sample_sets, config,
                                     es_hists_file, tmp_dir)
    # use ES null distributions to compute global statistics
    # and produce a report
    logging.info("Computing FDR q values")
    json_file = os.path.join(config.output_dir, Config.RESULTS_JSON_FILE)
    compute_qvalues_parallel(len(sample_sets), es_hists_file, worker_basenames, 
                             json_file) 
    # cleanup
    if os.path.exists(tmp_dir):
        shutil.rmtree(tmp_dir)
    logging.info("Finished")
