'''
Created on Oct 9, 2013

@author: mkiyer
'''
import os
import logging
import shutil
import gzip
from collections import namedtuple
from multiprocessing import Process
import numpy as np

# local imports
import ssea.cfisher as fisher
from ssea.kernel import ssea_kernel2, RandomState
from base import BOOL_DTYPE, Config, Result, chunk, quantile_sorted
from countdata import BigCountMatrix

# improves readability of code
KernelResult = namedtuple('KernelResult', ('ranks',
                                           'norm_counts', 
                                           'norm_counts_miss',
                                           'norm_counts_hit',
                                           'es_vals', 
                                           'es_ranks', 
                                           'es_runs'))

# enrichment score histogram bins
NUM_BINS = 10000
BINS_NEG = np.linspace(-1.0, 0.0, num=NUM_BINS+1)
BINS_POS = np.linspace(0.0, 1.0, num=NUM_BINS+1)
BIN_CENTERS_NEG = (BINS_NEG[:-1] + BINS_NEG[1:]) / 2.
BIN_CENTERS_POS = (BINS_POS[:-1] + BINS_POS[1:]) / 2.  
PRECISION = 4

# results directories
TMP_DIR = 'tmp'
TMP_JSON_FILE = 'tmp.json.gz'

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
    k = ssea_kernel2(counts, size_factors, membership, rng,
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
    for i in xrange(config.resampling_iterations):
        k = ssea_kernel2(counts, size_factors, membership, rng,
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
        k = ssea_kernel2(counts, size_factors, membership, rng,
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
    pvals = np.empty(membership.shape[1], dtype=np.float)
    fdr_q_values = np.empty(membership.shape[1], dtype=np.float)
    fwer_p_values = np.empty(membership.shape[1], dtype=np.float)
    es_null_means = np.empty(membership.shape[1], dtype=np.float)
    # separate the positive and negative sides of the null distribution
    # based on the observed enrichment scores
    es_neg_inds = (es_vals < 0).nonzero()[0]
    if len(es_neg_inds) > 0:
        # mask positive scores 
        es_null_neg = np.ma.masked_greater(null_es_vals[:,es_neg_inds], 0)
        # Adjust for variation in gene set size. Normalize ES(S,null)
        # and the observed ES(S), separately rescaling the positive and
        # negative scores by dividing by the mean of the ES(S,null) to
        # yield normalized scores NES(S,null)
        es_null_neg_means = es_null_neg.mean(axis=0)
        es_null_means[es_neg_inds] = es_null_neg_means
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
        pvals[es_neg_inds] = pneg
    # do the same for the positive enrichment scores (see above for
    # detailed comments
    es_pos_inds = (es_vals >= 0).nonzero()[0]
    if len(es_pos_inds) > 0:
        # mask negative scores 
        es_null_pos = np.ma.masked_less(null_es_vals[:,es_pos_inds], 0)
        # normalize
        es_null_pos_means = es_null_pos.mean(axis=0)
        es_null_means[es_pos_inds] = es_null_pos_means        
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
        pvals[es_pos_inds] = ppos
    # Control for multiple hypothesis testing and summarize results
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
            n = (1+(nes_null_neg <= nes).sum()) / (1+float(nes_null_neg_count))
            d = (nes_obs_neg <= nes).sum() / float(nes_obs_neg_count)
            fwerp = (1+(nes_null_min <= nes).sum()) / (1+float(len(nes_null_min)))
        else:
            n = (1+(nes_null_pos >= nes).sum()) / (1+float(nes_null_pos_count))
            d = (nes_obs_pos >= nes).sum() / float(nes_obs_pos_count)
            fwerp = (1+(nes_null_max >= nes).sum()) / (1+float(len(nes_null_max)))
        fdr_q_values[j] = n / d
        fwer_p_values[j] = fwerp
    # setup result objects    
    for j in xrange(membership.shape[1]):
        # create result object
        res = Result()
        res.rand_seed = rand_seed
        res.es = round(es_vals[j], PRECISION)
        res.es_rank = int(es_ranks[j])
        res.nominal_p_value = round(pvals[j],PRECISION)
        res.t_nes = round(nes_vals[j],PRECISION)
        res.t_fdr_q_value = round(fdr_q_values[j],PRECISION)
        res.t_fwer_p_value = round(fwer_p_values[j],PRECISION)
        # save some of the resampled es points 
        res.resample_es_vals = np.around(resample_es_vals[:Config.MAX_ES_POINTS,j],PRECISION)
        res.resample_es_ranks = resample_es_ranks[:Config.MAX_ES_POINTS,j]
        res.null_es_vals = np.around(null_es_vals[:Config.MAX_ES_POINTS,j], PRECISION)
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
                odds_ratio = np.inf if d == 0 else n/d
        elif np.isfinite(d):
            odds_ratio = np.inf
        else:
            odds_ratio = np.nan if n == 0 else 0.0
        # create dictionary result
        res.core_hits = int(core_hits)
        res.core_misses = int(core_misses)
        res.null_hits = int(null_hits)
        res.null_misses = int(null_misses)
        res.fisher_p_value = fisher_p_value
        res.odds_ratio = odds_ratio
        yield j, res, null_es_vals[:,j]

def ssea_serial(matrix_dir, shape, sample_sets, config, 
                output_json_file, output_hist_file, 
                startrow=None, endrow=None):
    '''
    main SSEA loop (single processor)
    
    weight_matrix_file: numpy memmap matrix containing numeric data 
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
    es_hists = {'null_pos': np.zeros((len(sample_sets),NUM_BINS), dtype=np.float),
                'null_neg': np.zeros((len(sample_sets),NUM_BINS), dtype=np.float),
                'obs_pos': np.zeros((len(sample_sets),NUM_BINS), dtype=np.float),
                'obs_neg': np.zeros((len(sample_sets),NUM_BINS), dtype=np.float)}
    # setup report file
    outfileh = gzip.open(output_json_file, 'wb')    
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
        for j,res,es_null in ssea_run(counts, size_factors, membership, 
                                      rng, config):
            # save row and column id
            res.t_id = i
            res.ss_id = j
            # convert to json
            print >>outfileh, res.to_json()
            # update ES histograms
            if res.es <= 0:
                es_hists['null_neg'][res.ss_id] += np.histogram(es_null, BINS_NEG)[0]
                es_hists['obs_neg'][res.ss_id] += np.histogram(res.es, BINS_NEG)[0]
            if res.es >= 0:
                es_hists['null_pos'][res.ss_id] += np.histogram(es_null, BINS_POS)[0]        
                es_hists['obs_pos'][res.ss_id] += np.histogram(res.es, BINS_POS)[0]
    # close report file
    outfileh.close()
    # save histograms to a file
    np.savez(output_hist_file, **es_hists)
    # cleanup
    bm.close()

def ssea_parallel(matrix_dir, shape, sample_sets, config, 
                  output_json_file, output_hist_file): 
    '''
    main SSEA loop (multiprocessing implementation)
    
    See ssea_serial function for documentation
    '''
    # start worker processes
    tmp_dir = os.path.join(config.output_dir, TMP_DIR)
    procs = []
    chunks = []
    worker_json_files = []
    worker_hist_files = []
    # divide matrix rows across processes
    for startrow,endrow in chunk(shape[0], config.num_processes):
        i = len(procs)
        logging.debug("Worker process %d range %d-%d (%d total rows)" % 
                      (i, startrow, endrow, (endrow-startrow)))
        # worker output files
        json_file = os.path.join(tmp_dir, "w%03d.json" % (i))
        hist_file = os.path.join(tmp_dir, "w%03d_hists.npz" % (i))
        worker_json_files.append(json_file)
        worker_hist_files.append(hist_file)
        args = (matrix_dir, shape, sample_sets, config, 
                json_file, hist_file, startrow, endrow)        
        p = Process(target=ssea_serial, args=args)
        p.start()
        procs.append(p)
        chunks.append((startrow,endrow))
    # join worker processes (wait for processes to finish)
    for p in procs:
        p.join()
    # merge workers
    logging.info("Merging %d worker results" % (len(procs)))
    fout = open(output_json_file, 'wb')
    # setup global ES histograms
    es_hists = {'null_pos': np.zeros((len(sample_sets),NUM_BINS), dtype=np.float),
                'null_neg': np.zeros((len(sample_sets),NUM_BINS), dtype=np.float),
                'obs_pos': np.zeros((len(sample_sets),NUM_BINS), dtype=np.float),
                'obs_neg': np.zeros((len(sample_sets),NUM_BINS), dtype=np.float)}
    for i in xrange(len(procs)):        
        # merge json files
        with open(worker_json_files[i], 'rb') as fin:
            shutil.copyfileobj(fin, fout)      
        # aggregate numpy arrays
        npzfile = np.load(worker_hist_files[i])
        es_hists['null_pos'] += npzfile['null_pos']
        es_hists['null_neg'] += npzfile['null_neg']
        es_hists['obs_pos'] += npzfile['obs_pos']
        es_hists['obs_neg'] += npzfile['obs_neg']
        npzfile.close()
    fout.close() 
    np.savez(output_hist_file, **es_hists)

def compute_global_stats(es_hists_file, input_json_file, output_json_file):
    # read es histograms
    npzfile = np.load(es_hists_file)
    hists_null_pos = npzfile['null_pos']
    hists_null_neg = npzfile['null_neg']
    hists_obs_pos = npzfile['obs_pos']
    hists_obs_neg = npzfile['obs_neg']
    npzfile.close()
    # compute per-sample-set means
    ss_null_pos_counts = hists_null_pos.sum(axis=1)
    ss_null_neg_counts = hists_null_neg.sum(axis=1)
    ss_obs_pos_counts = hists_obs_pos.sum(axis=1)
    ss_obs_neg_counts = hists_obs_neg.sum(axis=1)
    ss_null_neg_means = np.fabs((hists_null_neg * BIN_CENTERS_NEG).sum(axis=1) / 
                                ss_null_neg_counts.clip(1.0))
    ss_null_pos_means = np.fabs((hists_null_pos * BIN_CENTERS_POS).sum(axis=1) / 
                                ss_null_pos_counts.clip(1.0))
    # compute global means
    g_null_pos = hists_null_pos.sum(axis=0)
    g_null_neg = hists_null_neg.sum(axis=0)
    g_obs_pos = hists_obs_pos.sum(axis=0)
    g_obs_neg = hists_obs_neg.sum(axis=0)
    g_null_neg_mean = np.fabs((g_null_pos * BIN_CENTERS_NEG).sum() / 
                              g_null_neg.sum().clip(1.0))    
    g_null_pos_mean = np.fabs((g_null_pos * BIN_CENTERS_POS).sum() / 
                              g_null_pos.sum().clip(1.0))
    # parse report json and write new values
    fin = gzip.open(input_json_file, 'rb')
    fout = gzip.open(output_json_file, 'wb')
    for line in fin:
        # load json document (one per line)
        res = Result.from_json(line.strip())
        # get relevant columns
        i = res.ss_id
        es = res.es
        # compute sample set and global NES and FDR q-values
        if es < 0:
            if ss_null_neg_means[i] == 0:
                ss_nes = 0.0
            else:
                ss_nes = es / ss_null_neg_means[i]
            es_bin = np.digitize((es,), BINS_NEG)
            ss_n = (1+hists_null_neg[i,:es_bin].sum()) / (1+float(ss_null_neg_counts[i]))
            ss_d = hists_obs_neg[i,:es_bin].sum() / float(ss_obs_neg_counts[i])
            if g_null_neg_mean == 0:
                global_nes = 0.0
            else:
                global_nes = es / g_null_neg_mean
            global_n = (1+g_null_neg[:es_bin].sum()) / (1+float(g_null_neg.sum()))
            global_d = g_obs_neg[:es_bin].sum() / float(g_obs_neg.sum())
        else:
            if ss_null_pos_means[i] == 0:
                ss_nes = 0.0
            else:
                ss_nes = es / ss_null_pos_means[i]            
            es_bin = np.digitize((es,), BINS_POS) - 1
            ss_n = (1+hists_null_pos[i,es_bin:].sum()) / (1+float(ss_null_pos_counts[i]))
            ss_d = hists_obs_pos[i,es_bin:].sum() / float(ss_obs_pos_counts[i])
            if g_null_pos_mean == 0:
                global_nes = 0.0
            else:
                global_nes = es / g_null_pos_mean
            global_n = (1+g_null_pos[es_bin:].sum()) / (1+float(g_null_neg.sum()))
            global_d = g_obs_pos[es_bin:].sum() / float(g_obs_neg.sum())        
        ss_fdr_q_value = ss_n / ss_d
        global_fdr_q_value = global_n / global_d
        # update json dict
        res.ss_fdr_q_value = ss_fdr_q_value
        res.ss_nes = ss_nes
        res.global_fdr_q_value = global_fdr_q_value
        res.global_nes = global_nes
        # convert back to json
        print >>fout, res.to_json()
    fin.close()
    fout.close()

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
        logging.debug("\tCreating tmp directory '%s'" % (tmp_dir))
        os.makedirs(tmp_dir)
    # output files
    tmp_json_file = os.path.join(tmp_dir, TMP_JSON_FILE)
    es_hists_file = os.path.join(config.output_dir, Config.OUTPUT_HISTS_FILE)
    shape = (len(row_metadata),len(col_metadata))
    if config.num_processes > 1:
        logging.info("Running SSEA in parallel with %d processes" % 
                     (config.num_processes))
        ssea_parallel(config.matrix_dir, shape, sample_sets, config, 
                      tmp_json_file, es_hists_file)
    else:
        logging.info("Running SSEA in serial")
        ssea_serial(config.matrix_dir, shape, sample_sets, config, 
                    tmp_json_file, es_hists_file)
    # use ES null distributions to compute global statistics
    # and produce a report
    logging.info("Computing global statistics")
    json_file = os.path.join(config.output_dir, Config.RESULTS_JSON_FILE)
    compute_global_stats(es_hists_file, tmp_json_file, json_file)
    # cleanup
    if os.path.exists(tmp_dir):
        shutil.rmtree(tmp_dir)
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
    logging.info("Finished")
