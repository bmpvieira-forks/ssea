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

NUM_NES_BINS = 10000
NES_BINS_POS = np.logspace(-1,2,num=NUM_NES_BINS+1,base=10)
NES_BINS_NEG = -1.0 * np.logspace(-1,2,num=NUM_NES_BINS+1,base=10)[::-1]
NES_BIN_CENTERS_NEG = (NES_BINS_NEG[:-1] + NES_BINS_NEG[1:]) / 2.
NES_BIN_CENTERS_POS = (NES_BINS_POS[:-1] + NES_BINS_POS[1:]) / 2.  
NES_NEG_MIN = NES_BINS_NEG[0]
NES_NEG_MAX = NES_BINS_NEG[-1]
NES_POS_MIN = NES_BINS_POS[0]
NES_POS_MAX = NES_BINS_POS[-1]

def init_hists(nsets):
    return {'null_es_pos': np.zeros((nsets,NUM_BINS), dtype=np.float),
            'null_es_neg': np.zeros((nsets,NUM_BINS), dtype=np.float),
            'obs_es_pos': np.zeros((nsets,NUM_BINS), dtype=np.float),
            'obs_es_neg': np.zeros((nsets,NUM_BINS), dtype=np.float),
            'null_nes_pos': np.zeros(NUM_NES_BINS, dtype=np.float),
            'null_nes_neg': np.zeros(NUM_NES_BINS, dtype=np.float),
            'obs_nes_pos': np.zeros(NUM_NES_BINS, dtype=np.float),
            'obs_nes_neg': np.zeros(NUM_NES_BINS, dtype=np.float)}

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
    null_es_means = np.empty(membership.shape[1], dtype=np.float)
    nes_vals = np.empty(membership.shape[1], dtype=np.float)
    null_nes_vals = np.empty(shape, dtype=np.float)
    pvals = np.empty(membership.shape[1], dtype=np.float)
    fdr_q_values = np.empty(membership.shape[1], dtype=np.float)
    fwer_p_values = np.empty(membership.shape[1], dtype=np.float)
    # separate the positive and negative sides of the null distribution
    # based on the observed enrichment scores
    es_neg_inds = (es_vals < 0).nonzero()[0]
    if len(es_neg_inds) > 0:
        # mask positive scores 
        null_es_neg = np.ma.masked_greater_equal(null_es_vals[:,es_neg_inds], 0)
        # Adjust for variation in gene set size. Normalize ES(S,null)
        # and the observed ES(S), separately rescaling the positive and
        # negative scores by dividing by the mean of the ES(S,null) to
        # yield normalized scores NES(S,null)
        null_es_neg_means = null_es_neg.mean(axis=0)
        null_es_means[es_neg_inds] = null_es_neg_means
        null_nes_neg = null_es_neg / np.fabs(null_es_neg_means)
        null_nes_vals[:,es_neg_inds] = null_nes_neg
        null_nes_neg_count = null_nes_neg.count()
        # To compute FWER create a histogram of the maximum NES(S,null) 
        # over all S for each of the permutations by using the positive 
        # or negative values corresponding to the sign of the observed NES(S). 
        null_nes_min = null_nes_neg.min(axis=1).compressed()
        # Normalize the observed ES(S) by rescaling by the mean of
        # the ES(S,null) separately for positive and negative ES(S)
        nes_obs_neg = (np.ma.MaskedArray(es_vals[es_neg_inds]) / 
                       np.fabs(null_es_neg_means))
        nes_obs_neg_count = nes_obs_neg.count()
        nes_vals[es_neg_inds] = nes_obs_neg
        # estimate nominal p value for S from ES(S,null) by using the
        # positive or negative portion of the distribution corresponding
        # to the sign of the observed ES(S)
        pneg = 1.0 + (null_es_neg <= es_vals[es_neg_inds]).sum(axis=0)
        pneg = pneg / (1.0 + null_es_neg.count(axis=0).astype(float))
        pvals[es_neg_inds] = pneg
    # do the same for the positive enrichment scores (see above for
    # detailed comments
    es_pos_inds = (es_vals >= 0).nonzero()[0]
    if len(es_pos_inds) > 0:
        # mask negative scores 
        null_es_pos = np.ma.masked_less_equal(null_es_vals[:,es_pos_inds], 0)
        # normalize
        null_es_pos_means = null_es_pos.mean(axis=0)
        null_es_means[es_pos_inds] = null_es_pos_means        
        null_nes_pos = null_es_pos / np.fabs(null_es_pos_means)
        null_nes_vals[:,es_pos_inds] = null_nes_pos
        null_nes_pos_count = null_nes_pos.count()
        # store max NES for FWER calculation
        null_nes_max = null_nes_pos.max(axis=1).compressed()
        nes_obs_pos = (np.ma.MaskedArray(es_vals[es_pos_inds]) / 
                       np.fabs(null_es_pos_means))
        nes_obs_pos_count = nes_obs_pos.count()
        nes_vals[es_pos_inds] = nes_obs_pos
        # estimate p values
        ppos = 1.0 + (null_es_pos >= es_vals[es_pos_inds]).sum(axis=0)
        ppos = ppos / (1.0 + null_es_pos.count(axis=0).astype(np.float))
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
            n = (1+(null_nes_neg <= nes).sum()) / (1+float(null_nes_neg_count))
            d = (nes_obs_neg <= nes).sum() / float(nes_obs_neg_count)
            fwerp = (1+(null_nes_min <= nes).sum()) / (1+float(len(null_nes_min)))
        else:
            n = (1+(null_nes_pos >= nes).sum()) / (1+float(null_nes_pos_count))
            d = (nes_obs_pos >= nes).sum() / float(nes_obs_pos_count)
            fwerp = (1+(null_nes_max >= nes).sum()) / (1+float(len(null_nes_max)))
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
        # return null distributions for global fdr calculation
        yield j, res, null_es_vals[:,j], null_nes_vals[:,j]

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
    hists = init_hists(len(sample_sets))
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
        for j,res,null_es,null_nes in ssea_run(counts, size_factors, 
                                               membership, rng, config):
            # save row and column id
            res.t_id = i
            res.ss_id = j
            # convert to json
            print >>outfileh, res.to_json()
            # update ES histograms
            if res.es <= 0:
                hists['null_es_neg'][res.ss_id] += np.histogram(null_es, BINS_NEG)[0]
                hists['obs_es_neg'][res.ss_id] += np.histogram(res.es, BINS_NEG)[0]                
                nes = np.clip(res.t_nes, NES_NEG_MIN, NES_NEG_MAX)
                hists['null_nes_neg'] += np.histogram(null_nes.clip(NES_NEG_MIN,NES_NEG_MAX), NES_BINS_NEG)[0]
                hists['obs_nes_neg'] += np.histogram(nes, NES_BINS_NEG)[0]                
            if res.es >= 0:
                hists['null_es_pos'][res.ss_id] += np.histogram(null_es, BINS_POS)[0]        
                hists['obs_es_pos'][res.ss_id] += np.histogram(res.es, BINS_POS)[0]
                nes = np.clip(res.t_nes, NES_POS_MIN, NES_POS_MAX)                
                hists['null_nes_pos'] += np.histogram(null_nes.clip(NES_POS_MIN,NES_POS_MAX), NES_BINS_POS)[0]
                hists['obs_nes_pos'] += np.histogram(nes, NES_BINS_POS)[0]
    # close report file
    outfileh.close()
    # save histograms to a file
    np.savez(output_hist_file, **hists)
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
    hists = init_hists(len(sample_sets))
    for i in xrange(len(procs)):        
        # merge json files
        with open(worker_json_files[i], 'rb') as fin:
            shutil.copyfileobj(fin, fout)      
        # aggregate numpy arrays
        npzfile = np.load(worker_hist_files[i])
        for k in hists.iterkeys():
            hists[k] += npzfile[k]
        npzfile.close()
    fout.close() 
    np.savez(output_hist_file, **hists)

def compute_global_stats(hists_file, input_json_file, output_json_file):
    # read es histograms
    npzfile = np.load(hists_file)
    hists_null_es_pos = npzfile['null_es_pos']
    hists_null_es_neg = npzfile['null_es_neg']
    hists_obs_es_pos = npzfile['obs_es_pos']
    hists_obs_es_neg = npzfile['obs_es_neg']
    hists_null_nes_pos = npzfile['null_nes_pos']
    hists_null_nes_neg = npzfile['null_nes_neg']
    hists_obs_nes_pos = npzfile['obs_nes_pos']
    hists_obs_nes_neg = npzfile['obs_nes_neg']
    npzfile.close()
    # compute per-sample-set means
    null_es_pos_counts = hists_null_es_pos.sum(axis=1)
    null_es_neg_counts = hists_null_es_neg.sum(axis=1)
    obs_es_pos_counts = hists_obs_es_pos.sum(axis=1)
    obs_es_neg_counts = hists_obs_es_neg.sum(axis=1)
    null_es_neg_means = np.fabs((hists_null_es_neg * BIN_CENTERS_NEG).sum(axis=1) / 
                                null_es_neg_counts.clip(1.0))
    null_es_pos_means = np.fabs((hists_null_es_pos * BIN_CENTERS_POS).sum(axis=1) / 
                                null_es_pos_counts.clip(1.0))
    # compute global means    
    null_nes_pos_counts = hists_null_nes_pos.sum()
    null_nes_neg_counts = hists_null_nes_neg.sum()
    obs_nes_pos_counts = hists_obs_nes_pos.sum()
    obs_nes_neg_counts = hists_obs_nes_neg.sum()
    null_nes_neg_mean = np.fabs((hists_null_nes_neg * NES_BIN_CENTERS_NEG).sum() / 
                                null_nes_neg_counts)
    null_nes_pos_mean = np.fabs((hists_null_nes_pos * NES_BIN_CENTERS_POS).sum() / 
                                null_nes_pos_counts)
    # parse report json and write new values
    fin = gzip.open(input_json_file, 'rb')
    fout = gzip.open(output_json_file, 'wb')
    for line in fin:
        # load json document (one per line)
        res = Result.from_json(line.strip())
        # get relevant columns
        i = res.ss_id
        es = res.es
        nes = res.t_nes
        # compute sample set and global NES and FDR q-values
        if es < 0:
            if null_es_neg_means[i] == 0:
                ss_nes = 0.0
            else:
                ss_nes = es / null_es_neg_means[i]
            es_bin = np.digitize((es,), BINS_NEG)
            ss_n = (1+hists_null_es_neg[i,:es_bin].sum()) / (1+float(null_es_neg_counts[i]))
            ss_d = hists_obs_es_neg[i,:es_bin].sum() / float(obs_es_neg_counts[i])
            # NES uses different bins
            if null_nes_neg_mean == 0:
                global_nes = 0.0
            else:
                global_nes = nes / null_nes_neg_mean
            nes_bin = np.digitize((nes,), NES_BINS_NEG)
            global_n = (1+hists_null_nes_neg[:nes_bin].sum()) / (1+float(null_nes_neg_counts))
            global_d = hists_obs_nes_neg[:nes_bin].sum() / float(obs_nes_neg_counts)
        else:
            if null_es_pos_means[i] == 0:
                ss_nes = 0.0
            else:
                ss_nes = es / null_es_pos_means[i]            
            es_bin = np.digitize((es,), BINS_POS) - 1
            ss_n = (1+hists_null_es_pos[i,es_bin:].sum()) / (1+float(null_es_pos_counts[i]))
            ss_d = hists_obs_es_pos[i,es_bin:].sum() / float(obs_es_pos_counts[i])
            # NES uses different bins
            if null_nes_pos_mean == 0:
                global_nes = 0.0
            else:
                global_nes = nes / null_nes_pos_mean
            nes_bin = np.digitize((nes,), NES_BINS_POS) - 1
            global_n = (1+hists_null_nes_pos[nes_bin:].sum()) / (1+float(null_nes_pos_counts))
            global_d = hists_obs_nes_pos[nes_bin:].sum() / float(obs_nes_pos_counts)
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
        logging.debug("Creating tmp directory '%s'" % (tmp_dir))
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
