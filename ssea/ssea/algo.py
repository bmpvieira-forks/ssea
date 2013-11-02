'''
Created on Oct 9, 2013

@author: mkiyer
'''
import os
import logging
import shutil
import json
import gzip
import itertools
from multiprocessing import Process, JoinableQueue
import numpy as np

# local imports
from kernel import ssea_kernel
from base import BOOL_DTYPE, Config, Result, Metadata

# enrichment score histogram bins
NUM_BINS = 10000
BINS_NEG = np.linspace(-1.0, 0.0, num=NUM_BINS+1)
BINS_POS = np.linspace(0.0, 1.0, num=NUM_BINS+1)
BIN_CENTERS_NEG = (BINS_NEG[:-1] + BINS_NEG[1:]) / 2.
BIN_CENTERS_POS = (BINS_POS[:-1] + BINS_POS[1:]) / 2.  

# results directories
TMP_DIR = 'tmp'
TMP_JSON_FILE = 'tmp.json.gz'



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

def ssea_run(sample_ids, weights, sample_sets, 
             weight_method_miss='unweighted', 
             weight_method_hit='unweighted',
             weight_const=0.0, 
             weight_noise=0.0,
             perms=10000):
    '''
    
    sample_ids: list of unique integer ids
    weights: list of float values
    sample_sets: list of SampleSet objects
    '''
    weights = np.array(weights, dtype=np.float)
    # noise does not preserve rank so need to add first
    tweights = weights.copy()
    if weight_noise > 0.0:
        tweights += weight_noise * np.random.random(len(weights))
    # rank order the N samples in D to form L={s1...sn} 
    ranks = np.argsort(tweights)[::-1]
    sample_ids = [sample_ids[i] for i in ranks]
    weights = weights[ranks]
    tweights = tweights[ranks]
    # perform power transform and adjust by constant
    tweights += weight_const
    tweights_miss = np.fabs(transform_weights(tweights, weight_method_miss)) 
    tweights_hit = np.fabs(transform_weights(tweights, weight_method_hit))    
    # convert sample sets to membership vectors
    membership = np.zeros((len(sample_ids),len(sample_sets)), 
                          dtype=BOOL_DTYPE)
    for j,sample_set in enumerate(sample_sets):
        membership[:,j] = sample_set.get_array(sample_ids)
    # determine enrichment score (ES)
    perm = np.arange(len(sample_ids))
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
        es_null_neg = np.ma.masked_greater(es_null[:,es_neg_inds], 0)
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
        es_null_pos = np.ma.masked_less(es_null[:,es_pos_inds], 0)
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
        # get indexes of hits in this set
        m = membership[:,j]
        hit_inds = (m > 0).nonzero()[0]       
        num_hits = hit_inds.shape[0]
        num_misses = m.shape[0] - num_hits
        # calculate leading edge stats
        if es_vals[j] < 0:
            core_hits = sum(i >= es_run_inds[j] for i in hit_inds)
            core_misses = membership.shape[0] - es_run_inds[j]
        else:
            core_hits = sum(i <= es_run_inds[j] for i in hit_inds)
            core_misses = 1 + es_run_inds[j] - core_hits
        null_hits = num_hits - core_hits
        null_misses = num_misses - core_misses
        # TODO: fishers exact test? odds ratio?
        # histogram of ES null distribution
        h = np.histogram(es_null[:,j], bins=Config.ES_NULL_BINS)[0]        
        percent_neg = (100. * (es_null[:,j] < 0).sum() / es_null.shape[0])
        # create dictionary result
        res = Result()
        res.ss_id = sample_sets[j]._id    
        res.es = es_vals[j]
        res.nes = nes_vals[j]
        res.nominal_p_value = pvals[j]
        res.fdr_q_value = fdr_q_values[j]
        res.fwer_p_value = fwer_p_values[j]
        res.rank_at_max = int(es_run_inds[j])
        res.core_hits = int(core_hits)
        res.core_misses = int(core_misses)
        res.null_hits = int(null_hits)
        res.null_misses = int(null_misses)
        res.es_null_mean = es_null_means[j]
        res.es_null_percent_neg = percent_neg
        res.es_null_hist = h
        yield res, es_null[:,j]

def ssea_serial(weight_matrix, sample_sets, config, output_json_file, 
                output_hist_file, startrow=None, endrow=None):
    '''
    main SSEA loop (single processor)
    
    weight_matrix: matrix containing numeric data 
    sample_sets: list of SampleSet objects
    config: Config object
    output_json_file: filename for writing results (JSON format)
    output_hist_file: filename for writing ES histograms
    '''
    # determine range of matrix to process
    if startrow is None:
        startrow = 0
    if endrow is None:
        endrow = weight_matrix.shape[0]
    assert startrow < endrow
    # setup global ES histograms
    es_hists = {'null_pos': np.zeros((len(sample_sets),NUM_BINS), dtype=np.float),
                'null_neg': np.zeros((len(sample_sets),NUM_BINS), dtype=np.float),
                'obs_pos': np.zeros((len(sample_sets),NUM_BINS), dtype=np.float),
                'obs_neg': np.zeros((len(sample_sets),NUM_BINS), dtype=np.float)}
    # setup report file
    outfileh = gzip.open(output_json_file, 'wb')    
    for i in xrange(startrow, endrow):
        logging.debug("\tRow: %d/%d" % (i+1,endrow-startrow))
        # read from memmap
        weights = np.array(weight_matrix[i,:], dtype=np.float)
        # remove 'nan' values        
        sample_ids = np.isfinite(weights).nonzero()[0]
        weights = weights[sample_ids]
        # run ssea
        for res,es_null in ssea_run(sample_ids, weights, sample_sets,
                                    weight_method_miss=config.weight_miss,
                                    weight_method_hit=config.weight_hit,
                                    weight_const=config.weight_const,
                                    weight_noise=config.weight_noise,
                                    perms=config.perms):
            # save row id
            res.t_id = i
            # convert to json
            print >>outfileh, res.to_json()
            # update ES histograms
            if res.es <= 0:
                es_hists['null_neg'][i] += np.histogram(es_null, BINS_NEG)[0]
                es_hists['obs_neg'][i] += np.histogram(res.es, BINS_NEG)[0]
            if res.es >= 0:
                es_hists['null_pos'][i] += np.histogram(es_null, BINS_POS)[0]        
                es_hists['obs_pos'][i] += np.histogram(res.es, BINS_POS)[0]
    # close report file
    outfileh.close()
    # save histograms to a file
    np.savez(output_hist_file, **es_hists)

def ssea_parallel(weight_vecs, sample_sets, config, output_json_file, 
                  output_hist_file): 
    '''
    main SSEA loop (multiprocessing implementation)
    
    See ssea_serial function for documentation
    '''
    def worker(input_queue, sample_sets, config, json_file, hist_file): 
        def queue_iter(q):
            while True:
                obj = q.get()
                if (obj is None):
                    break
                yield obj
                input_queue.task_done()
            input_queue.task_done()
        # initialize output file
        ssea_serial(queue_iter(input_queue), sample_sets, config, 
                    json_file, hist_file)
    # create temp directory
    tmp_dir = os.path.join(config.output_dir, "tmp")
    if not os.path.exists(tmp_dir):
        logging.debug("\tCreating tmp directory '%s'" % (tmp_dir))
        os.makedirs(tmp_dir)
    # create multiprocessing queue for passing data
    input_queue = JoinableQueue(maxsize=config.num_processors*3)
    # start worker processes
    procs = []
    worker_json_files = []
    worker_hist_files = []
    try:
        for i in xrange(config.num_processors):
            json_file = os.path.join(tmp_dir, "w%03d.json" % (i))
            worker_json_files.append(json_file)
            hist_file = os.path.join(tmp_dir, "w%03d_hists.npz" % (i))
            worker_hist_files.append(hist_file)
            args = (input_queue, sample_sets, config, json_file, hist_file) 
            p = Process(target=worker, args=args)
            p.start()
            procs.append(p)
        # parse weight vectors
        for weight_vec in weight_vecs:
            input_queue.put(weight_vec)
    finally:
        # stop workers
        for p in procs:
            input_queue.put(None)
        # close queue
        input_queue.close()
        input_queue.join()
        # join worker processes
        for p in procs:
            p.join()
    # merge workers
    logging.info("Merging %d worker results" % (config.num_processors))
    fout = open(output_json_file, 'wb')
    # setup global ES histograms
    es_hists = {'null_pos': np.zeros((len(sample_sets),NUM_BINS), dtype=np.float),
                'null_neg': np.zeros((len(sample_sets),NUM_BINS), dtype=np.float),
                'obs_pos': np.zeros((len(sample_sets),NUM_BINS), dtype=np.float),
                'obs_neg': np.zeros((len(sample_sets),NUM_BINS), dtype=np.float)}
    for i in xrange(config.num_processors):        
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
    # read es arrays
    npzfile = np.load(es_hists_file)
    hists_null_pos = npzfile['null_pos']
    hists_null_neg = npzfile['null_neg']
    hists_obs_pos = npzfile['obs_pos']
    hists_obs_neg = npzfile['obs_neg']
    npzfile.close()
    # compute totals
    hists_null_pos_counts = hists_null_pos.sum(axis=1)
    hists_null_neg_counts = hists_null_neg.sum(axis=1)
    hists_obs_pos_counts = hists_obs_pos.sum(axis=1)
    hists_obs_neg_counts = hists_obs_neg.sum(axis=1)
    # compute means
    hists_null_neg_means = np.fabs((hists_null_neg * BIN_CENTERS_NEG).sum(axis=1) / 
                                   hists_null_neg_counts)
    hists_null_pos_means = np.fabs((hists_null_pos * BIN_CENTERS_POS).sum(axis=1) / 
                                   hists_null_pos_counts)
    # parse report json and write new values
    fin = gzip.open(input_json_file, 'rb')
    fout = gzip.open(output_json_file, 'wb')
    for line in fin:
        # load json document (one per line)
        res = Result.from_json(line.strip())
        # get relevant columns
        i = res.ss_id
        es = res.es
        # compute global nes and fdr        
        if es < 0:
            global_nes = es / hists_null_neg_means[i]            
            es_bin = np.digitize((es,), BINS_NEG)
            n = (1+hists_null_neg[i,:es_bin].sum()) / (1+float(hists_null_neg_counts[i]))
            d = hists_obs_neg[i,:es_bin].sum() / float(hists_obs_neg_counts[i])
        else:
            global_nes = es / hists_null_pos_means[i]            
            es_bin = np.digitize((es,), BINS_POS) - 1
            n = (1+hists_null_pos[i,es_bin:].sum()) / (1+float(hists_null_pos_counts[i]))
            d = hists_obs_pos[i,es_bin:].sum() / float(hists_obs_pos_counts[i])
        global_fdr_qval = n / d
        # update json dict
        res.global_fdr_q_value = global_fdr_qval
        res.global_nes = global_nes
        # convert back to json
        print >>fout, res.to_json()
    fin.close()
    fout.close()


def ssea_main(weight_matrix, row_metadata, col_metadata, sample_sets, 
              config):
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
    if config.num_processes > 1:
        logging.info("Running SSEA in parallel with %d processes" % 
                     (config.num_processors))
        # TODO: fix parallel
        ssea_parallel(weight_matrix, sample_sets, config, tmp_json_file, 
                      es_hists_file)
    else:
        logging.info("Running SSEA in serial")
        ssea_serial(weight_matrix, sample_sets, config, tmp_json_file, 
                    es_hists_file)
    # use ES null distributions to compute global statistics
    # and produce a report
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
