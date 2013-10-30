'''
Created on Oct 9, 2013

@author: mkiyer
'''
import os
import logging
import shutil
import json
from multiprocessing import Process, JoinableQueue

import numpy as np
import matplotlib.pyplot as plt

# local imports
from kernel import ssea_kernel
from base import BOOL_DTYPE, Result
from report import Report, create_detailed_report, create_html_report

# enrichment score histogram bins
NUM_BINS = 10000
BINS_NEG = np.linspace(-1.0, 0.0, num=NUM_BINS+1)
BINS_POS = np.linspace(0.0, 1.0, num=NUM_BINS+1)
BIN_CENTERS_NEG = (BINS_NEG[:-1] + BINS_NEG[1:]) / 2.
BIN_CENTERS_POS = (BINS_POS[:-1] + BINS_POS[1:]) / 2.  

# results directories
DETAILS_DIR = 'details'
TMP_DIR = 'tmp'
OUTPUT_TSV_FILE = 'out.tsv'
OUTPUT_HISTS_FILE = 'es_hists.npz'

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
        ppos = 1.0 + (es_null_pos >= es_vals[es_pos_inds]).sum(axis=0)
        ppos = ppos / (1.0 + es_null_pos.count(axis=0).astype(np.float))
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
            fwerp = (1+(nes_null_min <= nes).sum()) / (1+float(len(nes_null_min)))
        else:
            n = (nes_null_pos >= nes).sum() / float(nes_null_pos_count)
            d = (nes_obs_pos >= nes).sum() / float(nes_obs_pos_count)
            fwerp = (1+(nes_null_max >= nes).sum()) / (1+float(len(nes_null_max)))
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

def ssea_serial(weight_vecs, sample_sets, config, details_dir, 
                output_tsv_file, output_hist_file):
    '''
    main SSEA loop (single processor)
    
    weight_vec: list/iterator of WeightVector objects
    sample_sets: list of SampleSet objects
    config: Config object
    details_dir: path to store detailed report files
    output_tsv_file: filename for writing results
    output_hist_file: filename for writing ES histograms
    '''
    # setup global ES histograms
    es_hists = {'null_pos': np.zeros((len(sample_sets),NUM_BINS), dtype=np.float),
                'null_neg': np.zeros((len(sample_sets),NUM_BINS), dtype=np.float),
                'obs_pos': np.zeros((len(sample_sets),NUM_BINS), dtype=np.float),
                'obs_neg': np.zeros((len(sample_sets),NUM_BINS), dtype=np.float)}
    # setup report file
    outfileh = open(output_tsv_file, 'w')    
    print >>outfileh, '\t'.join(Report.FIELDS)
    # run SSEA on each weight vector
    for weight_vec in weight_vecs:
        logging.info("\tName: %s" % (weight_vec.name))
        results = ssea_run(weight_vec.samples, 
                           weight_vec.weights, 
                           sample_sets, 
                           weight_method_miss=config.weight_miss,
                           weight_method_hit=config.weight_hit,
                           weight_const=config.weight_const,
                           weight_noise=config.weight_noise,
                           perms=config.perms)
        name = weight_vec.name
        desc = ' '.join(weight_vec.metadata)
        for i,res in enumerate(results):
            # get output fields
            fields = res.get_report_fields(name, desc)
            # decide whether to create detailed report
            if res.qval <= config.detailed_report_threshold:
                details_dict = create_detailed_report(name, desc, res, 
                                                      details_dir, config)
                fields[-1] = json.dumps(details_dict)
            # output to text file
            print >>outfileh, '\t'.join(map(str, fields))
            # update ES histograms
            if res.es <= 0:
                es_hists['null_neg'][i] += np.histogram(res.es_null, BINS_NEG)[0]
                es_hists['obs_neg'][i] += np.histogram(res.es, BINS_NEG)[0]
            if res.es >= 0:
                es_hists['null_pos'][i] += np.histogram(res.es_null, BINS_POS)[0]        
                es_hists['obs_pos'][i] += np.histogram(res.es, BINS_POS)[0]
    # close report file
    outfileh.close()
    # save histograms to a file
    np.savez(output_hist_file, **es_hists)

def ssea_parallel(weight_vecs, sample_sets, config, details_dir, 
                  output_tsv_file, output_hist_file): 
    '''
    main SSEA loop (multiprocessing implementation)
    
    See ssea_serial function for documentation
    '''
    def worker(input_queue, sample_sets, config, details_dir, tsv_file, 
               hist_file):
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
                    details_dir, tsv_file, hist_file)
    # create temp directory
    tmp_dir = os.path.join(config.output_dir, "tmp")
    if not os.path.exists(tmp_dir):
        logging.debug("\tCreating tmp directory '%s'" % (tmp_dir))
        os.makedirs(tmp_dir)
    # create multiprocessing queue for passing data
    input_queue = JoinableQueue(maxsize=config.num_processors*3)
    # start worker processes
    procs = []
    worker_tsv_files = []
    worker_hist_files = []
    try:
        for i in xrange(config.num_processors):
            tsv_file = os.path.join(tmp_dir, "w%03d_report.tsv" % (i))
            worker_tsv_files.append(tsv_file)
            hist_file = os.path.join(tmp_dir, "w%03d_hists.npz" % (i))
            worker_hist_files.append(hist_file)
            args = (input_queue, sample_sets, config, details_dir, 
                    tsv_file, hist_file) 
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
    outfileh = open(output_tsv_file, 'w')
    print >>outfileh, '\t'.join(Report.FIELDS)
    # setup global ES histograms
    es_hists = {'null_pos': np.zeros((len(sample_sets),NUM_BINS), dtype=np.float),
                'null_neg': np.zeros((len(sample_sets),NUM_BINS), dtype=np.float),
                'obs_pos': np.zeros((len(sample_sets),NUM_BINS), dtype=np.float),
                'obs_neg': np.zeros((len(sample_sets),NUM_BINS), dtype=np.float)}
    for i in xrange(config.num_processors):
        # merge report tsv
        tsv_file = worker_tsv_files[i]
        with open(tsv_file, 'r') as fin:
            fin.next() # skip header
            for line in fin:
                print >>outfileh, line.rstrip()
        # aggregate numpy arrays
        npzfile = np.load(worker_hist_files[i])
        es_hists['null_pos'] += npzfile['null_pos']
        es_hists['null_neg'] += npzfile['null_neg']
        es_hists['obs_pos'] += npzfile['obs_pos']
        es_hists['obs_neg'] += npzfile['obs_neg']
        npzfile.close()
    outfileh.close() 
    np.savez(output_hist_file, **es_hists)

def compute_global_stats(sample_sets, es_hists_file, input_tsv_file,
                         output_tsv_file):
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
    # map sample set names to indexes
    sample_set_ind_map = \
        dict((ss.name, i) for i,ss in enumerate(sample_sets))
    # read report
    fout = open(output_tsv_file, 'w')
    print >>fout, '\t'.join(Report.FIELDS)
    fin = open(input_tsv_file)
    fin.next()
    es_col_ind = Report.FIELD_MAP['es']
    sample_set_col_ind = Report.FIELD_MAP['sample_set_name']
    global_nes_ind = Report.FIELD_MAP['global_nes']
    global_qval_ind = Report.FIELD_MAP['global_fdr_q_value']
    for line in fin:
        # read result
        fields = line.strip().split('\t')
        sample_set_name = fields[sample_set_col_ind]
        i = sample_set_ind_map[sample_set_name]
        es = float(fields[es_col_ind])
        # compute global nes and fdr        
        if es < 0:
            global_nes = es / hists_null_neg_means[i]            
            es_bin = np.digitize((es,), BINS_NEG)
            n = hists_null_neg[i,:es_bin].sum() / float(hists_null_neg_counts[i])
            d = hists_obs_neg[i,:es_bin].sum() / float(hists_obs_neg_counts[i])
        else:
            global_nes = es / hists_null_pos_means[i]            
            es_bin = np.digitize((es,), BINS_POS) - 1
            n = hists_null_pos[i,es_bin:].sum() / float(hists_null_pos_counts[i])
            d = hists_obs_pos[i,es_bin:].sum() / float(hists_obs_pos_counts[i])
        global_fdr_qval = n / d
        fields[global_nes_ind] = str(global_nes)
        fields[global_qval_ind] = str(global_fdr_qval)
        print >>fout, '\t'.join(fields)
    fin.close()
    fout.close()

def ssea_main(weight_vec_iter, sample_sets, config):
    # setup output directory
    if not os.path.exists(config.output_dir):
        logging.info("Creating output directory '%s'" % 
                     (config.output_dir))
        os.makedirs(config.output_dir)
    details_dir = os.path.join(config.output_dir, DETAILS_DIR)
    if not os.path.exists(details_dir):
        logging.debug("\tCreating details directory '%s'" % 
                      (details_dir))
        os.makedirs(details_dir)
    # create temp directory
    tmp_dir = os.path.join(config.output_dir, "tmp")
    if not os.path.exists(tmp_dir):
        logging.debug("\tCreating tmp directory '%s'" % (tmp_dir))
        os.makedirs(tmp_dir)
    # output files
    tmp_tsv_file = os.path.join(tmp_dir, OUTPUT_TSV_FILE)
    es_hists_file = os.path.join(config.output_dir, OUTPUT_HISTS_FILE)
    if config.num_processors > 1:
        logging.info("Running SSEA in parallel with %d processes" % 
                     (config.num_processors))
        ssea_parallel(weight_vec_iter, sample_sets, config, details_dir,
                      tmp_tsv_file, es_hists_file)
    else:
        logging.info("Running SSEA in serial")
        ssea_serial(weight_vec_iter, sample_sets, config, details_dir,
                    tmp_tsv_file, es_hists_file)
    # use ES null distributions to compute global statistics
    # and produce a report
    report_tsv_file = os.path.join(config.output_dir, 'out.tsv')
    compute_global_stats(sample_sets, es_hists_file, tmp_tsv_file,
                         report_tsv_file)
    # create html report
    if config.create_html:
        logging.info("Writing HTML Report")
        create_html_report(report_tsv_file, DETAILS_DIR, config)
    # free resources
    plt.close('all')
    # cleanup
    if os.path.exists(tmp_dir):
        shutil.rmtree(tmp_dir)    
    logging.info("Finished")
    return report_tsv_file
