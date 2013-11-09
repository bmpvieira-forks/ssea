'''
Created on Oct 29, 2013

@author: mkiyer
'''
import sys
import os
import argparse
import gzip
import logging
import shutil
import re
from multiprocessing import Process, Queue

# set matplotlib backend
import matplotlib
matplotlib.use('Agg')

# third-party packages
import numpy as np
from jinja2 import Environment, PackageLoader
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.cm as cm
from matplotlib import figure

# local imports
from ssea import __version__, __date__, __updated__
from ssea.kernel import ssea_kernel2, RandomState, power_transform
from base import BOOL_DTYPE, timestamp, Config, Result, Metadata, SampleSet, hist_quantile, quantile_sorted
from countdata import BigCountMatrix

# setup path to web files
import ssea
SRC_WEB_PATH = os.path.join(ssea.__path__[0], 'web')

# setup html template environment
env = Environment(loader=PackageLoader("ssea", "templates"),
                  extensions=["jinja2.ext.loopcontrols"])

# matplotlib static figure for plotting
global_fig = plt.figure(0)

# functions to test operator
OP_TEST_FUNCS = {'<=': lambda a,b: a<=b,
                 '>=': lambda a,b: a>=b,
                 '<': lambda a,b: a<b,
                 '>': lambda a,b: a>b}

class ReportConfig(object):
    def __init__(self):
        self.num_processes = 1
        self.input_dir = None
        self.output_dir = None
        self.thresholds = []
        self.plot_conf_int = True
        self.conf_int = 0.95
        self.create_html = True
        self.create_plots = True

    def get_argument_parser(self, parser=None):
        if parser is None:
            parser = argparse.ArgumentParser()
        grp = parser.add_argument_group("SSEA Report Options")
        grp.add_argument('-p', '--num-processes', dest='num_processes',
                         type=int, default=1,
                         help='Number of processor cores available '
                         '[default=%(default)s]')
        grp.add_argument('-o', '--output-dir', dest="output_dir", 
                         default=self.output_dir,
                         help='Output directory [default=%(default)s]')
        grp.add_argument('--no-html', dest="create_html", 
                         action="store_false", default=self.create_html,
                         help='Do not create detailed html reports')
        grp.add_argument('--no-plot', dest="create_plots", 
                         action="store_false", default=self.create_plots,
                         help='Do not create enrichment plots')
        grp.add_argument('-t', '--threshold', action="append", 
                         dest="thresholds",
                         help='Significance thresholds for generating '
                         'detailed reports [default=%(default)s]')
        grp.add_argument('--no-plot-conf-int', dest="plot_conf_int", 
                         action="store_false", default=self.plot_conf_int,
                         help='Do not show confidence intervals in '
                         'enrichment plot')
        grp.add_argument('--conf-int', dest="conf_int", type=float, 
                         default=self.conf_int,
                         help='Confidence interval level '
                         '[default=%(default)s]')
        grp.add_argument('input_dir')
        return parser        

    def log(self, log_func=logging.info):
        log_func("Parameters")
        log_func("----------------------------------")
        log_func("num processes:           %d" % (self.num_processes))
        log_func("input directory:         %s" % (self.input_dir))
        log_func("output directory:        %s" % (self.output_dir))
        log_func("create html report:      %s" % (self.create_html))
        log_func("create plots:            %s" % (self.create_plots))
        log_func("plot conf interval:      %s" % (self.plot_conf_int))
        log_func("conf interval:           %f" % (self.conf_int))
        log_func("thresholds:              %s" % (','.join(''.join(map(str,t)) for t in self.thresholds)))
        log_func("----------------------------------")

    def parse_args(self, parser, args):
        # process and check arguments
        self.create_html = args.create_html
        self.create_plots = args.create_plots
        self.plot_conf_int = args.plot_conf_int
        self.conf_int = args.conf_int
        self.num_processes = args.num_processes
        # parse threshold arguments of the form 'attribute,value'
        # for example: nominal_p_value,0.05
        if args.thresholds is not None:
            for arg in args.thresholds:
                m = re.match(r'(.+)([<>]=?)(.+)', arg)
                if m is None:
                    parser.error('error parsing threshold argument "%s"' % (arg))
                attr, op, value = m.groups()
                if attr not in Result.FIELDS:
                    parser.error('threshold attribute "%s" unknown' % (attr))
                if op not in OP_TEST_FUNCS:
                    parser.error('unrecognized operator "%s"' % (op))
                value = float(value)
                self.thresholds.append((attr,op,value))       
        # check input directory
        if not os.path.exists(args.input_dir):
            parser.error("input directory '%s' not found" % (args.input_dir))
        self.input_dir = args.input_dir
        # setup output directory
        if args.output_dir is None:
            self.output_dir = "Report_%s" % (timestamp())
        else:            
            self.output_dir = args.output_dir
        if os.path.exists(self.output_dir):
            parser.error("output directory '%s' already exists" % 
                         (self.output_dir))

class SSEAData:
    def get_details_table(self, sample_metadata):
        # show details of hits
        rows = [['index', 'sample', 'rank', 'raw_weights', 'transformed_weights',
                 'running_es', 'core_enrichment']]
        for i,ind in enumerate(self.hit_indexes):
            if self.es < 0:
                is_enriched = int(ind >= self.es_rank)
            else:                
                is_enriched = int(ind <= self.es_rank)
            meta = sample_metadata[self.sample_ids[ind]]
            rows.append([i, meta.name, ind+1, self.raw_weights[ind],
                         self.weights_hit[ind], self.es_run[ind], 
                         is_enriched])
        return rows

def ssea_rerun(sample_ids, counts, size_factors, sample_set, seed, config):
    '''    
    sample_ids: list of unique integer ids
    counts: list of floats
    size_factors: list of floats
    sample_set: SampleSet object
    '''
    # convert sample set to membership vector
    membership = np.empty((len(sample_ids),1), dtype=BOOL_DTYPE)
    membership[:,0] = sample_set.get_array(sample_ids)
    # reproduce previous run
    rng = RandomState(seed)
    (ranks, norm_counts, norm_counts_miss, norm_counts_hit, 
     es_vals, es_ranks, es_runs) = \
        ssea_kernel2(counts, size_factors, membership, rng,
                     resample_counts=False,
                     permute_samples=False,
                     add_noise=True,
                     noise_loc=config.noise_loc, 
                     noise_scale=config.noise_scale,
                     method_miss=config.weight_miss,
                     method_hit=config.weight_hit,
                     method_param=config.weight_param)
    # make object for plotting
    m = membership[ranks,0]
    hit_indexes = (m > 0).nonzero()[0]
    d = SSEAData()
    d.es = es_vals[0]
    d.es_run = es_runs[:,0]
    d.es_rank = es_ranks[0]
    d.hit_indexes = hit_indexes
    d.ranks = ranks
    d.sample_ids = sample_ids[ranks]
    d.raw_weights = norm_counts[ranks]
    d.weights_miss = norm_counts_miss[ranks]
    d.weights_hit = norm_counts_hit[ranks]
    return d

def plot_enrichment2(result, sseadata,
                     title, 
                     plot_conf_int=True, 
                     conf_int=0.95, 
                     fig=None):
    if fig is None:
        fig = plt.Figure()
    else:
        fig.clf()
    gs = gridspec.GridSpec(3, 1, height_ratios=[2,1,1])
    # running enrichment score
    ax0 = fig.add_subplot(gs[0])
    x = np.arange(len(sseadata.es_run))
    y = sseadata.es_run
    ax0.plot(x, y, lw=2, color='k', label='Enrichment profile')
    ax0.axhline(y=0, color='gray')
    ax0.axvline(x=sseadata.es_rank, lw=1, linestyle='--', color='black')
    # contour maps of resampled and null distributions
    xbins = np.linspace(0, len(sseadata.es_run), num=Config.ES_RANK_NUM_BINS)
    ybins = np.linspace(-1.0, 1.0, num=Config.ES_VAL_NUM_BINS)
    X = (xbins[:-1] + xbins[1:]) / 2.0
    Y = (ybins[:-1] + ybins[1:]) / 2.0
    # null distribution
    Z = np.array(result.null_es_hist2d)
    ax0.contour(X, Y, Z,               
                cmap=cm.winter,
                alpha=0.9,
                extend='both')
    # resampled distribution
    Z = np.array(result.resample_es_hist2d)    
    ax0.contour(X, Y, Z,
                cmap=cm.autumn,
                alpha=1.0,
                extend='both')
    ax0.set_xlim((0, len(sseadata.es_run)))
    ax0.set_ylim((-1.0, 1.0))
    ax0.grid(True)
    ax0.set_xticklabels([])
    ax0.set_ylabel('Enrichment score (ES)')
    ax0.set_title('Enrichment plot: %s' % (title))
    # membership in sample set
    ax1 = fig.add_subplot(gs[1])
    ax1.vlines(sseadata.hit_indexes, ymin=0, ymax=1, lw=0.25, 
               color='black', label='Hits')
    ax1.set_xlim((0, len(sseadata.es_run)))
    ax1.set_ylim((0, 1))
    ax1.set_xticks([])
    ax1.set_yticks([])
    ax1.set_xticklabels([])
    ax1.set_yticklabels([])
    ax1.set_ylabel('Set')
    # weights
    ax2 = fig.add_subplot(gs[2])
    ax2.plot(sseadata.weights_miss, color='blue')
    ax2.plot(sseadata.weights_hit, color='red')
    #ax2.plot(weights_hit, color='red')
    ax2.set_xlim((0, len(sseadata.es_run)))
    ax2.set_xlabel('Samples')
    ax2.set_ylabel('Weights')
    # draw
    fig.tight_layout()
    return fig

def plot_enrichment(running_es, rank_at_max, hit_indexes, weights_miss, 
                    weights_hit, es, null_es_mean, es_null_bins, 
                    null_es_val_hist, title, plot_conf_int=True, conf_int=0.95, 
                    fig=None):
    if fig is None:
        fig = plt.Figure()
    else:
        fig.clf()
    gs = gridspec.GridSpec(3, 1, height_ratios=[2,1,1])
    # running enrichment score
    ax0 = fig.add_subplot(gs[0])
    x = np.arange(len(running_es))
    y = running_es
    ax0.plot(x, y, lw=2, color='blue', label='Enrichment profile')
    ax0.axhline(y=0, color='gray')
    ax0.axvline(x=rank_at_max, lw=1, linestyle='--', color='black')
    # confidence interval
    if plot_conf_int:
        if es < 0:
            inds = (es_null_bins <= 0).nonzero()[0]
            left, right = -1.0, 0.0
        else:
            inds = (es_null_bins >= 0).nonzero()[0]
            left, right = 0.0, 1.0
        null_es_val_hist = np.array(null_es_val_hist, dtype=np.float)
        bins = es_null_bins[inds]
        hist = null_es_val_hist[inds[:-1]]
        ci_lower = hist_quantile(hist, bins, 1.0-conf_int, left, right)                
        ci_upper = hist_quantile(hist, bins, conf_int, left, right)
        lower_bound = np.repeat(ci_lower, len(x))
        upper_bound = np.repeat(ci_upper, len(x))
        ax0.axhline(y=null_es_mean, lw=2, color='red', ls=':')
        ax0.fill_between(x, lower_bound, upper_bound,
                         lw=0, facecolor='yellow', alpha=0.5,
                         label='%.2f CI' % (100. * conf_int))
        # here we use the where argument to only fill the region 
        # where the ES is above the confidence interval boundary
        if es < 0:
            ax0.fill_between(x, y, lower_bound, where=y<lower_bound, 
                             lw=0, facecolor='blue', alpha=0.5)
        else:
            ax0.fill_between(x, upper_bound, y, where=y>upper_bound, 
                             lw=0, facecolor='blue', alpha=0.5)
    ax0.set_xlim((0, len(running_es)))
    ax0.grid(True)
    ax0.set_xticklabels([])
    ax0.set_ylabel('Enrichment score (ES)')
    ax0.set_title('Enrichment plot: %s' % (title))
    # membership in sample set
    ax1 = fig.add_subplot(gs[1])
    ax1.vlines(hit_indexes, ymin=0, ymax=1, lw=0.25, 
               color='black', label='Hits')
    ax1.set_xlim((0, len(running_es)))
    ax1.set_ylim((0, 1))
    ax1.set_xticks([])
    ax1.set_yticks([])
    ax1.set_xticklabels([])
    ax1.set_yticklabels([])
    ax1.set_ylabel('Set')
    # weights
    ax2 = fig.add_subplot(gs[2])
    ax2.plot(weights_miss, color='blue')
    ax2.plot(weights_hit, color='red')
    #ax2.plot(weights_hit, color='red')
    ax2.set_xlim((0, len(running_es)))
    ax2.set_xlabel('Samples')
    ax2.set_ylabel('Weights')
    # draw
    fig.tight_layout()
    return fig

def plot_null_distribution(es, es_null_bins, null_es_val_hist, 
                           fig=None):
    if fig is None:
        fig = plt.Figure()
    else:
        fig.clf()
    # get coords of bars    
    left = es_null_bins[:-1]
    height = null_es_val_hist
    width = [(r-l) for l,r in zip(es_null_bins[:-1],es_null_bins[1:])]
    # make plot
    ax = fig.add_subplot(1,1,1)
    ax.bar(left, height, width)
    ax.axvline(x=es, linestyle='--', color='black')
    ax.set_title('Random ES distribution')
    ax.set_ylabel('P(ES)')
    # calculate percent neg
    percent_neg = sum(null_es_val_hist[i] for i in xrange(len(null_es_val_hist))
                      if es_null_bins[i] < 0)
    percent_neg /= float(sum(null_es_val_hist))
    ax.set_xlabel('ES (Sets with neg scores: %.0f%%)' % 
                  (percent_neg))
    return fig

def create_detailed_report(result, sseadata, rowmeta, colmeta, sample_set, 
                           reportconfig):
    '''
    Generate detailed report files including enrichment plots and 
    a tab-delimited text file showing the running ES score and rank
    of each member in the sample set
     
    returns dict containing files written
    '''    
    d = {}
    if reportconfig.create_plots:        
        # enrichment plot
        fig = plot_enrichment2(result, sseadata, 
                               title=sample_set.name, 
                               plot_conf_int=reportconfig.plot_conf_int, 
                               conf_int=reportconfig.conf_int,
                               fig=global_fig)
#         fig = plot_enrichment(sseadata.es_run, 
#                               sseadata.es_rank,
#                               sseadata.hit_indexes, 
#                               sseadata.weights_miss,
#                               sseadata.weights_hit,
#                               sseadata.es,  
#                               result.null_es_mean, 
#                               Config.ES_NULL_BINS,
#                               result.null_es_val_hist,
#                               title=sample_set.name, 
#                               plot_conf_int=reportconfig.plot_conf_int, 
#                               conf_int=reportconfig.conf_int,
#                               fig=global_fig)
        eplot_png = '%s.%s.eplot.png' % (rowmeta.name, sample_set.name)
        eplot_pdf = '%s.%s.eplot.pdf' % (rowmeta.name, sample_set.name)
        fig.savefig(os.path.join(reportconfig.output_dir, eplot_png))
        fig.savefig(os.path.join(reportconfig.output_dir, eplot_pdf))
        # null distribution plot
        H = np.array(result.null_es_hist2d)
        fig = plot_null_distribution(result.es, 
                                     Config.ES_VAL_BINS, 
                                     H.sum(axis=1),
                                     fig=global_fig)
        nplot_png = '%s.%s.null.png' % (rowmeta.name, sample_set.name)
        nplot_pdf = '%s.%s.null.pdf' % (rowmeta.name, sample_set.name)
        fig.savefig(os.path.join(reportconfig.output_dir, nplot_png))        
        fig.savefig(os.path.join(reportconfig.output_dir, nplot_pdf))
        d.update({'eplot_png': eplot_png,
                  'nplot_png': nplot_png})
        d.update({'eplot_pdf': eplot_pdf,
                  'nplot_pdf': nplot_pdf})

    # write tab-delimited details file
    details_rows = sseadata.get_details_table(colmeta)
    details_tsv = '%s.%s.tsv' % (rowmeta.name, sample_set.name)
    fp = open(os.path.join(reportconfig.output_dir, details_tsv), 'w')
    for row in details_rows:
        print >>fp, '\t'.join(map(str,row))
    fp.close()
    d['tsv'] = details_tsv
    # render to html
    if reportconfig.create_html:
        details_html = '%s.%s.html' % (rowmeta.name, sample_set.name)
        t = env.get_template('details.html')
        fp = open(os.path.join(reportconfig.output_dir, details_html), 'w')
        print >>fp, t.render(result=result,
                             sseadata=sseadata,
                             details=details_rows,
                             files=d,
                             rowmeta=rowmeta,
                             sample_set=sample_set)
        fp.close()
        d['html'] = details_html
    return d

def create_html_report(input_file, output_file, row_metadata, sample_sets, 
                       runconfig):
    def _result_parser(filename):
        with open(filename, 'r') as fp:
            for line in fp:
                result = Result.from_json(line.strip())
                rowmeta = row_metadata[result.t_id]
                sample_set = sample_sets[result.ss_id]
                result.sample_set_name = sample_set.name
                result.sample_set_desc = sample_set.desc
                result.sample_set_size = len(sample_set.sample_ids)
                result.name = rowmeta.name
                yield result
    t = env.get_template('report.html')
    with open(output_file, 'w') as fp:
        print >>fp, t.render(name=runconfig.name,
                             results=_result_parser(input_file))

def parse_and_filter_results(filename, thresholds):
    with gzip.open(filename, 'rb') as fin:
        for line in fin:
            # load json document (one per line)
            result = Result.from_json(line.strip())      
            # apply thresholds
            skip = False
            for attr,op,threshold in thresholds:
                value = getattr(result, attr)
                if not OP_TEST_FUNCS[op](value, threshold):
                    skip = True
                    break
            if skip:
                continue
            yield result

def _producer_process(input_queue, filename, config):
    for result in parse_and_filter_results(filename, config.thresholds):
        input_queue.put(result)
    # tell consumers to stop
    for i in xrange(config.num_processes):
        input_queue.put(None)
    logging.debug("Producer finished")

def _worker_process(input_queue, output_queue, sample_sets, row_metadata, 
                    col_metadata, runconfig, config):
    # open weight matrix memmap file
    weight_matrix_file = os.path.join(config.input_dir, 
                                      Config.WEIGHT_MATRIX_FILE)
    shape = (len(row_metadata),len(col_metadata))
    weight_matrix = np.memmap(weight_matrix_file, 
                              dtype=Config.MEMMAP_DTYPE, 
                              mode='r', 
                              shape=shape)
    # process results
    while True:
        result = input_queue.get()
        if result is None:
            break
        # process result
        weights = np.array(weight_matrix[result.t_id,:], dtype=np.float)
        # remove 'nan' values        
        sample_ids = np.isfinite(weights).nonzero()[0]
        weights = weights[sample_ids]
        # get sample set
        sample_set = sample_sets[result.ss_id]
        d = create_detailed_report(result, sample_ids, weights,
                                   row_metadata[result.t_id], 
                                   col_metadata,
                                   sample_set,
                                   runconfig,
                                   config)
        # update results with location of plot files
        result.files = d                
        output_queue.put(result)
    del weight_matrix
    # send done signal
    output_queue.put(None)
    logging.debug("Worker finished")

def report_parallel(config):
    # create output dir
    if not os.path.exists(config.output_dir):
        logging.debug("Creating output dir '%s'" % (config.output_dir))
        os.makedirs(config.output_dir)
    # create directory for static web files (CSS, javascript, etc)
    if config.create_html:
        web_dir = os.path.join(config.output_dir, 'web')
        if not os.path.exists(web_dir):
            logging.debug("Installing web files")
            shutil.copytree(SRC_WEB_PATH, web_dir)
    # load input files
    row_metadata_json_file = os.path.join(config.input_dir, 
                                          Config.METADATA_JSON_FILE)
    row_metadata = list(Metadata.parse_json(row_metadata_json_file))
    col_metadata_json_file = os.path.join(config.input_dir, 
                                          Config.SAMPLES_JSON_FILE)
    col_metadata = list(Metadata.parse_json(col_metadata_json_file))
    sample_sets_json_file = os.path.join(config.input_dir,
                                         Config.SAMPLE_SETS_JSON_FILE)
    sample_sets = dict((ss._id,ss) for ss in SampleSet.parse_json(sample_sets_json_file))
    config_json_file = os.path.join(config.input_dir, 
                                    Config.CONFIG_JSON_FILE)
    runconfig = Config.parse_json(config_json_file)
    # create multiprocessing queues for passing data
    input_queue = Queue(maxsize=config.num_processes*3)
    output_queue = Queue(maxsize=config.num_processes*3)    
    # start a producer process
    logging.debug("Starting producer process and %d workers" % (config.num_processes))
    json_file = os.path.join(config.input_dir, Config.RESULTS_JSON_FILE)
    args=(input_queue, json_file, config)
    producer = Process(target=_producer_process, args=args)
    producer.start()
    # start consumer processes
    procs = []
    for i in xrange(config.num_processes):
        args = (input_queue, output_queue, sample_sets, row_metadata, 
                col_metadata, runconfig, config)
        p = Process(target=_worker_process, args=args)
        p.start()
        procs.append(p)
    # get results from consumers
    num_alive = config.num_processes
    filtered_results_file = os.path.join(config.output_dir, 'filtered_results.json')
    with open(filtered_results_file, 'w') as fout:
        while num_alive > 0:
            result = output_queue.get()
            if result is None:
                num_alive -= 1
                logging.debug("Main process detected worker finished, %d still alive" % (num_alive))
            else:
                # write result to tab-delimited text
                print >>fout, result.to_json()
    logging.debug("Joining all processes")
    # wait for producer to finish
    producer.join()
    # wait for consumers to finish
    for p in procs:
        p.join()
    # produce report
    if config.create_html:
        logging.debug("Writing HTML report")
        html_file = os.path.join(config.output_dir, 'filtered_results.html')
        create_html_report(filtered_results_file, html_file,
                           row_metadata, sample_sets, runconfig)
    logging.debug("Done.")

def report(config):
    # create output dir
    if not os.path.exists(config.output_dir):
        logging.debug("Creating output dir '%s'" % (config.output_dir))
        os.makedirs(config.output_dir)
    # create directory for static web files (CSS, javascript, etc)
    if config.create_html:
        web_dir = os.path.join(config.output_dir, 'web')
        if not os.path.exists(web_dir):
            logging.info("Installing web files")
            shutil.copytree(SRC_WEB_PATH, web_dir)
    # link to input files
    row_metadata_json_file = os.path.join(config.input_dir, 
                                          Config.METADATA_JSON_FILE)
    col_metadata_json_file = os.path.join(config.input_dir, 
                                          Config.SAMPLES_JSON_FILE)
    sample_sets_json_file = os.path.join(config.input_dir,
                                         Config.SAMPLE_SETS_JSON_FILE)
    config_json_file = os.path.join(config.input_dir, 
                                    Config.CONFIG_JSON_FILE)
    # load input files
    row_metadata = list(Metadata.parse_json(row_metadata_json_file))
    col_metadata = list(Metadata.parse_json(col_metadata_json_file))
    sample_sets = dict((ss._id,ss) for ss in SampleSet.parse_json(sample_sets_json_file))
    runconfig = Config.parse_json(config_json_file)
    # open data matrix
    bm = BigCountMatrix.open(runconfig.matrix_dir)
    # write filtered results to output file
    filtered_results_file = os.path.join(config.output_dir, 
                                         'filtered_results.json')
    fout = open(filtered_results_file, 'w')
    # parse report json    
    logging.info("Processing results")
    json_file = os.path.join(config.input_dir, Config.RESULTS_JSON_FILE)
    for result in parse_and_filter_results(json_file, config.thresholds):
        # read from memmap
        counts = np.array(bm.counts[result.t_id,:], dtype=np.float)
        # remove 'nan' values        
        sample_ids = np.isfinite(counts).nonzero()[0]
        counts = counts[sample_ids]
        size_factors = bm.size_factors[sample_ids]
        # get sample set
        sample_set = sample_sets[result.ss_id]
        # rerun ssea
        sseadata = ssea_rerun(sample_ids, counts, size_factors, sample_set, 
                              result.rand_seed, runconfig)
        d = create_detailed_report(result, sseadata, 
                                   row_metadata[result.t_id], col_metadata, 
                                   sample_set, config)
        # update results with location of plot files
        result.files = d
        # write result to tab-delimited text
        print >>fout, result.to_json()
    fout.close()
    # produce report
    if config.create_html:
        logging.debug("Writing HTML report")
        html_file = os.path.join(config.output_dir, 'filtered_results.html')
        create_html_report(filtered_results_file, html_file,
                           row_metadata, sample_sets, runconfig)
    logging.debug("Done.")
    return 0

def main(argv=None):
    '''Command line options.'''    
    if argv is None:
        argv = sys.argv
    else:
        sys.argv.extend(argv)

    program_name = os.path.basename(sys.argv[0])
    program_version = "v%s" % __version__
    program_build_date = str(__updated__)
    program_version_message = '%s %s (%s)' % (program_name, program_version, program_build_date)
    program_shortdesc = __import__('__main__').__doc__.split("\n")[1]
    program_license = '''%s

  Created by mkiyer and yniknafs on %s.
  Copyright 2013 MCTP. All rights reserved.
  
  Licensed under the GPL
  http://www.gnu.org/licenses/gpl.html
  
  Distributed on an "AS IS" basis without warranties
  or conditions of any kind, either express or implied.

USAGE
''' % (program_shortdesc, str(__date__))

    # create instance of run configuration
    config = ReportConfig()
    # Setup argument parser
    parser = argparse.ArgumentParser(description=program_license)
    # Add command line parameters
    config.get_argument_parser(parser)
    parser.add_argument("-v", "--verbose", dest="verbose", 
                        action="store_true", default=False, 
                        help="set verbosity level [default: %(default)s]")
    parser.add_argument('-V', '--version', action='version', 
                        version=program_version_message)
    # Process arguments
    args = parser.parse_args()
    # setup logging
    if args.verbose > 0:
        level = logging.DEBUG
    else:
        level = logging.INFO
    logging.basicConfig(level=level,
                        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    # initialize configuration
    config.parse_args(parser, args)
    config.log()
    #return report_parallel(config)
    return report(config)

if __name__ == "__main__":
    sys.exit(main())


