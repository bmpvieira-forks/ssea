'''
Created on Oct 18, 2013

@author: mkiyer
'''
import argparse
import logging
import os
from datetime import datetime
from time import time

# local imports
from ssea import __version__
from base import WEIGHT_METHODS

DETAILS_DIR = 'details'

def timestamp():
    return datetime.fromtimestamp(time()).strftime('%Y-%m-%d-%H-%M-%S-%f')

class Config(object):
    def __init__(self):
        self.version = __version__
        self.weight_miss = 'weighted'
        self.weight_hit = 'weighted'
        self.perms = 1000
        self.fdr_qval_threshold = 0.05
        self.plot_conf_int = True
        self.create_html = True
        self.create_plots = True
        self.conf_int = 0.95
        self.sample_set_size_min = 10
        self.sample_set_size_max = 0
        self.smx_files = []
        self.smt_files = []
        self.weight_matrix_file = None
        self.output_dir = timestamp()
        self.details_dir = os.path.join(self.output_dir, 'details')
        self.name = 'myssea'
        self.num_processors = 1
    
    def get_argument_parser(self, parser=None):
        if parser is None:
            parser = argparse.ArgumentParser()
        grp = parser.add_argument_group("SSEA Options")
        grp.add_argument('-p', '--num-processors', dest='num_processors',
                         type=int, default=1,
                         help='Number of processor cores available '
                         '[default=%(default)s]')
        grp.add_argument('-o', '--output-dir', dest="output_dir", 
                         help='Output directory [default=%(default)s]')
        grp.add_argument('-n', '--name', dest="name", default=self.name,
                         help='Analysis name [default=%(default)s]')
        grp.add_argument('--weight-miss', dest='weight_miss',
                         choices=WEIGHT_METHODS, 
                         default=self.weight_miss,
                         help='Weighting method for elements not in set ' 
                         '[default=%(default)s]')
        grp.add_argument('--weight-hit', dest='weight_hit', 
                         choices=WEIGHT_METHODS, 
                         default=self.weight_hit,
                         help='Weighting method for elements in set '
                         '[default=%(default)s]')
        grp.add_argument('--perms', type=int, default=self.perms,
                         help='Number of permutations '
                         '[default=%(default)s]')
        grp.add_argument('--no-html', dest="create_html", 
                         action="store_false", default=self.create_html,
                         help='Do not create detailed html reports')
        grp.add_argument('--no-plot', dest="create_plots", 
                         action="store_false", default=self.create_plots,
                         help='Do not create enrichment plots')
        grp.add_argument('--fdr-qval-threshold', type=float,
                         dest="fdr_qval_threshold",
                         default=self.fdr_qval_threshold,
                         help='FDR q-value threshold for generating '
                         'reports [default=%(default)s]')
        grp.add_argument('--no-plot-conf-int', dest="plot_conf_int", 
                         action="store_false", default=self.plot_conf_int,
                         help='Do not show confidence intervals in '
                         'enrichment plot')
        grp.add_argument('--conf-int', dest="conf_int", type=float, 
                         default=self.conf_int,
                         help='Confidence interval level '
                         '[default=%(default)s]')
        grp.add_argument('--smin', dest="sample_set_size_min", type=int,
                         default=self.sample_set_size_min, metavar="N",
                         help='Exclude sample sets smaller than N '
                         'from the analysis')
        grp.add_argument('--smax', dest="sample_set_size_max", type=int,
                         default=self.sample_set_size_max, metavar="N",
                         help='Exclude sample sets larger than N '
                         'from the analysis')
        grp.add_argument('--smx', dest="smx_files", action='append',
                         help='File(s) containing sets in column format')
        grp.add_argument('--smt', dest="smt_files", action='append',
                         help='File(s) containing sets in row format')
        grp.add_argument('weight_matrix_file', 
                         help='File containing weight matrix')
        return parser

    def log(self, log_func=logging.info):
        log_func("Parameters")
        log_func("----------------------------------")
        log_func("\tname:                    %s" % (self.name))
        log_func("\tpermutations:            %d" % (self.perms))
        log_func("\tweight method miss:      %s" % (self.weight_miss))
        log_func("\tweight method hit:       %s" % (self.weight_hit))
        log_func("\tcreate html report:      %s" % (self.create_html))
        log_func("\tcreate plots:            %s" % (self.create_plots))
        log_func("\tFDR q-value threshold:   %f" % (self.fdr_qval_threshold))
        log_func("\tplot conf interval:      %s" % (self.plot_conf_int))
        log_func("\tconf interval:           %f" % (self.conf_int))
        log_func("\tsample set size min:     %d" % (self.sample_set_size_min))
        log_func("\tsample set size max:     %d" % (self.sample_set_size_max))
        log_func("\tsmx files:               %s" % (','.join(self.smx_files)))
        log_func("\tsmt files:               %s" % (','.join(self.smt_files)))
        log_func("\tweight matrix file:      %s" % (self.weight_matrix_file))
        log_func("\toutput directory:        %s" % (self.output_dir))
        log_func("----------------------------------")

    def get_json(self):
        d = {'name': self.name,
             'version': self.version,
             'perms': self.perms,
             'weight_miss': self.weight_miss,
             'weight_hit': self.weight_hit,
             'create_html': self.create_html,
             'create_plots': self.create_plots,
             'fdr_qval_threshold': self.fdr_qval_threshold,
             'plot_conf_int': self.plot_conf_int,
             'conf_int': self.conf_int,
             'sample_set_size_min': self.sample_set_size_min,
             'sample_set_size_max': self.sample_set_size_max,
             'smx_files': self.smx_files,
             'smt_files': self.smt_files,
             'weight_matrix_file': self.weight_matrix_file,
             'output_dir': self.output_dir}
        return d

    def parse_args(self, parser, args):
        # process and check arguments
        self.perms = max(1, args.perms)
        self.weight_miss = str(args.weight_miss)
        self.weight_hit = str(args.weight_hit)
        self.create_html = args.create_html
        self.create_plots = args.create_plots
        self.fdr_qval_threshold = args.fdr_qval_threshold
        self.plot_conf_int = args.plot_conf_int
        self.conf_int = args.conf_int
        self.sample_set_size_min = args.sample_set_size_min
        self.sample_set_size_max = args.sample_set_size_max
        self.name = args.name
        self.num_processors = args.num_processors
        # output directory
        self.output_dir = args.output_dir
        if os.path.exists(self.output_dir):
            parser.error("output directory '%s' already exists" % 
                         (self.output_dir))
        self.details_dir = os.path.join(self.output_dir, DETAILS_DIR)
        # read sample sets
        smx_files = []
        if args.smx_files is not None:
            for filename in args.smx_files:
                if not os.path.exists(filename):
                    parser.error("smx file '%s' not found" % (filename))
                smx_files.append(filename)
        self.smx_files = smx_files
        smt_files = []
        if args.smt_files is not None:
            for filename in args.smt_files:
                if not os.path.exists(filename):
                    parser.error("smt file '%s' not found" % (filename))
                smt_files.append(filename)
        self.smt_files = smt_files
        # read weights
        if not os.path.exists(args.weight_matrix_file):
            parser.error("weight matrix file '%s' not found" % 
                         (args.weight_matrix_file))
        self.weight_matrix_file = args.weight_matrix_file
