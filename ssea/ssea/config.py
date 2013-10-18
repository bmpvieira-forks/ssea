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
from base import WEIGHT_METHODS

def timestamp():
    return datetime.fromtimestamp(time()).strftime('%Y-%m-%d-%H-%M-%S-%f')

class Config(object):
    def __init__(self):
        self.weight_miss = 'weighted'
        self.weight_hit = 'weighted'
        self.perms = 1000
        self.plot_conf_int = True
        self.conf_int = 0.95
        self.smx_files = []
        self.smt_files = []
        self.weight_matrix_file = None
        self.output_dir = timestamp()
        self.name = 'myssea'
    
    def get_argument_parser(self, parser=None):
        if parser is None:
            parser = argparse.ArgumentParser()
        grp = parser.add_argument_group("SSEA Options")
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
        grp.add_argument('--no-plot-conf-int', dest="plot_conf_int", 
                         action="store_false", default=self.plot_conf_int,
                         help='Do not show confidence intervals in '
                         'enrichment plot')
        grp.add_argument('--conf-int', dest="conf_int", type=float, 
                         default=self.conf_int,
                         help='Confidence interval level '
                         '[default=%(default)s]')        
        grp.add_argument('--smx', dest="smx_files", action='append',
                         help='File(s) containing sets in column format')
        grp.add_argument('--smt', dest="smt_files", action='append',
                         help='File(s) containing sets in row format')
        grp.add_argument('-o', '--output-dir', dest="output_dir", 
                         help='Output directory [default=%(default)s]')
        grp.add_argument('-n', '--name', dest="name", default=self.name,
                         help='Analysis name [default=%(default)s]')
        grp.add_argument('weight_matrix_file', 
                         help='File containing weight matrix')
        return parser

    def log(self, log_func=logging.info):
        log_func("Parameters")
        log_func("----------------------------------")
        log_func("\tname:               %s" % (self.name))
        log_func("\tpermutations:       %d" % (self.perms))
        log_func("\tweight method miss: %s" % (self.weight_miss))
        log_func("\tweight method hit:  %s" % (self.weight_hit))
        log_func("\tplot conf interval: %s" % (self.plot_conf_int))
        log_func("\tconf interval:      %f" % (self.conf_int))
        log_func("\tsmx files:          %s" % (len(self.smx_files)))
        log_func("\tsmt files:          %s" % (len(self.smt_files)))
        log_func("\tweight matrix file: %s" % (self.weight_matrix_file))
        log_func("\toutput directory:   %s" % (self.output_dir))
        log_func("----------------------------------")

    def parse_args(self, parser, args):
        # process and check arguments
        self.perms = max(1, args.perms)
        self.weight_miss = args.weight_miss
        self.weight_hit = args.weight_hit
        self.plot_conf_int = args.plot_conf_int
        self.conf_int = args.conf_int
        self.name = args.name
        # output directory
        self.output_dir = args.output_dir
        if os.path.exists(self.output_dir):
            parser.error("output directory '%s' already exists" % 
                         (self.output_dir))
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
