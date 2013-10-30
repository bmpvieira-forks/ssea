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
        self.num_processors = 1
        self.name = 'myssea'
        self.weight_miss = 'log'
        self.weight_hit = 'log'
        self.weight_const = 1.1
        self.weight_noise = 0.1
        self.perms = 1000
        self.detailed_report_threshold = 0.05
        self.plot_conf_int = True
        self.conf_int = 0.95
        self.create_html = True
        self.output_dir = "SSEA_%s" % (timestamp())
    
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
        grp.add_argument('--weight-const', dest='weight_const', type=float,
                         default=self.weight_const,
                         help='Constant floating-point number to add to '
                         'all weights prior to transformation '
                         '[default=%(default)s]')
        grp.add_argument('--weight-noise', dest='weight_noise', type=float,
                         default=self.weight_noise,
                         help='Add uniform noise in the range [0.0-X) to '
                         'the weights to increase robustness '
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
        grp.add_argument('--threshold', type=float,
                         dest="detailed_report_threshold",
                         default=self.detailed_report_threshold,
                         help='FDR q-value threshold for generating '
                         'detailed reports [default=%(default)s]')
        grp.add_argument('--no-plot-conf-int', dest="plot_conf_int", 
                         action="store_false", default=self.plot_conf_int,
                         help='Do not show confidence intervals in '
                         'enrichment plot')
        grp.add_argument('--conf-int', dest="conf_int", type=float, 
                         default=self.conf_int,
                         help='Confidence interval level '
                         '[default=%(default)s]')
        return parser

    def log(self, log_func=logging.info):
        log_func("Parameters")
        log_func("----------------------------------")
        log_func("\tname:                    %s" % (self.name))
        log_func("\tpermutations:            %d" % (self.perms))
        log_func("\tweight method miss:      %s" % (self.weight_miss))
        log_func("\tweight method hit:       %s" % (self.weight_hit))
        log_func("\tweight constant:         %f" % (self.weight_const))
        log_func("\tweight noise:            %s" % (self.weight_noise))
        log_func("\tcreate html report:      %s" % (self.create_html))
        log_func("\tcreate plots:            %s" % (self.create_plots))
        log_func("\tdetailed report q-val:   %f" % (self.detailed_report_threshold))
        log_func("\tplot conf interval:      %s" % (self.plot_conf_int))
        log_func("\tconf interval:           %f" % (self.conf_int))
        log_func("\toutput directory:        %s" % (self.output_dir))
        log_func("----------------------------------")

    def get_json(self):
        d = {'name': self.name,
             'perms': self.perms,
             'weight_miss': self.weight_miss,
             'weight_hit': self.weight_hit,
             'weight_const': self.weight_const,
             'weight_noise': self.weight_noise,
             'create_html': self.create_html,
             'create_plots': self.create_plots,
             'detailed_report_threshold': self.detailed_report_threshold,
             'plot_conf_int': self.plot_conf_int,
             'conf_int': self.conf_int,
             'output_dir': self.output_dir}
        return d

    def parse_args(self, parser, args):
        # process and check arguments
        self.perms = max(1, args.perms)
        self.create_html = args.create_html
        self.create_plots = args.create_plots
        self.detailed_report_threshold = args.detailed_report_threshold
        self.plot_conf_int = args.plot_conf_int
        self.conf_int = args.conf_int
        self.na_value = args.na_value
        self.name = args.name
        self.num_processors = args.num_processors
        self.num_metadata_cols = args.num_metadata_cols
        # check weight methods and constant
        self.weight_miss = str(args.weight_miss)
        self.weight_hit = str(args.weight_hit)
        self.weight_const = args.weight_const
        self.weight_noise = args.weight_noise
        if self.weight_const < 0.0:
            parser.error('weight const < 0.0 invalid')
        if (self.weight_const - self.weight_noise) < 0.0:            
            parser.error('weight const minus noise < 0.0 invalid')
        elif ((self.weight_miss == 'log' or self.weight_hit == 'log')):
            if self.weight_const < 1.0:
                parser.error('weight constant %f < 1.0 not allowed with '
                             'log method' % (self.weight_const))
            if (self.weight_const - self.weight_noise) < 1.0:
                parser.error('weight constant minus noise is < 1.0')
        # output directory
        self.output_dir = args.output_dir
        if os.path.exists(self.output_dir):
            parser.error("output directory '%s' already exists" % 
                         (self.output_dir))
