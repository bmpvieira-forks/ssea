'''
Created on Oct 29, 2013

@author: mkiyer
'''
import numpy as np

# import classes needed to set up SSEA
from ssea.base import SampleSet, WeightVector
from ssea.config import Config
from ssea.algo import ssea_main
from ssea.report import Report

# set some parameters for testing
num_samples = 1000
num_transcripts = 10
sample_set_size = 50

# define "universe" of samples
samples = ["SAMPLE%03d" % (i) for i in xrange(num_samples)]

# create weight matrix data
transcript_ids = ["T%03d" % (i) for i in xrange(num_transcripts)]
# random data
mat = np.random.random(size=(num_transcripts, num_samples))
weight_vec_iter = WeightVector.from_data(rownames=transcript_ids,
                                         samples=samples,
                                         weight_matrix=mat)

# create a sample set
sample_set = SampleSet(name='test_sample_set',
                       desc='test description',
                       value=set(['SAMPLE%03d' % (x) for x in xrange(sample_set_size)]))
# pass list of sample sets to SSEA
sample_sets = [sample_set]

# configure SSEA by first creating a Config object
config = Config()
# modify options as needed
config.num_processors = 1
config.name = 'myssea'
config.weight_miss = 'log'
config.weight_hit = 'log'
config.weight_const = 1.1
config.weight_noise = 0.1
config.perms = 1000
config.detailed_report_threshold = 0.05
config.plot_conf_int = True
config.conf_int = 0.95
config.create_html = True
config.create_plots = True

# run SSEA
report_file = ssea_main(weight_vec_iter, sample_sets, config)

# parse output
for rowdict in Report.parse(report_file):
    es = rowdict['es']
    global_qval = rowdict['global_fdr_q_value']
    available_fields = rowdict.keys()
    print rowdict['name'], es, global_qval