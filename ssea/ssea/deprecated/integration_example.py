'''
Created on Oct 29, 2013

@author: mkiyer
'''

# THIS NO LONGER APPLIES

import numpy as np

# import classes needed to set up SSEA
from ssea.base import Metadata, SampleSet, WeightVector, Config
from ssea.algo import ssea_main

# set some parameters for testing
num_samples = 1000
num_transcripts = 100
sample_set_size = 50

config = Config()       
print config.__dict__

import sys
sys.exit(0)

# create samples. samples are now Metadata objects which have a 
# unique integer id as well as a dictionary of associated sample parameters
samples = []
for i in xrange(num_samples):
    samples.append(Metadata(i, "SAMPLE%03d" % i, {'a_random_param': np.random.random()}))    

# create weight metadata in the form of Metadata objects corresponding
# to transcripts with dictionary of associated parameters
transcripts = []
for i in xrange(num_transcripts):
    transcripts.append(Metadata(i, "T%03d" % i, {'bongo_kongo': np.random.random()}))    

# create weight matrix, a 2D numpy array, and fill it with random data 
mat = np.random.random(size=(num_transcripts, num_samples))

# create a weight vector iterator from the matrix, the transcript metadata,
# and the sample metadata
weight_vec_iter = WeightVector.from_data(mat, transcripts, samples)

# create a sample set
sample_set = SampleSet(_id=0,
                       name='test_sample_set',
                       desc='test description',
                       samples=samples[:sample_set_size])
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

# run SSEA
report_file = ssea_main(weight_vec_iter, sample_sets, config)

import sys
sys.exit(0)

# parse output
#for rowdict in Report.parse(report_file):
#    es = rowdict['es']
#    global_qval = rowdict['global_fdr_q_value']
#    available_fields = rowdict.keys()
#    print rowdict['name'], es, global_qval