'''
Created on Nov 12, 2013

@author: mkiyer
@author: yniknafs
'''
import os
import sys
import argparse
import logging
import pymongo
import subprocess
import ssea
import collections
import numpy as np

NUM_NES_BINS = 10001
NES_BINS = np.logspace(-1,2,num=NUM_NES_BINS,base=10)
LOG_NES_BINS = np.log10(NES_BINS)

#connects to the mongo db and returns a dictionary  
#containing each collection in the database

_package_dir = ssea.__path__[0]

#list of fields to add to the trans_meta dictionary
fields_trans = ['category', 
          'nearest_gene_names', 
          'num_exons', 
          'name',
          'gene_id']

#list of fields from the reports to use in the combined database
fields_results = ['t_id',
          'ss_id',
          'ss_fdr_q_value',
          'es',
          'nes',
          'nominal_p_value',
          'ss_rank',
          'ss_frac']


#function to connect to database and return all collections
def db_connect(name, host):
    logging.info('connecting to %s database on mongo server: %s' % (name, host))
    client = pymongo.MongoClient(host)
    db = client[name]
    transcripts = db['transcripts']
    samples = db['samples']
    sample_sets = db['sample_sets']
    config = db['config']
    results = db['results']
    hists = db['hists']
    merged = db['merged']
    colls = {'transcripts':transcripts, 'samples':samples, 'sample_sets':sample_sets, 
             'config':config, 'results':results, 'hists':hists, 'merged':merged}
    return colls

def expression_import(name, host):
    _ssea_path = ssea.__path__[0]
    _expression_path = os.path.join(_ssea_path, 'utils/expression.py')
    colls = db_connect(name, host)
    runconfig = Config.from_dict(colls['config'].find_one()) 
    dir = runconfig.matrix_dir
    logging.info('importing expression data (%s) to %s database on mongo server: %s' % (_expression_path, name, host))
    #p1 = subprocess.call(['python', _expression_path, dir])
    p1 = subprocess.Popen(['python', _expression_path, dir], stdout=subprocess.PIPE)
    p2 = subprocess.Popen(['mongoimport', '-c', 'expression', '--host', host, '-d', name], stdin=p1.stdout)
    p1.wait()
    p2.wait()
    logging.info('expression import complete')

