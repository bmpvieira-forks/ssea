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
from ssea.lib.countdata import BigCountMatrix
from ssea.lib.base import Result, SampleSet
import json

NUM_NES_BINS = 10001
NES_BINS = np.logspace(-1,2,num=NUM_NES_BINS,base=10)
LOG_NES_BINS = np.log10(NES_BINS)


_package_dir = ssea.__path__[0]

#list of fields to add to the trans_meta dictionary
fields_trans = ['category', 
          'nearest_gene_names', 
          'num_exons', 
          'name',
          'gene_id']

#list of fields from the reports to use in the combined database
fields_reports = ['t_id',
          'ss_id',
          'ss_fdr_q_value',
          'fdr_q_value',
          'es',
          'nes',
          'nominal_p_value',
          'ss_rank']



#connects to the mongo db and returns a dictionary  
#containing each collection in the database
def db_connect(name, host):
    logging.info('connecting to %s database on mongo server: %s' % (name, host))
    client = pymongo.MongoClient(host)
    db = client[name]
    row_metadata = db['metadata']
    col_metadata = db['samples']
    sample_sets = db['sample_sets']
    config = db['config']
    reports = db['reports']
    colls = {'row_meta':row_metadata, 'col_meta':col_metadata, 'ss':sample_sets, 'config':config, 'reports':reports}
    return colls


    config_json_file = os.path.join(input_dir, 
                                    'config.json')
    results_json_file = os.path.join(input_dir, 
                                    'results.json')
    hists_file = os.path.join(input_dir, 
                                    'hists.npz')

def db_ss_printJSON(ssea_dir, matrix_dir, ss_id):
    sample_sets_json_file = os.path.join(ssea_dir,
                                         'sample_set.json')
    bm = BigCountMatrix.open(matrix_dir)
    samples = bm.colnames
    ss = SampleSet.parse_json(sample_sets_json_file)
    membership = ss.get_array(samples)
    d = ss.to_dict(membership)
    d['_id'] = int(ss_id)
    print json.dumps(d)
    
def db_config_printJSON(ssea_dir, ss_id):
    config_json_file = os.path.join(ssea_dir,
                                         'config.json')
    s = open(config_json_file).read()
    d = json.loads(s)
    d['_id'] = int(ss_id)
    print json.dumps(d)

def db_results_printJSON(ssea_dir, ss_id):
    results_json_file = os.path.join(ssea_dir,
                                         'results.json')
    with open(results_json_file, 'r') as fin:
        for line in fin:
            # load json document (one per line)
            result = Result.from_json(line.strip())  
            result.ss_id = int(ss_id)
            print result.to_json()

def db_hists_printJSON(ssea_dir, ss_id):
    hists_file = os.path.join(ssea_dir, 
                                    'hists.npz')

    x = np.load(hists_file)
    hist_fields = ['obs_nes_neg', 
                   'obs_nes_pos', 
                   'null_nes_neg',
                   'null_nes_pos']
    d = {}
    for field in hist_fields: 
        d[field] =  list(x[field])
    
    d['nes_bins'] = list(NES_BINS)
    d['_id'] = int(ss_id)
    print json.dumps(d)
    

def main(argv=None):
    '''Command line options.'''    
    # Setup command line args
    parser = argparse.ArgumentParser()
    # Add command line parameters
    parser.add_argument("ssea_dir", 
                        help="directory containing ssea files to import into db")
    parser.add_argument("matrix_dir", 
                        help="directory containing matrix files")
    parser.add_argument("ss_id", 
                        help="Sample set ID")
    parser.add_argument("-s", dest="s", 
                        action="store_true", default=False, 
                        help="Print sample_set JSON")
    parser.add_argument("-c", dest="c", 
                        action="store_true", default=False, 
                        help="Print config JSON")
    parser.add_argument("-r", dest="r", 
                        action="store_true", default=False, 
                        help="Print results JSONs")
    parser.add_argument("--hist", dest="h", 
                        action="store_true", default=False, 
                        help="Print hists JSON")
    
    
    args = parser.parse_args()
    level = logging.DEBUG
    logging.basicConfig(level=level,
                        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    
    if args.s:
        db_ss_printJSON(args.ssea_dir, args.matrix_dir, args.ss_id)
    if args.c:
        db_config_printJSON(args.ssea_dir, args.ss_id)
    if args.r:
        db_results_printJSON(args.ssea_dir, args.ss_id)
    if args.h:
        db_hists_printJSON(args.ssea_dir, args.ss_id)
    
    
    
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
