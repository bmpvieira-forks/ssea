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
from mongo import db_connect
import json

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
fields_reports = ['t_id',
          'ss_id',
          'ss_fdr_q_value',
          'fdr_q_value',
          'es',
          'nes',
          'nominal_p_value',
          'ss_rank']


def db_ssea_import(ssea_dir, matrix_dir, name, host):
    sample_sets_json_file = os.path.join(ssea_dir,
                                         'sample_set.json')
    ss_name = json.loads(open(sample_sets_json_file).read())['name']
    colls = db_connect(name, host)
    ss = colls['sample_sets']
    ss_check = ss.find_one({'name':ss_name})
    if ss_check != None:
        logging.info('Sample set: \'%s\' already in database' % ss_name)
    else: 
        logging.info('importing sample set \'%s\' to %s database on mongo server: %s' % (ss_name, name, host))
        
        
        
#         logging.debug("Importing sample_set file")
        #get ss_id by checking current number of ss in database
        ss_id = str(ss.count()) 
        _ssea_path = ssea.__path__[0]
        _merge_path = os.path.join(_ssea_path, 'utils/mongo_ssea_printJSON.py')
        p1 = subprocess.Popen(['python', _merge_path, ssea_dir, matrix_dir, ss_id, '-s'], stdout=subprocess.PIPE)
        p2 = subprocess.Popen(['mongoimport', '-c', 'sample_sets', '--host', host, '-d', name, '--upsert'], stdin=p1.stdout)
        p1.wait()
        p2.wait()
        
#         logging.info("Importing config file")
        p1 = subprocess.Popen(['python', _merge_path, ssea_dir, matrix_dir, ss_id, '-c'], stdout=subprocess.PIPE)
        p2 = subprocess.Popen(['mongoimport', '-c', 'configs', '--host', host, '-d', name, '--upsert'], stdin=p1.stdout)
        p1.wait()
        p2.wait()
        
#         logging.info("Importing results file")
        p1 = subprocess.Popen(['python', _merge_path, ssea_dir, matrix_dir, ss_id, '-r'], stdout=subprocess.PIPE)
        p2 = subprocess.Popen(['mongoimport', '-c', 'results', '--host', host, '-d', name, '--upsert'], stdin=p1.stdout)
        p1.wait()
        p2.wait()
        
#         logging.info("Importing hists file")
        p1 = subprocess.Popen(['python', _merge_path, ssea_dir, matrix_dir, ss_id, '--hist'], stdout=subprocess.PIPE)
        p2 = subprocess.Popen(['mongoimport', '-c', 'hists', '--host', host, '-d', name, '--upsert'], stdin=p1.stdout)
        p1.wait()
        p2.wait()
        
#         logging.info("Creating merge collection")
        _merge_path = os.path.join(_ssea_path, 'utils/mongo_merge_printJSON.py')
        p1 = subprocess.Popen(['python', _merge_path, '--host', host, '--name', name], stdout=subprocess.PIPE)
        p2 = subprocess.Popen(['mongoimport', '-c', 'merged', '--host', host, '-d', name, '--upsert'], stdin=p1.stdout)
        p1.wait()
        p2.wait()
    
        logging.info("Finished importing \'%s\'" % ss_name)
    
def main(argv=None):
    '''Command line options.'''    
    # Setup command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("ssea_dir", 
                        help="directory containing files to import into db")
    parser.add_argument("matrix_dir", 
                        help="directory containing files to import into db")
    parser.add_argument("-n", "--name", dest = 'name',
                        default = 'compendia',
                        help = 'name for ssea run (will be name of database)')
    parser.add_argument("--host", dest = 'host',
                        default = 'localhost:27017',
                        help = 'name of mongodb server to connect to')
    args = parser.parse_args()
    level = logging.DEBUG
    logging.basicConfig(level=level,
                        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    
    db_ssea_import(args.ssea_dir, args.matrix_dir, args.name, args.host)
    
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
