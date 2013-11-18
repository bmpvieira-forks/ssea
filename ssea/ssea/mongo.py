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
from base import Config

import ssea
_package_dir = ssea.__path__[0]

def db_drop_colls(name):
    client = pymongo.MongoClient()
    db = client[name]
    row_metadata = db['metadata']
    col_metadata = db['samples']
    sample_sets = db['sample_sets']
    config = db['config']
    results = db['reports']
    colls = [row_metadata, col_metadata, sample_sets, config, results]
    
    for coll in colls:
        coll.drop()
    

def db_import(input_dir, name, host):
    config = Config()
    row_metadata_json_file = os.path.join(input_dir, 
                                          Config.METADATA_JSON_FILE)
    col_metadata_json_file = os.path.join(input_dir, 
                                          Config.SAMPLES_JSON_FILE)
    sample_sets_json_file = os.path.join(input_dir,
                                         Config.SAMPLE_SETS_JSON_FILE)
    config_json_file = os.path.join(input_dir, 
                                    Config.CONFIG_JSON_FILE)
    results_json_file = os.path.join(input_dir, 
                                    Config.RESULTS_JSON_FILE)
    
    p = subprocess.call(['mongoimport', '-c', 'metadata', '--file', row_metadata_json_file, '--host', host, '-d', name])
    p = subprocess.call(['mongoimport', '-c', 'samples', '--file', col_metadata_json_file, '--host', host, '-d', name])
    p = subprocess.call(['mongoimport', '-c', 'sample_sets', '--file', sample_sets_json_file, '--host', host,  '-d', name])
    p = subprocess.call(['mongoimport', '-c', 'config', '--file', config_json_file, '--host', host,  '-d', name])
    p1 = subprocess.Popen(['zcat', results_json_file], stdout=subprocess.PIPE)
    p2 = subprocess.Popen(['mongoimport', '-c', 'reports', '--host', host, '-d', name], stdin=p1.stdout)
    p1.wait()
    p2.wait()

def main(argv=None):
    '''Command line options.'''    
    # Setup argument parser
    parser = argparse.ArgumentParser()
    # Add command line parameters
    parser.add_argument("input_dir", 
                        help="directory containing files to import into db")
    parser.add_argument("-n", "--name", dest = 'name',
                        default = 'ssea',
                        help = 'name for ssea run (will be name of collections in db)')
    parser.add_argument("--host", dest = 'host',
                        default = 'localhost',
                        help = 'name of mongodb server to connect to')
    parser.add_argument("-d", "--delete", dest="delete", 
                        action="store_true", default=False, 
                        help="remove all collections from current database")
    # Process arguments
    args = parser.parse_args()
    # setup logging
    
    level = logging.INFO
    logging.basicConfig(level=level,
                        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    
    if args.delete == True:
        #function to delete all collections
        db_drop_colls(args.name)
    else: 
        # import data into mongodb  
        db_import(args.input_dir, args.name, args.host)
    
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
