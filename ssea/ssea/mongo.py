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

'''
can you see this Matthew?
'''

# host name of mongo server 
HOSTNAME = 'pathbio-dx11.path.med.umich.edu'


def db_drop_colls(name):
    client = pymongo.MongoClient()
    db = client.test
    row_metadata = db[name + '.metadata']
    col_metadata = db[name + '.samples']
    sample_sets = db[name + '.sample_sets']
    config = db[name + '.config']
    results = db[name + '.reports']
    dbs = [row_metadata, col_metadata, sample_sets, config, results]
    
    for coll in dbs:
        coll.drop()
    

def db_import(input_dir, name):
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
    
    p = subprocess.call(['mongoimport', '-c', name + '.metadata', '--file', row_metadata_json_file, '--host', HOSTNAME ])
    p = subprocess.call(['mongoimport', '-c', name + '.samples', '--file', col_metadata_json_file, '--host', HOSTNAME ])
    p = subprocess.call(['mongoimport', '-c', name + '.sample_sets', '--file', sample_sets_json_file, '--host', HOSTNAME ])
    p = subprocess.call(['mongoimport', '-c', name + '.config', '--file', config_json_file, '--host', HOSTNAME ])
    p1 = subprocess.Popen(['zcat', results_json_file], stdout=subprocess.PIPE)
    p2 = subprocess.Popen(['mongoimport', '-c', name + '.reports', '--host', HOSTNAME ], stdin=p1.stdout)
    p1.wait()
    p2.wait()

def main(argv=None):
    '''Command line options.'''    
    # create instance of run configuration
    
    # Setup argument parser
    parser = argparse.ArgumentParser()
    # Add command line parameters
    parser.add_argument("input_dir", 
                        help="directory containing files to import into db")
    parser.add_argument("-n", "--name", dest = 'name',
                        default = 'ssea',
                        help = 'name for ssea run (will be name of collections in db)')
    # Process arguments
    args = parser.parse_args()
    # setup logging
    
    level = logging.INFO
    logging.basicConfig(level=level,
                        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    
    # import data into mongodb  
    db_import(args.input_dir, args.name)
    
    #function to delete all collections     
    #db_drop_colls(args.name)
    
    

    
    return 0

if __name__ == "__main__":
    sys.exit(main())