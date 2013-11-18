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

#connects to the mongo db and returns a dictionary  
#containing each collection in the database


def db_connect(name, host):
    client = pymongo.MongoClient(host)
    db = client[name]
    row_metadata = db['metadata']
    col_metadata = db['samples']
    sample_sets = db['sample_sets']
    config = db['config']
    reports = db['reports']
    colls = {'row_meta':row_metadata, 'col_meta':col_metadata, 'ss':sample_sets, 'config':config, 'reports':reports}
    return colls

#removes all the collections in the database 'name'
def db_drop_colls(name, host):
    colls = db_connect(name, host)
    for key in colls.iterkeys():
        colls[key].drop()


def expression_import(name, host):
    _ssea_path = ssea.__path__[0]
    _expression_path = os.path.join(_ssea_path, 'expression.py')
    colls = db_connect(name, host)
    runconfig = Config.from_dict(colls['config'].find_one()) 
    dir = runconfig.matrix_dir
    #p1 = subprocess.call(['python', _expression_path, dir])
    p1 = subprocess.Popen(['python', _expression_path, dir], stdout=subprocess.PIPE)
    p2 = subprocess.Popen(['mongoimport', '-c', 'expression', '--host', host, '-d', name], stdin=p1.stdout)
    p1.wait()
    p2.wait()

def expression_drop(name, host):
    client= pymongo.MongoClient(host)
    db = client[name]
    coll = db.expression
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

def db_index_ssea(name, host):
    colls = db_connect(name, host)
    
    #index over transcript ID and gene ID for the row_meta collection
    row_meta = colls['row_meta']
    row_meta.ensure_index('name')
    row_meta.ensure_index('gene_id')
    
    #index over libraryID for the col_meta
    col_meta = colls['col_meta']
    col_meta.ensure_index('library_id')
    
    #index sample_sets over the sample set name
    sample_sets = colls['ss']
    sample_sets.ensure_index('name')

    #index reports with a compound index for t_id and ss_id. also for ss_id
    reports = colls['reports']
    reports.ensure_index([('t_id', pymongo.ASCENDING), 
                          ('ss_id', pymongo.ASCENDING)])
    reports.ensure_index('ss_id')

def db_index_expr(name, host):
    client= pymongo.MongoClient(host)
    db = client[name]
    coll = db.expression
    coll.ensure_index([('t_id', pymongo.ASCENDING) , 
                       ('s_id', pymongo.ASCENDING)])
    
    
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
    parser.add_argument("--host", dest = 'host',
                        default = 'mongodb://localhost:27017/',
                        help = 'name of mongodb server to connect to')
    parser.add_argument("-ds", "--delete", dest="delete", 
                        action="store_true", default=False, 
                        help="remove all ssea collections from current database")
    parser.add_argument("-e", "--expression", dest="expr", 
                        action="store_true", default=False, 
                        help="import expression matrix into database")
    parser.add_argument("-s", "--ssea", dest="colls", 
                        action="store_true", default=False, 
                        help="import all ssea output into database")
    parser.add_argument("-de", "--drop_expr", dest="de", 
                        action="store_true", default=False, 
                        help="delete expression data from database")
    parser.add_argument("-is", "--index_ssea", dest="index", 
                        action="store_true", default=False, 
                        help="create index for ssea collections")
    parser.add_argument("-ie", "--index_expr", dest="ie", 
                        action="store_true", default=False, 
                        help="create index for the expression matrix")
    # Process arguments
    args = parser.parse_args()
    # setup logging
    
    level = logging.INFO
    logging.basicConfig(level=level,
                        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    
    
    if args.delete == True:
        #function to delete all collections
        db_drop_colls(args.name, args.host)
    if args.colls == True:
        # import data into mongodb  
        db_import(args.input_dir, args.name, args.host)
    if args.expr == True: 
        expression_import(args.name, args.host)
    if args.de == True: 
        expression_drop(args.name, args.host)
    if args.index == True:
        db_index_ssea(args.name, args.host)
    if args.ie == True:  
        db_index_expr(args.name, args.host)
        
        
        
    return 0

if __name__ == "__main__":
    sys.exit(main())
