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
from ssea.utils.mongo import db_connect, fields_trans, fields_results

#connects to the mongo db and returns a dictionary  
#containing each collection in the database

_package_dir = ssea.__path__[0]


def db_index(name, host):
    
    colls = db_connect(name, host)
    print colls
    #index sample_sets collection
    logging.info('indexing sample_sets collection')
    ss = colls['sample_sets']
    ss.ensure_index('name')
    
    #indexing results collection
    logging.info('indexing results collection')
    results = colls['results']
    
    results.ensure_index([('t_id', pymongo.ASCENDING), 
                          ('ss_id', pymongo.ASCENDING)])
    results.ensure_index([('ss_id', pymongo.ASCENDING), 
                          ('nes', pymongo.ASCENDING)])
    
    #create indexes for the merged collection
    #create compound index for ss_id + each other field in reports
    client = pymongo.MongoClient(host)
    db = client[name]
    merged = db.merged
    for field in fields_results:
        if field != 'ss_id':
            merged.ensure_index([('ss_id', pymongo.ASCENDING),
                                 (field, pymongo.ASCENDING)])
            #create three item compound index to be used when a category filter is applied
            merged.ensure_index([('ss_id', pymongo.ASCENDING),
                                ('category', pymongo.ASCENDING),
                                (field, pymongo.ASCENDING)])
            #create indexes for regex queries 
            merged.ensure_index([('ss_id', pymongo.ASCENDING),
                                (field, pymongo.ASCENDING),
                                ('nearest_gene_names', pymongo.ASCENDING)])
    fields_trans.append('loc_strand')
    for field in fields_trans:
        if field != 'category': 
            merged.ensure_index([('ss_id', pymongo.ASCENDING),
                                 (field, pymongo.ASCENDING)])
            #create three item compound index to be used when a category filter is applied
            merged.ensure_index([('ss_id', pymongo.ASCENDING),
                                ('category', pymongo.ASCENDING),
                                (field, pymongo.ASCENDING)])
            #create indexes for regex queries 
            merged.ensure_index([('ss_id', pymongo.ASCENDING),
                                (field, pymongo.ASCENDING),
                                ('nearest_gene_names', pymongo.ASCENDING)])
    
    
def main(argv=None):
    '''Command line options.'''    
    # Setup command line arguments
    parser = argparse.ArgumentParser()
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
    
    db_index(args.name, args.host)
    
    logging.info("Index finished")
    return 0

if __name__ == "__main__":
    sys.exit(main())
