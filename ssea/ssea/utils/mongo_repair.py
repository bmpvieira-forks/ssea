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



def db_repair(name, host):
    colls = db_connect(name, host)
    sample_sets = colls['sample_sets']
    configs = colls['configs']
    results = colls['results']
    hists = colls['hists']
    merged = colls['merged']
    
    #check for incomplete imports
    logging.info('Removing incomplete imports')
    tmps = sample_sets.find({'name':'TMP'})
    tmp_ids = []
    for ss in tmps: 
        tmp_ids.append(ss['_id'])
    
    sample_sets.remove({'_id': {'$in': tmp_ids}})
    configs.remove({'_id': {'$in': tmp_ids}})
    hists.remove({'_id': {'$in': tmp_ids}})
    results.remove({'ss_id': {'$in': tmp_ids}})
    merged.remove({'ss_id': {'$in': tmp_ids}})
    
    

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
    
    db_repair(args.name, args.host)
    
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
