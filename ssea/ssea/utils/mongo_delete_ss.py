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



def db_delete_ss(name, host, ss_id):
    colls = db_connect(name, host)
    sample_sets = colls['sample_sets']
    configs = colls['configs']
    results = colls['results']
    hists = colls['hists']
    merged = colls['merged']
    
    ss_name = sample_sets.find_one({'_id':ss_id})['name']
    
    #check for incomplete imports
    logging.info('Removing sample set \'%s\'' % ss_name)
    
    sample_sets.remove({'_id': ss_id})
    configs.remove({'_id': ss_id})
    hists.remove({'_id': ss_id})
    results.remove({'ss_id': ss_id})
    merged.remove({'ss_id': ss_id})
    
    

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
    parser.add_argument("--ss_id", dest = 'ss_id',
                        help = 'Sample set ID to drop')
    args = parser.parse_args()
    level = logging.DEBUG
    logging.basicConfig(level=level,
                        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    
    db_delete_ss(args.name, args.host, int(args.ss_id))
    
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
