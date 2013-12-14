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
import json
import collections
from mongo import db_connect, fields_trans, fields_results


def main(argv=None):
    '''Command line options.'''    
    # create instance of run configuration
    
    # Setup argument parser
    parser = argparse.ArgumentParser()
    # Add command line parameters
    parser.add_argument("-n", "--name", dest = 'name',
                        default = 'compendia',
                        help = 'name for ssea run (will be name of database)')
    parser.add_argument("--host", dest = 'host',
                        default = 'localhost:27017',
                        help = 'name of mongodb server to connect to')
    
    # Process arguments
    args = parser.parse_args()
    # setup logging
    
    level = logging.DEBUG
    logging.basicConfig(level=level,
                        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    
    
    colls = db_connect(args.name, args.host)
    transcripts = colls['transcripts']
    results = colls['results']

    
    #parse through transcript metadata and build dict to be used during merge
    trans_dict = collections.defaultdict(lambda: {})
    logging.info('Parsing through transcript metadata to prepare merge')
    tot = transcripts.find().count()
    i = 0
    for x in transcripts.find():
        i+=1
        if (i % 25000) == 0:
            logging.debug('Finished %d/%d' % (i, tot))
            
        key = x['_id']
        #create a dict placeholder for this _id element
        id_dict = {}
        for field in fields_trans:
            id_dict[field] = x[field]
        #create a combined locus and strand field
        locus = x['locus']
        strand = x['strand']
        new_loc = locus + '(' + strand + ')'
        id_dict['loc_strand'] = new_loc
        trans_dict[key] = id_dict
    
    #print merged json
    tot = results.find().count()
    logging.info('Merging transcript metadata and results fields (%d total merged documents' % tot)
    fields_results.append('_id')
    for x in results.find():
        #create another dict placeholder to be printed as JSON
        dict = {}
        for field in fields_results: 
            dict[field] = str(x[field])
        id = x['t_id']
        trans_meta = trans_dict[id]
        
        for key in trans_meta.iterkeys():
            dict[key] = trans_meta[key]
        print json.dumps(dict)
    
           
    

if __name__ == "__main__":
    sys.exit(main())
