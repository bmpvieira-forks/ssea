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
from ssea.base import Config
from ssea.algo import ES_BINS_NEG, ES_BINS_POS
import ssea
import collections
import numpy as np

#connects to the mongo db and returns a dictionary  
#containing each collection in the database

_package_dir = ssea.__path__[0]

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
    hists_file = os.path.join(input_dir, 
                                    Config.OUTPUT_HISTS_FILE)
    
    logging.debug("Importing row metadata file %s" % (row_metadata_json_file))
    p = subprocess.call(['mongoimport', 
                         '--host', host, 
                         '-d', name, 
                         '-c', 'metadata', 
                         '--file', row_metadata_json_file])
    logging.debug("Importing column metadata file %s" % (col_metadata_json_file))
    p = subprocess.call(['mongoimport', '-c', 'samples', '--file', col_metadata_json_file, '--host', host, '-d', name])
    logging.debug("Importing sample sets file %s" % (sample_sets_json_file))
    p = subprocess.call(['mongoimport', '-c', 'sample_sets', '--file', sample_sets_json_file, '--host', host,  '-d', name])
    logging.debug("Importing config file %s" % (config_json_file))
    p = subprocess.call(['mongoimport', '-c', 'config', '--file', config_json_file, '--host', host,  '-d', name])
    logging.debug("Importing result file %s" % (results_json_file))
    p1 = subprocess.Popen(['zcat', results_json_file], stdout=subprocess.PIPE)
    p2 = subprocess.Popen(['mongoimport', '-c', 'reports', '--host', host, '-d', name], stdin=p1.stdout)
    p1.wait()
    p2.wait()
    
    #add the histogram data to the sample set collection
    client = pymongo.MongoClient(host)
    db = client[name]
    coll = db['sample_sets']
    x = np.load(hists_file)
    ss_hist = {}
    hist_fields = ['obs_es_neg', 
                   'obs_es_pos', 
                   'null_es_neg',
                   'null_es_pos']
    for i in xrange(coll.count()):
        for field in hist_fields: 
            ss_hist[field] =  list(x[field][i])
        ss_hist['es_bins_neg'] = list(ES_BINS_NEG)
        ss_hist['es_bins_pos'] = list(ES_BINS_POS)
        spec = {'_id':i}
        coll.update(spec, {'$set': ss_hist})
    
    logging.debug("Done")
    
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
    

def collection_merger(name, host):
    '''
     Row meta options: 
        category
        distance
        annotation_sources
        nearest_gene_names
        nearest_gene_ids
        transcript_length
        gene_id
        locus
        tss_id
        num_exons
        _id
        strand
        name
    '''
    '''
      Reports options: 
            ss_id
            t_id
            null_es_vals
            global_nes
            es_rank
            ss_fdr_q_value
            null_es_hist
            fdr_q_value
            fwer_p_value
            odds_ratio
            es
            nes
            core_hits
            null_misses
            rand_seed
            nominal_p_value
            global_fdr_q_value
            resample_es_vals
            fisher_p_value
            null_es_ranks
            null_hits
            core_misses
            ss_nes
            row_id
            resample_es_ranks
    '''
    
    #create a dictionary with key as _id for transcripts and value containing all the fields for row_meta
    trans_dict = collections.defaultdict(lambda: {})
    
    #list of fields to add to the trans_meta dictionary
    fields_trans = ['category', 
              'nearest_gene_names', 
              'num_exons', 
              'strand',
              'name']
    
    colls = db_connect(name, host)
    transcripts = colls['row_meta']
    
    #parse through the transcript collection and create dictionary
    for x in transcripts.find():
        key = x['_id']
        #create a dict placeholder for this _id element
        id_dict = {}
        for field in fields_trans: 
            id_dict[field] = x[field]
        trans_dict[key] = id_dict
    
    
    reports = colls['reports']
    #list of fields from the reports to use in the combined database
    fields_reports = ['t_id',
              'ss_id',
              'global_nes',
              'ss_fdr_q_value',
              'fdr_q_value',
              'es',
              'nes',
              'global_fdr_q_value',
              'ss_nes']
    #reconnect to database in order to create and add to new collection
    client = pymongo.MongoClient(host)
    db = client[name]
    merged = db.merged
    for x in reports.find():
        #create another dict placeholder to be added to the new collection
        dict = {}
        for field in fields_reports: 
            dict[field] = x[field]
        id = x['t_id']
        trans_meta = trans_dict[id]
        
        for key in trans_meta.iterkeys():
            dict[key] = trans_meta[key]
        merged.insert(dict)
    
    
    #create indexes over all the fields
    for field in fields_trans: 
        merged.ensure_index(field)
    for field in fields_reports:
        merged.ensure_index(field)
     


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
                        default = 'localhost:27017',
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
    parser.add_argument("-m", "--merge", dest="m", 
                        action="store_true", default=False, 
                        help="create merged collection for making datatables")
    # Process arguments
    args = parser.parse_args()
    # setup logging
    
    level = logging.DEBUG
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
    if args.m == True: 
        collection_merger(args.name, args.host)
        
    return 0

if __name__ == "__main__":
    sys.exit(main())
