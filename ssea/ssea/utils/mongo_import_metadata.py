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



def db_connect(name, host):
    logging.info('connecting to %s database on mongo server: %s' % (name, host))
    client = pymongo.MongoClient(host)
    db = client[name]
    samples = db['samples']
    transcripts = db['transcripts']
    colls = {'samples':samples, 'transcripts':transcripts}
    return colls


def db_metadata_import(input_dir, name, host):
    logging.info('importing ssea data to %s database on mongo server: %s' % (name, host))

    
    logging.debug("Importing samples file")
    _ssea_path = ssea.__path__[0]
    _merge_path = os.path.join(_ssea_path, 'utils/mongo_metadata_printJSON.py')
    p1 = subprocess.Popen(['python', _merge_path, input_dir,'-c'], stdout=subprocess.PIPE)
    p2 = subprocess.Popen(['mongoimport', '-c', 'samples', '--host', host, '-d', name], stdin=p1.stdout)
    p1.wait()
    p2.wait()
    logging.info("Importing transcripts file")
    p1 = subprocess.Popen(['python', _merge_path, input_dir,'-r'], stdout=subprocess.PIPE)
    p2 = subprocess.Popen(['mongoimport', '-c', 'transcripts', '--host', host, '-d', name], stdin=p1.stdout)
    p1.wait()
    p2.wait()
    sample_index_fields = ['library_id',
                           'tcga_disease_type',
                           'assembly_cohort',
                           'fragment_length_mean',
                           'cohort',
                           'cancer_progression',
                           'tcga_legacy_id']
    
    transcript_index_fields = ['category',
                               'nearest_gene_names',
                               'gene_id',
                               'locus',
                               'name']
    
    colls = db_connect(name, host)
    samples = colls['samples']
    logging.info('indexing samples collection')
    for field in sample_index_fields:
        samples.ensure_index(field)
    
    transcripts = colls['transcripts']
    logging.info('indexing transcripts collection')
    for field in sample_index_fields:
        transcripts.ensure_index(field)
    
    
def main(argv=None):
    '''Command line options.'''    
    # Setup command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("input_dir", 
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
    
    db_metadata_import(args.input_dir, args.name, args.host)
    
    logging.info("Import finished")
    return 0

if __name__ == "__main__":
    sys.exit(main())
