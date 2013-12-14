'''
Created on Dec 11, 2013

@author: mkiyer
'''
import argparse
import logging
import os
import sys
import itertools
import subprocess
import json

from ssea.lib.countdata import BigCountMatrix

            
def main():
    parser = argparse.ArgumentParser()            
    parser.add_argument('--colmeta', dest='col_metadata_file',
                        help='file containing metadata corresponding to each '
                        'column of the weight matrix file')
    parser.add_argument('--rowmeta', dest='row_metadata_file',
                        help='file containing metadata corresponding to each '
                        'row of the weight matrix file')
    parser.add_argument('matrix_dir')
    args = parser.parse_args()
    # check command line args
    if not os.path.exists(args.col_metadata_file):
        parser.error("Column metadata file '%s' not found" % (args.col_metadata_file))
    if not os.path.exists(args.row_metadata_file):
        parser.error("Row metadata file '%s' not found" % (args.row_metadata_file))
    if not os.path.exists(args.matrix_dir):
        parser.error('matrix path "%s" not found' % (args.matrix_dir))
    matrix_dir = os.path.abspath(args.matrix_dir)
    col_metadata_file = os.path.abspath(args.col_metadata_file)
    row_metadata_file = os.path.abspath(args.row_metadata_file)
    # open matrix
    bm = BigCountMatrix.open(matrix_dir)
    if bm.size_factors is None:
        parser.error("Size factors not found in count matrix")
    # read metadata
    logging.info("Reading row metadata")
    row_metadata = list(Metadata.parse_tsv(row_metadata_file, bm.rownames))
    logging.info("Reading column metadata")
    col_metadata = list(Metadata.parse_tsv(col_metadata_file, bm.colnames))
    # pipe row metadata into mongoimport 
    logging.debug("Importing row metadata")
    for m in row_metadata:
        print >>sys.stdout, m.to_json()
    logging.debug("Importing column metadata")
    for m in col_metadata:
        print >>sys.stdout, m.to_json()
    # cleanup
    bm.close()


if __name__ == "__main__":
    sys.exit(main())
