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
from ssea.countdata import BigCountMatrix
import json
import decimal


def main(argv=None):
    '''Command line options.'''    
    # create instance of run configuration
    
    # Setup argument parser
    parser = argparse.ArgumentParser()
    # Add command line parameters
    parser.add_argument("matrix_dir", 
                        help="directory containing matrix file")
    parser.add_argument("-nr", "--rows", dest = 'num_rows',
                        default = 1,
                        help = 'number of rows in matrix to grab')
    parser.add_argument("-nc", "--cols", dest = 'num_cols',
                        default = 1,
                        help = 'number of rows in matrix to grab')
    parser.add_argument("-in", "--input_number", dest="in_num", 
                        action="store_true", default=False, 
                        help="import defined number of rows/cols")
    # Process arguments
    args = parser.parse_args()
    # setup logging
    
    level = logging.INFO
    logging.basicConfig(level=level,
                        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    
    
    bcm = BigCountMatrix.open(args.matrix_dir)
    rows = bcm.rownames
    cols = bcm.colnames
    matrix = bcm.counts
    
    if args.in_num == True:
        num_rows = int(args.num_rows)
        num_cols = int(args.num_cols)
               
        for row in xrange(num_rows):
            for col in xrange(num_cols):
                d = {'t_id': row,
                     's_id': col,
                     'value': float(matrix[row,col])}
                print json.dumps(d)
    else: 
        for row in xrange(len(rows)):
            for col in xrange(len(cols)):
                d = {'t_id': row,
                     's_id': col,
                     'value': float(matrix[row,col])}
                print json.dumps(d)
        
    

if __name__ == "__main__":
    sys.exit(main())
