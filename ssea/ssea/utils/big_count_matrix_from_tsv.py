'''
Created on Dec 11, 2013

@author: mkiyer
'''
import argparse
import logging
import sys
import os

from ssea.lib.countdata import BigCountMatrix

DEFAULT_NA_VALUE = 'NA'

def main():
    logging.basicConfig(level=logging.DEBUG,
                        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument('--na-value', dest='matrix_na_values', 
                        default=[DEFAULT_NA_VALUE], action='append',
                        help='Value to interpret as missing/invalid '
                        'in weight matrix [default=%(default)s]')    
    parser.add_argument('input_tsv_file')
    parser.add_argument('output_dir')
    # parse args
    args = parser.parse_args()    
    input_tsv_file = args.input_tsv_file
    output_dir = args.output_dir
    matrix_na_values = args.matrix_na_values
    # check args
    if not os.path.exists(input_tsv_file):
        parser.error('Input file "%s" not found' % (input_tsv_file))
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    # convert matrix 
    logging.info("Converting text matrix file to binary format")
    bm = BigCountMatrix.from_tsv(input_tsv_file, output_dir, 
                                 na_values=matrix_na_values)
    logging.info("Estimating size factors")
    bm.estimate_size_factors('deseq')
    bm.close()

if __name__ == '__main__':
    sys.exit(main())
