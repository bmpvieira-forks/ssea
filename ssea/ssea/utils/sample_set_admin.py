'''
Created on Dec 20, 2013

@author: mkiyer
'''
import argparse
import logging
import sys
import os

from ssea.lib.base import SampleSet

def to_json(args):
    filename = args.sample_set_file
    sep = args.sep
    if not os.path.exists(filename):
        logging.error("Sample set file '%s' not found" % (filename))
        return 1
    ext = os.path.splitext(filename)[-1]
    if ext == '.smx':        
        for ss in SampleSet.parse_smx(filename, sep):
            print ss.to_json()
    elif ext == '.smt':
        for ss in SampleSet.parse_smt(filename, sep):
            print ss.to_json()

def main():
    logging.basicConfig(level=logging.DEBUG,
                        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()
    # repair runs that terminated prematurely
    parser_json = subparsers.add_parser('tojson')
    parser_json.add_argument('--sep', dest='sep', default='\t') 
    parser_json.add_argument('sample_set_file') 
    parser_json.set_defaults(func=to_json)
    args = parser.parse_args()
    return args.func(args)
 
if __name__ == '__main__':
    sys.exit(main())
