'''
Created on Nov 25, 2013

@author: mkiyer
'''
import os
import sys
import logging
import argparse
import glob

from ssea.base import Config
from ssea.algo import compute_global_stats_parallel, TMP_DIR

def main(argv=None):
    logging.basicConfig(level=logging.DEBUG,
                        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    # Setup argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument('input_dir')
    parser.add_argument('output_json_file')
    args = parser.parse_args()
    tmp_dir = os.path.join(args.input_dir, TMP_DIR)
    # divide matrix rows across processes
    worker_json_files = [glob.glob(os.path.join(tmp_dir, 'w*.json*'))]
    es_hists_file = os.path.join(args.input_dir, Config.OUTPUT_HISTS_FILE)
    compute_global_stats_parallel(es_hists_file, worker_json_files, args.output_json_file)
    return 0

if __name__ == "__main__":
    sys.exit(main())
