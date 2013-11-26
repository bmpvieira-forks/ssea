'''
Created on Nov 25, 2013

@author: mkiyer
'''
import os
import sys
import logging
import argparse
import glob
import subprocess

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
    es_hists_file = os.path.join(args.input_dir, Config.OUTPUT_HISTS_FILE)
    worker_json_files = []
    worker_json_stats_files = []
    for i,filename in glob.glob(os.path.join(tmp_dir, '*.json')):
        worker_json_files.append(filename)
        worker_json_stats_files.append(os.path.join(tmp_dir, 'w%03d.stats.json' % (i)))
    compute_global_stats_parallel(es_hists_file, worker_json_files, worker_json_stats_files)
    # run a shell 'cat' command because it is fast
    args = ['cat']
    args.extend(worker_json_stats_files)
    args.append('>')
    args.append(args.output_json_file)
    cmd = ' '.join(args)
    retcode = subprocess.call(cmd, shell=True)
    if retcode != 0:
        logging.error('Error concatenating worker json files')    
    return 0

if __name__ == "__main__":
    sys.exit(main())
