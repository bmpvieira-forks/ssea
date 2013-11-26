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
    if not os.path.exists(args.input_dir):
        parser.error('input directory %s not found' % (args.input_dir))
    if os.path.exists(args.output_json_file):
        parser.error('output json file %s exists' % (args.output_json_file))
    input_dir = args.input_dir
    tmp_dir = os.path.join(args.input_dir, TMP_DIR)
    if not os.path.exists(tmp_dir):
        parser.error('temp directory %s not found within input directory %s' % (tmp_dir, input_dir))
    # divide matrix rows across processes
    es_hists_file = os.path.join(input_dir, Config.OUTPUT_HISTS_FILE)
    worker_json_stats_files = []
    worker_json_files = glob.glob(os.path.join(tmp_dir, '*.json'))
    for i in xrange(len(worker_json_files)):
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
