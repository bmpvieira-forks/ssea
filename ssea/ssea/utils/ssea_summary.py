'''
Created on Jan 30, 2014

@author: mkiyer
'''
import logging
import argparse
import os
import sys
import glob
import collections
import numpy as np

# local imports
from ssea.lib.config import Config 
from ssea.lib.countdata import BigCountMatrix
from ssea.lib.base import Result, JobStatus

def check_path(path):
    if not os.path.exists(path):
        logging.error('Directory not found at path "%s"' % (path))
        return False
    elif not os.path.isdir(path):
        logging.error('Directory not valid at path "%s"' % (path))
        return False
    elif JobStatus.get(path) != JobStatus.DONE:
        logging.info('Result status not set to "DONE" at path "%s"' % (path))
        return False
    return True

def find_paths(input_dirs):
    paths = set()
    for input_dir in input_dirs:
        if not os.path.exists(input_dir):
            logging.error("input dir '%s' not found" % (input_dir))
        if not os.path.isdir(input_dir):
            logging.error("input dir '%s' not a valid directory" % (input_dir))
        input_dir = os.path.abspath(input_dir)
        for path in glob.iglob(os.path.join(input_dir, "*")):
            if os.path.exists(path) and os.path.isdir(path):
                if check_path(path):
                    paths.add(path)
    return paths

def parse_results(filename):
    with open(filename, 'r') as fp:
        for line in fp:
            result = Result.from_json(line.strip())
            yield result

def main():
    logging.basicConfig(level=logging.DEBUG,
                        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument('--fdr', dest='fdr', type=float, default=1.0)
    parser.add_argument('-a', '--attr', dest='ssea_attrs', action='append', default=[])
    parser.add_argument("-i", dest="input_paths_file", default=None)
    parser.add_argument('matrix_dir')
    args = parser.parse_args()
    # get args
    fdr_threshold = args.fdr
    matrix_dir = args.matrix_dir
    ssea_attrs = args.ssea_attrs
    input_paths_file = args.input_paths_file
    # check args
    if len(ssea_attrs) == 0:
        parser.error('Please specify one or more attributes using "-a" or "--attr"')
    input_paths = []
    if input_paths_file is None:
        logging.error('No input directories specified (use -i).. Exiting.')
        return 1
    if not os.path.exists(input_paths_file):
        logging.error('Input paths file "%s" not found' % (input_paths_file))
    else:
        with open(input_paths_file) as fileh:
            for line in fileh:
                path = line.strip()
                if path in input_paths:
                    continue
                if check_path(path):
                    input_paths.append(path)
    if len(input_paths) == 0:
        logging.error('No valid SSEA results directories found.. Exiting.')
        return 1
    # parse results
    logging.debug('SSEA results: %d' % (len(input_paths)))
    logging.debug('SSEA attributes: %s' % (','.join(ssea_attrs)))
    logging.debug('SSEA FDR threshold: %f' % fdr_threshold)
    bm = BigCountMatrix.open(matrix_dir)
    dat = collections.defaultdict(lambda: collections.defaultdict(lambda: []))
    for input_path in input_paths:
        logging.debug('Parsing path %s' % (input_path))
        results_file = os.path.join(input_path, Config.RESULTS_JSON_FILE)
        # extract data
        i = 0
        sig = 0
        for res in parse_results(results_file):
            # logging
            i += 1
            if (i % 10000) == 0:
                logging.debug('Parsed %d results' % (i))
            if res.ss_fdr_q_value > fdr_threshold:
                continue
            sig += 1
            transcript_id = bm.rownames[res.t_id]
            for a in ssea_attrs:
                dat[a][transcript_id].append(getattr(res, a))
        logging.debug('Found %d results for path %s (%d significant)' % (i, input_path, sig))
    bm.close()
    # output results
    header_fields = ['transcript_id', 'attr', 'min', 'max', 'absmax', 'mean', 'median']
    print '\t'.join(header_fields)
    for a in ssea_attrs:
        attrdict = dat[a]
        for transcript_id in sorted(attrdict):
            arr = np.array(attrdict[transcript_id])
            fields = [transcript_id, a, np.min(arr), np.max(arr), np.max(np.abs(arr)), np.median(arr), np.mean(arr)]
            print '\t'.join(map(str, fields))

if __name__ == '__main__':
    sys.exit(main())