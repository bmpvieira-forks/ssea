'''
Created on Jan 30, 2014

@author: mkiyer
'''
import logging
import argparse
import os
import sys
import glob
from multiprocessing import Pool

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

def worker(args):    
    (input_path, matrix_dir, meta, fdr_thresholds) = args 
    bm = BigCountMatrix.open(matrix_dir)
    ss_compname = os.path.basename(input_path)
    results_file = os.path.join(input_path, Config.RESULTS_JSON_FILE)
    sigup = [set() for x in fdr_thresholds]
    sigdn = [set() for x in fdr_thresholds]
    # extract data
    n = 0
    for res in parse_results(results_file):
        n += 1
        if (n % 10000) == 0:
            logging.debug('%s parsed %d' % (ss_compname, n))
        transcript_id = bm.rownames[res.t_id]
        if (meta is not None) and (transcript_id not in meta):
            continue
        for i,fdr_threshold in enumerate(fdr_thresholds):
            if res.ss_fdr_q_value > fdr_threshold:
                continue                
            if res.ss_frac > 0:
                sigup[i].add(res.t_id)
            else:
                sigdn[i].add(res.t_id)
    bm.close()
    return (ss_compname, sigup, sigdn)

def stats_parallel(input_paths, matrix_dir, transcripts, prefix,
                   fdr_thresholds, num_processes):
    tasklist = []
    for input_path in input_paths:
        tasklist.append((input_path, matrix_dir, transcripts, fdr_thresholds))
    # create pool
    pool = Pool(processes=num_processes)
    result_iter = pool.imap_unordered(worker, tasklist)
    sigup = [set() for x in fdr_thresholds]
    sigdn = [set() for x in fdr_thresholds]
    sigall = [set() for x in fdr_thresholds]
    bm = BigCountMatrix.open(matrix_dir)
    nrows = len(bm.rownames)
    header_fields = ['ss_compname', 'dir', 'fdr', 'count']
    print '\t'.join(header_fields)
    for ss_compname, ss_sigup, ss_sigdn in result_iter:
        for i,fdr_threshold in enumerate(fdr_thresholds):
            fields = [ss_compname, 'up', '%.1e' % (fdr_threshold), len(ss_sigup[i])]
            print '\t'.join(map(str, fields))
            fields = [ss_compname, 'dn', '%.1e' % (fdr_threshold), len(ss_sigdn[i])]
            print '\t'.join(map(str, fields))
            ss_sigall = ss_sigup[i].union(ss_sigdn[i])
            fields = [ss_compname, 'both', '%.1e' % (fdr_threshold), len(ss_sigall)]
            print '\t'.join(map(str, fields))
            num_none = nrows - len(ss_sigall)
            fields = [ss_compname, 'none', '%.1e' % (fdr_threshold), num_none]
            print '\t'.join(map(str, fields))
            sigup[i].update(ss_sigup[i])
            sigdn[i].update(ss_sigdn[i])
            sigall[i].update(ss_sigall)
    pool.close()
    pool.join()
    # global stats
    for i,fdr_threshold in enumerate(fdr_thresholds):
        filename = prefix + '_%.1e_up' % (fdr_threshold)
        with open(filename, 'w') as f:
            sig_t_ids = [bm.rownames[x] for x in sorted(sigup[i])]
            print >>f, '\n'.join(sig_t_ids)
        filename = prefix + '_%.1e_dn' % (fdr_threshold)
        with open(filename, 'w') as f:
            sig_t_ids = [bm.rownames[x] for x in sorted(sigdn[i])]
            print >>f, '\n'.join(sig_t_ids)
        filename = prefix + '_%.1e_both' % (fdr_threshold)
        sig_t_ids = [bm.rownames[x] for x in sorted(sigall[i])]
        with open(filename, 'w') as f:
            print >>f, '\n'.join(sig_t_ids)
        filename = prefix + '_%.1e_none' % (fdr_threshold)
        with open(filename, 'w') as f:
            none_t_ids = set(bm.rownames).difference(sig_t_ids)
            print >>f, '\n'.join(none_t_ids)
    bm.close()

def main():
    logging.basicConfig(level=logging.DEBUG,
                        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument('--fdr', dest='fdr', type=float, action='append', default=[])
    parser.add_argument('--transcripts', dest='transcripts_file', default=None)
    parser.add_argument('-p', dest='num_processes', type=int, default=1)
    parser.add_argument('-i', dest='input_paths_file', default=None)
    parser.add_argument('matrix_dir')
    parser.add_argument('prefix')
    args = parser.parse_args()
    # get args
    matrix_dir = args.matrix_dir
    fdr_thresholds = args.fdr
    num_processes = args.num_processes
    transcripts_file = args.transcripts_file
    input_paths_file = args.input_paths_file
    prefix = args.prefix
    # check args
    fdr_thresholds = [max(0.0, min(x, 1.0)) for x in fdr_thresholds]
    fdr_thresholds = sorted(set(fdr_thresholds))
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
    transcripts = None
    if transcripts_file is not None:
        logging.debug('Parsing transcript list')
        transcripts = set()
        with open(transcripts_file) as f:
            for line in f:
                fields = line.strip().split('\t')
                transcripts.add(fields[0])
        logging.debug('Found %d transcripts' % (len(transcripts)))
    # parse results
    logging.debug('SSEA results: %d' % (len(input_paths)))
    logging.debug('FDR thresholds: %s' % (','.join(map(str,fdr_thresholds))))
    logging.debug('Num processes: %d' % (num_processes))
    stats_parallel(input_paths, matrix_dir, transcripts, prefix,
                   fdr_thresholds, num_processes)

if __name__ == '__main__':
    sys.exit(main())