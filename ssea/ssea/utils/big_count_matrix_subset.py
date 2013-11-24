'''
Created on Nov 5, 2013

@author: mkiyer
'''
import os
import logging
import argparse
import numpy as np

from ssea.countdata import BigCountMatrix

def find_transcripts_to_skip(input_dir, skip_nan, skip_zero):
    '''
    find transcripts with 'nan' or 0.0 values
    '''
    skip_rownames = []
    bm = BigCountMatrix.open(input_dir)
    for i in xrange(bm.shape[0]):
        a = np.array(bm.counts[i,:], dtype=np.float)
        skip = False
        if skip_nan:
            num_finite = np.isfinite(a).sum()
            if num_finite == 0:                
                logging.debug('Row %d t_id %s all nan' % (i, bm.rownames[i]))
                skip = True
        if skip_zero:
            num_nonzero = (a > 0).sum()
            if num_nonzero == 0:
                logging.debug('Row %d t_id %s all zeros' % (i, bm.rownames[i]))
                skip = True
        if skip:
            skip_rownames.append(bm.rownames[i])
    bm.close()
    logging.debug('Found %d rows to skip' % (len(skip_rownames)))
    return skip_rownames
            
def main():
    logging.basicConfig(level=logging.DEBUG,
                      format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument('--skip-all-nan', dest='skip_nan', action='store_true')
    parser.add_argument('--skip-all-zero', dest='skip_zero', action='store_true')
    parser.add_argument('--libs', dest="library_ids", default=None)
    parser.add_argument('--transcripts', dest="transcript_ids", default=None)
    parser.add_argument('input_dir')
    parser.add_argument('output_dir')
    args = parser.parse_args()
    if not os.path.exists(args.input_dir):
        parser.error("input directory '%s' not found" % (args.input_dir))
    if os.path.exists(args.output_dir):
        parser.error("output directory '%s' already exists" % (args.output_dir))
    # open matrix
    bm = BigCountMatrix.open(args.input_dir)
    # get library and transcript ids
    if args.library_ids is not None:
        library_ids = set([line.strip() for line in open(args.library_ids)])
    else:
        library_ids = set()    
    if args.transcript_ids is not None:
        transcript_ids = set([line.strip() for line in open(args.transcript_ids)])
    else:
        transcript_ids = set(bm.rownames)
    if args.skip_nan or args.skip_zero:
        logging.debug('Checking matrix for rows of all zero and/or nan')
        skip_ids = set(find_transcripts_to_skip(args.input_dir, 
                                                args.skip_nan, 
                                                args.skip_zero))
        transcript_ids.difference_update(skip_ids)
    logging.debug('Creating subset with %d transcripts' % (len(transcript_ids)))
    bm.copy(args.output_dir, transcript_ids, library_ids)
    bm.close()
    
if __name__ == '__main__':
    main()
