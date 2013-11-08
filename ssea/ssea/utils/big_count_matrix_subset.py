'''
Created on Nov 5, 2013

@author: mkiyer
'''
import logging
import argparse

from ssea.countdata import BigCountMatrix

def main():
    logging.basicConfig(level=logging.DEBUG,
                      format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument('--libs', dest="library_ids", default=None)
    parser.add_argument('--transcripts', dest="transcript_ids", default=None)
    parser.add_argument('input_dir')
    parser.add_argument('output_dir')
    args = parser.parse_args()
    # get library and transcript ids
    if args.library_ids is not None:
        library_ids = [line.strip() for line in open(args.library_ids)]
    else:
        library_ids = None    
    if args.transcript_ids is not None:
        transcript_ids = [line.strip() for line in open(args.transcript_ids)]
    else:
        transcript_ids = None
    # open matrix
    bm = BigCountMatrix.open(args.input_dir)
    bm.copy(args.output_dir, transcript_ids, library_ids)

if __name__ == '__main__':
    main()
