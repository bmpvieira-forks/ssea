'''
Created on Apr 2, 2014

@author: mkiyer
'''
import logging
import argparse
import sys
import os
import collections
import numpy as np

MAD_CONSTANT = 1.0
MAD_SCALE_FACTOR = 1.0 / 0.6745

# local imports
from ssea.lib.countdata import BigCountMatrix

def parse_table(filename):
    d = {}
    with open(filename) as f:
        header_fields = f.next().strip().split('\t')
        for line in f:
            fields = line.strip().split('\t')
            d[fields[0]] = dict(zip(header_fields, fields))
    return d

def parse_library_table(filename, names):
    namedict = dict((name,i) for i,name in enumerate(names))
    col_inds = []
    with open(filename) as f:
        header_fields = f.next().strip().split('\t')
        for line in f:
            fields = line.strip().split('\t')
            d = dict(zip(header_fields, fields))
            if d['library_id'] not in namedict:
                logging.warning('Library %s not found in matrix' % (d['library_id']))
                continue
            i = namedict[d['library_id']]
            col_inds.append(i)
    return col_inds

def parse_transcript_table(filename, names):
    namedict = dict((name,i) for i,name in enumerate(names))
    row_inds = []
    lengths = []
    with open(filename) as f:
        header_fields = f.next().strip().split('\t')
        for line in f:
            fields = line.strip().split('\t')
            d = dict(zip(header_fields, fields))
            t_id = d['transcript_id']
            if t_id not in namedict:
                logging.warning('Transcript %s not in matrix' % (t_id))
                continue
            i = namedict[t_id]
            length = int(d['transcript_length']) / 1000.0
            row_inds.append(i)
            lengths.append(length)
    return row_inds, lengths

def main():
    logging.basicConfig(level=logging.DEBUG,
                        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument('--colmeta', dest='col_metadata_file', default=None)
    parser.add_argument('--rowmeta', dest='row_metadata_file', default=None)
    parser.add_argument('matrix_dir')
    args = parser.parse_args()
    col_metadata_file = args.col_metadata_file
    row_metadata_file = args.row_metadata_file
    matrix_dir = args.matrix_dir
    if not os.path.exists(matrix_dir):
        parser.error('matrix directory "%s" not found' % (matrix_dir))
    # open matrix
    bm = BigCountMatrix.open(matrix_dir)
    # get pheno data
    logging.debug('Reading library metadata')
    col_inds = parse_library_table(col_metadata_file, bm.colnames)
    # get transcript metadata
    logging.debug('Reading transcript metadata')
    row_inds, lengths = parse_transcript_table(row_metadata_file, bm.rownames)
    # get total counts per library
#    logging.debug('Getting total fragments per library')
#    lib_sizes = np.empty(len(col_inds), dtype=np.float)
#    for j in xrange(len(col_inds)):
#        a = bm.counts_t[j,:]
#        a = a[np.isfinite(a)]
#        lib_sizes[j] = a.sum()
#    lib_sizes /= 1.0e6
    # normalize
    logging.debug('Normalizing and summarizing counts per transcript')
    header_fields = ['transcript_id']
    header_fields.extend([bm.colnames[x] for x in col_inds])
    print '\t'.join(header_fields)
    for i,lengthkb in zip(row_inds,lengths):
        t_id = bm.rownames[i]
        counts = bm.counts[i, col_inds]
        # ignore nans
        valid_inds = np.isfinite(counts)
        # normalize counts
        a = (counts[valid_inds] / bm.size_factors[valid_inds])
        # log transform
        a = np.log2(a + 1.0)
        # z-score
        #mean = np.mean(a)
        #std = np.std(a)
        #a = (a - mean) / std
        # subtract median and divide by MAD
        med = np.median(a)
        mad = MAD_CONSTANT + (np.median(np.abs(a - med)) * MAD_SCALE_FACTOR)
        a = (a - med) / mad
        # write
        out = np.empty(len(col_inds), dtype=np.float)
        out[:] = np.nan
        out[valid_inds] = a
        fields = [t_id]
        fields.extend(map(str,out))
        print '\t'.join(fields)
        continue
        #a = (counts[valid_inds] / bm.size_factors[valid_inds])
        #a = (counts[valid_inds] / lib_sizes[valid_inds]) / lengthkb
        # center and scale
        med = np.median(a)
        mad = med + (np.median(np.abs(a - med)) * MAD_SCALE_FACTOR)
        #mad = MAD_CONSTANT + (np.median(np.abs(a - med)) * MAD_SCALE_FACTOR)
        a = (a - med) / mad
        # log transform
        a = np.sign(a) * np.log2(np.abs(a)+1.0)
        # normalize by transcript length
        #a -= np.log2(lengthkb)
        # output result
        out = np.empty(len(col_inds), dtype=np.float)
        out[:] = np.nan
        out[valid_inds] = a
        fields = [t_id]
        fields.extend(map(str,out))
        print '\t'.join(fields)
        
        #normcounts = (counts[valid_inds] / lib_sizes[valid_inds]) / length
        #normlogcounts = np.log2(normcounts + 1)
        # subtracting global median and dividing by the median absolute deviation
        #med = np.median(normlogcounts)
        #mad = MAD_CONSTANT + (np.median(np.abs(normlogcounts - med)) * MAD_SCALE_FACTOR)
        #normlogcounts = (normlogcounts - med) / mad
        # output final matrix
        #a = np.empty(len(col_inds), dtype=np.float)
        #a[:] = np.nan
        #a[valid_inds] = normlogcounts
        #fields = [t_id]
        #fields.extend(map(str,a))
        #print '\t'.join(fields)
    bm.close()    
    return 0

if __name__ == '__main__':
    sys.exit(main())