'''
Created on Dec 11, 2013

@author: mkiyer
'''
import argparse
import logging
import os
import sys
import itertools
import subprocess
import json

from ssea.lib.countdata import BigCountMatrix
                
class Metadata(object):
    __slots__ = ('_id', 'name', 'params')
    
    def __init__(self, _id=None, name=None, params=None):
        '''
        _id: unique integer id
        name: string name
        params: dictionary of parameter-value data
        '''
        self._id = _id
        self.name = name
        self.params = {}
        if params is not None:
            self.params.update(params)

    def __repr__(self):
        return ("<%s(_id=%d,name=%s,params=%s>" % 
                (self.__class__.__name__, self._id, self.name, 
                 self.params))
    def __eq__(self, other):
        return self._id == other._id
    def __ne__(self, other):
        return self._id != other._id
    def __hash__(self):
        return hash(self._id)

    def to_json(self):
        d = dict(self.params)
        d.update({'_id': self._id,
                  'name': self.name})
        return json.dumps(d)
    
    @staticmethod
    def from_json(s):
        d = json.loads(s)
        m = Metadata()
        m._id = d.pop('_id')
        m.name = d.pop('name')
        m.params = d
        return m
    
    @staticmethod
    def from_dict(d):
        m = Metadata()
        m._id = d.pop('_id')
        m.name = d.pop('name')
        m.params = d
        return m
    
    @staticmethod
    def parse_json(filename):
        with open(filename, 'r') as fp:
            for line in fp:
                yield Metadata.from_json(line.strip())

    @staticmethod
    def parse_tsv(filename, names, id_iter=None):
        '''
        parse tab-delimited file containing sample information        
        first row contains column headers
        first column must contain sample name
        remaining columns contain metadata
        '''
        if id_iter is None:
            id_iter = itertools.count()
        # read entire metadata file
        metadict = {}
        with open(filename) as fileh:
            header_fields = fileh.next().strip().split('\t')[1:]
            for line in fileh:
                fields = line.strip().split('\t')
                name = fields[0]            
                metadict[name] = fields[1:]
        # join with names
        for name in names:
            if name not in metadict:
                logging.error("Name %s not found in metadata" % (name))
            assert name in metadict
            fields = metadict[name]
            metadata = dict(zip(header_fields,fields))
            yield Metadata(id_iter.next(), name, metadata)  
            
def main():
    parser = argparse.ArgumentParser()            
#     parser.add_argument('--colmeta', dest='col_metadata_file',
#                         help='file containing metadata corresponding to each '
#                         'column of the weight matrix file')
#     parser.add_argument('--rowmeta', dest='row_metadata_file',
#                         help='file containing metadata corresponding to each '
#                         'row of the weight matrix file')
    parser.add_argument("-r", dest="row", 
                        action="store_true", default=False, 
                        help="Print row_meta JSONs")
    parser.add_argument("-c", dest="col", 
                        action="store_true", default=False, 
                        help="Print col_meta JSONs")
    parser.add_argument('matrix_dir')
    args = parser.parse_args()
    # check command line args
    matrix_dir = os.path.abspath(args.matrix_dir)
    col_metadata_file = os.path.join(matrix_dir, 'colmeta.tsv')
    row_metadata_file = os.path.join(matrix_dir, 'rowmeta.tsv')
    
    if not os.path.exists(col_metadata_file):
        parser.error("Column metadata file '%s' not found" % (args.col_metadata_file))
    if not os.path.exists(row_metadata_file):
        parser.error("Row metadata file '%s' not found" % (args.row_metadata_file))
    if not os.path.exists(args.matrix_dir):
        parser.error('matrix path "%s" not found' % (args.matrix_dir))

#     col_metadata_file = os.path.abspath(args.col_metadata_file)
#     row_metadata_file = os.path.abspath(args.row_metadata_file)
    # open matrix
    bm = BigCountMatrix.open(matrix_dir)
    if bm.size_factors is None:
        parser.error("Size factors not found in count matrix")
    # read metadata
    logging.info("Reading row metadata")
    row_metadata = list(Metadata.parse_tsv(row_metadata_file, bm.rownames))
    logging.info("Reading column metadata")
    col_metadata = list(Metadata.parse_tsv(col_metadata_file, bm.colnames))
    # pipe row metadata into mongoimport 
    if args.row:
        logging.debug("Importing row metadata")
        for m in row_metadata:
            print >>sys.stdout, m.to_json()
    if args.col:
        logging.debug("Importing column metadata")
        for m in col_metadata:
            print >>sys.stdout, m.to_json()
    # cleanup
    bm.close()


if __name__ == "__main__":
    sys.exit(main())
