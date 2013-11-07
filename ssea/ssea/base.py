'''
Created on Oct 18, 2013

@author: mkiyer
'''
import os
import argparse
import logging
import json
from datetime import datetime
from time import time
import itertools
import numpy as np

BOOL_DTYPE = np.uint8
FLOAT_DTYPE = np.float
WEIGHT_METHODS = ['unweighted', 'weighted', 'log']

class NumpyJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist() # or map(int, obj)
        return json.JSONEncoder.default(self, obj)

def timestamp():
    return datetime.fromtimestamp(time()).strftime('%Y-%m-%d-%H-%M-%S-%f')

def quantile(a, frac, limit=(), interpolation_method='fraction'):
    '''copied verbatim from scipy code (scipy.org)'''
    def _interpolate(a, b, fraction):
        return a + (b - a)*fraction;
    values = np.sort(a, axis=0)
    if limit:
        values = values[(limit[0] <= values) & (values <= limit[1])]

    idx = frac * (values.shape[0] - 1)
    if (idx % 1 == 0):
        score = values[idx]
    else:
        if interpolation_method == 'fraction':
            score = _interpolate(values[int(idx)], values[int(idx) + 1],
                                 idx % 1)
        elif interpolation_method == 'lower':
            score = values[np.floor(idx)]
        elif interpolation_method == 'higher':
            score = values[np.ceil(idx)]
        else:
            raise ValueError("interpolation_method can only be 'fraction', " \
                             "'lower' or 'higher'")
    return score

def hist_quantile(hist, bins, frac, left=None, right=None):
    '''
    '''
    assert frac >= 0.0
    assert frac <= 1.0
    hist_norm = np.zeros(len(bins), dtype=np.float)
    hist_norm[1:] = (hist.cumsum() / float(hist.sum()))
    return np.interp(frac, hist_norm, bins, left, right)

def chunk(n, nchunks):
    '''
    divide the integer 'n' into 'nchunks' equal sized ranges
    '''
    chunk_size, remainder = divmod(n, nchunks)            
    start = 0
    while True:
        end = start + chunk_size
        if remainder > 0:
            end += 1
            remainder -= 1
        yield (start, end)
        if end == n:
            break
        start = end
    assert end == n

class ParserError(Exception):
    '''Error parsing a file.'''
    def __init__(self, msg):
        super(ParserError).__init__(type(self))
        self.msg = "ERROR: %s" % msg
    def __str__(self):
        return self.msg
    def __unicode__(self):
        return self.msg

class Config(object):
    # constants
    ES_NULL_NUM_BINS = 101
    ES_NULL_BINS = np.linspace(-1.0, 1.0, num=ES_NULL_NUM_BINS)
    # constants
    MEMMAP_DTYPE = 'float32'
    SAMPLES_JSON_FILE = 'samples.json'
    METADATA_JSON_FILE = 'metadata.json'
    SAMPLE_SETS_JSON_FILE = 'sample_sets.json'
    CONFIG_JSON_FILE = 'config.json'
    WEIGHT_MATRIX_FILE = 'weight_matrix.memmap'
    RESULTS_JSON_FILE = 'results.json.gz'
    OUTPUT_HISTS_FILE = 'es_hists.npz'
    
    def __init__(self):
        self.num_processes = 1
        self.output_dir = "SSEA_%s" % (timestamp())
        self.name = 'myssea'
        self.perms = 1000
        self.weight_miss = 'log'
        self.weight_hit = 'log'
        self.weight_params = 1.0

        self.count_model = 'poisson'
        
        self.noise_model = 'gaussian'
        self.noise_mean = 1.0
        
        self.weight_const = 1.1
        self.weight_noise = 0.1

    def to_json(self):
        return json.dumps(self.__dict__)
    
    @staticmethod
    def from_json(s):
        c = Config()
        d = json.loads(s)
        c.__dict__ = d
        return c

    @staticmethod
    def parse_json(filename):
        with open(filename, 'r') as fp:
            line = fp.next()
            return Config.from_json(line.strip())

    def update_argument_parser(self, parser=None):
        if parser is None:
            parser = argparse.ArgumentParser()
        grp = parser.add_argument_group("SSEA Options")
        grp.add_argument('-p', '--num-processes', dest='num_processes',
                         type=int, default=1,
                         help='Number of processor cores available '
                         '[default=%(default)s]')
        grp.add_argument('-o', '--output-dir', dest="output_dir", 
                         help='Output directory [default=%(default)s]')
        grp.add_argument('-n', '--name', dest="name", default=self.name,
                         help='Analysis name [default=%(default)s]')
        grp.add_argument('--perms', type=int, default=self.perms,
                         help='Number of permutations '
                         '[default=%(default)s]')
        grp.add_argument('--weight-miss', dest='weight_miss',
                         choices=WEIGHT_METHODS, 
                         default=self.weight_miss,
                         help='Weighting method for elements not in set ' 
                         '[default=%(default)s]')
        grp.add_argument('--weight-hit', dest='weight_hit', 
                         choices=WEIGHT_METHODS, 
                         default=self.weight_hit,
                         help='Weighting method for elements in set '
                         '[default=%(default)s]')
        grp.add_argument('--weight-const', dest='weight_const', type=float,
                         default=self.weight_const,
                         help='Constant floating-point number to add to '
                         'all weights prior to transformation '
                         '[default=%(default)s]')
        grp.add_argument('--weight-noise', dest='weight_noise', type=float,
                         default=self.weight_noise,
                         help='Add uniform noise in the range [0.0-X) to '
                         'the weights to increase robustness '
                         '[default=%(default)s]') 
        return parser

    def log(self, log_func=logging.info):
        log_func("Parameters")
        log_func("----------------------------------")
        log_func("name:                    %s" % (self.name))
        log_func("output directory:        %s" % (self.output_dir))
        log_func("num processes:           %d" % (self.num_processes))
        log_func("permutations:            %d" % (self.perms))
        log_func("weight method miss:      %s" % (self.weight_miss))
        log_func("weight method hit:       %s" % (self.weight_hit))
        log_func("weight constant:         %f" % (self.weight_const))
        log_func("weight noise:            %s" % (self.weight_noise))
        log_func("----------------------------------")

    def parse_args(self, parser, args):
        # process and check arguments
        self.name = args.name
        self.num_processes = args.num_processes
        self.perms = max(1, args.perms)
        # check weight methods and constant
        self.weight_miss = str(args.weight_miss)
        self.weight_hit = str(args.weight_hit)
        self.weight_const = args.weight_const
        self.weight_noise = args.weight_noise
        if self.weight_const < 0.0:
            parser.error('weight const < 0.0 invalid')
        if (self.weight_const - self.weight_noise) < 0.0:            
            parser.error('weight const minus noise < 0.0 invalid')
        elif ((self.weight_miss == 'log' or self.weight_hit == 'log')):
            if self.weight_const < 1.0:
                parser.error('weight constant %f < 1.0 not allowed with '
                             'log method' % (self.weight_const))
            if (self.weight_const - self.weight_noise) < 1.0:
                parser.error('weight constant minus noise is < 1.0')
        # output directory
        self.output_dir = args.output_dir
        if os.path.exists(self.output_dir):
            parser.error("output directory '%s' already exists" % 
                         (self.output_dir))

class Metadata(object):
    __slots__ = ('_id', 'name', 'params')
    
    def __init__(self, _id=None, name=None, params=None):
        '''
        _id: unique integer id (will be auto-generated if not provided)
        name: string name
        params: dictionary of parameter-value data
        '''
        self._id = _id
        self.name = name
        self.params = {}
        if params is not None:
            self.params.update(params)

    def __repr__(self):
        return ("<%s(_id=%d,name=%s,params=%s" % 
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
    def parse_json(filename):
        with open(filename, 'r') as fp:
            for line in fp:
                yield Metadata.from_json(line.strip())
    
    @staticmethod
    def parse_tsv(filename, id_iter=None):
        '''
        parse tab-delimited file containing sample information        
        first row contains column headers
        first column must contain sample name
        remaining columns contain metadata
        '''
        if id_iter is None:
            id_iter = itertools.count()
        fileh = open(filename)
        header_fields = fileh.next().strip().split('\t')
        for line in fileh:
            fields = line.strip().split('\t')
            name = fields[0]
            metadata = dict(zip(header_fields[1:],fields[1:]))
            yield Metadata(id_iter.next(), name, metadata)
        fileh.close()
        

class SampleSet(object): 
    def __init__(self, _id=None, name=None, desc=None, sample_ids=None):
        '''
        _id: unique integer id (will be auto-generated if not provided)
        name: string name of sample set
        desc: string description of sample set 
        sample_ids: list of unique integer ids corresponding to samples
        '''
        self._id = _id
        self.name = name
        self.desc = desc
        self.sample_ids = set()
        if sample_ids is not None:
            self.sample_ids.update(sample_ids)

    def __repr__(self):
        return ("<%s(_id=%d,name=%s,desc=%s,sample_ids=%s" % 
                (self.__class__.__name__, self._id, self.name, self.desc,
                 self.sample_ids))
    
    def __len__(self):
        return len(self.sample_ids)

    def get_array(self, all_ids):
        return np.array([x in self.sample_ids for x in all_ids], 
                        dtype=BOOL_DTYPE)

    def to_json(self):
        d = {'_id': self._id,
             'name': self.name,
             'desc': self.desc,
             'sample_ids': list(self.sample_ids)}
        return json.dumps(d)

    @staticmethod
    def from_json(s):
        d = json.loads(s)
        ss = SampleSet()
        ss._id = d['_id']
        ss.name = d['name']
        ss.desc = d['desc']
        ss.sample_ids = set(d['sample_ids'])
        return ss

    @staticmethod
    def parse_json(filename):
        with open(filename, 'r') as fp:
            for line in fp:
                yield SampleSet.from_json(line.strip())

    @staticmethod
    def remove_duplicates(sample_sets):
        '''
        compare all sample sets and remove duplicates
        
        sample_sets: list of SampleSet objects
        '''
        # TODO: write this
        pass

    @staticmethod
    def parse_smx(filename, samples, id_iter=None):
        '''
        filename: smx formatted file
        samples: list of all samples in the experiment
        '''
        if id_iter is None:
            id_iter = itertools.count()
        fileh = open(filename)
        names = fileh.next().rstrip('\n').split('\t')
        descs = fileh.next().rstrip('\n').split('\t')
        if len(names) != len(descs):
            raise ParserError('Number of fields in differ in columns 1 and 2 '
                              'of sample set file')
        # get name -> _id map of samples
        name_id_map = dict((s.name, s._id) for s in samples)            
        # create empty sample sets
        sample_sets = [SampleSet(id_iter.next(),n,d) 
                       for n,d in zip(names,descs)]
        lineno = 3
        for line in fileh:
            if not line:
                continue
            line = line.rstrip('\n')
            if not line:
                continue
            fields = line.split('\t')
            for i,name in enumerate(fields):
                if not name:
                    continue
                if name not in name_id_map:
                    raise ParserError('Unrecognized sample name "%s" in '
                                      'sample set "%s"' 
                                      % (name, sample_sets[i].name))
                sample_id = name_id_map[name]
                sample_sets[i].sample_ids.add(sample_id)
            lineno += 1
        fileh.close()
        return sample_sets

    @staticmethod
    def parse_smt(filename, samples, id_iter=None):
        '''
        filename: smt formatted file
        samples: list of all samples in the experiment
        '''
        if id_iter is None:
            id_iter = itertools.count()
        # get name -> _id map of samples
        name_id_map = dict((s.name, s._id) for s in samples)
        sample_sets = []
        fileh = open(filename)
        for line in fileh:
            fields = line.strip().split('\t')
            name = fields[0]
            desc = fields[1]
            sample_ids = []
            for sample_name in fields[2:]:
                if sample_name not in name_id_map:
                    raise ParserError('Unrecognized sample name "%s" in '
                                      'sample set "%s"' 
                                      % (sample_name, name))
                sample_ids.append(name_id_map[sample_name])
            sample_sets.append(SampleSet(id_iter.next(), name, desc, sample_ids))
        fileh.close()
        return sample_sets

def weight_matrix_tsv_to_memmap(tsv_file, output_file,
                                row_metadata=None,
                                col_metadata=None, 
                                na_values=None,
                                dtype='float32'): 
    '''
    convert/copy a tab-delimited file containing numeric weight data
    to a numpy memmap file.    
    
    tsv_file: string path to tab-delimited matrix file
    mmap_file: string path to mmap output file
    row_metadata: list of Metadata objects corresponding to rows in file
    col_metadata: list of Metadata objects corresponding to columns in file 
    na_val: set of values corresponding to missing data 
    '''
    if na_values is None:
        na_values = set(['NA'])
    else:
        na_values = set(na_values)
        
    if col_metadata is None:
        # create new Metadata from column names
        fileh = open(tsv_file, 'r')
        header_fields = fileh.next().strip().split('\t')
        id_iter = itertools.count()
        col_metadata = [Metadata(id_iter.next(), f) 
                        for f in header_fields[1:]]
        fileh.close()
    if row_metadata is None:
        # create new Metadata from row names
        row_metadata = []
        id_iter = itertools.count()
        with open(tsv_file, 'r') as fileh:
            fileh.next() # skip header
            for line in fileh:
                name = line.strip().split('\t',1)[0]
                row_metadata.append(Metadata(id_iter.next(), name))
    # check column metadata for correctness 
    fileh = open(tsv_file)
    header_fields = fileh.next().strip().split('\t')
    assert len(col_metadata) == (len(header_fields)-1)
    # check that column names are in correct order
    for i in xrange(len(col_metadata)):
        assert col_metadata[i].name == header_fields[i+1]
    # create mmap file
    mat = np.memmap(output_file, dtype=dtype, mode='w+', 
                    shape=(len(row_metadata), len(col_metadata)))
    # write matrix data (converting missing values to np.nan)
    for i,line in enumerate(fileh):
        fields = line.strip().split('\t')
        # check row metadata
        assert fields[0] == row_metadata[i].name
        # convert to floats and store in mmap      
        mat[i,:] = [(np.nan if x in na_values else float(x))
                    for x in fields[1:]]
    fileh.close()
    del mat
    return row_metadata, col_metadata

def weight_matrix_memmap_copy(input_file, output_file,
                              row_metadata,
                              col_metadata,
                              dtype='float32'):
    '''
    copy a numpy memmap file
    
    input_file: input memmap file
    output_file: output memmap file
    row_metadata: Metadata objects for each row
    col_metadata: Metadata objects for each column
    dtype: numpy dtype of input array e.g. 'float32'
    '''
    # input memmap file
    inp = np.memmap(input_file, dtype=dtype, mode='r', 
                    shape=(len(row_metadata),len(col_metadata)))
    # output memmap file
    outp = np.memmap(output_file, dtype=dtype, mode='w+', 
                     shape=(len(row_metadata), len(col_metadata)))
    # copy
    outp[:] = inp
    del outp
    del inp
     
class Result(object):
    FIELDS = ('t_id', 'ss_id', 'es', 'nes', 'nominal_p_value',
              'fdr_q_value', 'fwer_p_value', 'rank_at_max',
              'core_hits', 'core_misses', 'null_hits', 'null_misses',
              'es_null_hist', 'es_null_mean', 'es_null_percent_neg',
              'global_nes', 'global_fdr_q_value')
    
    def __init__(self):
        for x in Result.FIELDS:
            setattr(self, x, None)
    
    def to_json(self):
        return json.dumps(self.__dict__, cls=NumpyJSONEncoder)
    
    @staticmethod
    def from_json(s):
        d = json.loads(s)
        res = Result()
        res.__dict__ = d
        return res