'''
Created on Oct 18, 2013

@author: mkiyer
'''
import logging
import json
import itertools
import numpy as np

INT_DTYPE = np.int
FLOAT_DTYPE = np.float
MISSING_VALUE = -1

class WeightMethod:
    UNWEIGHTED = 0
    WEIGHTED = 1
    EXP = 2
    LOG = 3
WEIGHT_METHODS = {'unweighted': WeightMethod.UNWEIGHTED,
                  'weighted': WeightMethod.WEIGHTED,
                  'exp': WeightMethod.EXP,
                  'log': WeightMethod.LOG}
WEIGHT_METHOD_STR = dict((v,k) for k,v in WEIGHT_METHODS.iteritems())

class NumpyJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

def quantile_sorted(a, frac):
    def _interpolate(a, b, fraction):
        return a + (b - a)*fraction;
    idx = frac * (a.shape[0] - 1)
    if (idx % 1 == 0):
        score = a[idx]
    else:
        score = _interpolate(a[int(idx)], a[int(idx) + 1], idx % 1)
    return score

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

def interp(x, xp, yp):
    '''
    linear interpolation
    x: x value to interpolate
    xp: x coords of points
    yp: y coords of points
    returns y value corresponding to x
    '''
    b = np.searchsorted(xp, x, side='left')
    if b == 0:
        return yp[0]
    if b >= xp.shape[0]:
        return yp[-1]
    a = b - 1
    frac = (x - xp[a]) / (xp[b] - xp[a])
    y = yp[a] + (yp[b] - yp[a]) * frac
    return y

def hist_quantile(hist, bins, frac, left=None, right=None):
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

class SampleSet(object): 
    DTYPE = np.int
    MISSING_VALUE = -1
    
    def __init__(self, name=None, desc=None, values=None):
        '''
        name: string name of sample set
        desc: string description of sample set 
        values: list of tuples of (sample_name,0 or 1) for 
        valid samples in this set
        '''
        self.name = name
        self.desc = desc
        self.value_dict = {}
        if values is not None:
            for sample,value in values:
                value = int(value)
                assert (value == 0) or (value == 1)
                self.value_dict[sample] = value

    def __repr__(self):
        return ("<%s(name=%s,desc=%s,value_dict=%s" % 
                (self.__class__.__name__, self._id, self.name, self.desc,
                 str(self.value_dict)))
        
    def __len__(self):
        return sum(self.value_dict.itervalues())
    
    def get_array(self, samples):
        return np.array([self.value_dict.get(x, MISSING_VALUE) 
                         for x in samples], dtype=SampleSet.DTYPE)
    
    @staticmethod
    def from_json(s):
        d = json.loads(s)
        return SampleSet(**d)
       
    @staticmethod
    def parse_json(filename):
        with open(filename, 'r') as f:
            return SampleSet.from_json(f.next().strip())

    def to_json(self):
        d = {'name': self.name,
             'desc': self.desc,
             'values': self.value_dict.items()}
        return json.dumps(d)

    @staticmethod
    def parse_smx(filename, sep='\t'):
        '''
        filename: smx (column) formatted file
        '''
        fileh = open(filename)
        names = fileh.next().rstrip('\n').split(sep)[1:]
        descs = fileh.next().rstrip('\n').split(sep)[1:]
        if len(names) != len(descs):
            raise ParserError('Number of fields in differ in columns 1 and 2 '
                              'of sample set file')
        # create empty sample sets
        sample_sets = [SampleSet(n,d) for n,d in zip(names,descs)]
        lineno = 3
        for line in fileh:
            if not line:
                continue
            line = line.rstrip('\n')
            if not line:
                continue
            fields = line.split(sep)
            sample = fields[0]
            for i,value in enumerate(fields[1:]):
                if not value:
                    continue
                value = int(value)
                sample_sets[i].value_dict[sample] = value
            lineno += 1
        fileh.close()
        return sample_sets

    @staticmethod
    def parse_smt(filename, sep='\t'):
        '''
        filename: smt (row) formatted file
        '''
        sample_sets = []
        fileh = open(filename)
        samples = fileh.next().rstrip('\n').split(sep)
        for line in fileh:
            fields = line.rstrip('\n').split(sep)
            name = fields[0]
            desc = fields[1]
            values = []
            for i in xrange(2, len(fields)):
                value = fields[i]
                if not value:
                    continue
                value = int(value)
                values.append((samples[i],value))
            sample_sets.append(SampleSet(name, desc, values))
        fileh.close()
        return sample_sets

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

class Result(object):
    MAX_POINTS = 100
    FIELDS = ('t_id', 'rand_seed', 'es', 'es_rank', 'nominal_p_value',
              'core_hits', 'core_misses', 'null_hits', 'null_misses',
              'fisher_p_value', 'odds_ratio', 'nes', 'ss_fdr_q_value', 
              'ss_rank', 'ss_percentile', 'resample_es_vals', 
              'resample_es_ranks', 'null_es_vals', 'null_es_ranks', 
              'null_es_mean')
    
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

    @staticmethod
    def default():
        '''setup a Result object with default (non-significant) values'''
        res = Result()
        res.t_id = None
        res.rand_seed = None
        res.es = 0.0
        res.es_rank = 0
        res.nominal_p_value = 1.0
        res.core_hits = 0
        res.core_misses = 0
        res.null_hits = 0
        res.null_misses = 0
        res.fisher_p_value = 1.0
        res.odds_ratio = 1.0
        res.nes = 0.0
        res.ss_fdr_q_value = 1.0
        res.ss_rank = None
        res.ss_frac = 0.0
        res.resample_es_vals = np.zeros(Result.MAX_POINTS, dtype=np.float)
        res.resample_es_ranks = np.zeros(Result.MAX_POINTS, dtype=np.float)
        res.null_es_vals = np.zeros(Result.MAX_POINTS, dtype=np.float)
        res.null_es_ranks = np.zeros(Result.MAX_POINTS, dtype=np.float)
        res.null_es_mean = 0.0
        return res
