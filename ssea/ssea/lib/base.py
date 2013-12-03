'''
Created on Oct 18, 2013

@author: mkiyer
'''
import logging
import json
import itertools
import numpy as np

BOOL_DTYPE = np.uint8
FLOAT_DTYPE = np.float

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
        ss = SampleSet(**d)
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
                    logging.warning('Unrecognized sample name "%s" in '
                                    'sample set "%s"' 
                                    % (name, sample_sets[i].name))
                    continue
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
                    logging.warning('Unrecognized sample name "%s" in '
                                    'sample set "%s"' 
                                    % (sample_name, name))
                    continue
                sample_ids.append(name_id_map[sample_name])
            sample_sets.append(SampleSet(id_iter.next(), name, desc, sample_ids))
        fileh.close()
        return sample_sets
    
class Result(object):
    FIELDS = ('row_id', 'ss_id', 'rand_seed', 'es', 'es_rank', 'nominal_p_value',
              'core_hits', 'core_misses', 'null_hits', 'null_misses',
              'fisher_p_value', 'odds_ratio', 'nes',
              'fdr_q_value', 'ss_fdr_q_value', 'ss_rank',
              'resample_es_vals', 'resample_es_ranks', 
              'null_es_vals', 'null_es_ranks', 'null_es_hist')
    
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
