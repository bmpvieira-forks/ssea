'''
Created on Oct 18, 2013

@author: mkiyer
'''
import numpy as np

BOOL_DTYPE = np.uint8
FLOAT_DTYPE = np.float
WEIGHT_METHODS = ['unweighted', 'weighted', 'log']

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
    def __init__(self, name=None, desc=None, value=None):
        self.name = name
        self.desc = desc
        self.value = value
    
    def __len__(self):
        return 0 if self.value is None else len(self.value)

    def get_array(self, samples):
        return np.array([x in self.value for x in samples], 
                        dtype=BOOL_DTYPE)

    @staticmethod
    def parse_smx(filename):
        fileh = open(filename)
        names = fileh.next().rstrip('\n').split('\t')
        descs = fileh.next().rstrip('\n').split('\t')
        if len(names) != len(descs):
            raise ParserError("Number of fields in differ in columns 1 and 2 of sample set file")
        sample_sets = [SampleSet(name=n,desc=d,value=set()) for n,d in zip(names,descs)]
        lineno = 3
        for line in fileh:
            if not line:
                continue
            line = line.rstrip('\n')
            if not line:
                continue
            fields = line.split('\t')
            for i,f in enumerate(fields):
                if not f:
                    continue
                sample_sets[i].value.add(f)
            lineno += 1
        fileh.close()
        return sample_sets

    @staticmethod
    def parse_smt(filename):
        sample_sets = []
        fileh = open(filename)    
        for line in fileh:
            fields = line.strip().split('\t')
            name = fields[0]
            desc = fields[1]
            values = set(fields[2:])
            sample_sets.append(SampleSet(name, desc, values))
        fileh.close()
        return sample_sets

class WeightVector(object):
    METADATA_COLS=2
    
    def __init__(self, name=None, desc=None, samples=None, weights=None):
        self.name = name
        self.desc = desc
        self.samples = samples
        self.weights = weights

    @staticmethod
    def parse_wmt(filename):
        '''generator function to parse weight matrix files'''
        fileh = open(filename)
        header_fields = fileh.next().strip().split('\t')
        samples = header_fields[WeightVector.METADATA_COLS:]
        lineno = 2
        for line in fileh:
            fields = line.strip().split('\t')
            if len(fields[WeightVector.METADATA_COLS:]) != len(samples):
                raise ParserError("Number of fields in line %d of weight " 
                                  "matrix file %s does not match number of "
                                  "samples" %
                                  (lineno, filename))
            name = fields[0]
            desc = fields[1]
            try:
                weights = map(float,fields[2:])
            except ValueError:
                raise ParserError("Values at line number %d cannot be "
                                  "converted to a floating point numbers" 
                                  % (lineno))    
            yield WeightVector(name, desc, samples, weights)
            lineno += 1
        fileh.close()

    @staticmethod
    def parse_wmx(filename):
        fileh = open(filename)
        fields = fileh.next().strip().split('\t')
        names = fields[1:]
        fields = fileh.next().strip().split('\t')
        descs = fields[1:]
        samples = []
        weights = [list() for x in len(names)]
        lineno = 3
        for line in fileh:
            fields = line.strip().split('\t')
            if len(fields) == 0:
                continue
            elif len(fields[1:]) != len(names):
                raise ParserError("Number of fields in line %d of weight " 
                                  "matrix file %s does not match number of "
                                  "columns" %
                                  (lineno, filename))
            samples.append(fields[0])
            try:
                vals = map(float, fields[1:])
            except ValueError:
                raise ParserError("Values at line number %d cannot be "
                                  "converted to a floating point number" % 
                                  (lineno))
            for i,val in enumerate(vals):
                weights[i].append(val)
            lineno += 1        
        fileh.close()
        wobjs = []
        for i in xrange(len(names)):
            wobjs.append(WeightVector(names[i], descs[i], samples, 
                                      weights[i]))
        return wobjs
