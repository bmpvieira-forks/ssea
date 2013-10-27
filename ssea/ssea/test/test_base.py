'''
Created on Oct 22, 2013

@author: mkiyer
'''
import unittest

import random
import os

from ssea.base import SampleSet, WeightVector

def generate_random_sample_sets(N, minsize, maxsize, population):
    names = []
    descs = []
    sample_sets = []
    for i in xrange(N):
        size = random.randint(minsize, maxsize)
        sample_set = map(str,random.sample(population, size))
        sample_sets.append(sample_set)
        names.append("SS%d" % (i))
        descs.append("Sample Set %d" % (i))
    return names, descs, sample_sets

class TestBase(unittest.TestCase):

    def test_sample_set_smx_parser(self):
        # generate sample sets
        N = 1000
        minsize = 1
        maxsize = 1000
        population = range(100000)
        names, descs, sample_sets = generate_random_sample_sets(N, minsize, maxsize, population)
        # write to a temp file
        fileh = open('tmp', 'w')
        print >>fileh, '\t'.join(names)
        print >>fileh, '\t'.join(descs)
        for i in xrange(maxsize):
            fields = []
            for j in xrange(N):
                if i >= len(sample_sets[j]):
                    fields.append('')
                else:
                    fields.append(sample_sets[j][i])
            print >>fileh, '\t'.join(fields)
        fileh.close()
        # read into sample sets
        read_sample_sets = SampleSet.parse_smx('tmp')
        self.assertTrue(len(read_sample_sets) == N)
        self.assertTrue(len(read_sample_sets) == len(sample_sets))
        for i in xrange(N):
            self.assertEqual(read_sample_sets[i].name, names[i])
            self.assertEqual(read_sample_sets[i].desc, descs[i])
            self.assertEqual(read_sample_sets[i].value, set(sample_sets[i]))
        os.remove('tmp')

    def test_sample_set_smt_parser(self):
        # generate sample sets
        N = 1000
        minsize = 1
        maxsize = 1000
        population = range(100000)
        names, descs, sample_sets = generate_random_sample_sets(N, minsize, maxsize, population)
        # write to a temp file
        fileh = open('tmp', 'w')
        for i in xrange(maxsize):
            fields = [names[i], descs[i]]
            fields.extend(sample_sets[i])
            print >>fileh, '\t'.join(fields)
        fileh.close()
        # read into sample sets
        read_sample_sets = SampleSet.parse_smt('tmp')
        self.assertTrue(len(read_sample_sets) == N)
        self.assertTrue(len(read_sample_sets) == len(sample_sets))
        for i in xrange(N):
            self.assertEqual(read_sample_sets[i].name, names[i])
            self.assertEqual(read_sample_sets[i].desc, descs[i])
            self.assertEqual(read_sample_sets[i].value, set(sample_sets[i]))
        os.remove('tmp')

    def test_weight_vector_parser(self):
        pass
    


if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()