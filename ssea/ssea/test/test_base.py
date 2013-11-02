'''
Created on Oct 22, 2013

@author: mkiyer
'''
import unittest

import random
import os

from ssea.base import Metadata, SampleSet

def generate_random_sample_sets(N, minsize, maxsize, samples):
    sample_sets = []
    for i in xrange(N):
        size = random.randint(minsize, maxsize)
        randsamples = random.sample(samples, size)
        sample_ids = [s._id for s in randsamples]
        sample_sets.append(SampleSet(name="SS%d" % (i),
                                     desc="Sample Set %d" % (i),
                                     sample_ids=sample_ids))
    return sample_sets

class TestBase(unittest.TestCase):

    def test_sample_set_smx_parser(self):
        # generate sample sets
        N = 1000
        minsize = 1
        maxsize = 1000
        population = map(str,range(100000))
        samples = [Metadata(name=n) for n in population]
        sample_id_name_map = dict((s._id,s.name) for s in samples)
        sample_sets = generate_random_sample_sets(N, minsize, maxsize, samples)
        # write to a temp file
        fileh = open('tmp', 'w')
        print >>fileh, '\t'.join([ss.name for ss in sample_sets])
        print >>fileh, '\t'.join([ss.desc for ss in sample_sets])
        # force sample id sets to be lists for writing
        for ss in sample_sets:
            ss.sample_ids = list(ss.sample_ids)
        for i in xrange(maxsize):
            fields = []
            for j in xrange(N):
                if i >= len(sample_sets[j]):
                    fields.append('')
                else:
                    ss = sample_sets[j]
                    sample = sample_id_name_map[ss.sample_ids[i]]
                    fields.append(sample)
            print >>fileh, '\t'.join(fields)
        # convert back to sets
        for ss in sample_sets:
            ss.sample_ids = set(ss.sample_ids)
        fileh.close()
        # read into sample sets
        read_sample_sets = SampleSet.parse_smx('tmp', samples)
        self.assertTrue(len(read_sample_sets) == N)
        self.assertTrue(len(read_sample_sets) == len(sample_sets))
        for i in xrange(N):
            self.assertEqual(read_sample_sets[i].name, sample_sets[i].name)
            self.assertEqual(read_sample_sets[i].desc, sample_sets[i].desc)
            self.assertEqual(read_sample_sets[i].sample_ids, sample_sets[i].sample_ids)
        os.remove('tmp')

    def test_sample_set_smt_parser(self):
        # generate sample sets
        N = 1000
        minsize = 1
        maxsize = 1000
        population = map(str,range(100000))
        samples = [Metadata(name=n) for n in population]
        sample_id_name_map = dict((s._id,s.name) for s in samples)
        sample_sets = generate_random_sample_sets(N, minsize, maxsize, samples)
        # write to a temp file
        fileh = open('tmp', 'w')
        for i in xrange(len(sample_sets)):
            ss = sample_sets[i]
            fields = [ss.name, ss.desc]
            fields.extend(sample_id_name_map[s_id] for s_id in ss.sample_ids)
            print >>fileh, '\t'.join(fields)
        fileh.close()
        # read into sample sets
        read_sample_sets = SampleSet.parse_smt('tmp', samples)
        self.assertTrue(len(read_sample_sets) == N)
        self.assertTrue(len(read_sample_sets) == len(sample_sets))
        for i in xrange(N):
            self.assertEqual(read_sample_sets[i].name, sample_sets[i].name)
            self.assertEqual(read_sample_sets[i].desc, sample_sets[i].desc)
            self.assertEqual(read_sample_sets[i].sample_ids, sample_sets[i].sample_ids)
        os.remove('tmp')




if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()