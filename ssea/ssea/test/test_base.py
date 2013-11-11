'''
Created on Oct 22, 2013

@author: mkiyer
'''
import unittest

import random
import os
import numpy as np

from ssea.base import Metadata, SampleSet, chunk, hist_quantile, interp

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

    def test_chunk(self):
        self.assertEquals(list(chunk(100,1)), [(0, 100)])
        self.assertEquals(list(chunk(100,2)), [(0, 50), (50, 100)])
        self.assertEquals(list(chunk(100,3)), [(0, 34), (34, 67), (67, 100)])
        self.assertEquals(list(chunk(100,4)), [(0, 25), (25, 50), (50, 75), (75, 100)])
        self.assertEquals(list(chunk(100,6)), [(0, 17), (17, 34), (34, 51), (51, 68), (68, 84), (84, 100)])
        self.assertEquals(list(chunk(1,4)), [(0, 1)])
        self.assertEquals(list(chunk(2,4)), [(0, 1), (1, 2)])
        self.assertEquals(list(chunk(3,4)), [(0, 1), (1, 2), (2, 3)])
        self.assertEquals(list(chunk(4,4)), [(0, 1), (1, 2), (2, 3), (3, 4)])
        self.assertEquals(list(chunk(5,4)), [(0, 2), (2, 3), (3, 4), (4, 5)])

    def test_interp(self):
        x = np.random.normal(loc=0.5, scale=0.1, size=100000)
        bins = np.linspace(0.0, 1.0, num=51)
        h = np.histogram(x, bins=bins)[0]        
        fp = np.zeros(len(bins), dtype=np.float)
        fp[1:] = h.cumsum()
        x = np.linspace(0,1,21)
        y = np.interp(x, bins, fp) 
        y2 = np.array([interp(v, bins, fp) for v in x])
        self.assertTrue(np.array_equal(y, y2))
        
    def test_hist_quantile(self):
        # TODO: write test cases
        #hist = np.array([0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0,1.0,1.0,2.0,4.0,10.0,10.0,13.0,22.0,21.0,32.0,40.0,36.0,36.0,51.0,68.0,61.0,54.0,61.0,55.0,55.0,36.0,24.0,8.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0])
        #bins = np.array([-1.0,-0.98,-0.96,-0.94,-0.92,-0.9,-0.88,-0.86,-0.84,-0.82,-0.8,-0.78,-0.76,-0.74,-0.72,-0.7,-0.68,-0.66,-0.64,-0.62,-0.6,-0.58,-0.56,-0.54,-0.52,-0.5,-0.48,-0.46,-0.44,-0.42,-0.4,-0.38,-0.36,-0.34,-0.32,-0.3,-0.28,-0.26,-0.24,-0.22,-0.2,-0.18,-0.16,-0.14,-0.12,-0.1,-0.08,-0.06,-0.04,-0.02,0.0])
        #print 'h', hist_quantile(hist,bins, 0.05)
        #print 'h', hist_quantile(hist,bins, 0.95)
        #print len(hist)
        #print len(bins)
        #es -0.27963404631
        #es null mean -0.308334586515
        return

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