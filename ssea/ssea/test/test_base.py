'''
Created on Oct 22, 2013

@author: mkiyer
'''
import unittest

import random
import os
import numpy as np

from ssea.lib.base import chunk, interp, SampleSet
   
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
        self.assertTrue(np.allclose(y, y2, rtol=1e-6, atol=1e-6))


def generate_random_sample_set(minsize, maxsize, samples):
    size = random.randint(minsize, maxsize)
    randsamples = random.sample(samples, size)
    randvalues = np.random.random_integers(0,1,size=size)
    values = []
    for i in xrange(len(samples)):
        if samples[i] in randsamples:
            index = randsamples.index(samples[i])
            values.append((samples[i],randvalues[index]))
    return SampleSet(name="SS",
                     desc="Sample Set",
                     values=values)
    
class TestSampleSet(unittest.TestCase):
    
    def test_sample_set_smx_parser(self):
        # generate samples
        samples = ['S%d' % (i) for i in range(10000)]
        # generate sample sets
        N = 100
        minsize = 1
        maxsize = N
        sample_sets = []
        for i in xrange(N):
            sample_sets.append(generate_random_sample_set(minsize,maxsize,samples))
        # write to a temp file
        names = ['Name'] + [ss.name for ss in sample_sets]
        descs = ['Desc'] + [ss.desc for ss in sample_sets]
        with open('tmp', 'w') as fileh:
            print >>fileh, '\t'.join(names)
            print >>fileh, '\t'.join(descs)
            for i in xrange(len(samples)):
                fields = [samples[i]]
                for j in xrange(len(sample_sets)):
                    if samples[i] in sample_sets[j].value_dict:
                        fields.append(sample_sets[j].value_dict[samples[i]])
                    else:
                        fields.append('')
                print >>fileh, '\t'.join(map(str,fields))
        fileh.close()
        # read into sample sets
        read_sample_sets = SampleSet.parse_smx('tmp')
        self.assertTrue(len(read_sample_sets) == N)
        self.assertTrue(len(read_sample_sets) == len(sample_sets))
        for i in xrange(N):
            ss = sample_sets[i]
            rss = read_sample_sets[i]
            self.assertEqual(rss.name, ss.name)
            self.assertEqual(rss.desc, ss.desc)
            self.assertTrue(set(rss.value_dict.items()) == 
                            set(ss.value_dict.items()))
            a = ss.get_array(samples)
            b = rss.get_array(samples)
            self.assertTrue(np.array_equal(a, b))
        os.remove('tmp')

    def test_sample_set_smt_parser(self):
        # generate samples
        samples = ['S%d' % (i) for i in range(10000)]
        # generate sample sets
        N = 100
        minsize = 1
        maxsize = N
        sample_sets = []
        for i in xrange(N):
            sample_sets.append(generate_random_sample_set(minsize,maxsize,samples))
        # write to a temp file
        fileh = open('tmp', 'w')
        fields = ['Name', 'Description']
        fields.extend(samples)
        print >>fileh, '\t'.join(fields)
        for i in xrange(len(sample_sets)):
            ss = sample_sets[i]
            fields = [ss.name, ss.desc]
            for j in xrange(len(samples)):
                if samples[j] in ss.value_dict:
                    fields.append(ss.value_dict[samples[j]])
                else:
                    fields.append('')
            print >>fileh, '\t'.join(map(str,fields))
        fileh.close()
        # read into sample sets
        read_sample_sets = SampleSet.parse_smt('tmp')
        self.assertTrue(len(read_sample_sets) == N)
        self.assertTrue(len(read_sample_sets) == len(sample_sets))
        for i in xrange(N):
            ss = sample_sets[i]
            rss = read_sample_sets[i]
            self.assertEqual(rss.name, ss.name)
            self.assertEqual(rss.desc, ss.desc)
            self.assertTrue(set(rss.value_dict.items()) == 
                            set(ss.value_dict.items()))
            a = ss.get_array(samples)
            b = rss.get_array(samples)
            self.assertTrue(np.array_equal(a, b))            
        os.remove('tmp')

if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()