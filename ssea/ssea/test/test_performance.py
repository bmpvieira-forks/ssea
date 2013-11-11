'''
Created on Oct 24, 2013

@author: mkiyer
'''
import numpy as np
import sys

from ssea.base import interp
#from ssea.kernel import ssea_kernel
#from ssea.algo import transform_weights

def interp_setup(numbins):
    import numpy as np
    d = np.random.normal(loc=0.5, scale=0.1, size=100000)
    bins = np.linspace(0.0, 1.0, num=numbins)
    h = np.histogram(d, bins=bins)[0]        
    fp = np.zeros(len(bins), dtype=np.float)
    fp[1:] = h.cumsum()
    x = np.linspace(0,1,num=100000)
    return x, bins, fp

def interp_test1():
    x, bins, fp = interp_setup(10001)
    for v in x:
        y = np.interp(v, bins, fp) 

def interp_test2():
    x, bins, fp = interp_setup(10001)
    for v in x:
        y = interp(v, bins, fp) 

if __name__ == "__main__":
    import timeit
    print 'hello'
    print timeit.timeit('interp_test1()', setup='from __main__ import interp_test1', number=1)
    print 'hello'
    print timeit.timeit('interp_test2()', setup='from __main__ import interp_test2', number=1)

#     import cProfile
#     import pstats
#     profile_filename = '_profile.bin'
#     cProfile.run('unittest.main()', profile_filename)
#     statsfile = open("profile_stats.txt", "wb")
#     p = pstats.Stats(profile_filename, stream=statsfile)
#     stats = p.strip_dirs().sort_stats('cumulative')
#     stats.print_stats()
#     statsfile.close()
#     sys.exit(0)
