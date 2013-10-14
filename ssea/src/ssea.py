#!/bin/env python2.7
# encoding: utf-8
'''
 -- Sample Set Enrichment Analysis (SSEA) --

Assessment of enrichment in a ranked list of quantitative measurements 

@author:     mkiyer
@author:     yniknafs
        
@copyright:  2013 Michigan Center for Translational Pathology. All rights reserved.
        
@license:    GPL2

@contact:    mkiyer@umich.edu
@deffield    updated: Updated
'''
import sys
import os
import argparse
import logging

from algo import ssea, SampleSet

__all__ = []
__version__ = 0.1
__date__ = '2013-10-09'
__updated__ = '2013-10-09'

DEBUG = 1
TESTRUN = 0
PROFILE = 0

class CLIError(Exception):
    '''Generic exception to raise and log different fatal errors.'''
    def __init__(self, msg):
        super(CLIError).__init__(type(self))
        self.msg = "ERROR: %s" % msg
    def __str__(self):
        return self.msg
    def __unicode__(self):
        return self.msg

class ParserError(Exception):
    '''Error parsing a file.'''
    def __init__(self, msg):
        super(ParserError).__init__(type(self))
        self.msg = "ERROR: %s" % msg
    def __str__(self):
        return self.msg
    def __unicode__(self):
        return self.msg

def parse_gmx(filename):
    fileh = open(filename)
    names = fileh.next().strip().split('\t')
    descs = fileh.next().strip().split('\t')
    if len(names) != len(descs):
        raise ParserError("Number of fields in differ in columns 1 and 2 of sample set file")
    sample_sets = [SampleSet(name=n,desc=d,value=set()) for n,d in zip(names,descs)]
    lineno = 3
    for line in fileh:
        if not line:
            continue
        line = line.strip()
        if not line:
            continue
        fields = line.split('\t')
        if len(fields) != len(names):
            raise ParserError("Incorrect number of fields in line %d" % (lineno))
        for i,f in enumerate(fields):
            sample_sets[i].value.add(f)
        lineno += 1
    fileh.close()
    return sample_sets

def parse_weights(filename):
    samples = []
    weights = []
    fileh = open(filename)
    lineno = 1
    for line in open(filename):
        fields = line.strip().split('\t')
        if len(fields) == 0:
            continue
        elif len(fields) == 1:
            raise ParserError("Only one field at line number %d" % (lineno))
        sample = fields[0]
        try:
            rank = float(fields[1])
        except ValueError:
            raise ParserError("Value at line number %d cannot be converted to a floating point number" % (lineno))    
        samples.append(sample)
        weights.append(rank)
        lineno += 1
    fileh.close()
    return samples,weights

def main(argv=None):
    '''Command line options.'''    
    if argv is None:
        argv = sys.argv
    else:
        sys.argv.extend(argv)

    program_name = os.path.basename(sys.argv[0])
    program_version = "v%s" % __version__
    program_build_date = str(__updated__)
    program_version_message = '%%(prog)s %s (%s)' % (program_version, program_build_date)
    program_shortdesc = __import__('__main__').__doc__.split("\n")[1]
    program_license = '''%s

  Created by mkiyer and yniknafs on %s.
  Copyright 2013 MCTP. All rights reserved.
  
  Licensed under the GPL
  http://www.gnu.org/licenses/gpl.html
  
  Distributed on an "AS IS" basis without warranties
  or conditions of any kind, either express or implied.

USAGE
''' % (program_shortdesc, str(__date__))

    try:
        # Setup argument parser
        parser = argparse.ArgumentParser(description=program_license, 
                                         formatter_class=argparse.RawDescriptionHelpFormatter)
        parser.add_argument("-v", "--verbose", dest="verbose", 
                            action="store_true", default=False, 
                            help="set verbosity level [default: %(default)s]")
        parser.add_argument('-V', '--version', action='version', version=program_version_message)
        parser.add_argument('--weight-methods', dest='weight_methods', 
                            default='weighted,weighted')
        parser.add_argument('--weight-params', dest='weight_params', 
                            default=None)
        parser.add_argument('--perms', type=int, default=1000)
        parser.add_argument('--gmx', dest="gmx_file", default=None)
        parser.add_argument('weights_file') 
        # Process arguments
        args = parser.parse_args()
        verbose = args.verbose
        weights_file = args.weights_file
        gmx_file = args.gmx_file
        perms = args.perms
        weight_methods = args.weight_methods
        weight_params = args.weight_params
        
        # setup logging
        if DEBUG or (verbose > 0):
            level = logging.DEBUG
        else:
            level = logging.INFO
        logging.basicConfig(level=level,
                            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

        # check command line arguments
        if not os.path.exists(weights_file):
            parser.error("weights file '%s' not found" % (weights_file))
        # read weights
        samples, weights = parse_weights(weights_file)        
        # read sample sets
        sample_sets = []
        if gmx_file:
            if not os.path.exists(gmx_file):
                parser.error("gmx file '%s' not found" % (gmx_file))
            sample_sets.extend(parse_gmx(gmx_file))
        # check parameters
        perms = max(1, perms)
        # check weight method, params
        fields = weight_methods.split(',')
        if len(fields) == 1:
            weight_methods = (fields[0], fields[0])
        else:
            weight_methods = fields[:2]
        # run
        ssea(samples, weights, sample_sets, 
             weight_methods=weight_methods,
             weight_params=weight_params, 
             perms=perms)            
    except KeyboardInterrupt:
        ### handle keyboard interrupt ###
        pass
#     except Exception, e:
#         pass
#         if DEBUG or TESTRUN:
#             raise(e)
#         indent = len(program_name) * " "
#         logging.error(program_name + ": " + repr(e) + "\n")
#         logging.error(indent + "  for help use --help")
#         return 2
    return 0

if __name__ == "__main__":
    if DEBUG:
        pass
    if TESTRUN:
        pass
        #import doctest
        #doctest.testmod()
    if PROFILE:
        import cProfile
        import pstats
        profile_filename = '_profile.txt'
        cProfile.run('main()', profile_filename)
        statsfile = open("profile_stats.txt", "wb")
        p = pstats.Stats(profile_filename, stream=statsfile)
        stats = p.strip_dirs().sort_stats('cumulative')
        stats.print_stats()
        statsfile.close()
        sys.exit(0)
    sys.exit(main())