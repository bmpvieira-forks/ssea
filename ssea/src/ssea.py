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

from algo import ssea

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

def read_sample_set(sample_set_file):
    fileh = open(sample_set_file)
    name = fileh.next()
    desc = fileh.next()
    sample_set = set()
    for line in fileh:
        if not line:
            continue
        line = line.strip()
        if not line:
            continue
        fields = line.split('\t')
        if not fields[0]:
            continue
        sample_set.add(fields[0])
    fileh.close()
    return name, desc, sample_set

def read_ranks(ranks_file):
    samples = []
    ranks = []
    fileh = open(ranks_file)
    lineno = 1
    for line in open(ranks_file):
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
        ranks.append(rank)
        lineno += 1
    fileh.close()
    return samples,ranks

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
        parser.add_argument('sample_set_file')
        parser.add_argument('ranks_file')        
        # Process arguments
        args = parser.parse_args()
        verbose = args.verbose
        sample_set_file = args.sample_set_file
        ranks_file = args.ranks_file

        # setup logging
        if DEBUG or (verbose > 0):
            level = logging.DEBUG
        else:
            level = logging.INFO
        logging.basicConfig(level=level,
                            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

        # check command line arguments
        if not os.path.exists(sample_set_file):
            parser.error("sample set file '%s' not found" % (sample_set_file))
        if not os.path.exists(ranks_file):
            parser.error("ranks file '%s' not found" % (ranks_file))

        name, desc, sample_set = read_sample_set(sample_set_file)
        samples, weights = read_ranks(ranks_file)
        
        print name, desc, sample_set
        print samples, weights
        
        ssea(samples, weights, sample_set)
        #raise CLIError("include and exclude pattern are equal! Nothing will be processed.")
        return 0
    except KeyboardInterrupt:
        ### handle keyboard interrupt ###
        return 0
    except Exception, e:
        if DEBUG or TESTRUN:
            raise(e)
        indent = len(program_name) * " "
        logging.error(program_name + ": " + repr(e) + "\n")
        logging.error(indent + "  for help use --help")
        return 2

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