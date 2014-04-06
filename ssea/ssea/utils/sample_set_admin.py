'''
Created on Dec 20, 2013

@author: mkiyer
'''
import argparse
import logging
import sys
import os
import collections

from ssea.lib.base import SampleSet, computerize_name

def _parse_sample_sets(filename, sep):
    if not os.path.exists(filename):
        logging.error("Sample set file '%s' not found" % (filename))
        sys.exit(1)
    ext = os.path.splitext(filename)[-1]
    if ext == '.smx':        
        for ss in SampleSet.parse_smx(filename, sep):
            yield ss
    elif ext == '.smt':
        for ss in SampleSet.parse_smt(filename, sep):
            yield ss
    elif ext == '.json':
        for ss in SampleSet.parse_json(filename):
            yield ss
    else:
        logging.error('suffix not recognized (.smx, .smt, or .json)')

def info(args):
    filename = args.sample_set_file
    sep = args.sep
    if not os.path.exists(filename):
        logging.error("Sample set file '%s' not found" % (filename))
        return 1
    print '\t'.join(['compname', 'name', 'desc', 'total', 'hits', 
                     'misses', 'nas'])
    for ss in _parse_sample_sets(filename, sep):
        hits = 0
        misses = 0
        nas = 0
        tot = 0
        for v in ss.value_dict.itervalues():
            if v == 1:
                hits += 1
            elif v == 0:
                misses += 1
            else:
                nas += 1
            tot += 1
        print '\t'.join(map(str, [computerize_name(ss.name), ss.name, ss.desc, tot, hits, misses, nas]))
        #print 'Name: %s Description: %s Samples: %d Hits: %d Misses: %d NAs: %d' % (ss.name, ss.desc, tot, hits, misses, nas)  

def getcohort(args):
    filename = args.sample_set_file
    sep = args.sep
    samples = set()
    for ss in _parse_sample_sets(filename, sep):
        for k,v in ss.value_dict.iteritems():
            if v != 1 and v != 0:
                continue
            samples.add(k)
    for sample in sorted(samples):
        print sample

def newcohort(args):
    sample_set_file = args.sample_set_file
    cohort_file = args.cohort_file
    sep = args.sep
    cohort_samples = set(line.strip() for line in open(cohort_file))
    for ss in _parse_sample_sets(sample_set_file, sep):
        new_value_dict = {}
        hits = 0
        for k,v in ss.value_dict.iteritems(): 
            if k in cohort_samples:
                if v == 1:
                    hits += 1
                new_value_dict[k] = v
        if hits > 0:
            ss = SampleSet(ss.name, ss.desc, new_value_dict.items())
            print ss.to_json()
        else:
            logging.warning('Sample set %s has no hits' % (ss.name))

def rename(args):
    filename = args.sample_set_file
    sep = args.sep
    fromname = args.fromname
    toname = args.toname
    todesc = args.todesc
    if not os.path.exists(filename):
        logging.error("Sample set file '%s' not found" % (filename))
        return 1
    sample_sets = list(_parse_sample_sets(filename, sep))
    if len(sample_sets) == 1:
        ss = sample_sets[0]
        ss.name = toname
        ss.desc = todesc
        print ss.to_json()
    else:
        for ss in sample_sets:
            if ss.name == fromname:
                ss.name = toname
                ss.desc = todesc
            print ss.to_json()

def extract(args):
    list_file = args.sample_set_list_file
    sample_set_file = args.sample_set_file
    sep = args.sep
    if not os.path.exists(sample_set_file):
        logging.error("Sample set file '%s' not found" % (sample_set_file))
        return 1
    compname_dict = {}
    name_dict = {}
    for ss in _parse_sample_sets(sample_set_file, sep):
        compname_dict[computerize_name(ss.name)] = ss
        name_dict[ss.name] = ss
    with open(list_file) as fileh:
        for line in fileh:
            name = line.strip()
            found = False
            if name in compname_dict:
                print compname_dict[name].to_json()
                found = True
            elif name in name_dict:
                print name_dict[name].to_json()
                found = True
            if not found:
                logging.error('Sample set "%s" not found' % (name))

def to_json(args):
    filename = args.sample_set_file
    sep = args.sep
    if not os.path.exists(filename):
        logging.error("Sample set file '%s' not found" % (filename))
        return 1
    for ss in _parse_sample_sets(filename, sep):
        print ss.to_json()
        
def to_smx(args):
    filename = args.sample_set_file
    sep = args.sep
    prefix = args.prefix
    if not os.path.exists(filename):
        logging.error("Sample set file '%s' not found" % (filename))
        return 1
    for ss in _parse_sample_sets(filename, sep):
        lines = []
        lines.append('\t'.join(['name', ss.name]))
        lines.append('\t'.join(['desc', ss.desc]))
        for sample in sorted(ss.value_dict):
            lines.append('\t'.join([sample, str(ss.value_dict[sample])]))
        suffix = computerize_name(ss.name)
        path = prefix + '.' + suffix + '.smx'
        with open(path, 'w') as f:
            for line in lines:
                print >>f, line

def subset(args):
    filename = args.sample_set_file
    sep = args.sep
    name = args.name
    desc = args.desc
    hit_set_names = args.hit_sets
    miss_set_names = args.miss_sets
    # check arguments
    if name is None:
        name = 'subset'
    if desc is None:
        desc = name
    if not hit_set_names or not miss_set_names:
        logging.error('Sample sets to be considered "hits" or "misses" '
                      'should be specified using --hit and --miss')
        return 1
    sample_sets = dict((ss.name,ss) for ss in _parse_sample_sets(filename, sep))
    new_value_dict = {}
    for ss_name in hit_set_names:
        if ss_name not in sample_sets:
            logging.error('Sample set name "%s" not found.. Exiting.' % (ss_name))
            return 1
        ss = sample_sets[ss_name]
        for k,v in ss.value_dict.iteritems():
            if v == 1:
                new_value_dict[k] = 1
    for ss_name in miss_set_names:
        if ss_name not in sample_sets:
            logging.error('Sample set name "%s" not found.. Exiting.' % (ss_name))
            return 1
        ss = sample_sets[ss_name]
        for k,v in ss.value_dict.iteritems():
            if v == 1:
                new_value_dict[k] = 0
    ss = SampleSet(name, desc, new_value_dict.items())
    print ss.to_json()

def merge(args):
    filename = args.sample_set_file
    sep = args.sep
    name = args.name
    desc = args.desc
    sample_set_names = args.sample_set_names
    # check arguments
    if name is None:
        name = 'merge'
    if desc is None:
        desc = name
    if not sample_set_names:
        logging.error('No sample set names specified')
        return 1
    sample_sets = dict((ss.name,ss) for ss in _parse_sample_sets(filename, sep))
    new_value_dict = collections.defaultdict(lambda: 0)
    for ss_name in sample_set_names:
        if ss_name not in sample_sets:
            logging.error('Sample set name "%s" not found.. Exiting.' % (ss_name))
            return 1
        ss = sample_sets[ss_name]
        for k,v in ss.value_dict.iteritems():
            if k in new_value_dict:
                continue
            new_value_dict[k] = v
    ss = SampleSet(name, desc, new_value_dict.items())
    print ss.to_json()

def main():
    logging.basicConfig(level=logging.DEBUG,
                        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument('--sep', dest='sep', default='\t') 
    subparsers = parser.add_subparsers()
    # list sample sets
    subparser = subparsers.add_parser('info')
    subparser.add_argument('sample_set_file') 
    subparser.set_defaults(func=info)
    # extract certain sets from a larger file
    subparser = subparsers.add_parser('extract')
    subparser.add_argument('sample_set_list_file')
    subparser.add_argument('sample_set_file')
    subparser.set_defaults(func=extract)
    # convert smt/smx to json format
    subparser = subparsers.add_parser('tojson')
    subparser.add_argument('sample_set_file') 
    subparser.set_defaults(func=to_json)
    # convert to smx format
    subparser = subparsers.add_parser('tosmx')
    subparser.add_argument('sample_set_file') 
    subparser.add_argument('prefix') 
    subparser.set_defaults(func=to_smx)
    # rename
    subparser = subparsers.add_parser('rename')
    subparser.add_argument('--fromname', dest='fromname', default=None)
    subparser.add_argument('--name', dest='toname', default=None)
    subparser.add_argument('--desc', dest='todesc', default=None)
    subparser.add_argument('sample_set_file')
    subparser.set_defaults(func=rename)
    # merge (union) of sets 
    subparser = subparsers.add_parser('merge')
    subparser.add_argument('--name', dest='name', default=None)
    subparser.add_argument('--desc', dest='desc', default=None)
    subparser.add_argument('--set', dest='sample_set_names', action='append') 
    subparser.add_argument('sample_set_file')
    subparser.set_defaults(func=merge)
    # merge multiple sets into a single set
    subparser = subparsers.add_parser('subset')
    subparser.add_argument('--name', dest='name', default=None)
    subparser.add_argument('--desc', dest='desc', default=None)
    subparser.add_argument('--hit', dest='hit_sets', action='append') 
    subparser.add_argument('--miss', dest='miss_sets', action='append') 
    subparser.add_argument('sample_set_file')
    subparser.set_defaults(func=subset)
    # only include certain samples
    subparser = subparsers.add_parser('newcohort')
    subparser.add_argument('sample_set_file')
    subparser.add_argument('cohort_file')
    subparser.set_defaults(func=newcohort)
    # get samples
    subparser = subparsers.add_parser('getcohort')
    subparser.add_argument('sample_set_file')
    subparser.set_defaults(func=getcohort)
    args = parser.parse_args()    
    return args.func(args)
 
if __name__ == '__main__':
    sys.exit(main())
