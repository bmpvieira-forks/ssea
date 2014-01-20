'''
Created on Dec 17, 2013

@author: mkiyer
'''
import argparse
import logging
import sys
import os
import glob
import shutil

from ssea.lib.base import JobStatus, SampleSet, computerize_name
from ssea.lib.config import Config
 
def check_path(path):
    if not os.path.exists(path):
        logging.error('Directory not found at path "%s"' % (path))
        return False
    elif not os.path.isdir(path):
        logging.error('Directory not valid at path "%s"' % (path))
        return False
    elif JobStatus.get(path) != JobStatus.DONE:
        logging.info('Result status not set to "DONE" at path "%s"' % (path))
        return False
    return True

def find_paths(input_dirs):
    paths = set()
    for input_dir in input_dirs:
        if not os.path.exists(input_dir):
            logging.error("input dir '%s' not found" % (input_dir))
        if not os.path.isdir(input_dir):
            logging.error("input dir '%s' not a valid directory" % (input_dir))
        input_dir = os.path.abspath(input_dir)
        for path in glob.iglob(os.path.join(input_dir, "*")):
            if os.path.exists(path) and os.path.isdir(path):
                if check_path(path):
                    paths.add(path)
    return paths

def find(args):
    input_dirs = args.input_dirs
    if input_dirs is None:
        logging.error('Please specify SSEA directories using "-d" or "--dir"')
        return 1
    for path in find_paths(input_dirs):
        print path
        
def info(args):
    input_dirs = args.input_dirs
    if input_dirs is None:
        logging.error('Please specify SSEA directories using "-d" or "--dir"')
        return 1
    for path in find_paths(input_dirs):
        sample_set_json_file = os.path.join(path, Config.SAMPLE_SET_JSON_FILE)
        sample_set = SampleSet.parse_json(sample_set_json_file)[0]
        compname = computerize_name(sample_set.name)
        print '\t'.join([compname, path])        

def merge(args):
    # get args
    input_paths_file = args.input_paths_file
    input_dirs = args.input_dirs
    output_dir = args.output_dir
    # check args
    input_paths = set()
    if input_paths_file is not None:
        if not os.path.exists(input_paths_file):
            logging.error('Input paths file "%s" not found' % (input_paths_file))
        else:
            with open(input_paths_file) as fileh:
                for line in fileh:
                    path = line.strip()
                    if check_path(path):
                        input_paths.add(path)
    if input_dirs is not None:
        input_paths.update(find_paths(input_dirs))
    if len(input_paths) == 0:
        logging.error('No valid SSEA results directories found.. Exiting.')
        return 1
    if not os.path.exists(output_dir):
        logging.debug('Creating output directory "%s"' % (output_dir))
        os.makedirs(output_dir)
    # read paths already in output directory
    existing_paths = set()
    for path in glob.iglob(os.path.join(output_dir, "*")):
        if os.path.exists(path) and os.path.isdir(path):
            existing_paths.add(path)
    # merge input paths
    for src in input_paths:
        sample_set_json_file = os.path.join(src, Config.SAMPLE_SET_JSON_FILE)
        sample_set = SampleSet.parse_json(sample_set_json_file)[0]
        dirname = computerize_name(sample_set.name)
        dst = os.path.join(output_dir, dirname)
        if os.path.exists(dst):
            logging.error('Conflict when merging sample set name "%s" from path "%s"' % (sample_set.name, src))
        else:
            logging.debug('Moving sample set "%s" from "%s" to "%s"' % (sample_set.name, src, dst))
            shutil.move(src, dst)

def main():
    logging.basicConfig(level=logging.DEBUG,
                        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()
    # info table
    subparser = subparsers.add_parser('info')
    subparser.add_argument("-d", "--dir", dest="input_dirs", action="append", default=None)
    subparser.set_defaults(func=info)
    # find valid ssea results paths
    subparser = subparsers.add_parser('find')
    subparser.add_argument("-d", "--dir", dest="input_dirs", action="append", default=None)
    subparser.set_defaults(func=find)
    # combine ssea output from different folders
    subparser = subparsers.add_parser('merge')
    subparser.add_argument("-i", dest="input_paths_file", default=None)
    subparser.add_argument("-d", "--dir", dest="input_dirs", action="append", default=None)
    subparser.add_argument('output_dir')
    subparser.set_defaults(func=merge)
    args = parser.parse_args()
    args.func(args)

if __name__ == '__main__':
    sys.exit(main())
