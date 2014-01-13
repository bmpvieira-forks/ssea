'''
Created on Dec 17, 2013

@author: mkiyer
'''
import argparse
import logging
import sys

def repair(args):
    pass

def merge(args):
    pass

def main():
    logging.basicConfig(level=logging.DEBUG,
                        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()
    # repair runs that terminated prematurely
    parser_repair = subparsers.add_parser('repair')
    parser_repair.add_argument('output_dir') 
    parser_repair.set_defaults(func=repair)
    # combine ssea output from different folders
    parser_merge = subparsers.add_parser('merge')
    parser_merge.add_argument('output_dirs', nargs='+')
    parser_merge.set_default(func=merge)
    args = parser.parse_args()
    args.func(args)

if __name__ == '__main__':
    sys.exit(main())
