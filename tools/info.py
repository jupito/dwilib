#!/usr/bin/env python2

"""Print information about columns of number values."""

from __future__ import division, print_function
import argparse
import sys

import numpy as np

from dwi import asciifile
from dwi import util

def parse_args():
    """Parse command-line arguments."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('input', metavar='FILENAME', nargs='+', default=[],
            help='input ASCII files')
    p.add_argument('-v', '--verbose',
            action='count',
            help='increase verbosity')
    p.add_argument('--basic', '-b', action='store_true',
            help='show median, mean, variance, min, max, sum')
    p.add_argument('--five', '-f', action='store_true',
            help='show Tukey\'s five numbers (min, q1, median, q3, max)')
    args = p.parse_args()
    return args

args = parse_args()
for filename in args.input:
    af = asciifile.AsciiFile(filename)
    if args.verbose:
        print(filename)
        if af.d.has_key('description'):
            print(af.d['description'])
    params = af.params()
    for i, a in enumerate(af.a.T):
        d = dict(i=i, param=params[i])
        if args.basic:
            d.update(mean=a.mean(), std=a.std(), var=a.var(), sum=sum(a))
            print('{i}\t{param}'\
                    '\t{mean:g}\t{std:g}\t{var:g}'\
                    '\t{sum:g}'.format(**d))
        if args.five:
            d.update(util.fivenumd(a))
            print('{i}\t{param}'\
                    '\t{min:g}\t{q1:g}\t{median:g}\t{q3:g}\t{max:g}'.format(**d))
