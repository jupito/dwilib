#!/usr/bin/env python2

"""Print information about columns of number values."""

from __future__ import absolute_import, division, print_function
import argparse

import dwi.asciifile
import dwi.util


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('input', metavar='FILENAME', nargs='+', default=[],
                   help='input ASCII files')
    p.add_argument('-v', '--verbose', action='count',
                   help='increase verbosity')
    p.add_argument('--basic', '-b', action='store_true',
                   help='show median, mean, variance, min, max, sum')
    p.add_argument('--five', '-f', action='store_true',
                   help="show Tukey's five numbers (min, q1, median, q3, max)")
    return p.parse_args()


def main():
    args = parse_args()
    for filename in args.input:
        af = dwi.asciifile.AsciiFile(filename)
        if args.verbose:
            print(filename)
            if 'description' in af.d:
                print(af.d['description'])
        params = af.params()
        for i, a in enumerate(af.a.T):
            d = dict(i=i, param=params[i])
            if args.basic:
                s = '{i} {param} {mean:g} {std:g} {var:g} {sum:g}'
                d.update(mean=a.mean(), std=a.std(), var=a.var(), sum=sum(a))
                print(s.format(**d))
            if args.five:
                s = '{i} {param} {min:g} {q1:g} {median:g} {q3:g} {max:g}'
                d.update(dwi.util.fivenumd(a))
                print(s.format(**d))


if __name__ == '__main__':
    main()
