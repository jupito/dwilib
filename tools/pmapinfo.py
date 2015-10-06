#!/usr/bin/env python2

"""Print information about pmaps."""

from __future__ import absolute_import, division, print_function
import argparse

import numpy as np

import dwi.files
import dwi.util


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('input', metavar='PATH', nargs='+',
                   help='input pmap files')
    p.add_argument('-v', '--verbose', action='count',
                   help='increase verbosity')
    return p.parse_args()


def main():
    args = parse_args()
    for path in args.input:
        img, attrs = dwi.files.read_pmap(path)
        d = dict(path=path, paramlen=max(len(x) for x in attrs['parameters']))
        for i, param in enumerate(attrs['parameters']):
            a = img[..., i]
            shape, size = a.shape, a.size
            nans = np.isnan(a)
            if np.any(nans):
                a = img[-nans]
            d.update(param=param, shape=shape, size=a.size,
                     nonnans=a.size/size, min=np.min(a), max=np.max(a),
                     mean=np.mean(a), median=np.median(a), p='.4')
            s = '{path} {param:{paramlen}} {shape} {size} {nonnans:{p}%} {min:{p}f} {max:{p}f}'
            print(s.format(**d))
            # if args.basic:
            #     s = '{i} {param} {mean:g} {std:g} {var:g} {sum:g}'
            #     d.update(mean=a.mean(), std=a.std(), var=a.var(), sum=sum(a))
            #     print(s.format(**d))
            # if args.five:
            #     s = '{i} {param} {min:g} {q1:g} {median:g} {q3:g} {max:g}'
            #     d.update(dwi.util.fivenumd(a))
            #     print(s.format(**d))


if __name__ == '__main__':
    main()
