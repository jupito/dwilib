#!/usr/bin/env python2

"""Print information about pmaps."""

from __future__ import absolute_import, division, print_function
import argparse
import os.path

import numpy as np

import dwi.files
import dwi.util


class Pmap(object):
    def __init__(self, path):
        self.path = path
        self._img, self._attrs = dwi.files.read_pmap(path)

    keys = property(lambda self: ','.join(sorted(x for x in dir(self) if not
                                                 x.startswith('_'))))
    file = property(lambda self: os.path.basename(self.path))
    root = property(lambda self: os.path.splitext(self.file)[0])
    shape = property(lambda self: shorten(self._img.shape))
    type = property(lambda self: self._img.dtype)
    size = property(lambda self: self._img.size)
    finite = property(lambda self: np.count_nonzero(np.isfinite(self._img)))
    rfinite = property(lambda self: self.finite / self.size)
    nonzero = property(lambda self: np.count_nonzero(self._img))
    rnonzero = property(lambda self: self.nonzero / self.size)
    sum = property(lambda self: np.nansum(self._img))
    mean = property(lambda self: np.nanmean(self._img))
    std = property(lambda self: np.nanstd(self._img))
    min = property(lambda self: np.nanmin(self._img))
    median = property(lambda self: np.nanmedian(self._img))
    max = property(lambda self: np.nanmax(self._img))
    spacing = property(lambda self: shorten(self._attrs['voxel_spacing']))
    errors = property(lambda self: len(self._attrs.get('errors', [])))


def shorten(o):
    return str(o).translate(None, ' \t')


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('path', nargs='+',
                   help='input pmap files')
    p.add_argument('-v', '--verbose', action='count',
                   help='increase verbosity')
    p.add_argument('-k', '--keys', default='shape,path',
                   help='comma-separated keys')
    return p.parse_args()


def main():
    args = parse_args()
    keys = args.keys.split(',')
    for path in args.path:
        pmap = Pmap(path)
        fmt = '{k}={v}' if args.verbose else '{v}'
        fields = (fmt.format(k=x, v=getattr(pmap, x)) for x in keys)
        print(*fields, sep='\t')


if __name__ == '__main__':
    main()
