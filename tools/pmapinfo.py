#!/usr/bin/env python2

"""Print information about pmaps."""

from __future__ import absolute_import, division, print_function
import argparse
import os.path

import numpy as np

import dwi.files
import dwi.util


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('path', nargs='+',
                   help='input pmap files')
    p.add_argument('-v', '--verbose', action='count',
                   help='increase verbosity')
    p.add_argument('-k', '--keys', default='shape,path',
                   help='comma-separated keys')
    return p.parse_args()


class Pmap(object):
    def __init__(self, path):
        self.path = path
        self._img, self._attrs = dwi.files.read_pmap(path)

    keys = property(lambda self: ','.join(sorted(x for x in dir(self) if not
                                                 x.startswith('_'))))
    file = property(lambda self: os.path.basename(self.path))
    root = property(lambda self: os.path.splitext(self.file)[0])
    shape = property(lambda self: self._img.shape)
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
    spacing = property(lambda self: self._attrs['voxel_spacing'])


def main():
    args = parse_args()
    keys = args.keys.split(',')
    # keys = []
    # for k in args.keys.split(','):
    #     if k in d:
    #         keys.append(k)
    #     else:
    #         keys.extend(x for x in d.keys() if x.startswith(k))
    for path in args.path:
        pmap = Pmap(path)
        fmt = '{k}={v}' if args.verbose else '{v}'
        fields = (fmt.format(k=x, v=getattr(pmap, x)) for x in keys)
        print(*fields)

        # d = dict(path=path, paramlen=max(len(x) for x in attrs['parameters']))
        # for i, param in enumerate(attrs['parameters']):
        #     a = img[..., i]
        #     # a[a==0] = np.nan
        #     nans = np.isnan(a)
        #     if np.any(nans):
        #         a = a[-nans]
        #     d.update(param=param, nonnans=a.size/nans.size,
        #              min=np.min(a), max=np.max(a), mean=np.mean(a),
        #              median=np.median(a), p='.4')
        #     s = '{path}  {param:{paramlen}}  {nonnans:{p}%}  {min:{p}f}  {max:{p}f}  {mean:{p}f}  {median:{p}f}'
        #     #print(s.format(**d))
        #     print(img.shape, path)
        #     # if args.basic:
        #     #     s = '{i} {param} {mean:g} {std:g} {var:g} {sum:g}'
        #     #     d.update(mean=a.mean(), std=a.std(), var=a.var(), sum=sum(a))
        #     #     print(s.format(**d))
        #     # if args.five:
        #     #     s = '{i} {param} {min:g} {q1:g} {median:g} {q3:g} {max:g}'
        #     #     d.update(dwi.util.fivenumd(a))
        #     #     print(s.format(**d))


if __name__ == '__main__':
    main()
