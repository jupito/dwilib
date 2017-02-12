#!/usr/bin/env python2

"""Print information about pmaps."""

from __future__ import absolute_import, division, print_function

from pathlib2 import Path
import numpy as np

import dwi.conf
import dwi.files
import dwi.image
import dwi.util


class Pmap(object):
    """Open pmap to request properties from."""

    def __init__(self, path, **kwargs):
        self.path = Path(path)
        self._img = dwi.image.Image.read(str(path), **kwargs)

    name = property(lambda self: self.path.name)
    stem = property(lambda self: self.path.stem)
    type = property(lambda self: self._img.dtype)
    shape = property(lambda self: shorten(self._img.shape))
    size = property(lambda self: self._img.size)
    mbb = property(lambda self: shorten(self._img[self._img.mbb()].shape))
    spacing = property(lambda self: shorten(self._img.spacing))

    finite = property(lambda self: np.count_nonzero(np.isfinite(self._img)))
    nonzero = property(lambda self: np.count_nonzero(self._img))
    neg = property(lambda self: np.count_nonzero(self._img < 0))

    rfinite = property(lambda self: self.finite / self.size)
    rnonzero = property(lambda self: self.nonzero / self.size)
    rneg = property(lambda self: self.neg / self.size)

    sum = property(lambda self: np.nansum(self._img))
    mean = property(lambda self: np.nanmean(self._img))
    std = property(lambda self: np.nanstd(self._img))
    var = property(lambda self: np.nanvar(self._img))
    min = property(lambda self: np.nanmin(self._img))
    median = property(lambda self: np.nanmedian(self._img))
    max = property(lambda self: np.nanmax(self._img))
    five = property(lambda self: shorten(dwi.util.fivenums(self._img)))

    errors = property(lambda self: len(self._img.info['attrs'].get('errors',
                                                                   ())))
    ce16 = property(lambda self: cast_errors(self._img, np.float16))
    ce32 = property(lambda self: cast_errors(self._img, np.float32))


def cast_errors(a, dtype):
    """Return the number of finite ndarray elements that do not stay close to
    the original value after type casting. See numpy.isclose().
    """
    a = a[np.isfinite(a)]
    return a.size - np.count_nonzero(np.isclose(a, a.astype(dtype)))


def shorten(o):
    """Make object string and remove all whitespace."""
    if isinstance(o, np.ndarray):
        o = list(o)
    return ''.join(str(o).split())


def parse_args():
    available_keys = sorted(x for x in dir(Pmap) if not x.startswith('_'))
    epilog = 'Available keys: {}'.format(','.join(available_keys))
    p = dwi.conf.get_parser(description=__doc__, epilog=epilog)
    p.add('path', nargs='+',
          help='input pmap files')
    p.add('-p', '--params', nargs='*',
          help='parameters')
    p.add('-m', '--masks', metavar='MASKFILE', nargs='+',
          help='mask files')
    p.add('-k', '--keys', default='shape,path',
          help='comma-separated keys for specifiying requested info')
    return p.parse_args()


def main():
    args = parse_args()
    keys = args.keys.split(',')
    if args.masks:
        mask = dwi.util.unify_masks(dwi.files.read_mask(x) for x in args.masks)
    for path in args.path:
        pmap = Pmap(path, params=args.params)
        if args.masks:
            pmap._img.apply_mask(mask)
        fmt = '{k}={v}' if args.verbose else '{v}'
        fields = (fmt.format(k=x, v=getattr(pmap, x)) for x in keys)
        print(*fields, sep='\t')


if __name__ == '__main__':
    main()
