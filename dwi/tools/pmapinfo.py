#!/usr/bin/python3

"""Print information about pmaps."""

import numpy as np

import dwi.conf
import dwi.files
from dwi.files import Path
import dwi.image
import dwi.mask
import dwi.util

lambdas = dict(
    path=lambda x: x.info['path'],
    name=lambda x: x.info['path'].name,
    stem=lambda x: x.info['path'].stem,
    type=lambda x: x.dtype,
    shape=lambda x: shorten(x.shape),
    size=lambda x: x.size,
    mbb=lambda x: shorten(x[x.mbb()].shape),
    spacing=lambda x: shorten(x.spacing),
    centroid=lambda x: ','.join(str(int(round(y))) for y in x.centroid()),

    finite=lambda x: np.count_nonzero(np.isfinite(x)),
    nonzero=lambda x: np.count_nonzero(x),
    neg=lambda x: np.count_nonzero(x < 0),

    rfinite=lambda x: lambdas['finite'](x) / x.size,
    rnonzero=lambda x: lambdas['nonzero'](x) / x.size,
    rneg=lambda x: lambdas['neg'](x) / x.size,

    sum=lambda x: np.nansum(x),
    mean=lambda x: np.nanmean(x),
    std=lambda x: np.nanstd(x),
    var=lambda x: np.nanvar(x),
    min=lambda x: np.nanmin(x),
    median=lambda x: np.nanmedian(x),
    max=lambda x: np.nanmax(x),
    five=lambda x: shorten(dwi.util.fivenums(x)),

    regs=lambda x: dwi.mask.nregions(x.squeeze()),

    errors=lambda x: len(x.info['attrs'].get('errors', ())),
    ce16=lambda x: cast_errors(x, np.float16),
    ce32=lambda x: cast_errors(x, np.float32),
    )


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
    epilog = 'Available keys: {}'.format(','.join(sorted(lambdas.keys())))
    p = dwi.conf.get_parser(description=__doc__, epilog=epilog)
    p.add('path', nargs='+', type=Path,
          help='input pmap files')
    p.add('-p', '--params', nargs='*',
          help='parameters')
    p.add('-m', '--masks', metavar='MASKFILE', nargs='+',
          help='mask files')
    p.add('-k', '--keys', default='shape,path',
          help='comma-separated keys for specifiying requested info')
    p.add('--ofs', default='\t',
          help='output field separator')
    return p.parse_args()


def main():
    args = parse_args()
    keys = args.keys.split(',')
    if args.masks:
        mask = dwi.util.unify_masks(dwi.files.read_mask(x) for x in args.masks)
    for path in args.path:
        try:
            pmap = dwi.image.Image.read(path, params=args.params)
        except Exception as e:
            print('Could not read image: {}: {}'.format(path, e))
            continue
        if args.masks:
            pmap.apply_mask(mask)
        fmt = '{k}={v}' if args.verbose else '{v}'
        fields = (fmt.format(k=x, v=lambdas[x](pmap)) for x in keys)
        print(*fields, sep=args.ofs)


if __name__ == '__main__':
    main()
