#!/usr/bin/python3

"""Select voxels from image and write them into another file. Output may
include all voxels, or a selection made by subwindow specification or mask
file.

Multiple same-size images may be combined by overlaying the parameters. In this
case, the output file will receive its attributes from the first input file,
except for the 'parameters' attribute, which will be aggregated.
"""

import argparse
from functools import reduce
from operator import add
import os.path

import numpy as np

import dwi.files
import dwi.mask
import dwi.util


def parse_args():
    """Parse command-line arguments."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('-v', '--verbose', action='count',
                   help='increase verbosity')
    p.add_argument('-i', '--input', metavar='INFILE', nargs='+', required=True,
                   help='input parametric map files')
    p.add_argument('-o', '--output', metavar='OUTFILE',
                   help='output parametric map file')
    p.add_argument('-m', '--mask', metavar='MASKFILE', required=False,
                   help='mask file (applied within subwindow size)')
    p.add_argument('-s', '--subwindow', metavar='I', nargs=6, type=int,
                   default=[], required=False,
                   help='use subwindow (specified by 6 one-based indices)')
    p.add_argument('-p', '--param', type=int,
                   help='parameter index')
    p.add_argument('--keepmasked', action='store_true',
                   help='keep masked voxels (as NaN values)')
    p.add_argument('--subwindow-mask', action='store_true',
                   help='apply subwindow on mask, too')
    p.add_argument('--rename_params', metavar='NAME', nargs='+',
                   help='parameter rename list')
    p.add_argument('--source_attrs', action='store_true',
                   help='output attributes include source files')
    p.add_argument('--astype',
                   help='converted type (e.g. bool, float16, ...)')
    return p.parse_args()


def merge(tuples):
    """Merge pmaps. Parameter names are aggregated; for other attributes, only
    the first file is used.
    """
    image = np.concatenate([x for x, _ in tuples], axis=-1)
    attrs = tuples[0][1]
    attrs['parameters'] = reduce(add, (x['parameters'] for _, x in tuples))
    return image, attrs


def main():
    """Main."""
    args = parse_args()
    image, attrs = merge([dwi.files.read_pmap(x) for x in args.input])

    if args.param is not None:
        image = image[..., [args.param]]
        attrs['parameters'] = [attrs['parameters'][args.param]]

    if args.astype:
        attrs['source_type'] = str(image.dtype)
        image = image.astype(args.astype)

    if args.subwindow:
        if args.verbose:
            print('Using subwindow %s' % args.subwindow)
        image = dwi.util.crop_image(image, args.subwindow,
                                    onebased=True).copy()

    if args.mask:
        mask = dwi.mask.read_mask(args.mask)
        if args.subwindow and args.subwindow_mask:
            mask = mask.get_subwindow(args.subwindow)
        if args.verbose:
            print('Using mask %s' % mask)
        if args.keepmasked:
            image = mask.apply_mask(image, value=np.nan)
        else:
            image = mask.selected(image)

    if args.rename_params:
        attrs['parameters'] = args.rename_params
    if args.source_attrs:
        attrs['source_files'] = args.input

    # Write output voxels. Unless output filename is specified, one will be
    # constructed from (first) input filename.
    outfile = args.output or os.path.basename(args.input[0]) + '.txt'
    if args.verbose:
        print('Writing image {s} with {n} attributes to {f}'.format(
            s=image.shape, n=len(attrs), f=outfile))
    dwi.files.write_pmap(outfile, image, attrs)


if __name__ == '__main__':
    main()
