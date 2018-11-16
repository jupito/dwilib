#!/usr/bin/python3

"""Show information on masks and write subregions of them."""

import argparse
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
    p.add_argument('--input', '-i', metavar='INFILE', required=True,
                   help='input mask file')
    p.add_argument('--subregion', '-s', metavar='OUTFILE',
                   help='output subregion file')
    p.add_argument('--pad', metavar='I', type=int, default=10,
                   help='subregion padding size (default: 10)')
    args = p.parse_args()
    return args


def write_subregion(mask, pad, infile, filename):
    """Write a subregion file from mask."""
    bb = mask.bounding_box(pad=(np.inf, pad, pad))
    bb = tuple(np.ravel(bb))
    comment = '%s, %i' % (os.path.basename(infile), pad)
    print('Writing subregion to %s with padding of %s...' % (filename, pad))
    dwi.files.write_subregion_file(filename, bb, comment=comment)


def main():
    args = parse_args()
    mask = dwi.mask.read_mask(args.input)

    selected_slices = list(mask.selected_slices())
    mbb = mask.bounding_box()
    mbb_shape = tuple([b - a for a, b in mbb])
    d = dict(infile=args.input, shape=mask.shape(), size=np.prod(mask.shape()),
             nsel=mask.n_selected(), mbb=mbb, mbb_shape=mbb_shape,
             mbb_size=np.prod(mbb_shape), mbb_all=mask.mbb_equals_selection(),
             nsl=len(selected_slices), sl=selected_slices,
             msl=mask.max_slices())
    s = (
        'mask: {infile}\n'
        'mask shape: {shape}\n'
        'mask size: {size}\n'
        'selected voxels: {nsel}\n'
        'minimum bounding box coordinates: {mbb}\n'
        'minimum bounding box shape: {mbb_shape}\n'
        'minimum bounding box size: {mbb_size} (equals selection: {mbb_all})\n'
        'selected slices: {nsl}: {sl}\n'
        'maximum slices: {msl}'
    )
    print(s.format(**d))

    if args.subregion:
        write_subregion(mask, args.pad, args.input, args.subregion)


if __name__ == '__main__':
    main()
