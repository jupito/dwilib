#!/usr/bin/python3

"""Check if there are voxels in the 'other' mask (e.g. lesion) that don't
overlap the 'container' mask (e.g. prostate).
"""

import argparse

import numpy as np
import matplotlib.pyplot as plt

import dwi.files
import dwi.mask
import dwi.util


def parse_args():
    """Parse command-line arguments."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('container',
                   help='container mask (e.g. prostate)')
    p.add_argument('other',
                   help='other mask (e.g. lesion)')
    p.add_argument('--verbose', '-v', action='count',
                   help='increase verbosity')
    p.add_argument('--fig',
                   help='output figure')
    return p.parse_args()


def read_mask(path):
    """Read pmap as a mask."""
    mask, _ = dwi.files.read_pmap(path)
    assert mask.ndim == 4, mask.ndim
    assert mask.shape[-1] == 1, mask.shape
    mask = mask[..., 0].astype(np.bool).astype(np.int8)
    return mask


def get_overlap(container, other):
    """Get overlap array."""
    overlap = np.zeros_like(container, dtype=np.int8)
    overlap[container == 1] = 1  # Set container voxels.
    overlap[other == 1] = 3  # Set other voxels.
    overlap[container - other == -1] = 2  # Set non-container other voxels.
    return overlap


def write_figure(overlap, d, path):
    overlap = overlap.astype(np.float32) / 3
    plt.rcParams['image.aspect'] = 'equal'
    plt.rcParams['image.cmap'] = 'jet'
    plt.rcParams['image.interpolation'] = 'none'
    n_cols = 4
    n_rows = np.ceil(len(overlap) / n_cols)
    fig = plt.figure(figsize=(n_cols*6, n_rows*6))
    s = ('{container}\n'
         '{other}\n'
         '{ncontainer},  {nother},  {noutside},  {noutsider:.2%}')
    fig.suptitle(s.format(**d), y=1.05, fontsize=30, color='darkgreen')
    for i, a in enumerate(overlap):
        ax = fig.add_subplot(n_rows, n_cols, i+1)
        ax.set_title('Slice {}/{}'.format(i, len(overlap)))
        plt.imshow(a, vmin=0, vmax=1)
    plt.tight_layout()
    print('Writing figure:', path)
    dwi.files.ensure_dir(path)
    plt.savefig(path, bbox_inches='tight')
    plt.close()


def main():
    args = parse_args()
    container = read_mask(args.container)
    other = read_mask(args.other)
    assert container.shape == other.shape, (container.shape, other.shape)
    overlap = get_overlap(container, other)
    d = dict(
        container=args.container,
        other=args.other,
        ncontainer=np.count_nonzero(overlap == 1),
        nother=np.count_nonzero(overlap == 3),
        noutside=np.count_nonzero(overlap == 2),
        )
    d['noutsider'] = d['noutside'] / d['nother']
    d['nregc'] = dwi.mask.nregions(container)
    d['nrego'] = dwi.mask.nregions(other)
    s = ('{container}, {other}: '
         '{ncontainer}, {nother}, {noutside}, {noutsider:.2%}, '
         '{nregc}/{nrego}')
    print(s.format(**d))
    if args.fig:
        write_figure(overlap, d, args.fig)


if __name__ == '__main__':
    main()
