#!/usr/bin/env python2

"""Visualize texture map alongside pmap with lesion highlighted."""

from __future__ import absolute_import, division, print_function
import argparse

import numpy as np

import dwi.asciifile
import dwi.dataset
import dwi.texture
import dwi.util


def parse_args():
    """Parse command-line arguments."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--verbose', '-v', action='count',
                   help='increase verbosity')
    p.add_argument('--subregiondir',
                   help='subregion bounding box directory')
    p.add_argument('--pmapdir',
                   help='input parametric map directory')
    p.add_argument('--af',
                   help='input parametric map as ASCII file')
    p.add_argument('--params', nargs='+', default=['ADCm'],
                   help='image parameter to use')
    p.add_argument('--case', type=int, required=True,
                   help='case number')
    p.add_argument('--scan', required=True,
                   help='scan identifier')
    p.add_argument('--pmaskdir',
                   help='prostate mask directory')
    p.add_argument('--lmaskdir',
                   help='lesion mask directory')
    p.add_argument('--method', default='stats',
                   help='texture method')
    p.add_argument('--winsize', type=int, default=5,
                   help='window side length')
    args = p.parse_args()
    return args


def plot(pmaps, titles, lmask, n_rows, filename):
    import matplotlib.pyplot as plt

    assert pmaps[0].shape == lmask.shape
    # plt.rcParams['image.cmap'] = 'jet'
    # plt.rcParams['image.cmap'] = 'coolwarm'
    plt.rcParams['image.cmap'] = 'YlGnBu_r'
    plt.rcParams['image.interpolation'] = 'nearest'
    # plt.rcParams['text.usetex'] = True
    n_cols = len(pmaps) // n_rows
    fig = plt.figure(figsize=(n_cols*6, n_rows*6))

    for i, (pmap, title) in enumerate(zip(pmaps, titles)):
        ax = fig.add_subplot(n_rows, n_cols, i+1)
        ax.set_title(title)
        if i % n_cols == 0:
            impmap = plt.imshow(pmap)
            view = np.zeros(lmask.shape + (4,), dtype=np.float_)
            view[..., 0] = view[..., 3] = lmask
            plt.imshow(view, alpha=0.6)
        else:
            vmin, vmax = 0, None
            if title == 'LBP':
                vmax = 0.28
            impmap = plt.imshow(pmap, vmin=vmin, vmax=vmax)
        fig.colorbar(impmap, ax=ax, shrink=0.65)

    plt.tight_layout()
    print('Writing figure:', filename)
    plt.savefig(filename, bbox_inches='tight')
    plt.close()


def main():
    args = parse_args()
    if args.verbose:
        print('Reading data...')
    data = dwi.dataset.dataset_read_samples([(args.case, args.scan)])
    dwi.dataset.dataset_read_subregions(data, args.subregiondir)
    if args.af:
        af = dwi.asciifile.AsciiFile(args.af)
        print('params:', af.params())
        # Fix switched height/width.
        subwindow = af.subwindow()
        subwindow = subwindow[2:] + subwindow[:2]
        assert len(subwindow) == 4
        subwindow_shape = dwi.util.subwindow_shape(subwindow)
        image = af.a
        image.shape = (20,) + subwindow_shape + (len(af.params()),)
        data[0]['subregion'] = (0, 20) + subwindow
        data[0]['image'] = image
        print(data[0]['subregion'])
    else:
        dwi.dataset.dataset_read_pmaps(data, args.pmapdir, args.params)
    dwi.dataset.dataset_read_prostate_masks(data, args.pmaskdir)
    dwi.dataset.dataset_read_lesion_masks(data, args.lmaskdir)

    data = data[0]
    print('image shape:', data['image'].shape)
    print('lesion mask sizes:', [m.n_selected() for m in data['lesion_masks']])

    # Find maximum lesion and use it.
    lesion = max((m.n_selected(), i) for i, m in
                 enumerate(data['lesion_masks']))
    print('max lesion:', lesion)
    max_lesion = lesion[1]

    max_slice = data['lesion_masks'][max_lesion].max_slices()[0]
    pmap = data['image'][max_slice]
    proste_mask = data['prostate_mask'].array[max_slice]
    lesion_mask = data['lesion_masks'][max_lesion].array[max_slice]

    pmaps = []
    titles = []
    for i, param in enumerate(args.params):
        p = pmap[:, :, i]
        print(param, dwi.util.fivenum(p))
        dwi.util.clip_outliers(p, out=p)
        pmaps.append(p)
        titles.append(param)
        tmaps, names = dwi.texture.stats_map(p, 5, ['q1'], mask=proste_mask)
        pmaps.append(tmaps[0])
        titles.append('1st quartile')
        tmaps, names = dwi.texture.gabor_map(p, 5, [1], [0.1],
                                             mask=proste_mask)
        pmaps.append(tmaps[0])
        titles.append('Gabor')
        tmaps, names = dwi.texture.moment_map(p, 9, 2, mask=proste_mask)
        pmaps.append(tmaps[-1])
        titles.append('Moment')
        tmaps, names = dwi.texture.lbp_freq_map(p, 9, mask=proste_mask)
        pmaps.append(tmaps[8])
        titles.append('LBP')
        tmaps, names = dwi.texture.sobel_map(p, None, mask=proste_mask)
        pmaps.append(tmaps[-1])
        titles.append('Sobel')
        tmaps, names = dwi.texture.haar_map(p, 7, 2, mask=proste_mask)
        pmaps.append(tmaps[8])
        titles.append('Haar level 2')
        # tmaps, names = dwi.texture.texture_map(args.method, p, args.winsize,
        #                                        proste_mask)
        pmaps += list(tmaps)
        titles += names

    filename = 'texture_{case}_{scan}'.format(**data)
    plot(pmaps, titles, lesion_mask, len(args.params), filename)


if __name__ == '__main__':
    main()
