#!/usr/bin/env python2

"""Visualize texture map alongside pmap with lesion highlighted."""

import argparse

import numpy as np

import dwi.asciifile
import dwi.dataset
import dwi.patient
import dwi.dwimage
import dwi.mask
import dwi.texture
import dwi.util

def parse_args():
    """Parse command-line arguments."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--verbose', '-v', action='count',
            help='increase verbosity')
    p.add_argument('--subregiondir', default='bounding_box_100_10pad',
            help='subregion bounding box directory')
    p.add_argument('--pmapdir', default='dicoms_Mono_combinedDICOM',
            help='input parametric map directory')
    p.add_argument('--af',
            help='input parametric map as ASCII file')
    p.add_argument('--params', nargs='+', default=['ADCm'],
            help='image parameter to use')
    p.add_argument('--case', type=int, required=True,
            help='case number')
    p.add_argument('--scan', required=True,
            help='scan identifier')
    p.add_argument('--pmaskdir', default='masks_prostate',
            help='prostate mask directory')
    p.add_argument('--lmaskdir', default='masks_lesion_DWI',
            help='lesion mask directory')
    p.add_argument('--method', default='stats',
            help='texture method')
    p.add_argument('--winsize', type=int, default=[5],
            help='window side length')
    args = p.parse_args()
    return args

def plot(pmap, lmask, tmaps, names, filename):
    import matplotlib
    import matplotlib.pyplot as plt

    assert pmap.shape == lmask.shape

    #plt.rcParams['image.cmap'] = 'coolwarm'
    plt.rcParams['image.cmap'] = 'YlGnBu_r'
    plt.rcParams['image.interpolation'] = 'nearest'
    n_cols, n_rows = len(tmaps)+1, 1
    fig = plt.figure(figsize=(n_cols*6, n_rows*6))

    ax1 = fig.add_subplot(1, n_cols, 1)
    ax1.set_title('parametric map')
    #view = np.ones(pmap.shape + (3,), dtype=float)
    #view[...,0] = view[...,1] = view[...,2] = pmap / pmap.max()
    #plt.imshow(view)
    impmap = plt.imshow(pmap)
    view = np.zeros(lmask.shape + (4,), dtype=float)
    view[...,0] = view[...,3] = lmask
    plt.imshow(view, alpha=0.6)

    for i, (tmap, name) in enumerate(zip(tmaps, names)):
        ax = fig.add_subplot(1, n_cols, i+2)
        ax.set_title(name)
        plt.imshow(tmap, vmin=0)

    fig.colorbar(impmap, ax=ax1, shrink=0.65)

    plt.tight_layout()
    print 'Writing figure:', filename
    plt.savefig(filename, bbox_inches='tight')
    plt.close()


args = parse_args()
if args.verbose:
    print 'Reading data...'
data = dwi.dataset.dataset_read_samples([(args.case, args.scan)])
dwi.dataset.dataset_read_subregions(data, args.subregiondir)
if args.af:
    af = dwi.asciifile.AsciiFile(args.af)
    print 'params:', af.params()
    # Fix switched height/width.
    subwindow = af.subwindow()
    subwindow = subwindow[2:] + subwindow[:2]
    assert len(subwindow) == 4
    subwindow_shape = dwi.util.subwindow_shape(subwindow)
    image = af.a
    image.shape = (20,) + subwindow_shape + (len(af.params()),)
    data[0]['subregion'] = (0, 20) + subwindow
    data[0]['image'] = image
    print data[0]['subregion']
else:
    dwi.dataset.dataset_read_pmaps(data, args.pmapdir, args.params)
dwi.dataset.dataset_read_prostate_masks(data, args.pmaskdir)
dwi.dataset.dataset_read_lesion_masks(data, args.lmaskdir)

data = data[0]
print 'image shape:', data['image'].shape
print 'lesion mask sizes:', [m.n_selected() for m in data['lesion_masks']]

# Find maximum lesion and use it.
lesion = max((m.n_selected(), i) for i, m in enumerate(data['lesion_masks']))
print 'max lesion:', lesion
max_lesion = lesion[1]

max_slice = data['lesion_masks'][max_lesion].max_slices()[0]
pmap = data['image'][max_slice]
proste_mask = data['prostate_mask'].array[max_slice]
lesion_mask = data['lesion_masks'][max_lesion].array[max_slice]

#dwi.util.clip_pmap(pmap, args.param)
pmap = pmap[:,:,0]
print dwi.util.fivenum(pmap)
pmap = dwi.util.clip_outliers(pmap, out=pmap)
tmaps, names = dwi.texture.texture_map(args.method, pmap, args.winsize,
        mask=proste_mask)
print names

filename = 'texture_{case}_{scan}'.format(**data)
plot(pmap, lesion_mask, tmaps, names, filename)
