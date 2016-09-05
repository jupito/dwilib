#!/usr/bin/env python2

"""Draw some texture maps with focus on lesions."""

# This is for the MedPhys texturepaper.

from __future__ import absolute_import, division, print_function
import argparse
import logging

import numpy as np

import dwi.files
import dwi.mask
import dwi.paths
import dwi.plot
import dwi.util


def parse_args():
    """Parse command-line arguments."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--verbose', '-v', action='count',
                   help='be more verbose')
    p.add_argument('--featlist', '-f', default='feats.txt')
    p.add_argument('--samplelist', '-s', default='all')
    p.add_argument('--outdir', '-o', default='figs')
    return p.parse_args()


def show_image(plt, image, color, colorbar, **kwargs):
    """Show image."""
    if color:
        cmap = 'viridis'
        # cmap = 'coolwarm'
    else:
        cmap = 'gray'
    im = plt.imshow(image, cmap=cmap, **kwargs)
    if colorbar:
        dwi.plot.add_colorbar(im, pad_fraction=0)


def show_outline(plt, mask):
    """Show outline."""
    view = np.full_like(mask, np.nan, dtype=np.float16)
    view = dwi.mask.border(mask, out=view)
    # cmap = 'viridis'
    cmap = 'coolwarm'
    kwargs = dict(cmap=cmap, vmin=0, vmax=1, interpolation='nearest')
    plt.imshow(view, **kwargs)


def get_lesion_mask(masks):
    """Get unified single-slice lesion mask and index to most relevan slice."""
    def max_slices(mask):
        """Return indices of maximum slices."""
        counts = [np.count_nonzero(x) for x in mask]
        maxcount = max(counts)
        return [i for i, c in enumerate(counts) if c == maxcount]

    # Use slice with maximum lesion volume.
    mask = dwi.util.unify_masks(masks)
    centroids = [int(round(np.mean(max_slices(x)))) for x in masks]
    centroid = int(round(np.mean(max_slices(mask))))

    # centroids = [int(round(dwi.util.centroid(x)[0])) for x in masks]
    # centroid = int(round(dwi.util.centroid(mask)[0]))

    logging.debug('Lesion centroids (total): %s (%s)', centroids, centroid)
    mask = mask[centroid]

    return mask, centroid


def read_lmask(mode, case, scan):
    mode = dwi.util.ImageMode(mode)
    paths = []
    try:
        for i in range(1, 4):
            paths.append(dwi.paths.mask_path(mode, 'lesion', case, scan, i))
    except IOError:
        pass
    masks = [dwi.files.read_mask(x) for x in paths]
    lmask, img_slice = get_lesion_mask(masks)
    return lmask, img_slice


def read_pmap(mode, case, scan, img_slice):
    mode = dwi.util.ImageMode(mode)
    path = dwi.paths.pmap_path(mode, case, scan)
    pmap, _ = dwi.files.read_pmap(path, ondisk=True, params=[0])
    pmap = pmap[img_slice, :, :, 0]
    pmap = dwi.util.normalize(pmap, mode)
    return pmap


def read_tmap(mode, case, scan, img_slice, texture_spec):
    mode = dwi.util.ImageMode(mode)
    tmap = dwi.paths.texture_path(mode, case, scan, None, 'prostate', 'all', 0,
                                  texture_spec['method'],
                                  texture_spec['winsize'], voxel='all')
    param = '{winsize}-{method}({feature})'.format(**texture_spec)
    tmap, attrs = dwi.files.read_pmap(tmap, ondisk=True, params=[param])
    tmap = tmap[img_slice, :, :, 0]
    assert param == attrs['parameters'][0]
    return tmap, param


def read_pinkimage(case):
    """Read pink image."""
    from glob import glob
    import PIL
    pattern = '/mri/pink_images/extracted/{}-*'.format(case)
    paths = glob(pattern)
    if not paths:
        raise IOError('Pink image not found: {}'.format(pattern))
    arrays = [np.array(PIL.Image.open(x)) for x in sorted(paths)]
    min_width = min(x.shape[1] for x in arrays)
    arrays = [x[:, 0:min_width, :] for x in arrays]
    pink = np.concatenate(arrays)
    return pink


def rescale(image, factor):
    from scipy.ndimage import interpolation
    return interpolation.zoom(image, factor, order=0)


def rescale_(image, factor):
    from scipy.ndimage import interpolation
    typ = image.dtype
    image = image.astype(np.float)
    image = interpolation.zoom(image, factor)
    if typ == np.bool:
        image = dwi.util.asbool(image)
    else:
        image = image.astype(typ)
    return image


def read(mode, case, scan, texture_spec):
    """Read files."""
    lmask, img_slice = read_lmask(mode, case, scan)
    pmap = read_pmap(mode, case, scan, img_slice)
    tmap, param = read_tmap(mode, case, scan, img_slice, texture_spec)

    bb = dwi.util.bbox(np.isfinite(tmap), 10)
    pmap = pmap[bb]
    tmap = tmap[bb]
    lmask = lmask[bb]

    if mode.startswith('DWI'):
        pmap = rescale(pmap, 2)
        tmap = rescale(tmap, 2)
        lmask = rescale_(lmask, 2)

    tmap_lesion = np.where(lmask, tmap, np.nan)
    pmask = np.isfinite(tmap)

    try:
        pink = read_pinkimage(case)
    except IOError:
        pink = np.eye(5)

    images = dict(pmap=pmap, tmap=tmap, lmask=lmask, tmap_lesion=tmap_lesion,
                  pmask=pmask)
    assert len(set(x.shape for x in images.values())) == 1
    images['pink'] = pink
    return images, param


def plot(images, title, path):
    """Plot."""
    trange = dict(vmin=np.nanmin(images['tmap']),
                  vmax=np.nanmax(images['tmap']))
    def pink_image():
        plt.imshow(images['pink'])
    def prostate_outline():
        show_image(plt, images['pmap'], color=0, colorbar=1)
        show_outline(plt, images['pmask'])
    def lesion_outline():
        show_image(plt, images['pmap'], color=0, colorbar=1)
        show_outline(plt, images['lmask'])
    def prostate_texture():
        show_image(plt, images['pmap'], color=0, colorbar=0)
        show_image(plt, images['tmap'], color=1, colorbar=1)
        show_outline(plt, images['lmask'])
    def lesion_texture():
        show_image(plt, images['pmap'], color=0, colorbar=0)
        show_image(plt, images['tmap_lesion'], color=1, colorbar=1, **trange)
    funcs = [pink_image, prostate_outline, lesion_outline, prostate_texture,
             lesion_texture]
    it = dwi.plot.generate_plots(ncols=len(funcs), suptitle=title, path=path)
    for i, plt in enumerate(it):
        dwi.plot.noticks(plt)
        f = funcs[i]
        plt.title(f.__name__)
        f()


def cases_scans_lesions(mode, samplelist):
    mode = dwi.util.ImageMode(mode)
    path = dwi.paths.samplelist_path(mode, samplelist)
    patients = dwi.files.read_patients_file(path)
    for p in patients:
        for scan in p.scans:
            yield p.num, scan, p.lesions


def main():
    """Main."""
    args = parse_args()
    logging.basicConfig(level=logging.INFO)

    for line in dwi.files.valid_lines(args.featlist):
        mode, ws, mt, ft = line.split()
        texture_spec = dict(winsize=ws, method=mt, feature=ft)
        print(mode, texture_spec)
        for c, s, l in cases_scans_lesions(mode, args.samplelist):
            # if c not in [42, 111]:
            #     continue
            scores = '/'.join(str(x.score) for x in l)
            images, param = read(mode, c, s, texture_spec)
            title = '{}, {}-{}, {}, {}'.format(mode, c, s, scores, param)
            path = '{o}/tmap_{m}_{p}_{c:03}-{s}.png'.format(o=args.outdir,
                m=mode, p=param, c=c, s=s)
            plot(images, title, path)


if __name__ == '__main__':
    main()
