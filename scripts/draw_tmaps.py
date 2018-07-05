#!/usr/bin/python3

"""Draw some texture maps with focus on lesions."""

# This is for the MedPhys texturepaper.

import argparse
import logging

import numpy as np

import dwi.files
import dwi.mask
import dwi.patient
import dwi.paths
import dwi.plot
from dwi.types import ImageMode, Path, TextureSpec
import dwi.util


def parse_args():
    """Parse command-line arguments."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('-v', '--verbose', action='count',
                   help='increase verbosity')
    p.add_argument('-f', '--featlist', default='feats.txt')
    p.add_argument('-s', '--samplelist', default='all')
    p.add_argument('-o', '--outdir', default='figs')
    return p.parse_args()


def show_image(plt, image, colorbar=True, scale=None, **kwargs):
    """Show image."""
    d = {}
    if scale is not None:
        d['vmin'], d['vmax'] = scale
    d.update(kwargs)
    im = plt.imshow(image, **d)
    if colorbar:
        dwi.plot.add_colorbar(im, pad_fraction=0, format='')


def show_outline(plt, masks, cmaps=None):
    """Show outline."""
    if cmaps is None:
        # cmaps = ('coolwarm', 'viridis', 'hot')
        # cmaps = ['spring'] * 3
        # cmaps = ['rainbow'] * 3
        # cmaps = 'spring', 'summer', 'autumn', 'winter'
        cmaps = ['Wistia', 'cool_r', 'spring']
    assert len(masks) <= len(cmaps)
    for mask, cmap in zip(masks, cmaps):
        view = np.full_like(mask, np.nan, dtype=np.float)
        view = dwi.mask.border(mask, out=view)
        d = dict(cmap=cmap, interpolation='nearest', vmin=0, vmax=1, alpha=1.0)
        plt.imshow(view, **d)


def get_lesion_mask(masks, slice_index=None):
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
    logging.info('Mask shape: %s, centroid: %i, slice: %s', mask.shape,
                 centroid, slice_index)
    if slice_index is None:
        slice_index = centroid
    mask = mask[slice_index]

    return mask, slice_index


def read_lmask(mode, case, scan):
    mode = ImageMode(mode)
    paths = []
    try:
        for i in range(1, 4):
            path = Path(dwi.paths.mask_path(mode, 'lesion', case, scan,
                                            lesion=i))
            if path.exists():
                paths.append(path)
    except IOError:
        pass
    masks = [dwi.files.read_mask(x) for x in paths]

    # # Manually override slice index.
    # slice_indices = {
    #     (64, '1a', 'T2w-std'): 7,
    #     (64, '1a', 'T2-fitted'): 5,
    #     }

    # slice_index = slice_indices.get((case, scan, str(mode)))
    slice_index = None
    lmask, img_slice = get_lesion_mask(masks, slice_index=slice_index)
    return lmask, img_slice, [x[img_slice] for x in masks]


def read_pmap(mode, case, scan, img_slice):
    mode = ImageMode(mode)
    path = dwi.paths.pmap_path(mode, case, scan)
    pmap, _ = dwi.files.read_pmap(path, ondisk=True, params=[0])
    pmap = pmap[img_slice, :, :, 0]
    pmap = dwi.util.normalize(pmap, mode)
    return pmap


def read_tmap(mode, case, scan, img_slice, texture_spec):
    mode = ImageMode(mode)
    path = dwi.paths.texture_path(mode, case, scan, None, 'prostate', 'all', 0,
                                  texture_spec, voxel='all')

    # TODO: Kludge to remove `_mbb` from `glcm_mbb`. Filenames don't have it.
    t = texture_spec._replace(method=texture_spec.method.split('_')[0])

    param = '{t.winsize}-{t.method}({t.feature})'.format(t=t)
    tmap, attrs = dwi.files.read_pmap(path, ondisk=True, params=[param])
    tscale = tuple(np.nanpercentile(tmap[:, :, :, 0], (1, 99)))
    tmap = tmap[img_slice, :, :, 0]
    assert param == attrs['parameters'][0]
    return tmap, param, tscale


def read_histology(case):
    """Read histology section image."""
    from glob import glob
    import PIL
    pattern = '/mri/hist/pink_images/extracted/{}-*'.format(case)
    paths = glob(pattern)
    if not paths:
        raise IOError('Histology image not found: {}'.format(pattern))
    images = [np.array(PIL.Image.open(x)) for x in sorted(paths)]
    # If several, concatenate by common width.
    min_width = min(x.shape[1] for x in images)
    images = [x[:, 0:min_width, :] for x in images]
    image = np.concatenate(images)
    return image


# def rescale(image, factor, order=0):
#     """Rescale."""
#     from scipy.ndimage import interpolation
#     return interpolation.zoom(image, factor, order=order)


# def rescale_as_float(image, factor):
#     """Convert to float, rescale, convert back. Special boolean handling."""
#     from scipy.ndimage import interpolation
#     typ = image.dtype
#     image = image.astype(np.float)
#     image = interpolation.zoom(image, factor)
#     if typ == np.bool:
#         image = dwi.util.asbool(image)
#     else:
#         image = image.astype(typ)
#     return image


rescale = dwi.util.zoom
rescale_as_float = dwi.util.zoom_as_float


def read(mode, case, scan, texture_spec):
    """Read files."""
    try:
        histology = read_histology(case)
    except IOError:
        # histology = np.eye(5)
        raise

    lmask, img_slice, lmasks = read_lmask(mode, case, scan)
    pmap = read_pmap(mode, case, scan, img_slice)
    tmap, param, tscale = read_tmap(mode, case, scan, img_slice, texture_spec)

    bb = dwi.util.bbox(np.isfinite(tmap), 10)
    pmap = pmap[bb].copy()
    tmap = tmap[bb].copy()
    lmask = lmask[bb].copy()
    lmasks = [x[bb].copy() for x in lmasks]

    # if mode.startswith('DWI'):
    #     pmap = rescale(pmap, 2)
    #     tmap = rescale(tmap, 2)
    #     lmask = rescale_as_float(lmask, 2)
    #     lmasks = [rescale_as_float(x, 2) for x in lmasks]

    # Remove lesion voxels outside prostate.
    lmask[np.isnan(tmap)] = False
    for mask in lmasks:
        lmask[np.isnan(tmap)] = False
    pmap_prostate = np.where(np.isfinite(tmap), pmap, np.nan)
    tmap_lesion = np.where(lmask, tmap, np.nan)
    pmask = np.isfinite(tmap)

    images = dict(pmap=pmap, tmap=tmap, lmask=lmask,
                  pmap_prostate=pmap_prostate, tmap_lesion=tmap_lesion,
                  pmask=pmask)
    assert len({x.shape for x in images.values()} |
               {x.shape for x in lmasks}) == 1
    images['lmasks'] = lmasks
    images['histology'] = histology
    images['tscale'] = tscale
    return images, param


def plot(images, title, path):
    """Plot."""
    pscale = (0, 1)
    # tscale = tuple(np.nanpercentile(images['tmap'], (1, 99)))
    tscale = images['tscale']

    def histology_image(plt):
        plt.imshow(images['histology'])
        # plt.title('histology section')

    # def pmap(plt):
    #     show_image(plt, images['pmap'], scale=pscale, cmap='gray')
    #

    def prostate_pmap(plt):
        # XXX: Scale in these funcs?
        show_image(plt, images['pmap_prostate'], scale=pscale, cmap='gray')
        show_outline(plt, images['lmasks'])

    def prostate_texture(plt):
        show_image(plt, images['tmap'], scale=tscale)
        show_image(plt, images['tmap_lesion'])
        show_outline(plt, images['lmasks'])

    def lesion_texture(plt):
        # show_image(plt, images['tmap_lesion'], scale=tscale)
        show_image(plt, images['tmap_lesion'])

    funcs = [histology_image, prostate_pmap, prostate_texture]
    it = dwi.plot.generate_plots(ncols=len(funcs), suptitle=title, path=path)
    for i, plt in enumerate(it):
        plt.rcParams['savefig.dpi'] = '300'
        dwi.plot.noticks(plt)
        f = funcs[i]
        # plt.title(f.__name__.replace('_', ' '))
        plt.title('')
        f(plt)


def cases_scans_lesions(mode, samplelist, thresholds=None):
    """Iterate (case_id, scan_id, lesions)."""
    mode = ImageMode(mode)
    path = dwi.paths.samplelist_path(mode, samplelist)
    patients = dwi.files.read_patients_file(path)
    dwi.patient.label_lesions(patients, thresholds=thresholds)
    return ((p.num, s, p.lesions) for p in patients for s in p.scans)


def main():
    """Main."""
    args = parse_args()
    logging.basicConfig()
    # logging.basicConfig(level=logging.INFO)

    # thresholds = None
    # thresholds = ('3+3', '3+4')
    thresholds = ('3+3',)
    blacklist = []  # + [21, 22, 27, 42, 74, 79]
    # whitelist = []  # + [23, 24, 26, 29, 64]
    whitelist = [26, 42, 64]

    for i, line in enumerate(dwi.files.valid_lines(args.featlist)):
        words = line.split()
        mode = words[0]
        texture_spec = TextureSpec(*words[1:])
        it = cases_scans_lesions(mode, args.samplelist, thresholds=thresholds)
        for c, s, l in it:
            if blacklist and c in blacklist:
                continue
            if whitelist and c not in whitelist:
                continue
            # if 0 not in (x.label for x in l):
            #     continue  # Exclude if there's no first score group present.
            print(i, mode, texture_spec, c, s, l)
            try:
                images, _ = read(mode, c, s, texture_spec)
            except IOError as e:
                logging.error(e)
                continue

            labelnames = ['low', 'high']
            lesions = ', '.join('{} {} {}'.format(x.score, x.location,
                                                  labelnames[x.label])
                                for x in l)
            d = dict(m=mode, c=c, s=s, l=lesions, tw=texture_spec.winsize,
                     tm=texture_spec.method, tf=texture_spec.feature,
                     suffix='png')
            title = '{c}-{s} ({l})\n{m} {tm}({tf})-{tw}'.format(**d)
            path = '{c:03}-{s}_{m}_{tm}({tf})-{tw}.{suffix}'.format(**d)
            plot(images, title, Path(args.outdir, path))


if __name__ == '__main__':
    main()
