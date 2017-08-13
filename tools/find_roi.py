#!/usr/bin/python3

"""Find most interesting ROI's in a DWI image."""

import argparse
# import logging

import dwi.autoroi
import dwi.dataset
import dwi.mask
import dwi.patient
import dwi.util
from dwi.types import AlgParams, Path

# import matplotlib
import matplotlib.pyplot as plt
import numpy as np

DEFAULT_OUT_IMAGE_DIR = Path('find_roi_images')
DEFAULT_OUT_MASK_DIR = Path('masks_auto')


def parse_args():
    """Parse command-line arguments."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('-v', '--verbose', action='count',
                   help='increase verbosity')
    p.add_argument('--patients', type=Path,
                   help='patients file')
    p.add_argument('--pmapdir', type=Path,
                   help='input parametric map directory')
    p.add_argument('--subregiondir', type=Path,
                   help='subregion bounding box directory')
    p.add_argument('--prostatemaskdir', type=Path,
                   help='ROI mask directory')
    p.add_argument('--roimaskdir', type=Path,
                   help='ROI mask directory')
    p.add_argument('-p', '--param', default='ADCm',
                   help='image parameter to use')
    p.add_argument('--roidim', metavar='I', nargs=3, type=int,
                   default=[1, 5, 5],
                   help='dimensions of wanted ROI (3 integers; default 1 5 5)')
    p.add_argument('--algparams', metavar='I', nargs=5, type=int,
                   default=[2, 3, 10, 10, 500],
                   help='algorithm params (ROI side min, max, number of ROIs)')
    p.add_argument('--cases', metavar='I', nargs='*', type=int, default=[],
                   help='case numbers')
    p.add_argument('--scans', metavar='S', nargs='*', default=[],
                   help='scan identifiers')
    p.add_argument('--outmask', type=Path,
                   help='output mask file')
    p.add_argument('--outfig', type=Path,
                   help='output figure file')
    p.add_argument('--clip', action='store_true',
                   help='clip image intensity values on load')
    return p.parse_args()


def clip_pmap(img, params):
    """Clip pmap's parameter-specific intensity outliers in-place."""
    # XXX: Obsolete, use dwi.util.normalize().
    for i in range(img.shape[-1]):
        if params[i].startswith('ADC'):
            img[..., i].clip(0, 0.002, out=img[..., i])
        elif params[i].startswith('K'):
            img[..., i].clip(0, 2, out=img[..., i])


def draw_roi(img, pos, color):
    """Draw a rectangle ROI on a layer."""
    y, x = pos
    # img[y:y+5, x:x+5] = color
    img[y:y+5, x+0] = color
    img[y:y+5, x+4] = color
    img[y+0, x:x+5] = color
    img[y+4, x:x+5] = color


def get_roi_layer(img, pos, color):
    """Get a layer with a rectangle ROI for drawing."""
    layer = np.zeros(img.shape + (4,), dtype=np.float32)
    draw_roi(layer, pos, color)
    return layer


def draw(data, param, path):
    plt.rcParams['image.cmap'] = 'gray'
    plt.rcParams['image.interpolation'] = 'nearest'
    n_cols, n_rows = 3, 1
    fig = plt.figure(figsize=(n_cols*6, n_rows*6))

    colors = dict(
        cancer=(1.0, 0.0, 0.0, 1.0),
        normal=(0.0, 1.0, 0.0, 1.0),
        auto=(1.0, 1.0, 0.0, 1.0),
    )

    slice_index = data['roi_corner'][0]
    pmap = data['image'][slice_index, :, :, 0:1].copy()
    clip_pmap(pmap, [param])
    pmap = pmap[..., 0]

    # cancer_pos = (-1, -1)
    # normal_pos = (-1, -1)
    # distance = -1
    auto_pos = (data['roi_coords'][1][0], data['roi_coords'][2][0])
    # if 'cancer_mask' in data:
    #     cancer_pos = data['cancer_mask'].where()[0][1:3]
    #     distance = dwi.util.distance(cancer_pos, auto_pos)
    # if 'normal_mask' in data:
    #     normal_pos = data['normal_mask'].where()[0][1:3]

    ax1 = fig.add_subplot(1, n_cols, 1)
    # ax1.set_title('Slice %i %s' % (slice_index, param))
    plt.imshow(pmap)

    ax2 = fig.add_subplot(1, n_cols, 2)
    # ax2.set_title('Calculated score map')
    scoremap = data['scoremap'][slice_index]
    scoremap /= scoremap.max()
    imgray = plt.imshow(pmap, alpha=1)
    imjet = plt.imshow(scoremap, alpha=0.8, cmap='jet')

    ax3 = fig.add_subplot(1, n_cols, 3)
    # ax3.set_title('ROIs: %s, %s, distance: %.2f' % (cancer_pos, auto_pos,
    #                                                 distance))
    view = np.zeros(pmap.shape + (3,), dtype=np.float32)
    view[..., 0] = pmap / pmap.max()
    view[..., 1] = pmap / pmap.max()
    view[..., 2] = pmap / pmap.max()
    # for i, a in enumerate(pmap):
    #     for j, v in enumerate(a):
    #         if v < dwi.autoroi.ADCM_MIN:
    #             view[i, j, :] = [0.5, 0, 0]
    #         elif v > dwi.autoroi.ADCM_MAX:
    #             view[i, j, :] = [0, 0.5, 0]
    plt.imshow(view)
    d = dict(alpha=0.7)
    # if 'cancer_mask' in data:
    #     plt.imshow(get_roi_layer(pmap, cancer_pos, colors['cancer']), **d)
    # if 'normal_mask' in data:
    #     plt.imshow(get_roi_layer(pmap, normal_pos, colors['normal']), **d)
    # plt.imshow(get_roi_layer(pmap, auto_pos, colors['auto']), **d)
    plt.imshow(get_roi_layer(pmap, auto_pos, colors['cancer']), **d)

    fig.colorbar(imgray, ax=ax1, shrink=0.65)
    fig.colorbar(imjet, ax=ax2, shrink=0.65)
    fig.colorbar(imgray, ax=ax3, shrink=0.65)

    plt.tight_layout()
    print('Writing figure: {}'.format(path))
    plt.savefig(str(path), bbox_inches='tight')
    plt.close()


def write_mask(d, path):
    """Write mask. XXX: Here only single-slice ones."""
    slice_index = d['roi_corner'][0]
    a = np.zeros((d['original_shape'][1:3]), dtype=np.int8)
    _, y, x = d['roi_coords']
    y_offset, x_offset = d['subregion'][2], d['subregion'][4]
    y = (y[0]+y_offset, y[1]+y_offset)
    x = (x[0]+x_offset, x[1]+x_offset)
    a[y[0]:y[1], x[0]:x[1]] = 1
    mask = dwi.mask.Mask(slice_index+1, a)
    print('Writing mask: {}'.format(path))
    mask.write(str(path))


def main():
    args = parse_args()
    # depthmin, depthmax, sidemin, sidemax, nrois = args.algparams
    # ap = AlgParams(depthmin=depthmin, depthmax=depthmax,
    #                 sidemin=sidemin, sidemax=sidemax, nrois=nrois)
    ap = AlgParams(*args.algparams)
    if ap.sidemin > ap.sidemax or ap.depthmin > ap.depthmax:
        raise ValueError('Invalid ROI size limits')

    print('Reading data...')
    params = args.param.split('+')  # Can be like 'ADCk+K'.

    data = dwi.dataset.dataset_read_samplelist(args.patients, args.cases,
                                               args.scans)
    dwi.dataset.dataset_read_patientinfo(data, args.patients)
    dwi.dataset.dataset_read_subregions(data, args.subregiondir)
    dwi.dataset.dataset_read_pmaps(data, args.pmapdir, params)
    dwi.dataset.dataset_read_prostate_masks(data, args.prostatemaskdir)
    dwi.dataset.dataset_read_roi_masks(data, args.roimaskdir)
    if args.clip:
        for d in data:
            clip_pmap(d['image'], params)

    for d in data:
        print('{case} {scan}: {score} {subregion}'.format(**d))
        if args.verbose:
            print(d['image'].shape)
            print([len(x.selected_slices()) for x in [d['cancer_mask'],
                                                      d['normal_mask'],
                                                      d['prostate_mask']]])
        d.update(dwi.autoroi.find_roi(d['image'], args.roidim, params,
                                      prostate_mask=d['prostate_mask'], ap=ap))
        print('{case} {scan}: Optimal ROI at {roi_corner}'.format(**d))
        path = args.outfig or (DEFAULT_OUT_IMAGE_DIR /
                               '{case}_{scan}.png'.format(**d))
        draw(d, params[0], path)
        path = args.outmask or (DEFAULT_OUT_MASK_DIR /
                                '{case}_{scan}_auto.mask'.format(**d))
        write_mask(d, path)

    # if args.verbose:
    #     for i, p in enumerate(params):
    #         z, y, x = coords
    #         a = img[z[0]:z[1], y[0]:y[1], x[0]:x[1], i]
    #         print(p, a.min(), a.max(), np.median(a))
    #         print(dwi.util.fivenum(a.flatten()))
    #         a = img[..., i]
    #         print(p, a.min(), a.max(), np.median(a))
    #         print(dwi.util.fivenum(a.flatten()))


if __name__ == '__main__':
    main()
