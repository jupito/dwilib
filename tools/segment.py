#!/usr/bin/env python2

"""Prostate segementation."""

from __future__ import absolute_import, division, print_function
import argparse
import logging

import numpy as np
from scipy import ndimage
import skimage.segmentation
import sklearn.preprocessing

import dwi.files
import dwi.mask
import dwi.plot
import dwi.util


log = logging.getLogger('segment')


def parse_args():
    """Parse command-line arguments."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--verbose', '-v', action='count',
                   help='increase verbosity')
    p.add_argument('image',
                   help='input image file')
    p.add_argument('--params', '-p', nargs='+', type=int,
                   help='input image parameters')
    p.add_argument('--mask', '-m',
                   help='mask file')
    p.add_argument('--fig', '-f',
                   help='output figure file')
    return p.parse_args()


def scale_mask(mask, factor):
    mask = mask.astype(np.float_)
    mask = ndimage.interpolation.zoom(mask, factor, order=0)
    mask = dwi.util.asbool(mask)
    return mask


def preprocess(img):
    assert img.ndim == 4
    log.info(dwi.util.fivenums(img))
    original_shape = img.shape
    # img.shape = (-1, img.shape[-1])
    img = img.reshape((-1, img.shape[-1]))

    # img = sklearn.preprocessing.minmax_scale(img)
    # img = sklearn.preprocessing.scale(img)
    img = sklearn.preprocessing.robust_scale(img)

    # img.shape = original_shape
    img = img.reshape(original_shape)
    log.info(dwi.util.fivenums(img))
    return img


def label_groups(a, thresholds):
    labels = np.zeros_like(a, dtype=np.uint8)
    for i, t in enumerate(sorted(thresholds)):
        labels[a > t] = i + 1
    return labels


def get_markers(img):
    assert img.ndim == 4
    markers = np.zeros(img.shape[0:3], dtype=np.int8)

    # Based on absolute value thresholds (non-scaled image).
    bg, fg1, fg2 = np.percentile(img, 50), 1400, 1600
    # bg, fg1, fg2 = np.percentile(img, 50), 100, 300  # B=2000
    markers[img[..., 0] < bg] = 1
    markers[8:12][img[8:12][..., 0] > fg1] = 2
    markers[:3][img[:3][..., 0] > fg1] = 3
    markers[-3:][img[-3:][..., 0] > fg1] = 4
    markers[img[..., 0] > fg2] = 0

    # # Based on percentile thresholds.
    # thresholds = np.percentile(img, [50, 97, 98, 99.5])
    # log.info('Seed thresholds: %s', thresholds)
    # markers[img[..., 0] < thresholds[0]] = 1
    # markers[9:11][img[9:11][..., 0] > thresholds[1]] = 2
    # markers[:2][img[:2][..., 0] > thresholds[1]] = 3
    # markers[-2:][img[-2:][..., 0] > thresholds[1]] = 4
    # # markers[img[..., 0] > thresholds[2]] = 3

    # # Based on position.
    # pos = [x/2 for x in markers.shape]
    # slices = [slice(int(round(p-0.03*s)), int(round(p+0.03*s))) for p, s in
    #           zip(pos, markers.shape)]
    # # slices = [slice(int(0.47*x), int(-0.47*x)) for x in markers.shape]
    # log.info('Seed position: %s', slices)
    # # # markers[9:-9, 100:-100, 100:-100] = 2
    # markers[slices] = 2

    # pos = dwi.util.centroid(img[..., 0])
    # slices = [slice(int(round(p-0.03*s)), int(round(p+0.03*s))) for p, s in
    #           zip(pos, markers.shape)]
    # log.info('Seed position: %s', slices)
    # markers[slices] = 4

    return markers


def segment(img, markers, spacing):
    kwargs = dict(
        # beta=10,  # Default is 130.
        multichannel=True,
        spacing=spacing,
        )
    labels = skimage.segmentation.random_walker(img, markers, **kwargs)
    return labels


def plot(img, mask, path):
    assert img.ndim == 3
    vmin, vmax = np.min(img), np.max(img)
    titles = [str(x) for x in range(len(img))]
    it = dwi.plot.generate_plots(nrows=4, ncols=5, titles=titles, path=path)
    for i, plt in enumerate(it):
        plt.imshow(img[i], vmin=vmin, vmax=vmax)
        if mask is not None:
            view = np.zeros(img.shape[1:3] + (4,), dtype=np.float16)
            view[dwi.mask.border(mask[i])] = (1, 0, 0, 1)
            plt.imshow(view)


def main():
    args = parse_args()
    if args.verbose == 0:
        loglevel = logging.WARNING
    elif args.verbose == 1:
        loglevel = logging.INFO
    elif args.verbose > 1:
        loglevel = logging.DEBUG
    logging.basicConfig(level=loglevel)

    img, attrs = dwi.files.read_pmap(args.image, params=args.params)
    spacing = attrs['voxel_spacing']

    log.info(img.shape)
    log.info(attrs['parameters'])
    log.info(spacing)
    mask = dwi.files.read_mask(args.mask) if args.mask else None
    # img_scale = img.min(), img.max()
    # img = img[5:-5]
    # mask = mask[5:-5]

    # img = preprocess(img)

    # Downsample.
    factor = (1, 0.5, 0.5)
    img = ndimage.interpolation.zoom(img, factor + (1,), order=0)
    spacing = [s/f for s, f in zip(spacing, factor)]
    mask = scale_mask(mask, factor)

    # labels = label_groups(img[..., 0], np.percentile(img, [50, 99.5]))
    # labels = label_groups(img[0], [img[mask].min(), img[mask].max()])
    # labels = np.zeros(img.shape[0:3], dtype=np.uint8)
    # for i in range(len(img)):
    #     thresholds = np.percentile(img[i], [50, 99.5])
    #     labels[i] = label_groups(img[i, :, :, 0], thresholds)
    #     # labels[i] = segment(img[i])

    # B=2000
    # labels = label_groups(img[..., 0], [50, 100, 150, 200, 250, 300, 350])

    markers = get_markers(img)
    labels = segment(img, markers, spacing)
    plot(labels, mask, args.fig)


if __name__ == '__main__':
    main()
