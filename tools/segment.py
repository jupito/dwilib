#!/usr/bin/python3

"""Prostate segementation."""

import logging
from pathlib import Path

import numpy as np
from scipy import ndimage
import skimage.segmentation
import sklearn.preprocessing

import dwi.conf
import dwi.dataset
import dwi.files
import dwi.image
import dwi.mask
import dwi.plot
import dwi.util


def parse_args():
    """Parse command-line arguments."""
    p = dwi.conf.get_parser(description=__doc__)
    p.add('-m', '--modes', nargs='*', type=dwi.util.ImageMode, default=['DWI'],
          help='imaging modes')
    p.add('-s', '--samplelist', default='all',
          help='samplelist identifier')
    p.add('-c', '--cases', nargs='*', type=int,
          help='cases to include, if not all')
    p.add('image', type=Path,
          help='input image file')
    p.add('-p', '--params', nargs='+', type=int,
          help='input image parameters')
    p.add('--mask', type=Path,
          help='mask file')
    p.add('--fig', type=Path,
          help='output figure file')
    p.add('--hist', type=Path,
          help='output histogram file')
    return dwi.conf.parse_args(p)


def scale_mask(mask, factor):
    mask = mask.astype(np.float32)
    mask = ndimage.interpolation.zoom(mask, factor, order=0)
    mask = dwi.util.asbool(mask)
    return mask


def array_info(a):
    """Array info for debugging."""
    return [a.__class__.__name__, a.shape, a.dtype, dwi.util.fivenums(a),
            getattr(a, 'spacing', None)]


def preprocess(img):
    assert img.ndim == 4
    logging.info('Preprocessing: %s', array_info(img))
    info = img.info
    original_shape = img.shape
    img = img.reshape((-1, img.shape[-1]))

    img = sklearn.preprocessing.minmax_scale(img)
    # img = sklearn.preprocessing.scale(img)
    # img = sklearn.preprocessing.robust_scale(img)

    img = img.reshape(original_shape)
    img = dwi.image.Image(img, info=info)
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
    # logging.info('Seed thresholds: %s', thresholds)
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
    # logging.info('Seed position: %s', slices)
    # # # markers[9:-9, 100:-100, 100:-100] = 2
    # markers[slices] = 2

    # pos = dwi.util.centroid(img[..., 0])
    # slices = [slice(int(round(p-0.03*s)), int(round(p+0.03*s))) for p, s in
    #           zip(pos, markers.shape)]
    # logging.info('Seed position: %s', slices)
    # markers[slices] = 4

    return markers


def segment(img, markers, spacing):
    logging.info('Segmenting: %s, %s', array_info(img), spacing)
    logging.info('...with markers: %s', array_info(markers))
    d = dict(
        # beta=10,  # Default is 130.
        # mode='cg_mg',
        mode=None,
        multichannel=True,
        spacing=spacing,
        )
    labels = skimage.segmentation.random_walker(img, markers, **d)
    return labels


def histogram(img, mask, rng=None, path=None):
    d = dict(titles=img.params, path=path)
    # d.update(nrows=3, ncols=4)
    it = dwi.plot.generate_plots(**d)
    lst = list(img.each_param())
    for i, plt in enumerate(it):
        param, a = lst[i]
        d = dict(bins='auto', range=rng, histtype='step', label=param)
        plt.hist(a.ravel(), **d)
        plt.hist(a[mask], **d)


def plot(img, mask, path):
    assert img.ndim == 3
    vmin, vmax = np.min(img), np.max(img)
    titles = [str(x) for x in range(len(img))]
    it = dwi.plot.generate_plots(nrows=4, ncols=5, titles=titles, path=path)
    for i, plt in enumerate(it):
        plt.imshow(img[i], vmin=vmin, vmax=vmax)
        if mask is not None:
            view = np.zeros(img.shape[1:3] + (4,), dtype=np.float32)
            view[dwi.mask.border(mask[i])] = (1, 0, 0, 1)
            plt.imshow(view)


def process_image(imagepath, params, maskpath, histpath, figpath):
    img = dwi.image.Image.read(imagepath, params=params, dtype=np.float32)
    mask = dwi.files.read_mask(str(maskpath)) if maskpath else None

    logging.info('Image: %s, %s', array_info(img),
                 np.count_nonzero(mask) / mask.size)
    mbb = img[..., 0].mbb()
    img = img[mbb]
    mask = mask[mbb]
    logging.info('Image: %s', array_info(img))
    logging.info('...masked: %s, %s', array_info(img[mask]))
    for p, a in img.each_param():
        logging.info('Param: %s, %s, %s', p, array_info(a))

    # pc = [50, 99.5]
    pc = [90, 99.9]
    rng = np.percentile(img, pc)
    if histpath:
        histogram(img, mask, rng=rng, path=str(histpath))

    # img_scale = img.min(), img.max()
    # img = img[5:-5]
    # mask = mask[5:-5]

    # Downsample.
    logging.info('Scaling: %s', array_info(img))
    info = img.info
    factor = (1, 0.5, 0.5)
    img = ndimage.interpolation.zoom(img, factor + (1,), order=0)
    info['spacing'] = [s/f for s, f in zip(info['spacing'], factor)]
    mask = scale_mask(mask, factor)
    assert img[..., 0].shape == mask.shape, (img.shape, mask.shape)
    img = dwi.image.Image(img, info=info)

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
    img = preprocess(img)
    plot(img[..., 0], mask, str(figpath))
    labels = segment(img, markers, img.spacing)
    plot(labels, mask, str(figpath))


def main():
    args = parse_args()
    process_image(args.image, args.params, args.mask, args.hist, args.fig)


if __name__ == '__main__':
    main()
