#!/usr/bin/python3

"""Prostate segementation."""

import logging

import numpy as np
from scipy import ndimage
from skimage import filters, segmentation
from sklearn import preprocessing

import dwi.conf
import dwi.files
import dwi.image
import dwi.mask
import dwi.plot
import dwi.util
from dwi import ImageMode, Path


def parse_args():
    """Parse command-line arguments."""
    p = dwi.conf.get_parser(description=__doc__)
    p.add('-m', '--modes', nargs='*', type=ImageMode, default=['DWI'],
          help='imaging modes')
    p.add('-c', '--cases', nargs='*', type=int,
          help='cases to include, if not all')
    p.add('-i', '--image', type=Path,  # required=True,
          help='input image file')
    p.add('-p', '--params', nargs='+', type=int,
          help='input image parameters')
    p.add('--mask', type=Path,
          help='mask file')
    p.add('--fig', type=Path,
          help='output figure path')
    return p.parse_args()


def array_info(a):
    """Array info for debugging."""
    return ', '.join(str(x) for x in [a.__class__.__name__, a.shape,
                                      a.dtype.name, dwi.util.fivenums(a),
                                      getattr(a, 'spacing', None)])


def get_mbb(img, mask):
    """Get image MBB."""
    mbb = img[..., 0].mbb()
    logging.info('Taking MBB: %s', [(x.start, x.stop) for x in mbb])
    img = img[mbb]
    mask = mask[mbb]
    return img, mask


def normalize(img):
    """Normalize image intensity."""
    assert img.ndim == 4
    logging.info('Preprocessing: %s', array_info(img))
    info = img.info
    original_shape = img.shape
    img = img.reshape((-1, img.shape[-1]))

    img = preprocessing.minmax_scale(img)
    # img = preprocessing.scale(img)
    # img = preprocessing.robust_scale(img)

    img = img.reshape(original_shape)
    img = dwi.image.Image(img, info=info)
    return img


def rescale_mask(mask, factor):
    """Rescale mask."""
    mask = mask.astype(np.float32)  # Float16 seems to jam zoom().
    mask = ndimage.interpolation.zoom(mask, factor, order=0)
    mask = dwi.util.asbool(mask)
    return mask


def rescale(img, mask, factor):
    """Rescale image and mask."""
    logging.info('Scaling: %s', array_info(img))
    info = img.info
    img = ndimage.interpolation.zoom(img, factor + (1,), order=0)
    info['spacing'] = [s/f for s, f in zip(info['spacing'], factor)]
    mask = rescale_mask(mask, factor)
    assert img[..., 0].shape == mask.shape, (img.shape, mask.shape)
    img = dwi.image.Image(img, info=info)
    return img, mask


def smoothen(img):
    """Smoothen image."""
    return filters.gaussian(img, 1, multichannel=False)


def label_groups(a, thresholds):
    """Get initial labels for segmentation, based on intensity groups."""
    labels = np.zeros_like(a, dtype=np.uint8)
    for i, t in enumerate(sorted(thresholds)):
        labels[a > t] = i + 1
    return labels


def get_markers(img, mode):
    """Get initial labels for segmentation."""
    assert img.ndim == 4
    markers = np.zeros(img.shape[0:3], dtype=np.int16)

    if mode == 'DWI':
        # Based on absolute value thresholds (non-scaled image).
        bg, fg1, fg2 = np.percentile(img, 50), 1400, 1600
        # bg, fg1, fg2 = np.percentile(img, 50), 100, 300  # B=2000
        markers[img[..., 0] < bg] = 1
        markers[8:12][img[8:12][..., 0] > fg1] = 2
        markers[:3][img[:3][..., 0] > fg1] = 3
        markers[-3:][img[-3:][..., 0] > fg1] = 4
        markers[img[..., 0] > fg2] = 0
    else:
        # Based on percentile thresholds.
        pc = [50, 97, 98, 99.5]
        # pc = list(range(0, 100, 5))
        thresholds = np.percentile(img, pc)
        logging.info('Seed thresholds: %s', thresholds)

        # markers[img[..., 0] < thresholds[0]] = 1
        # markers[9:11][img[9:11][..., 0] > thresholds[1]] = 2
        # markers[:2][img[:2][..., 0] > thresholds[1]] = 3
        # markers[-2:][img[-2:][..., 0] > thresholds[1]] = 4
        # # markers[img[..., 0] > thresholds[2]] = 3

        def do(m, a):
            for t in list(thresholds):
                m = m[:, ::5, ::5]
                a = a[:, ::5, ::5]
                m[a > t] = next(it)
        a = img[..., 0]
        x, y, z = [_//3 for _ in a.shape]
        it = iter(range(1, 10000))
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    slices = (
                        slice(i*x, (i+1)*x),
                        slice(j*y, (j+1)*y),
                        slice(k*z, (k+1)*z)
                        )
                    do(markers[slices], a[slices])

        # # Based on position.
        # pos = [x/2 for x in markers.shape]
        # slices = [slice(int(round(p-0.03*s)), int(round(p+0.03*s))) for p, s
        #           in zip(pos, markers.shape)]
        # # slices = [slice(int(0.47*x), int(-0.47*x)) for x in markers.shape]
        # logging.info('Seed position: %s', slices)
        # # # markers[9:-9, 100:-100, 100:-100] = 2
        # markers[slices] = 2

        # pos = dwi.util.centroid(img[..., 0])
        # slices = [slice(int(round(p-0.03*s)), int(round(p+0.03*s))) for p, s
        #           in zip(pos, markers.shape)]
        # logging.info('Seed position: %s', slices)
        # markers[slices] = 4

    return markers


def segment(img, markers):
    """Segment image."""
    logging.info('Segmenting: %s', array_info(img))
    logging.info('...with markers: %s', array_info(markers))
    d = dict(
        # beta=10,  # Default is 130.
        mode='cg_mg',
        # mode=None,
        multichannel=True,
        spacing=img.spacing,
        )
    labels = segmentation.random_walker(img, markers, **d)
    return labels


def plot_histogram(img, mask, rng=None, path=None):
    """Plot histogram."""
    d = dict(titles=img.params, path=path)
    # d.update(nrows=3, ncols=4)
    it = dwi.plot.generate_plots(**d)
    lst = list(img.each_param())
    for i, plt in enumerate(it):
        param, a = lst[i]
        d = dict(bins='auto', range=rng, histtype='step', label=param)
        # plt.hist(a.ravel(), **d)
        plt.hist(a[mask], **d)
        plt.hist(a[~mask], **d)


def plot_image(img, mask, path):
    """Plot image slice by slice."""
    assert img.ndim == 3
    if mask is not None:
        centroid = tuple(int(round(x)) for x in mask.centroid())
    vmin, vmax = np.min(img), np.max(img)
    titles = [str(x) for x in range(len(img))]
    it = dwi.plot.generate_plots(nrows=10, ncols=5, titles=titles, path=path)
    for i, plt in enumerate(it):
        plt.imshow(img[i], vmin=vmin, vmax=vmax)
        if mask is not None:
            view = np.zeros(img.shape[1:3] + (4,), dtype=np.float32)
            view[dwi.mask.border(mask[i])] = (1, 0, 0, 1)
            if i == centroid[0]:
                view[centroid[1:]] = (1, 0, 0, 1)
            plt.imshow(view)


def fig_path(path, *specs, suffix='.png'):
    """Get output path for figure."""
    stem = '-'.join((path.stem,) + specs)
    return (path.parent / stem).with_suffix(suffix)


def process_image(mode, imagepath, params, maskpath, figpath):
    """Process an image."""
    logging.info('Reading: %s', imagepath)
    img = dwi.image.Image.read(imagepath, params=params, dtype=np.float32)
    mask = dwi.files.read_mask(str(maskpath)) if maskpath else None
    mask = dwi.image.Image(mask)

    logging.info('Image: %s, %s', array_info(img),
                 np.count_nonzero(mask) / mask.size)
    img, mask = get_mbb(img, mask)
    logging.info('Image: %s', array_info(img))
    logging.info('...masked: %s', array_info(img[mask]))
    # logging.info('Mask centroid: %s', [round(x) for x in mask.centroid()])
    for p, a in img.each_param():
        logging.info('Param: %s, %s', p, array_info(a))
    plot_image(img[..., 0], mask, fig_path(figpath, 'image', 'original'))

    # pc = [50, 99.5]
    pc = [90, 99.9]
    rng = np.percentile(img, pc)
    if figpath:
        plot_histogram(img, mask, rng=rng, path=fig_path(figpath, 'hist'))

    # img, mask = rescale(img, mask, (1, 0.5, 0.5))

    ####

    # labels = label_groups(img[..., 0], np.percentile(img, [50, 99.5]))
    # labels = label_groups(img[0], [img[mask].min(), img[mask].max()])
    # labels = np.zeros(img.shape[0:3], dtype=np.uint8)
    # for i in range(len(img)):
    #     thresholds = np.percentile(img[i], [50, 99.5])
    #     labels[i] = label_groups(img[i, :, :, 0], thresholds)
    #     # labels[i] = segment(img[i])

    # B=2000
    # labels = label_groups(img[..., 0], [50, 100, 150, 200, 250, 300, 350])
    # markers = label_groups(img[..., 0], [50, 100, 150, 200, 250, 300, 350])

    ####

    markers = get_markers(img, mode)
    plot_image(markers, mask, fig_path(figpath, 'markers'))

    img = normalize(img)
    # img = dwi.util.normalize(img, mode)
    plot_image(img[..., 0], mask, fig_path(figpath, 'image'))

    # img[..., 0] = smoothen(img[..., 0])
    # plot_image(img[..., 0], mask, fig_path(figpath, 'image', 'gaussian'))

    labels = segment(img, markers)
    plot_image(labels, mask, fig_path(figpath, 'labels'))


def main():
    """Main."""
    args = parse_args()
    mode = args.modes[0]
    process_image(mode, args.image, args.params, args.mask, args.fig)


if __name__ == '__main__':
    main()
