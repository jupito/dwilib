"""Detect lesion code."""

# http://scikit-image.org/docs/dev/_downloads/a35d9b7dfd6d18cf5e01952d15cc3af6/plot_blob.py

import logging
from functools import reduce
from math import sqrt
from operator import add

import numpy as np
# import matplotlib as mpl
import matplotlib.pyplot as plt
# import scipy as sp
from skimage import color, draw, feature, measure

import dwi.files
import dwi.mask
import dwi.readnib
import dwi.stats
import dwi.util

log = logging.getLogger(__name__)


def maskratio(large, small):
    """Calculate and validate mask size ratio."""
    sizes = [np.count_nonzero(x) for x in [large, small]]
    if 0 in sizes or sizes[1] >= sizes[0]:
        log.error('Mask size mismatch: large %i, small %i', *sizes)
        return np.nan
    return sizes[1] / sizes[0]


def find_blobs(image, voxel_size):
    """Find blobs in image."""
    log.info([image.shape, image.dtype, voxel_size])
    assert np.issubdtype(image.dtype, np.float), image.dtype

    image_gray = color.rgb2gray(image)
    # inverted = -image_gray + image_gray.max()
    inverted = dwi.util.flip_minmax(image_gray)

    # blob_args = dict(min_sigma=2, max_sigma=5)
    # l = np.min(image.shape)
    # blob_args = dict(min_sigma=l / 20, max_sigma=l / 5)
    blob_args = dict(min_sigma=3 / voxel_size,
                     max_sigma=10 / voxel_size,
                     overlap=0.33)
    # num_sigma, sigma_ratio = 10, 1.6
    num_sigma, sigma_ratio = 5, 1.6
    log.info([blob_args, num_sigma, sigma_ratio])

    # skimage.feature.blob_log(image, min_sigma=1, max_sigma=50,
    #                          num_sigma=10, threshold=0.2, overlap=0.5,
    #                          log_scale=False)
    blobs_log = feature.blob_log(inverted, **blob_args, num_sigma=num_sigma,
                                 threshold=0.1, log_scale=False)
    blobs_log[:, 2] = blobs_log[:, 2] * sqrt(2)  # Compute radii in 3rd column.

    # skimage.feature.blob_dog(image, min_sigma=1, max_sigma=50,
    #                          sigma_ratio=1.6, threshold=2.0, overlap=0.5)
    blobs_dog = feature.blob_dog(inverted, **blob_args,
                                 sigma_ratio=sigma_ratio, threshold=0.1)
    blobs_dog[:, 2] = blobs_dog[:, 2] * sqrt(2)  # Compute radii in 3rd column.

    # skimage.feature.blob_doh(image, min_sigma=1, max_sigma=30,
    #                          num_sigma=10, threshold=0.01, overlap=0.5,
    #                          log_scale=False)
    blobs_doh = feature.blob_doh(inverted, **blob_args, num_sigma=num_sigma,
                                 threshold=0.01, log_scale=False)
    # Here, radius is approx. equal to sigma.

    blobs_list = [blobs_log, blobs_dog, blobs_doh]
    titles = ['Laplacian of Gaussian', 'Difference of Gaussian',
              'Determinant of Hessian']
    short_titles = [dwi.util.abbrev(x) for x in titles]
    return dict(blobs_log=blobs_log,
                blobs_dog=blobs_dog,
                blobs_doh=blobs_doh,
                blobs_list=blobs_list,
                titles=titles, short_titles=short_titles)


def is_blob_in_mask(blob, mask):
    y, x, _ = (int(np.round(x)) for x in blob)
    return bool(mask[y, x])


def blob_performance(blobs, masks):
    """Return numbers of blobs each mask contains."""
    return tuple(sum(is_blob_in_mask(x, y) for x in blobs) for y in masks)


def find_image_contours(image):
    """Find contours in image."""
    # level = np.ptp(image) / 2
    level = 1050  # TODO: Different contour levels for ADC and T2w.
    kwargs = dict(fully_connected='low', positive_orientation='low')
    # kwargs = dict(fully_connected='high', positive_orientation='high')
    return measure.find_contours(image, level, **kwargs)


def select_blob(image, pmask, blobs, avg=np.nanmedian):
    """Select best blob from candidates."""
    # Pick blob disk with minimum average value.
    blobs_in_prostate = [x for x in blobs if is_blob_in_mask(x, pmask)]
    circles = [draw.circle(y, x, r, shape=image.shape) for
               y, x, r in blobs_in_prostate]
    pimage = image.copy()
    pimage[~pmask.astype(np.bool)] = np.nan
    avgs = [avg(pimage[x]) for x in circles]
    assert all(avgs), avgs
    i = np.argmin(avgs)
    return blobs_in_prostate[i], avgs[i]


def plot_blobs(image, masks, blobs_list, rois, titles, performances,
               suptitle=None, outpath=None):
    """Plot found blobs."""
    # outlines = dwi.mask.overlay_masks(masks)
    mask_contours = reduce(add, (dwi.mask.contours(x) for x in masks))
    # image_contours = find_image_contours(image)
    seq = zip(blobs_list, titles, performances)
    # imshow_args = dict(interpolation='nearest')
    imshow_args = dict(aspect='equal', interpolation='none')

    colors = dict(
        roi='lime',
        lesion='yellow',
        prostate='lightblue',
        outside='red',
        contours=['blue'] + ['orange'] * 3,
        level='pink',
        )

    def circle_color(blob, masks, roi):
        """Color circle by occupied blob."""
        if np.all(np.isclose(blob, roi)):
            return colors['roi']
        if any(is_blob_in_mask(blob, x) for x in masks[1:]):
            return colors['lesion']
        if is_blob_in_mask(blob, masks[0]):
            return colors['prostate']
        return colors['outside']

    fig, axes = plt.subplots(1, len(blobs_list), figsize=(9, 3), sharex=True,
                             sharey=True)
    if suptitle is not None:
        fig.suptitle(suptitle)
    ax = axes.ravel()
    for i, (blobs, title, perf) in enumerate(seq):
        ax[i].set_title(f'{title}: {perf[1]} / {len(blobs)}')
        ax[i].imshow(image, cmap='gray', **imshow_args)
        # ax[i].imshow(outlines, cmap='viridis_r', **imshow_args)
        for contour, c in zip(mask_contours, colors['contours']):
            ax[i].plot(contour[:, 1], contour[:, 0], color=c, linewidth=1.2)
        # for contour in image_contours:
        #     ax[i].plot(contour[:, 1], contour[:, 0], color=colors['level'],
        #                linewidth=1)
        for blob in blobs:
            y, x, r = blob
            c = circle_color(blob, masks, rois[i])
            circle = plt.Circle((x, y), r, color=c, linewidth=1,
                                linestyle='-.', fill=False)
            ax[i].add_patch(circle)
        ax[i].set_axis_off()

    if outpath is None:
        plt.tight_layout()
        plt.show()
    else:
        log.info('Plotting to %s', outpath)
        plt.savefig(str(outpath), bbox_inches='tight')
    fig.clf()
    plt.close('all')


def detect_blob(bundle, outdir=None):
    """Detect blobs and plot them. Return performance info."""
    assert bundle.voxel_shape[0] == bundle.voxel_shape[1], bundle.voxel_shape
    # i_slice, (img, *masks) = dwi.readnib.get_slices(bundle)
    i_slice = bundle.p_max_slice_index()
    img = bundle.image_slice()
    pmask, lmask = bundle.pmask_slice(), bundle.lmask_slice()
    # img = dwi.util.normalize(img, 'ADCm')
    ret = dict(i_slice=i_slice, n_slices=bundle.image.shape[0],
               maskratio=maskratio(pmask, lmask))

    blob_info = find_blobs(img, bundle.voxel_shape[0])
    blobs_list = blob_info['blobs_list']
    short_titles = blob_info['short_titles']
    performances = [blob_performance(x, [pmask, lmask]) for x in blobs_list]
    ret['performances'] = [x[1] for x in performances]
    log.info(performances)

    kwargs = dict(
        avg=np.nanmedian,
        # avg=np.nanmean,
        # avg=lambda x: np.nanpercentile(x, 50),
        )
    rois_avgs = [select_blob(img, pmask, x, **kwargs) for x in blobs_list]
    ret['rois'] = [x[0] for x in rois_avgs]
    ret['roi_avgs'] = [x[1] for x in rois_avgs]
    print([tuple(x) for x in ret['rois']])

    suptitle = ', '.join([f'{bundle.mode.param}',
                          f'case {bundle.target.case}',
                          f'slice {i_slice}/{ret["n_slices"]}'])
    if outdir is None:
        outpath = None
    else:
        stem = f'{bundle.mode.param}_{bundle.target.case}'
        outpath = (outdir / stem).with_suffix('.png')
        dwi.files.ensure_dir(outpath)
    plot_blobs(img, [pmask, lmask], blobs_list, ret['rois'], short_titles,
               performances, suptitle=suptitle, outpath=outpath)
    return ret


# def detect_blobs(case, modes, outdir):
#     """Detect blobs for a case."""
#     bundles = dwi.readnib.load_images(case, modes)
#     cases = []
#     strings = []
#     roi_avgs = []
#     for bundle in bundles:
#         if bundle.exists():
#             s1 = f'{bundle.mode.param} {bundle.target.case} '
#             print(s1, end='', flush=True)
#             d = detect_blob(bundle, outdir=outdir)
#             s2 = ' '.join([f'{d["i_slice"]:2d}/{d["n_slices"]}',
#                            f'{d["maskratio"]:4.0%}', f'{d["performances"]}'])
#             print(s2)
#             cases.append(bundle.target.case)
#             strings.append(s1 + s2)
#             roi_avgs.append(d['roi_avgs'])
#     return cases, strings, roi_avgs
