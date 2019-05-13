"""Detect lesion code."""

# http://scikit-image.org/docs/dev/_downloads/a35d9b7dfd6d18cf5e01952d15cc3af6/plot_blob.py

import logging
from functools import lru_cache, reduce
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


class BlobDetector:
    """..."""

    def __init__(self, image, voxel_size, **kwargs):
        """..."""
        image = np.asfarray(image)
        self.image = color.rgb2gray(image)
        self.image = dwi.util.flip_minmax(self.image)
        self.voxel_size = voxel_size
        self.set_kwargs(**kwargs)

    def set_kwargs(self, blob_ka={}, log_ka={}, dog_ka={}, doh_ka={}):
        """Set keyword arguments for blob detection functions."""
        mm = 1 / self.voxel_size  # A millimetre in voxel lengths.
        # self.blob_ka = dict(min_sigma=3 * mm, max_sigma=10 * mm,
        #                     overlap=0.33)
        self.blob_ka = dict(min_sigma=5 * mm, max_sigma=12 * mm, overlap=0.5)
        self.log_ka = dict(num_sigma=5, threshold=0.1, log_scale=False)
        self.dog_ka = dict(sigma_ratio=1.6, threshold=0.1)
        self.doh_ka = dict(num_sigma=5, threshold=0.01, log_scale=False)
        self.blob_ka.update(blob_ka)
        self.log_ka.update(log_ka)
        self.dog_ka.update(dog_ka)
        self.doh_ka.update(doh_ka)

    @lru_cache(None)
    def log(self):
        """Laplacian of Gaussian."""
        # skimage.feature.blob_log(image, min_sigma=1, max_sigma=50,
        #     num_sigma=10, threshold=0.2, overlap=0.5, log_scale=False)
        blobs = feature.blob_log(self.image, **self.blob_ka, **self.log_ka)
        blobs[:, 2] = blobs[:, 2] * sqrt(2)  # Compute radii in 3rd column.
        return blobs

    @lru_cache(None)
    def dog(self):
        """Difference of Gaussian."""
        # skimage.feature.blob_dog(image, min_sigma=1, max_sigma=50,
        #     sigma_ratio=1.6, threshold=2.0, overlap=0.5)
        blobs = feature.blob_dog(self.image, **self.blob_ka, **self.dog_ka)
        blobs[:, 2] = blobs[:, 2] * sqrt(2)  # Compute radii in 3rd column.
        return blobs

    @lru_cache(None)
    def doh(self):
        """Determinant of Hessian."""
        # blob_doh requires double.
        image = np.asfarray(self.image, dtype=np.double)
        # skimage.feature.blob_doh(image, min_sigma=1, max_sigma=30,
        #     num_sigma=10, threshold=0.01, overlap=0.5, log_scale=False)
        blobs = feature.blob_doh(image, **self.blob_ka, **self.doh_ka)
        # Here, radius is approx. equal to sigma.
        return blobs

    def find_blobs(self):
        """..."""
        funcs = [self.log, self.dog, self.doh]
        # blobs_list = [self.log(), self.dog(), self.doh()]
        blobs_list = [x() for x in funcs]
        # titles = ['Laplacian of Gaussian', 'Difference of Gaussian',
        #           'Determinant of Hessian']
        titles = [x.__doc__ for x in funcs]
        short_titles = [dwi.util.abbrev(x) for x in titles]
        return dict(blobs_log=self.log(), blobs_dog=self.dog(),
                    blobs_doh=self.doh(), blobs_list=blobs_list,
                    titles=titles, short_titles=short_titles)


# def find_blobs(image, voxel_size):
#     """Find blobs in image."""
#     log.info([image.shape, image.dtype, voxel_size])
#     image = np.asfarray(image)
#     image_gray = color.rgb2gray(image)
#     # inverted = -image_gray + image_gray.max()
#     inverted = dwi.util.flip_minmax(image_gray)
#
#     mm = 1 / voxel_size  # A millimetre in voxel lengths.
#     # blob_args = dict(min_sigma=3 * mm, max_sigma=10 * mm, overlap=0.33)
#     blob_args = dict(min_sigma=5 * mm, max_sigma=12 * mm, overlap=0.5)
#     log_args = dict(num_sigma=5, threshold=0.1, log_scale=False)
#     dog_args = dict(sigma_ratio=1.6, threshold=0.1)
#     doh_args = dict(num_sigma=5, threshold=0.01, log_scale=False)
#
#     # skimage.feature.blob_log(image, min_sigma=1, max_sigma=50,
#     #     num_sigma=10, threshold=0.2, overlap=0.5, log_scale=False)
#     blobs_log = feature.blob_log(inverted, **blob_args, **log_args)
#     blobs_log[:, 2] = blobs_log[:, 2] * sqrt(2)  # Compute radii in 3rd col.
#
#     # skimage.feature.blob_dog(image, min_sigma=1, max_sigma=50,
#     #     sigma_ratio=1.6, threshold=2.0, overlap=0.5)
#     blobs_dog = feature.blob_dog(inverted, **blob_args, **dog_args)
#     blobs_dog[:, 2] = blobs_dog[:, 2] * sqrt(2)  # Compute radii in 3rd col.
#
#     # skimage.feature.blob_doh(image, min_sigma=1, max_sigma=30,
#     #     num_sigma=10, threshold=0.01, overlap=0.5, log_scale=False)
#     dbl = np.asfarray(inverted, dtype=np.double)  # blob_doh requires double.
#     blobs_doh = feature.blob_doh(dbl, **blob_args, **doh_args)
#     # Here, radius is approx. equal to sigma.
#
#     blobs_list = [blobs_log, blobs_dog, blobs_doh]
#     titles = ['Laplacian of Gaussian', 'Difference of Gaussian',
#               'Determinant of Hessian']
#     short_titles = [dwi.util.abbrev(x) for x in titles]
#     return dict(blobs_log=blobs_log, blobs_dog=blobs_dog,
#                 blobs_doh=blobs_doh, blobs_list=blobs_list,
#                 titles=titles, short_titles=short_titles)


def maskratio(large, small):
    """Calculate and validate mask size ratio."""
    sizes = [np.count_nonzero(x) for x in [large, small]]
    if 0 in sizes or sizes[1] >= sizes[0]:
        log.error('Mask size mismatch: large %i, small %i', *sizes)
        return np.nan
    return sizes[1] / sizes[0]


def is_blob_in_mask(blob, mask):
    """..."""
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


# def select_blob(image, pmask, blobs, avg=np.nanmedian):
#     """Select best blob from candidates."""
#     # Pick blob disk with minimum average value.
#     blobs_in_prostate = [x for x in blobs if is_blob_in_mask(x, pmask)]
#     circles = [draw.circle(y, x, r, shape=image.shape) for
#                y, x, r in blobs_in_prostate]
#     pimage = image.copy()
#     pimage[~pmask.astype(np.bool)] = np.nan
#     avgs = [avg(pimage[x]) for x in circles]
#     assert all(avgs), avgs
#     i = np.argmin(avgs)
#     return blobs_in_prostate[i], avgs[i]


def select_best_blob(bundle, blobs, avg=np.nanmedian):
    """Select best blob from candidates."""
    # Pick blob disk with minimum average value.
    blobs_in_prostate = [x for x in blobs
                         if is_blob_in_mask(x, bundle.pmask_slice())]
    assert blobs_in_prostate, (blobs, bundle.image_slice().shape)
    circles = [draw.circle(y, x, r, shape=bundle.image_slice().shape) for
               y, x, r in blobs_in_prostate]
    pimage = bundle.pmasked_image_slice()
    avgs = [avg(pimage[x]) for x in circles]
    assert all(avgs), avgs
    i = np.argmin(avgs)
    return blobs_in_prostate[i]


def get_blob_avg(bundle, blob, avg=np.nanmedian):
    """Return average value of a blob."""
    y, x, r = blob
    circle = draw.circle(y, x, r, shape=bundle.image_slice().shape)
    pimage = bundle.pmasked_image_slice()
    return avg(pimage[circle])


def plot_blobs(image, masks, blobs_list, rois, titles, performances,
               suptitle=None, figpath=None):
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

    fig, axes = plt.subplots(1, len(blobs_list), figsize=(9, 3),
                             sharex=True, sharey=True)
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
    show_fig(fig, figpath)


def show_fig(fig, path):
    """Show or save figure."""
    if path is None:
        plt.tight_layout()
        plt.show()
    else:
        log.info('Plotting to %s', path)
        plt.savefig(str(path), bbox_inches='tight')
    fig.clf()
    plt.close('all')


def get_figpath(bundle, directory, suffix='.png'):
    """Get figure path (if directory given) and ensure it exists."""
    if directory is None:
        return None
    stem = f'{bundle.mode.param}_{bundle.target.case}'
    path = (directory / stem).with_suffix(suffix)
    return dwi.files.ensure_dir(path)


def plot_bundle_blobs(bundle, blob_info, rois, figdir):
    """..."""
    i_slice = bundle.p_max_slice_index()
    n_slices = bundle.image.shape[0]
    suptitle = ', '.join([f'{bundle.mode.param}',
                          f'case {bundle.target.case}',
                          f'slice {i_slice}/{n_slices}'])
    performances = [[-1] * 2] * 3
    plot_blobs(bundle.image_slice(),
               [bundle.pmask_slice(), bundle.lmask_slice()],
               blob_info['blobs_list'], rois, blob_info['short_titles'],
               performances, suptitle=suptitle,
               figpath=get_figpath(bundle, figdir))


# def detect_blob(bundle, figdir=None):
#     """Detect blobs and plot them. Return performance info."""
#     # i_slice, (img, *masks) = dwi.readnib.get_slices(bundle)
#     i_slice = bundle.p_max_slice_index()
#     img = bundle.image_slice()
#     pmask, lmask = bundle.pmask_slice(), bundle.lmask_slice()
#     # img = dwi.util.normalize(img, 'ADCm')
#     ret = dict(i_slice=i_slice, n_slices=bundle.image.shape[0],
#                maskratio=maskratio(pmask, lmask))
#
#     blob_info = find_blobs(img, bundle.voxel_size())
#     blobs_list = blob_info['blobs_list']
#     short_titles = blob_info['short_titles']
#     performances = [blob_performance(x, [pmask, lmask]) for x in blobs_list]
#     ret['performances'] = [x[1] for x in performances]
#     log.info(performances)
#
#     kwargs = dict(
#         avg=np.nanmedian,
#         # avg=np.nanmean,
#         # avg=lambda x: np.nanpercentile(x, 50),
#         )
#     rois_avgs = [select_blob(img, pmask, x, **kwargs) for x in blobs_list]
#     ret['rois'] = [x[0] for x in rois_avgs]
#     ret['roi_avgs'] = [x[1] for x in rois_avgs]
#     print([tuple(x) for x in ret['rois']])
#
#     suptitle = ', '.join([f'{bundle.mode.param}',
#                           f'case {bundle.target.case}',
#                           f'slice {i_slice}/{ret["n_slices"]}'])
#     if figdir is None:
#         figpath = None
#     else:
#         stem = f'{bundle.mode.param}_{bundle.target.case}'
#         figpath = (figdir / stem).with_suffix('.png')
#         dwi.files.ensure_dir(figpath)
#     plot_blobs(img, [pmask, lmask], blobs_list, ret['rois'], short_titles,
#                performances, suptitle=suptitle, figpath=figpath)
#     return ret


# def detect_blobs(case, modes, figdir):
#     """Detect blobs for a case."""
#     bundles = dwi.readnib.load_images(case, modes)
#     cases = []
#     strings = []
#     roi_avgs = []
#     for bundle in bundles:
#         if bundle.exists():
#             s1 = f'{bundle.mode.param} {bundle.target.case} '
#             print(s1, end='', flush=True)
#             d = detect_blob(bundle, figdir=figdir)
#             s2 = ' '.join([f'{d["i_slice"]:2d}/{d["n_slices"]}',
#                            f'{d["maskratio"]:4.0%}', f'{d["performances"]}'])
#             print(s2)
#             cases.append(bundle.target.case)
#             strings.append(s1 + s2)
#             roi_avgs.append(d['roi_avgs'])
#     return cases, strings, roi_avgs
