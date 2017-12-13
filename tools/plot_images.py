#!/usr/bin/python3

"""Draw images."""

import logging

import numpy as np
from skimage import measure

import dwi.conf
import dwi.dataset
import dwi.mask
import dwi.plot
import dwi.util
from dwi.types import ImageMode


def parse_args():
    """Parse command-line arguments."""
    p = dwi.conf.get_parser(description=__doc__)
    p.add('-m', '--modes', nargs='+', type=ImageMode,
          default=['DWI-Mono-ADCm',
                   # 'DWI-Kurt-ADCk', 'DWI-Kurt-K',
                   # 'T2w-std',
                   ],
          help='imaging modes')
    p.add('-s', '--samplelist', default='all',
          help='samplelist identifier')
    p.add('-c', '--cases', nargs='+', type=int,
          help='cases to include, if not all')
    p.add('-o', '--only_prostate_slices', action='store_true',
          help='include only prostate slices')
    p.add('-r', '--include_raw', action='store_true',
          help='include "raw" b=2000 mode')
    return p.parse_args()


def read_case(mode, case, scan, lesions, only_prostate_slices, include_raw):
    """Read case. Return raw DWI, pmap, pmask, lmasks, initial prostate slice
    index.
    """
    img = dwi.dataset.read_tmap(mode, case, scan)[:, :, :, 0]
    img = dwi.util.normalize(img, mode)
    pmask = dwi.dataset.read_prostate_mask(mode, case, scan)
    lmasks = [dwi.dataset.read_lesion_mask(mode, case, scan, x) for x in
              lesions]

    mbb = img.mbb()
    img = img[mbb]
    pmask = pmask[mbb]
    lmasks = [x[mbb] for x in lmasks]

    if only_prostate_slices:
        pad = (0, np.inf, np.inf)
    else:
        pad = (np.inf, np.inf, np.inf)
    prostate_slices = pmask.mbb(pad=pad)
    img = img[prostate_slices]
    pmask = pmask[prostate_slices]
    lmasks = [x[prostate_slices] for x in lmasks]
    images = [img]

    if include_raw:
        rawmode = ImageMode('DWI-b2000')
        raw = dwi.dataset.read_tmap(rawmode[0], case, scan,
                                    params=[-1])[:, :, :, 0]
        raw = dwi.util.normalize(raw, rawmode)
        raw = raw[mbb]
        raw = raw[prostate_slices]
        images = [raw] + images

    return images, pmask, lmasks, prostate_slices[0].start


def get_label_plot(mask):
    """Get connected regions for visualization."""
    labels, n = measure.label(mask, return_num=True, connectivity=None)
    labels = (labels / n).astype(np.float32)
    logging.info([mask.info['path'], n])
    return labels


def plot_case(imgs, masks, label, path):
    """Plot case."""
    overlay = dwi.mask.overlay_masks(masks)
    masks = [get_label_plot(x) for x in masks]
    d = dict(nrows=len(imgs) + len(masks), ncols=len(imgs[0]),
             suptitle=label, path=path)
    for i, plt in enumerate(dwi.plot.generate_plots(**d)):
        row, col = i // d['ncols'], i % d['ncols']
        kwargs = dict(vmin=0, vmax=1, cmap='hot')
        if row < len(imgs):
            plt.imshow(imgs[row][col], cmap='gray')
            plt.imshow(overlay[col], **kwargs, alpha=1.0)
        else:
            mask = masks[row - len(imgs)]
            plt.imshow(mask[col], **kwargs)


def draw_dataset(ds, only_prostate_slices, include_raw):
    """Process a dataset."""
    logging.info('Mode: %s', ds.mode)
    logging.info('Samplelist: %s', ds.samplelist)
    for case, scan, lesions in ds.each_image_id():
        imgs, pmask, lmasks, _ = read_case(ds.mode, case, scan, lesions,
                                           only_prostate_slices, include_raw)
        label = '{}-{} ({})'.format(case, scan, ds.mode)
        outdir = 'fig/masks'
        path = '{}/{}-{}.png'.format(outdir, case, scan)
        # print(path, label, img.shape, pmask.shape, [x.shape for x in lmasks])
        plot_case(imgs, [pmask] + lmasks, label, path)


def main():
    """Main."""
    args = parse_args()
    datasets = (dwi.dataset.Dataset(x, args.samplelist, cases=args.cases)
                for x in args.modes)
    for ds in datasets:
        draw_dataset(ds, args.only_prostate_slices, args.include_raw)


if __name__ == '__main__':
    main()
