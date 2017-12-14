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
    p.add('-A', '--all_slices', action='store_true',
          help='include all slices')
    p.add('-r', '--include_raw', action='store_true',
          help='include "raw" b=2000 mode')
    p.add('-C', '--connected_regions', action='store_true',
          help='colorize connected regions')
    p.add('-L', '--label', action='store_true',
          help='write text label')
    return p.parse_args()


def read_case(mode, case, scan, lesions, all_slices, include_raw):
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

    if all_slices:
        pad = (np.inf, np.inf, np.inf)
    else:
        pad = (0, np.inf, np.inf)
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


def plot_case(imgs, masks, label, path, connected_regions):
    """Plot case."""
    overlay = dwi.mask.overlay_masks(masks)
    if connected_regions:
        masks = [get_label_plot(x) for x in masks]
    d = dict(nrows=len(imgs) + len(masks), ncols=len(imgs[0]), suptitle=label,
             path=path)
    d['titles'] = [None] * (d['nrows'] * d['ncols'])
    for i, plt in enumerate(dwi.plot.generate_plots(**d)):
        row, col = i // d['ncols'], i % d['ncols']
        kwargs = dict(vmin=0, vmax=1, cmap='hot')
        if row < len(imgs):
            plt.imshow(imgs[row][col], cmap='gray')
            plt.imshow(overlay[col], **kwargs, alpha=1.0)
        else:
            mask = masks[row - len(imgs)]
            plt.imshow(mask[col], **kwargs)


def draw_dataset(ds, all_slices, include_raw, connected_regions,
                 label='{c}-{s} ({m})'):
    """Process a dataset."""
    logging.info('Mode: %s', ds.mode)
    logging.info('Samplelist: %s', ds.samplelist)
    for i, (case, scan, lesions) in enumerate(ds.each_image_id(), 1):
        imgs, pmask, lmasks, _ = read_case(ds.mode, case, scan, lesions,
                                           all_slices, include_raw)
        if label:
            label = label.format(c=case, s=scan, m=ds.mode)
        outdir = 'fig/masks'
        path = '{od}/{i}_{c}-{s}.png'.format(od=outdir, i=i, c=case, s=scan)
        # print(path, label, img.shape, pmask.shape, [x.shape for x in lmasks])
        plot_case(imgs, [pmask] + lmasks, label, path, connected_regions)


def main():
    """Main."""
    args = parse_args()
    datasets = (dwi.dataset.Dataset(x, args.samplelist, cases=args.cases)
                for x in args.modes)
    label = '{c}-{s} ({m})' if args.label else None
    for ds in datasets:
        draw_dataset(ds, args.all_slices, args.include_raw,
                     args.connected_regions, label=label)


if __name__ == '__main__':
    main()
