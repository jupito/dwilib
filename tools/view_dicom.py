#!/usr/bin/env python2

"""View a multi-slice, multi-b-value DWI DICOM image via the matplotlib GUI."""

# TODO Rename to general image viewer, not just dicom.

from __future__ import absolute_import, division, print_function
import argparse
import sys

import numpy as np
import matplotlib.pyplot as plt

import dwi.dicomfile
import dwi.files
import dwi.standardize
import dwi.util


class Gui(object):
    """A GUI widget for viewing 4D images (from DICOM etc.)."""
    def __init__(self, image, params):
        assert image.ndim == 4
        assert image.shape[-1] == len(params)
        self.image = image
        self.params = params
        self.max_param_length = max(len(_) for _ in params)
        self.pos = [0, 0]  # Slice, parameter index.
        self.update = [True, True]  # Update horizontal, vertical?
        self.reverse_cmap = False
        self.cmaps = dict(
            b='Blues_r',
            c='coolwarm',
            j='jet',
            o='bone',
            r='gray',
            y='YlGnBu_r',
        )
        fig = plt.figure()
        fig.canvas.mpl_connect('key_press_event', self.on_key)
        fig.canvas.mpl_connect('button_release_event', self.on_click)
        fig.canvas.mpl_connect('motion_notify_event', self.on_motion)
        view = self.image[self.pos[0], :, :, self.pos[1]]
        vmin, vmax = self.image.min(), self.image.max()
        self.im = plt.imshow(view, interpolation='none', vmin=vmin, vmax=vmax)
        self.show_help()
        plt.show()

    def on_key(self, event):
        if event.key == 'q':
            plt.close()
        if event.key == 'h':
            self.update[0] = not self.update[0]
        if event.key == 'v':
            self.update[1] = not self.update[1]
        if event.key == 'e':
            name = plt.get_cmap().name
            name = reverse_cmap(name)
            plt.set_cmap(name)
            self.reverse_cmap = not self.reverse_cmap
        if event.key in self.cmaps:
            name = self.cmaps[event.key]
            if self.reverse_cmap:
                name = reverse_cmap(name)
            plt.set_cmap(name)
        self.redraw(event)

    def on_click(self, event):
        if event.button == 1:
            self.update = [not _ for _ in self.update]

    def on_motion(self, event):
        h, w = self.im.get_size()
        if self.update[0] and event.xdata:
            relx = event.xdata / w
            self.pos[0] = int(relx * self.image.shape[0])
        if self.update[1] and event.ydata:
            rely = event.ydata / h
            self.pos[1] = int(rely * self.image.shape[-1])
        self.redraw(event)

    def redraw(self, event):
        if event.xdata and event.ydata:
            row = int(event.ydata)
            col = int(event.xdata)
            val = self.image[self.pos[0], row, col, self.pos[1]]
            s = ('\rPos: {s:2d},{r:3d},{c:3d},{p:2d}'
                 ' Value: {v:10g} Param: {n:{l}} ')
            d = dict(r=row, c=col, s=self.pos[0], p=self.pos[1], v=val,
                     n=self.params[self.pos[1]], l=self.max_param_length)
            sys.stdout.write(s.format(**d))
            sys.stdout.flush()
        view = self.image[self.pos[0], :, :, self.pos[1]]
        self.im.set_data(view)
        event.canvas.draw()

    def show_help(self):
        text = '''Usage:
    Horizontal mouse move: change slice (in update mode)
    Vertical mouse move: change b-value (in update mode)
    Click: toggle update mode
    h: toggle horizontal update
    v: toggle vertical update
    e: toggle reverse colormap
    g: toggle grid
    {cmap_keys}: select colormap: {cmap_names}
    q: quit'''.format(cmap_keys=', '.join(self.cmaps.keys()),
                      cmap_names=', '.join(self.cmaps.values()))
        print(text)
        print('Slices, rows, columns, b-values: {}'.format(self.image.shape))


def parse_args():
    """Parse command-line arguments."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('path',
                   help='DICOM directory or HDF5 file')
    p.add_argument('--params', type=int, nargs='+',
                   help='included parameter indices')
    p.add_argument('--subwindow', '-s', metavar='i',
                   nargs=6, default=[], type=int,
                   help='ROI (6 integers, zero-based)')
    p.add_argument('--verbose', '-v', action='count',
                   help='be more verbose')
    p.add_argument('--std',
                   help='standardization file to use')
    p.add_argument('--normalize', '-n', action='store_true',
                   help='normalize signal intensity curves')
    p.add_argument('--scale', action='store_true',
                   help='scale each parameter independently')
    p.add_argument('--info', '-i', action='store_true',
                   help='show information only')
    args = p.parse_args()
    return args


def reverse_cmap(name):
    """Return the name of the reverse version of given colormap."""
    if name.endswith('_r'):
        return name[:-2]
    else:
        return name + '_r'


def replace_nans(img):
    """Set any NaN voxels to parameter-wise background (minimum minus 10% of
    range). Return the number of replaced voxels.
    """
    nans = np.isnan(img)
    n = np.count_nonzero(nans)
    if n:
        for i in range(img.shape[-1]):
            v = img[..., i]
            nans = np.isnan(v)
            mn, mx = np.nanmin(v), np.nanmax(v)
            replacement = mn - (mx - mn) / 10
            v[nans] = replacement
        print('Replaced {} NaN voxels with sub-parameter background'.format(n))
    return n


def scale(img):
    """Scale image parameter-wise."""
    if not np.issubdtype(img.dtype, float):
        img = img.astype(np.float_)  # Integers cannot be scaled.
    for i in range(img.shape[-1]):
        img[..., i] = dwi.util.scale(img[..., i])
    print('Scaled to range: [{}, {}]'.format(img.min(), img.max()))
    return img


def main():
    args = parse_args()
    img, attrs = dwi.files.read_pmap(args.path, ondisk=True,
                                     params=args.params)
    # Image must be read in memory, because it will probably be modified. We
    # are using ondisk=True only to save memory.
    try:
        img = img[:]
    except MemoryError as e:
        print(e)
        with img.astype('float16'):
            img = img[:]

    print('Attributes:')
    for k, v in attrs.items():
        print('    {k}: {v}'.format(k=k, v=v))

    if args.subwindow:
        # If we don't take a copy here, the normalization below fails. :I
        img = dwi.util.crop_image(img, args.subwindow, onebased=False).copy()

    n = replace_nans(img)

    if args.std:
        if args.verbose:
            print('Standardizing...')
        img = dwi.standardize.standardize(img, args.std)

    print('Image shape: {s}, type: {t}'.format(s=img.shape, t=img.dtype))
    print('Voxels: {nv}, non-zero: {nz}, non-NaN: {nn}'.format(
        nv=img.size, nz=np.count_nonzero(img), nn=img.size-n))
    print('Five-num: {}'.format(dwi.util.fivenums(img)))

    if args.normalize:
        for si in img.reshape((-1, img.shape[-1])):
            dwi.util.normalize_si_curve_fix(si)
        print('Normalized to range: [{}, {}]'.format(img.min(), img.max()))

    if args.scale:
        img = scale(img)

    if not args.info:
        Gui(img, attrs['parameters'])


if __name__ == '__main__':
    main()
