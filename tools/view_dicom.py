#!/usr/bin/python3

"""View a multi-slice, multi-parameter DICOM image or pmap via the matplotlib
GUI."""

# TODO Rename to general image viewer, not just dicom.

import argparse
import logging
import sys

import numpy as np
import matplotlib.pyplot as plt

import dwi.files
import dwi.mask
import dwi.plot
import dwi.standardize
import dwi.util


class Gui(object):
    """A GUI widget for viewing 4D images (from DICOM etc.)."""
    cmaps = [
        'viridis',
        'Blues_r',
        'YlGnBu_r',
        'bone',
        'coolwarm',
        'gray',
        'jet',
        ]
    usage = '''Usage:
    Horizontal mouse move: change slice (if enabled)
    Vertical mouse move: change parameter (if enabled)
    Mouse button 1: toggle update on mouse move for both axes
    h: toggle horizontal update on mouse move
    v: toggle vertical update on mouse move
    g: toggle grid
    1-{}: select colormap: {}
    r: toggle reverse colormap
    q: quit'''.format(len(cmaps), ' '.join(cmaps))

    def __init__(self, image, params, label=None, masks=None):
        assert image.ndim == 4, image.shape
        assert image.shape[-1] == len(params), (image.shape, params)
        assert image.dtype in (np.bool, np.float32, np.float64), image.dtype
        self.image = image
        self.params = params
        self.max_param_length = max(len(x) for x in params)
        self.pos = [0, 0]  # Slice, parameter index.
        self.update = [True, True]  # Update horizontal, vertical?
        self.is_reverse_cmap = False
        if label is None:
            label = str(image.shape)
        self.label = label
        if masks:
            dwi.mask.overlay_masks(masks, self.image)
        self.im = None

    def show(self):
        """Activate the GUI."""
        print(self.usage)
        print('Slices, rows, columns, parameters: {}'.format(self.image.shape))
        fig = plt.figure()
        fig.canvas.mpl_connect('key_press_event', self.on_key)
        fig.canvas.mpl_connect('button_release_event', self.on_click)
        fig.canvas.mpl_connect('motion_notify_event', self.on_motion)
        view = self.image[self.pos[0], :, :, self.pos[1]]
        vmin, vmax = np.nanmin(self.image), np.nanmax(self.image)
        self.im = plt.imshow(view, interpolation='none', vmin=vmin, vmax=vmax)
        plt.show()

    def on_key(self, event):
        """Handle keypress."""
        if event.key == 'q':
            plt.close()
            return
        if event.key == 'h':
            self.update[0] = not self.update[0]
        if event.key == 'v':
            self.update[1] = not self.update[1]
        if event.key == 'r':
            self.toggle_reverse_cmap()
        if event.key.isdigit():
            self.set_cmap(int(event.key) - 1)
        self.redraw(event)

    def on_click(self, event):
        """Handle mouse click."""
        if event.button == 1:
            self.update = [not x for x in self.update]

    def on_motion(self, event):
        """Handle mouse move."""
        h, w = self.im.get_size()
        if self.update[0] and event.xdata:
            relx = event.xdata / w
            self.pos[0] = int(relx * self.image.shape[0])
        if self.update[1] and event.ydata:
            rely = event.ydata / h
            self.pos[1] = int(rely * self.image.shape[-1])
        self.redraw(event)

    def redraw(self, event):
        """Redraw window."""
        slc, param = self.pos
        if event.xdata and event.ydata:
            row = int(event.ydata)
            col = int(event.xdata)
            val = self.image[slc, row, col, param]
            s = ('\rPos: {s:2d},{r:3d},{c:3d},{p:2d}'
                 ' Value: {v:10g} Param: {n:{l}} ')
            d = dict(r=row, c=col, s=slc, p=param, v=val, n=self.params[param],
                     l=self.max_param_length)
            sys.stdout.write(s.format(**d))
            sys.stdout.flush()
        view = self.image[slc, :, :, param]
        self.im.set_data(view)
        event.canvas.draw()
        plt.xlabel(slc, fontsize='large')
        plt.ylabel(param, fontsize='large', rotation='horizontal')
        title = '\n'.join([self.label, self.params[param]])
        plt.title(title, fontsize='large')

    def toggle_reverse_cmap(self):
        """Reverse colormap."""
        name = plt.get_cmap().name
        name = dwi.plot.reverse_cmap(name)
        plt.set_cmap(name)
        self.is_reverse_cmap = not self.is_reverse_cmap

    def set_cmap(self, i):
        """Set colormap."""
        if 0 <= i < len(self.cmaps):
            name = self.cmaps[i]
            if self.is_reverse_cmap:
                name = dwi.plot.reverse_cmap(name)
            plt.set_cmap(name)


def parse_args():
    """Parse command-line arguments."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('-v', '--verbose', action='count',
                   help='be more verbose')
    p.add_argument('-p', '--params', nargs='+',
                   help='included parameter indices')
    p.add_argument('--mbb', action='store_true',
                   help='take minimum bounding box')
    p.add_argument('-w', '--subwindow', metavar='i',
                   nargs=6, default=[], type=int,
                   help='ROI (6 integers, zero-based)')
    p.add_argument('-m', '--mask', nargs='+',
                   help='mask files')
    p.add_argument('--std',
                   help='standardization file to use')
    p.add_argument('-n', '--normalize', metavar='MODE',
                   help='normalize signal intensity curves')
    p.add_argument('-s', '--scale', action='store_true',
                   help='scale each parameter independently')
    p.add_argument('-z', '--zoom', action='store_true',
                   help='rescale image dimensions')
    p.add_argument('-i', '--info', action='store_true',
                   help='show information only')
    p.add_argument('path',
                   help='image (DICOM directory, zip file, HDF5 file)')
    args = p.parse_args()
    return args


def read_to_mem(img):
    """Read image to memory, converting its type if too big."""
    # Image must be read in memory, because it will probably be modified. We
    # are using ondisk=True only to save memory.
    try:
        img = img[:]
    except MemoryError:
        newtype = np.float16
        logging.warning('Cannot fit image to memory as %s, converting to %s',
                        img.dtype, newtype)
        with img.astype(newtype):
            img = img[:]
    return img


def replace_nans(img):
    """Set any NaN voxels to parameter-wise background (minimum minus 10% of
    range). Return the number of replaced voxels.
    """
    def _replacement(a):
        mn, mx = np.nanmin(a), np.nanmax(a)
        return mn - (mx - mn) / 10
    nans = np.isnan(img)
    n = np.count_nonzero(nans)
    if n:
        for i in range(img.shape[-1]):
            v = img[..., i]
            nans = np.isnan(v)
            if np.any(nans):
                v[nans] = _replacement(v)
        print('Replaced {} NaN voxels with sub-param background.'.format(n))
    return n


def scale(img):
    """Scale image parameter-wise. Must not contain nans."""
    if not np.issubdtype(img.dtype, float):
        img = img.astype(np.float_)  # Integers cannot be scaled.
    for i in range(img.shape[-1]):
        img[..., i] = dwi.util.scale(img[..., i])
    print('Scaled to range: [{}, {}]'.format(img.min(), img.max()))
    return img


def zoom(img, masks, spacing=None, order=1):
    """Zoom."""
    if spacing is None:
        spacing = (5, 1, 1)
    # factor = [10, 2, 2]
    factor = [x / min(spacing) * 2 for x in spacing]
    print('Zooming', img.shape, img.dtype, img.mean())
    img = dwi.util.zoom(img, factor + [1], order=order)
    print('Zoomed', img.shape, img.dtype, img.mean())
    if masks:
        _means = [x.mean() for x in masks]
        print('Zooming masks', _means)
        masks = [dwi.util.zoom_as_float(x, factor, order=order) for x in masks]
        _means = [x.mean() for x in masks]
        print('Zoomed masks', _means)
    return img, masks


def main():
    """Main."""
    args = parse_args()
    img, attrs = dwi.files.read_pmap(args.path, ondisk=True,
                                     params=args.params)
    img = read_to_mem(img)

    if args.mbb:
        print('Taking minimum bounding_box from {}'.format(img.shape))
        bb = dwi.util.bbox(img[:, :, :, 0])
        img = img[bb]
    else:
        bb = None

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

    # Normalize wrt. signal intensity curve start or imaging parameter specs.
    if args.normalize:
        print('Normalizing as {}...'.format(args.normalize))
        if args.normalize == 'DWI':
            for si in img.reshape((-1, img.shape[-1])):
                dwi.util.normalize_si_curve_fix(si)
        else:
            img = dwi.util.normalize(img, args.normalize).astype(np.float32)

    if args.scale:
        img = scale(img)

    if args.mask:
        masks = [dwi.files.read_mask(x) for x in args.mask]
        if bb:
            masks = [x[bb] for x in masks]
    else:
        masks = None

    if args.zoom:
        img, masks = zoom(img, masks, spacing=attrs.get('voxel_spacing'))

    if not args.info:
        gui = Gui(img, attrs['parameters'], label=args.path, masks=masks)
        gui.show()


if __name__ == '__main__':
    main()
    print()  # The line has leftover characters.
