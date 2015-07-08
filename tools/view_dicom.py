#!/usr/bin/env python2

"""View a multi-slice, multi-b-value DWI DICOM image via the matplotlib GUI."""

# TODO Take only one path as argument.
# TODO Rename to general image viewer, not just dicom.

from __future__ import division, print_function
import argparse
import sys

import numpy as np
import matplotlib.pyplot as plt

import dwi.files
import dwi.util

def parse_args():
    """Parse command-line arguments."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--files', '-f', metavar='PATH',
                   nargs='+', default=[], required=True,
                   help='DICOM directory or file(s)')
    p.add_argument('--subwindow', '-s', metavar='i',
                   nargs=6, default=[], type=int,
                   help='ROI (6 integers, one-based)')
    p.add_argument('--verbose', '-v', action='count',
                   help='be more verbose')
    p.add_argument('--normalize', '-n', action='store_true',
                   help='normalize signal intensity curves')
    p.add_argument('--scale', action='store_true',
                   help='scale each parameter independently')
    p.add_argument('--info', '-i', action='store_true',
                   help='show information only')
    args = p.parse_args()
    return args

class Gui(object):
    """A GUI widget for viewing 4D images (from DICOM etc.)."""
    def __init__(self, image, params):
        assert image.ndim == 4
        self.image = image
        self.params = params
        self.i = 0
        self.j = 0
        self.update_x = True
        self.update_y = True
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
        view = self.image[self.i,:,:,self.j]
        self.im = plt.imshow(view, interpolation='none', vmin=self.image.min(),
                             vmax=self.image.max())
        self.show_help()
        plt.show()

    def on_key(self, event):
        if event.key == 'q':
            plt.close()
        if event.key == '1':
            self.update_x = not self.update_x
        if event.key == '2':
            self.update_y = not self.update_y
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
            self.update_x = not self.update_x
            self.update_y = not self.update_y

    def on_motion(self, event):
        if self.update_x and event.xdata:
            h, w = self.im.get_size()
            relx = event.xdata / w
            self.i = int(relx * self.image.shape[0])
        if self.update_y and event.ydata:
            h, w = self.im.get_size()
            rely = event.ydata / h
            self.j = int(rely * self.image.shape[-1])
        self.redraw(event)

    def redraw(self, event):
        if event.xdata and event.ydata:
            s = '\rPosition: {s:2d}, {r:3d}, {c:3d}, {b:2d} Param: {p} '
            d = dict(r=int(event.ydata), c=int(event.xdata),
                     s=self.i, b=self.j, p=self.params[self.j])
            sys.stdout.write(s.format(**d))
            sys.stdout.flush()
        view = self.image[self.i,:,:,self.j]
        self.im.set_data(view)
        event.canvas.draw()

    def show_help(self):
        text = '''Usage:
    Horizontal mouse move: change slice (in update mode)
    Vertical mouse move: change b-value (in update mode)
    Click: toggle update mode
    1: toggle horizontal update mode
    2: toggle vertical update mode
    e: toggle reverse colormap
    g: toggle grid
    {cmap_keys}: select colormap: {cmap_names}
    q: quit'''.format(cmap_keys=', '.join(self.cmaps.keys()),
                      cmap_names=', '.join(self.cmaps.values()))
        print(text)
        print('Slices, rows, columns, b-values: {}'.format(self.image.shape))


def reverse_cmap(name):
    """Return the name of the reverse version of given colormap."""
    if name.endswith('_r'):
        return name[:-2]
    else:
        return name + '_r'

def main():
    args = parse_args()

    if len(args.files) == 1:
        img, attrs = dwi.files.read_pmap(args.files[0])
    else:
        attrs = dwi.dicomfile.read_files(args.files)
        img = attrs['image']
        attrs['parameters'] = attrs['bvalues']

    if args.subwindow:
        # Use one-based indexing.
        z0, z1, y0, y1, x0, x1 = [i-1 for i in args.subwindow]
        # If we don't take a copy here, the normalization below fails. :I
        img = img[z0:z1, y0:y1, x0:x1, :].copy()

    print('Image shape: {s}, voxels: {nv}, non-zero voxels: {nz}'.format(
        s=img.shape, nv=img.size, nz=np.count_nonzero(img)))
    print('Image intensity range: [{}, {}]'.format(img.min(), img.max()))
    print('Image attributes:')
    for k, v in attrs.items():
        print('    {k}: {v}'.format(k=k, v=v))

    if args.normalize:
        for si in img.reshape((-1, img.shape[-1])):
            dwi.util.normalize_si_curve_fix(si)
        print('Normalized to range: [{}, {}]'.format(img.min(), img.max()))

    if args.scale:
        img = dwi.util.scale(img)
        print('Scaled to range: [{}, {}]'.format(img.min(), img.max()))

    if not args.info:
        #plt.switch_backend('gtk')
        Gui(img, attrs['parameters'])

if __name__ == '__main__':
    main()
