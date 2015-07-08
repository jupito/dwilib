#!/usr/bin/env python2

"""View a multi-slice, multi-b-value DWI DICOM image via the matplotlib GUI."""

# TODO Take only one path as argument.
# TODO Rename to general image viewer, not just dicom.

from __future__ import division, print_function
import argparse
import sys

import numpy as np
import matplotlib.pyplot as plt

import dwi.dwimage
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
    p.add_argument('--info', '-i', action='store_true',
                   help='show information only')
    args = p.parse_args()
    return args

class Gui(object):
    def __init__(self, image):
        self.image = image
        self.i = 0
        self.j = 0
        self.update_x = True
        self.update_y = True
        self.colormaps = dict(
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
        if event.key == '1':
            self.update_x = not self.update_x
        if event.key == '2':
            self.update_y = not self.update_y
        if event.key == 'q':
            plt.close()
        if event.key in self.colormaps:
            plt.set_cmap(self.colormaps[event.key])
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
            s = '\rslice {s:2d}, row {r:3d}, column {c:3d}, b-value {b:2d} '
            d = dict(r=int(event.ydata)+1, c=int(event.xdata)+1,
                     s=self.i+1, b=self.j+1)
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
    g: toggle grid
    {cmap_keys}: select colormap: {cmap_names}
    q: quit'''.format(cmap_keys=', '.join(self.colormaps.keys()),
                      cmap_names=', '.join(self.colormaps.values()))
        print(text)

def main():
    args = parse_args()

    if len(args.files) == 1:
        dwimage = dwi.dwimage.load(args.files[0])[0]
    else:
        dwimage = dwi.dwimage.load_dicom(args.files)[0]

    if args.subwindow:
        dwimage = dwimage.get_roi(args.subwindow, onebased=True)

    if args.normalize:
        for si in dwimage.sis:
            dwi.util.normalize_si_curve(si)

    print(dwimage)
    d = dict(min=dwimage.image.min(), max=dwimage.image.max(),
             vs=dwimage.voxel_spacing, nz=np.count_nonzero(dwimage.image))
    print('Image intensity min/max: {min}/{max}'.format(**d))
    print('Voxel spacing: {vs}'.format(**d))
    print('Non-zero voxels: {nz}'.format(**d))

    if not args.info:
        #plt.switch_backend('gtk')
        Gui(dwimage.image)

if __name__ == '__main__':
    main()
