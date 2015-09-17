"""Utilities for handling DWI images.

Class DWImage contains a DWI image, with some functionality like fitting. Use
load() to load an image from DICOM, ASCII, HDF5, or MATLAB source.
"""

from __future__ import absolute_import, division, print_function
import os
from time import time

import numpy as np

import dwi.util


class DWImage(object):
    """DWI image, a single slice of signal intensities at several b-values.

    Variables
    ---------
    image : ndarray, shape = [depth, height, width, n_bvalues]
        Array of voxels.
    sis : ndarray, shape = [depth * height * width, n_bvalues]
        Flattened view of the image.
    bset : ndarray, shape = [n_bvalues]
        Different b-values.
    """

    def __init__(self, image, bset):
        """Create a new DWI image.

        Parameters
        ----------
        image : array_like, shape = [((depth,) height,) width, n_bvalues]
            Array of signal intensities at different b-values.
        bset : sequence
            Different b-values.
        """
        self.image = np.array(image, dtype=float, ndmin=4)
        self.sis = self.image.view()
        self.sis.shape = (-1, self.image.shape[-1])
        # self.bset = np.array(sorted(set(bset)), dtype=float)
        self.bset = np.array(bset, dtype=float)
        self.start_time = self.end_time = -1
        if len(self.image.shape) != 4:
            raise Exception('Invalid image dimensions.')
        if not self.image.shape[-1] == self.sis.shape[-1] == len(self.bset):
            raise Exception('Image size does not match with b-values.')
        self.filename = None
        self.basename = None
        self.roislice = None
        self.name = None
        self.number = None
        self.subwindow = None
        self.voxel_spacing = None

    def __repr__(self):
        return '%s:%i' % (self.filename, self.number)

    def __str__(self):
        d = dict(fn=self.filename, n=self.number, nb=len(self.bset),
                 b=list(self.bset), size=self.size(), shape=self.shape(),
                 w=self.subwindow, ws=self.subwindow_shape())
        s = ('File: {fn}\n'
             'Number: {n}\n'
             'B-values: {nb}: {b}\n'
             'Voxels: {size}, {shape}\n'
             'Window: {w}, {ws}'.format(**d))
        return s

    def subwindow_shape(self):
        return dwi.util.subwindow_shape(self.subwindow)

    def shape(self):
        """Return image height and width."""
        return self.image.shape[0:-1]

    def size(self):
        """Return number of voxels."""
        return len(self.sis)

    def get_roi(self, position, bvalues=None, onebased=True):
        """Get a view of a specific ROI (region of interest)."""
        if onebased:
            position = [i-1 for i in position]  # One-based indexing.
        z0, z1, y0, y1, x0, x1 = position
        if bvalues is None:
            bvalues = range(len(self.bset))
        image = self.image[z0:z1, y0:y1, x0:x1, bvalues]
        bset = self.bset[bvalues]
        dwimage = DWImage(image, bset)
        dwimage.filename = self.filename
        dwimage.roislice = self.roislice
        dwimage.name = self.name
        dwimage.number = self.number
        dwimage.subwindow = (self.subwindow[0] + z0,
                             self.subwindow[0] + z1,
                             self.subwindow[2] + y0,
                             self.subwindow[2] + y1,
                             self.subwindow[4] + x0,
                             self.subwindow[4] + x1)
        if onebased:
            dwimage.subwindow = tuple(i+1 for i in dwimage.subwindow)
        dwimage.voxel_spacing = self.voxel_spacing
        return dwimage

    def start_execution(self):
        """Start taking time for an operation."""
        self.start_time = time()

    def end_execution(self):
        """End taking time for an operation."""
        self.end_time = time()

    def execution_time(self):
        """Return time consumed by previous operation."""
        return self.end_time - self.start_time

    def fit(self, model, average=False):
        """Fit model to whole image.

        Parameters
        ----------
        model : dwi.fit.Model
            Model used for fitting.
        average : bool, optional
            Fit just the mean of all voxels.

        Returns
        -------
        pmap : ndarray
            Result parameters and RMSE.
        """
        self.start_execution()
        xdata = self.bset
        ydatas = self.sis
        if average == 'mean' or average is True:
            ydatas = np.mean(ydatas, axis=0, keepdims=True)
        elif average == 'median':
            ydatas = dwi.util.median(ydatas, axis=0, keepdims=True)
        elif average:
            raise Exception('Invalid averaging method: {}'.format(average))
        pmap = model.fit(xdata, ydatas)
        self.end_execution()
        return pmap


def load(filename, nrois=1, varname='ROIdata'):
    """Load images from a file or directory."""
    _, ext = os.path.splitext(filename)
    if ext == '.mat':
        return load_matlab(filename, varname)
    elif ext in ('.txt', '.ascii'):
        return load_ascii(filename, nrois)
    elif ext in ('.hdf5', '.h5'):
        return load_hdf5(filename)
    else:
        return load_dicom([filename])


def load_matlab(filename, varname='ROIdata'):
    """Load images from a MATLAB file."""
    import scipy.io
    mat = scipy.io.loadmat(filename, struct_as_record=False)
    r = []
    for window in mat[varname][0]:
        win = window[0, 0]
        sis = win.SIs.T
        bset = win.bset[0]
        dwimage = DWImage(sis, bset)
        dwimage.filename = filename
        dwimage.basename = os.path.basename(filename)
        dwimage.roislice = '-'  # Not implemented.
        dwimage.name = '-'  # Not implemented.
        dwimage.number = int(win.number[0, 0])
        try:
            dwimage.subwindow = tuple(map(int, win.subwindow[0]))
        except:
            dwimage.subwindow = dwi.util.fabricate_subwindow(len(sis))
        dwimage.voxel_spacing = (1.0, 1.0, 1.0)
        r.append(dwimage)
    return r


def load_ascii(filename, nrois=1):
    """Load images from an ASCII file."""
    import dwi.asciifile
    af = dwi.asciifile.AsciiFile(filename)
    a = af.a.reshape(nrois, -1, af.a.shape[-1])
    r = []
    for i in range(nrois):
        sis = a[i]
        bset = af.bset()
        dwimage = DWImage(sis, bset)
        dwimage.filename = filename
        dwimage.basename = os.path.basename(filename)
        dwimage.roislice = af.roislice
        dwimage.name = af.name
        dwimage.number = af.number + i
        dwimage.subwindow = af.subwindow()
        dwimage.voxel_spacing = (1.0, 1.0, 1.0)
        r.append(dwimage)
    return r


def load_hdf5(filename):
    """Load image from an HDF5 file."""
    import dwi.hdf5
    a, d = dwi.hdf5.read_hdf5(filename)
    if a.ndim != 4:
        a = a.reshape(1, 1, -1, a.shape[-1])
    dwimage = DWImage(a, d.get('bset') or range(a.shape[-1]))
    dwimage.filename = os.path.abspath(filename)
    dwimage.basename = os.path.basename(filename)
    dwimage.number = 0
    dwimage.subwindow = (0, a.shape[0], 0, a.shape[1], 0, a.shape[2])
    dwimage.voxel_spacing = (1.0, 1.0, 1.0)
    return [dwimage]


def load_dicom(filenames):
    """Load a 3d image from DICOM files with slices combined.

    If only one filename is given, it is assumed to be a directory.
    """
    import dwi.dicomfile
    if len(filenames) == 1 and os.path.isdir(filenames[0]):
        d = dwi.dicomfile.read_dir(filenames[0])  # Directory.
    else:
        d = dwi.dicomfile.read_files(filenames)  # File list.
    img = d['image']
    dwimage = DWImage(img, d['bvalues'])
    dwimage.filename = os.path.abspath(filenames[0])
    dwimage.basename = os.path.basename(filenames[0])
    dwimage.roislice = '-'
    dwimage.name = '-'
    dwimage.number = 0
    dwimage.subwindow = (0, img.shape[0], 0, img.shape[1], 0, img.shape[2])
    dwimage.voxel_spacing = d['voxel_spacing']
    return [dwimage]
