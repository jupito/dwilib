"""Utilities for handling DWI images."""

from time import time
import os
import numpy as np
import scipy as sp

import fit
import util

def load(filename, nrois=1, varname='ROIdata'):
    """Load images from a file."""
    root, ext = os.path.splitext(filename)
    if ext == '.mat':
        return load_matlab(filename, varname)
    else:
        return load_ascii(filename, nrois)

def load_matlab(filename, varname='ROIdata'):
    """Load images from a MATLAB file."""
    mat = sp.io.loadmat(filename, struct_as_record=False)
    r = []
    for window in mat[varname][0]:
        win = window[0,0]
        sis = win.SIs.T
        bset = win.bset[0]
        dwi = DWImage(sis, bset)
        dwi.filename = filename
        dwi.roislice = '-' # Not implemented.
        dwi.name = '-' # Not implemented.
        dwi.number = int(win.number[0,0])
        try:
            dwi.subwindow = map(int, win.subwindow[0])
        except:
            dwi.subwindow = util.fabricate_subwindow(len(sis))
        r.append(dwi)
    return r

def load_ascii(filename, nrois=1):
    """Load images from an ASCII file."""
    import asciifile
    af = asciifile.AsciiFile(filename)
    a = af.a.reshape(nrois,-1,af.a.shape[-1])
    r = []
    for i in range(nrois):
        sis = a[i]
        bset = af.bset()
        dwi = DWImage(sis, bset)
        dwi.filename = filename
        dwi.roislice = af.roislice()
        dwi.name = af.name()
        dwi.number = af.number + i
        dwi.subwindow = af.subwindow()
        r.append(dwi)
    return r

def load_dicom(filenames):
    """Load images from DICOM files."""
    import dicomfile
    bset, slices = dicomfile.read_files(filenames)
    r = []
    for i, s in enumerate(slices):
        dwi = DWImage(s, bset)
        dwi.filename = filenames[0] + '...'
        dwi.roislice = '-'
        dwi.name = '-'
        dwi.number = i
        dwi.subwindow = (0, s.shape[0], 0, s.shape[1])
        r.append(dwi)
    return r

class DWImage(object):
    """DWI image, a single slice of signal intensities at several b-values.

    Variables
    ---------
    image : ndarray, shape = [height, width, n_bvalues]
        Slice of voxels.
    sis : ndarray, shape = [height * width, n_bvalues]
        Flattened view of the image.
    bset : ndarray, shape = [n_bvalues]
        Different b-values.
    execution_time : number
        Number of seconds spent executing last calculation, or -1.
    """

    def __init__(self, image, bset):
        """Create a new DWI image.

        Parameters
        ----------
        image : array_like, shape = [(height,) width, n_bvalues]
            Array of signal intensities at different b-values.
        bset : sequence
            Different b-values.
        """
        self.image = np.array(image, dtype=float, ndmin=3)
        self.sis = self.image.view()
        self.sis.shape = (-1,self.image.shape[-1])
        self.bset = np.array(sorted(set(bset)), dtype=float)
        self.execution_time = -1
        if not len(self.image.shape) == 3:
            raise Exception('Image has invalid dimensions.')
        if not self.image.shape[-1] == self.sis.shape[-1] == len(self.bset):
            raise Exception('Image size does not match with b-values.')

    def __repr__(self):
        return '%s:%i' % (self.filename, self.number)

    def __str__(self):
        d = dict(fn=self.filename, n=self.number,
                nb=len(self.bset), b=list(self.bset),
                size=self.size(), shape=self.shape(),
                w=self.subwindow, ws=self.subwindow_shape())
        s = 'File: {fn}\n'\
                'Number: {n}\n'\
                'B-values: {nb}: {b}\n'\
                'Voxels: {size}, {shape}\n'\
                'Window: {w}, {ws}'.format(**d)
        return s

    def subwindow_shape(self):
        return tuple((b-a for a, b in util.chunks(self.subwindow, 2)))

    def shape(self):
        """Return image height and width."""
        return self.image.shape[0:-1]

    def size(self):
        """Return number of voxels."""
        return len(self.sis)

    def fit_elem(self, model, elem, bvalues=[], mean=False):
        """Fit model to an image element.

        Parameters
        ----------
        model : fit.Model
            Model used for fitting.
        elem : integer
            Voxel index.
        bvalues : sequence of integers, optional
            Selected b-values (empty for all).
        mean : bool, optional
            Use mean of all voxels.

        Returns
        -------
        params : ndarray
            Result parameters.
        err : float
            RMSE.
        """
        if not bvalues:
            bvalues = range(len(self.bset))
        xdata = self.bset[bvalues]
        if mean:
            ydata = self.sis.mean(axis=0)[bvalues]
        else:
            ydata = self.sis[elem][bvalues]
        return model.fit_mi(xdata, ydata)

    def fit_whole(self, model, bvalues=[], log=None, mean=False):
        """Fit model to whole image.

        Parameters
        ----------
        model : fit.Model
            Model used for fitting.
        bvalues : sequence of integers, optional
            Selected b-values (empty for all).
        log : OutStream, optional
            Logging stream.
        mean : bool, optional
            Use mean of all voxels.

        Returns
        -------
        pmap : ndarray
            Result parameters and RMSE.
        """
        start = time()
        size = 1 if mean else self.size()
        pmap = np.zeros((size, len(model.params) + 1))
        cnt_errors = 0
        cnt_warnings = 0
        if log:
            log('Fitting %i elements to %s...\n' % (size, model))
        for i in range(size):
            if log:
                log('\r%i...' % i)
            params, err = self.fit_elem(model, i, bvalues, mean)
            pmap[i, -1] = err
            if np.isfinite(err):
                pmap[i, :-1] = params
            else:
                pmap[i, :-1].fill(np.nan)
                cnt_errors += 1
        if cnt_errors:
            map(util.impute, pmap.T)
        self.execution_time = time() - start
        if log:
            log('\nFinished with %i errors, %i warnings.\n'\
                    % (cnt_errors, cnt_warnings))
        return pmap
