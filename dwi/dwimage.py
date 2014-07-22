"""Utilities for handling DWI images."""

from time import time
import os.path
import numpy as np
import scipy as sp
import scipy.io

import asciifile
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
        try:
            dwi.subwindow = af.subwindow()
        except:
            dwi.subwindow = util.fabricate_subwindow(len(sis))
        r.append(dwi)
    return r

class DWImage(object):
    """DWI image, stored single-dimensionally."""

    def __init__(self, sis, bset):
        """Create a new DWI image.

        Parameters
        ----------
        sis : array_like, shape = (n_voxels, n_bvalues)
            Voxels representing signal intensities.
        bset : sequence
            Different b-values.
        """
        self.sis = np.array(sis, dtype=float)
        self.bset = np.array(sorted(set(bset)), dtype=float)
        self.execution_time = -1
        if self.sis.shape != (len(self.sis), len(self.bset)):
            raise Exception('Image size does not match with b-values.')

    def height(self):
        return self.subwindow[1] - self.subwindow[0] + 1

    def width(self):
        return self.subwindow[3] - self.subwindow[2] + 1

    def size(self):
        """Return number of voxels."""
        return len(self.sis)

    def __repr__(self):
        return '%s:%i' % (self.filename, self.number)

    def __str__(self):
        d = dict(fn=self.filename, n=self.number,
                nb=len(self.bset), b=list(self.bset),
                s=self.size(), win=self.subwindow,
                h=self.height(), w=self.width())
        s = 'File: {fn}\n'\
                'Number: {n}\n'\
                'B-values: {nb}: {b}\n'\
                'Window: {s}, {win}, {h}x{w}'.format(**d)
        return s

    def fit_elem(self, model, elem, bvalues=[], mean=False):
        """Curve fitting for an image element."""
        if not bvalues:
            bvalues = range(len(self.bset))
        xdata = self.bset[bvalues]
        if mean:
            ydata = self.sis.mean(axis=0)[bvalues]
        else:
            ydata = self.sis[elem][bvalues]
        return model.fit_mi(xdata, ydata)

    def fit_whole(self, model, bvalues=[], log=None, mean=False):
        """Curve fitting for the whole image."""
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
