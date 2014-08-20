"""Parametric model classes and fitting functionality."""

import numpy as np
from leastsqbound import leastsqbound

import util

class Parameter(object):
    """Parameter used in model definitions."""

    def __init__(self, name, steps, bounds, use_stepsize=True):
        """Create a new model parameter.

        Parameters
        ----------
        name : string
            Parameter name.
        steps : tuple
            Steps as (start, stop, size/number).
        bounds : tuple
            Constraints as (start, end).
        use_stepsize : bool, optional
            Use step size instead of number.
        """
        self.name = name
        self.steps = steps
        self.bounds = bounds
        self.use_stepsize = use_stepsize

    def __str__(self):
        return self.name

    def __repr__(self):
        return '%s=%s' % (self.name, self.steps)

    def guesses(self):
        """Return initial guesses."""
        if self.use_stepsize:
            g = np.arange(*self.steps)
        else:
            g = np.linspace(*self.steps)
        return g


class Model(object):
    def __init__(self, name, desc, func, params,
            preproc=None, postproc=None):
        """Create a new model definition.

        Parameters
        ----------
        name : string
            Model name.
        desc : string
            Model description.
        func : callable
            Fitted function.
        params : Parameter
            Parameter definitions.
        preproc : callable, optional
            Preprocessing function for data.
        postproc : callable, optional
            Postprocessing function for fitted parameters.
        """
        self.name = name
        self.desc = desc
        self.func = func
        self.params = params
        self.preproc = preproc
        self.postproc = postproc

    def __str__(self):
        return self.name

    def __repr__(self):
        return '%s %s' % (self.name, ' '.join(map(repr, self.params)))

    def bounds(self):
        """Return bounds of all parameters."""
        return [p.bounds for p in self.params]

    def guesses(self):
        """Return all combinations of initial guesses."""
        return util.combinations(map(lambda p: p.guesses(), self.params))

    def fit(self, xdata, ydatas):
        """Fit model to multiple voxels."""
        ydatas = prepare_for_fitting(ydatas)
        if self.preproc:
            for ydata in ydatas:
                self.preproc(ydata)
        shape = (len(ydatas), len(self.params) + 1)
        pmap = np.zeros(shape)
        if self.func:
            fit_curves_mi(self.func, xdata, ydatas,
                    self.guesses(), self.bounds(), pmap)
        else:
            pmap[:, :-1] = ydatas # Fill with original data.
        if self.postproc:
            for params in pmap:
                self.postproc(params[:-1])
        return pmap


def prepare_for_fitting(voxels):
    """Return a copy of voxels, prepared for fitting."""
    voxels = voxels.copy()
    for v in voxels:
        if v[0] == 0:
            # S(0) is not expected to be 0, set whole curve to 1 (ADC 0).
            v[:] = 1
    return voxels

def fit_curve(f, xdata, ydata, guess, bounds):
    """Fit a curve to data."""
    residual = lambda p, x, y: f(p, x) - y
    params, ier = leastsqbound(residual, guess, args=(xdata, ydata),
            bounds=bounds)
    if 0 < ier < 5:
        err = rmse(f, params, xdata, ydata)
    else:
        err = np.inf
    return params, err

def fit_curve_mi(f, xdata, ydata, guesses, bounds):
    """Fit a curve to data with multiple initializations."""
    guess = guesses[0]
    best_params = []
    best_err = np.inf
    for guess in guesses:
        params, err = fit_curve(f, xdata, ydata, guess, bounds)
        if err < best_err:
            best_params = params
            best_err = err
    return best_params, best_err

def fit_curves_mi(f, xdata, ydatas, guesses, bounds, out_pmap):
    """Fit curves to data with multiple initializations."""
    for i, ydata in enumerate(ydatas):
        params, err = fit_curve_mi(f, xdata, ydata, guesses, bounds)
        out_pmap[i, -1] = err
        if np.isfinite(err):
            out_pmap[i, :-1] = params
        else:
            out_pmap[i, :-1].fill(np.nan)

def rmse(f, p, xdata, ydata):
    """Root-mean-square error."""
    sqerr = (f(p, xdata) - ydata) ** 2
    return np.sqrt(sqerr.mean())
