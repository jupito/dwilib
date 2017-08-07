"""Parametric model classes and fitting functionality."""

from itertools import product

import numpy as np

import dwi.fit_one_by_one

# Select fitting implementation.
fit_curves_mi = dwi.fit_one_by_one.fit_curves_mi


class Parameter(object):
    """Parameter used in model definitions."""

    def __init__(self, name, steps, bounds, use_stepsize=True, relative=False):
        """Create a new model parameter.

        Parameters
        ----------
        name : string
            Parameter name.
        steps : tuple
            Steps as (start, stop, size/number).
        bounds : tuple
            Constraints as (start, end).
        use_stepsize : bool, optional, default True
            Use step size instead of number.
        relative : bool, optional, default False
            Values are relative to a constant given upon request.
        """
        self.name = name
        self.steps = steps
        self.bounds = bounds
        self.use_stepsize = use_stepsize
        self.relative = relative

    def __repr__(self):
        return '%s=%s' % (self.name, self.steps)

    def __str__(self):
        return self.name

    def guesses(self, c):
        """Return initial guesses."""
        if self.use_stepsize:
            g = np.arange(*self.steps)
        else:
            g = np.linspace(*self.steps)
        if self.relative:
            g *= c
        return g


class Model(object):
    def __init__(self, name, desc, func, params, preproc=None, postproc=None):
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

    def __repr__(self):
        return '%s %s' % (self.name, ' '.join(repr(x) for x in self.params))

    def __str__(self):
        return self.name

    def bounds(self):
        """Return bounds of all parameters."""
        return [x.bounds for x in self.params]

    def guesses(self, c):
        """Return all combinations of initial guesses."""
        return product(*[x.guesses(c) for x in self.params])

    def fit(self, xdata, ydatas):
        """Fit model to multiple voxels."""
        xdata = np.asanyarray(xdata)
        ydatas = np.asanyarray(ydatas)
        ydatas = prepare_for_fitting(ydatas)
        if self.preproc:
            for ydata in ydatas:
                self.preproc(ydata)
        shape = (len(ydatas), len(self.params) + 1)
        pmap = np.zeros(shape)
        if self.func:
            fit_curves_mi(self.func, xdata, ydatas, self.guesses,
                          self.bounds(), pmap)
        else:
            pmap[:, :-1] = ydatas  # Fill with original data.
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
