"""Parametric model definitions and fitting."""

import numpy as np
from leastsqbound import leastsqbound

import util

class Parameter(object):
    """Parameter used in model definitions."""

    def __init__(self, name, steps, bounds,
            use_stepsize=True, relative=False):
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
        relative : bool, optional
            Consider steps and bounds relative to SI(0).
        """
        self.name = name
        self.steps = steps
        self.bounds = bounds
        self.use_stepsize = use_stepsize
        self.relative = relative

    def __str__(self):
        return self.name

    def __repr__(self):
        return '%s=%s' % (self.name, self.steps)

    def steps_rel(self, si0):
        c = si0 if self.relative else 1.0
        return tuple(np.array(self.steps) * c)

    def bounds_rel(self, si0):
        c = si0 if self.relative else 1.0
        return tuple(np.array(self.bounds) * c)

    def guesses(self, si0):
        """Return initial guesses."""
        if self.use_stepsize:
            g = np.arange(*self.steps)
        else:
            g = np.linspace(*self.steps)
        if self.relative:
            g = g * si0
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

    def bounds(self, si0):
        """Return bounds of all parameters."""
        return [p.bounds_rel(si0) for p in self.params]

    def guesses(self, si0):
        """Return all combinations of initial guesses."""
        return util.combinations(map(lambda p: p.guesses(si0), self.params))

    def fit_mi(self, xdata, ydata):
        """Fit model to data with multiple initializations."""
        if self.preproc:
            self.preproc(ydata)
        if self.func:
            si0 = ydata[0]
            guesses = self.guesses(si0)
            bounds = self.bounds(si0)
            params, err = fit_curve_mi(self.func, xdata, ydata, guesses, bounds)
        else:
            params, err = ydata, 0.
        if self.postproc:
            self.postproc(params)
        return params, err


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

def rmse(f, p, xdata, ydata):
    """Root-mean-square error."""
    sqerr = (f(p, xdata) - ydata) ** 2
    return np.sqrt(sqerr.mean())

def biexp_flip(params):
    """If Df < Ds, flip them."""
    if params[1] < params[2]:
        params[1], params[2] = params[2], params[1]
        params[0] = 1.-params[0]


# Model functions.

def adcm(b, ADCm, C=1):
    """ADC mono: single exponential decay.

    C * exp(-b * ADCm)
    """
    return C * np.exp(-b * ADCm)

def adck(b, ADCk, K, C=1):
    """ADC kurtosis: K reflects deviation from Gaussian shape.

    C * exp(-b * ADCk + 1/6 * b^2 * ADCk^2 * K)
    """
    return C * np.exp(-b * ADCk + 1./6. * b**2 * ADCk**2 * K)

def adcs(b, ADCs, alpha, C=1):
    """ADC stretched.

    C * exp(-(b * ADCs)^alpha)
    """
    return C * np.exp(-(b * ADCs)**alpha)

def biexp(b, Af, Df, Ds, C=1):
    """Bi-exponential.

    C * ((1 - Af) * exp(-b * Ds) + Af * exp(-b * Df))
    """
    return C * ((1-Af) * np.exp(-b*Ds) + Af * np.exp(-b*Df))


# Model definitions.

# General C parameter used in non-normalized models.
ParamC = Parameter('C', (0.5, 1.25, 0.25), (0, 2), relative=True)

Models = []

Models.append(Model('Si',
        'Signal intensity values',
        None,
        []))
Models.append(Model('SiN',
        'Normalized signal intensity values',
        None,
        [],
        preproc=util.normalize_si_curve))

Models.append(Model('Mono',
        'ADC monoexponential',
        lambda p, x: adcm(x, *p),
        [
            Parameter('ADCm', (0.0001, 0.003, 0.00001), (0, 1)),
            ParamC
        ]))
Models.append(Model('MonoN',
        'Normalized ADC monoexponential',
        lambda p, x: adcm(x, *p),
        [
            Parameter('ADCmN', (0.0001, 0.003, 0.00001), (0, 1)),
        ],
        preproc=util.normalize_si_curve))

Models.append(Model('Kurt',
        'ADC kurtosis',
        lambda p, x: adck(x, *p),
        [
            Parameter('ADCk', (0.0001, 0.003, 0.00002), (0, 1)),
            Parameter('K', (0.0, 2.0, 0.1), (0, 10)),
            ParamC
        ]))
Models.append(Model('KurtN',
        'Normalized ADC kurtosis',
        lambda p, x: adck(x, *p),
        [
            Parameter('ADCkN', (0.0001, 0.003, 0.00002), (0, 1)),
            Parameter('KN', (0.0, 2.0, 0.1), (0, 10)),
        ],
        preproc=util.normalize_si_curve))

Models.append(Model('Stretched',
        'ADC stretched',
        lambda p, x: adcs(x, *p),
        [
            Parameter('ADCs', (0.0001, 0.003, 0.00002), (0, 1)),
            Parameter('Alpha', (0.1, 1.0, 0.05), (0, 1)),
            ParamC
        ]))
Models.append(Model('StretchedN',
        'Normalized ADC stretched',
        lambda p, x: adcs(x, *p),
        [
            Parameter('ADCsN', (0.0001, 0.003, 0.00002), (0, 1)),
            Parameter('AlphaN', (0.1, 1.0, 0.05), (0, 1)),
        ],
        preproc=util.normalize_si_curve))

Models.append(Model('Biexp',
        'Bi-exponential',
        lambda p, x: biexp(x, *p),
        [
            Parameter('Af', (0.2, 1.0, 0.1), (0, 1)),
            Parameter('Df', (0.001, 0.009, 0.0002), (0, 1)),
            Parameter('Ds', (0.000, 0.004, 0.00002), (0, 1)),
            ParamC
        ],
        postproc=biexp_flip))
Models.append(Model('BiexpN',
        'Normalized Bi-exponential',
        lambda p, x: biexp(x, *p),
        [
            Parameter('AfN', (0.2, 1.0, 0.1), (0, 1)),
            Parameter('DfN', (0.001, 0.009, 0.0002), (0, 1)),
            Parameter('DsN', (0.000, 0.004, 0.00002), (0, 1)),
        ],
        preproc=util.normalize_si_curve,
        postproc=biexp_flip))

"""
Im Matlab, we are using lsqnonlin and following initializations:
1. Mono-exponential model:
   ADCm from 0.1 um2/ms to 3.0 um2/ms with step size of 0.01 um2/ms.
2. Stretched exponential model:
   ADCs from 0.1 um2/ms to 3.0 um2/ms with step size of 0.02 um2/ms;
   alpha from 0.1 to 1.0 with step size of 0.05
3. Kurtosis model:
   ADCk from 0.1 um2/ms to 3.0 um2/ms with step size of 0.02 um2/ms;
   K from 0.0 to 2.0 with step size of 0.1.
4. Bi-exponential model:
   Df from 1.0 um2/ms to 9.0 um2/ms with step size of 0.2 um2/ms;
   Ds from 0.0 um2/ms to 4.0 um2/ms with step size of 0.02 um2/ms;
   f from 0.2 to 1.0 with step size of (0.1).

ADCm    0.0001 mm2/ms to 0.003 mm2/ms with step size of 0.00001 mm2/ms.

ADCk    0.0001 mm2/ms to 0.003 mm2/ms with step size of 0.00002 mm2/ms;
K       0.0 to 2.0 with step size of 0.1.

ADCs    0.0001 mm2/ms to 0.003 mm2/ms with step size of 0.00002 mm2/ms;
Alpha   0.1 to 1.0 with step size of 0.05

f       0.2 to 1.0 with step size of (0.1).
Df      0.001 mm2/ms to 0.009 mm2/ms with step size of 0.0002 mm2/ms;
Ds      0.000 mm2/ms to 0.004 mm2/ms with step size of 0.00002 mm2/ms;
"""
