# coding=utf-8

# Fitting.

import numpy as np
import scipy as sp
import scipy.optimize
from leastsqbound import leastsqbound

import util

class Parameter(object):
    def __init__(self, name, steps, bounds,
            use_stepsize=False, relative=False):
        self.name = name
        self.steps = steps
        self.bounds = bounds
        self.use_stepsize = use_stepsize
        self.relative = relative # Steps are considered relative to SI(0).

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
        '''Return initial guesses.'''
        if self.use_stepsize:
            g = np.arange(*self.steps)
        else:
            g = np.linspace(*self.steps)
        if self.relative:
            g = g * si0
        return g

class Model(object):
    def __init__(self, name, description, func, params,
            preproc=None, postproc=None):
        self.name = name
        self.description = description
        self.func = func
        self.params = params
        self.preproc = preproc # Preprocessing for SI curve.
        self.postproc = postproc # Postprocessing for fitted parameters.

    def __str__(self):
        return self.name

    def __repr__(self):
        return '%s %s' % (self.name, ' '.join(map(repr, self.params)))

    def bounds(self, si0):
        '''Return bounds of all parameters.'''
        #return map(lambda p: p.bounds, self.params)
        return [p.bounds_rel(si0) for p in self.params]

    def guesses(self, si0):
        '''Return all combinations of initial guesses.'''
        return util.combinations(map(lambda p: p.guesses(si0), self.params))

    def fit_mi(self):
        pass # TODO

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

def fit_model_mi(model, xdata, ydata):
    """Fit a model to data with multiple initializations."""
    if model.preproc:
        model.preproc(ydata)
    if model.func:
        si0 = ydata[0]
        guesses = model.guesses(si0)
        bounds = model.bounds(si0)
        params, err = fit_curve_mi(model.func, xdata, ydata, guesses, bounds)
    else:
        params, err = ydata, 0.
    if model.postproc:
        model.postproc(params)
    return params, err

def rmse(f, p, xdata, ydata):
    """Root-mean-square error."""
    sqerr = (f(p, xdata) - ydata) ** 2
    return np.sqrt(sqerr.mean())

def biexp(p, x):
    Af, Df, Ds, C = p
    return C * ((1-Af) * np.exp(-x*Ds) + Af * np.exp(-x*Df))

def biexp_normalized(p, x):
    Af, Df, Ds = p
    return (1-Af) * np.exp(-x*Ds) + Af * np.exp(-x*Df)

def biexp_flip(params):
    """If Df < Ds, flip them."""
    if params[1] < params[2]:
        params[1], params[2] = params[2], params[1]
        params[0] = 1.-params[0]

# General C parameter used in non-normalized models.
ParamC = Parameter('C', (0.5, 1.25, 0.25), (0, 2), True, relative=True)

Models = []
Models.append(Model('SiN',
        '''Normalized SI values.''',
        None,
        [],
        preproc=util.normalize_si_curve))

Models.append(Model('Mono',
        '''ADC mono: single exponential decay.
        C * exp(-b * ADCm)''',
        lambda p, x:\
            p[1] * np.exp(-x * p[0]),
        [
            Parameter('ADCm', (0.0001, 0.003, 0.00001), (0, 1), True),
            ParamC
        ]))
Models.append(Model('MonoN',
        '''Normalized ADC mono: single exponential decay.
        exp(-b * ADCm)''',
        lambda p, x:\
            np.exp(-x * p[0]),
        [
            Parameter('ADCmN', (0.0001, 0.003, 0.00001), (0, 1), True),
        ],
        preproc=util.normalize_si_curve))

Models.append(Model('Kurt',
        '''ADC kurtosis: K reflects deviation from Gaussian shape.
        C * exp(-b * ADCk + 1/6 * b^2 * ADCk^2 * K)''',
        lambda p, x:\
            p[2] * np.exp(-x * p[0] + 1.0/6.0 * x**2 * p[0]**2 * p[1]),
        [
            Parameter('ADCk', (0.0001, 0.003, 0.00002), (0, 1), True),
            Parameter('K', (0.0, 2.0, 0.1), (0, 10), True),
            ParamC
        ]))
Models.append(Model('KurtN',
        '''Normalized ADC kurtosis: K reflects deviation from Gaussian shape.
        exp(-b * ADCk + 1/6 * b^2 * ADCk^2 * K)''',
        lambda p, x:\
            np.exp(-x * p[0] + 1.0/6.0 * x**2 * p[0]**2 * p[1]),
        [
            Parameter('ADCkN', (0.0001, 0.003, 0.00002), (0, 1), True),
            Parameter('KN', (0.0, 2.0, 0.1), (0, 10), True),
        ],
        preproc=util.normalize_si_curve))

Models.append(Model('Stretched',
        '''ADC stretched.
        C * exp(-(b * ADCs)^alpha)''',
        lambda p, x:\
            p[2] * np.exp(-(x * p[0])**p[1]),
        [
            Parameter('ADCs', (0.0001, 0.003, 0.00002), (0, 1), True),
            Parameter('Alpha', (0.1, 1.0, 0.05), (0, 1), True),
            ParamC
        ]))
Models.append(Model('StretchedN',
        '''Normalized ADC stretched.
        exp(-(b * ADCs)^alpha)''',
        lambda p, x:\
            np.exp(-(x * p[0])**p[1]),
        [
            Parameter('ADCsN', (0.0001, 0.003, 0.00002), (0, 1), True),
            Parameter('AlphaN', (0.1, 1.0, 0.05), (0, 1), True),
        ],
        preproc=util.normalize_si_curve))

Models.append(Model('Biexp',
        '''Bi-exponential.
        C * ((1 - Af) * exp(-b * Ds) + Af * exp(-b * Df))''',
        biexp,
        [
            Parameter('Af', (0.2, 1.0, 0.1), (0, 1), True),
            Parameter('Df', (0.001, 0.009, 0.0002), (0, 1), True),
            Parameter('Ds', (0.000, 0.004, 0.00002), (0, 1), True),
            ParamC
        ],
        postproc=biexp_flip))

Models.append(Model('BiexpN',
        '''Normalized Bi-exponential.
        (1 - Af) * exp(-b * Ds) + Af * exp(-b * Df)''',
        biexp_normalized,
        [
            Parameter('AfN', (0.2, 1.0, 0.1), (0, 1), True),
            Parameter('DfN', (0.001, 0.009, 0.0002), (0, 1), True),
            Parameter('DsN', (0.000, 0.004, 0.00002), (0, 1), True),
        ],
        preproc=util.normalize_si_curve,
        postproc=biexp_flip))

"""
Im Matlab, we are using lsqnonlin and following initializations:
1. Mono-exponential model:
   ADCm from 0.1 µm2/ms to 3.0 µm2/ms with step size of 0.01 µm2/ms.
2. Stretched exponential model:
   ADCs from 0.1 µm2/ms to 3.0 µm2/ms with step size of 0.02 µm2/ms;
   α from 0.1 to 1.0 with step size of 0.05
3. Kurtosis model:
   ADCk from 0.1 µm2/ms to 3.0 µm2/ms with step size of 0.02 µm2/ms;
   K from 0.0 to 2.0 with step size of 0.1.
4. Bi-exponential model:
   Df from 1.0 µm2/ms to 9.0 µm2/ms with step size of 0.2 µm2/ms;
   Ds from 0.0 µm2/ms to 4.0 µm2/ms with step size of 0.02 µm2/ms;
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
