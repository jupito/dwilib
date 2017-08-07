"""Fitting implementation that fits curves simply one by one in serial.

This is an alternative implementation that uses a simple self-written
fixed-step gradient descent minimization. The aim is easy parallelization.

The simple fixed-step gradient descent minimizer seems to work sufficiently
well and fast, if you give it enough initial guesses and keep the number of
iterations small.

I didn't get the nonlinear conjugate gradient method working with our data. I
had better results with some simpler test functions, though, so it might work
with some tweaking.

NOTE: Bounds are not supported, and are thus ignored.
"""

import numpy as np

import dwi.minimize


def fit_curves_mi(f, xdata, ydatas, guesses, bounds, out_pmap, step=1.0e-7):
    """Fit curves to data with multiple initializations.

    Parameters
    ----------
    f : callable
        Cost function used for fitting in form of f(parameters, x).
    xdata : ndarray, shape = [n_bvalues]
        X data points, i.e. b-values
    ydatas : ndarray, shape = [n_curves, n_bvalues]
        Y data points, i.e. signal intensity curves
    guesses : callable
        A callable that returns an iterable of all combinations of parameter
        initializations, i.e. starting guesses, as tuples
    bounds : sequence of tuples (NOTE: not implemented)
        Constraints for parameters, i.e. minimum and maximum values
    out_pmap : ndarray, shape = [n_curves, n_parameters+1]
        Output array
    step : step size
        Task-specific step size used in minimization

    For each signal intensity curve, the resulting parameters with best fit
    will be placed in the output array, along with an RMSE value (root mean
    square error). In case of error, curve parameters will be set to NaN and
    RMSE to infinite.

    See files fit.py and models.py for more information on usage.
    """
    for i, ydata in enumerate(ydatas):
        d = fit_curve_mi(f, xdata, ydata, guesses(ydata[0]), bounds, step)
        out_pmap[i, -1] = d['y']
        if np.isfinite(d['y']):
            out_pmap[i, :-1] = d['x']
        else:
            out_pmap[i, :-1].fill(np.nan)


def fit_curve_mi(f, xdata, ydata, guesses, bounds, step=1.0e-7):
    """Fit a curve to data with multiple initializations.

    Try all given combinations of parameter initializations, and return the
    parameters and RMSE of best fit.
    """
    best = dict(y=np.inf)
    for guess in guesses:
        d = fit_curve(f, xdata, ydata, guess, step)
        if d['y'] < best['y']:
            best = d
    return best


def fit_curve(f, xdata, ydata, guess, bounds, step=1.0e-7):
    """Fit a curve to data."""
    def residual(p, x, y):
        return rmse(f, p, xdata, ydata)

    d = dwi.minimize.gradient_descent(residual, init=guess, step=step,
                                      args=[xdata, ydata])
    return d


def rmse(f, p, xdata, ydata):
    """Root-mean-square error."""
    results = np.asarray([f(p, x) for x in xdata])
    sqerr = (results - ydata)**2
    return np.sqrt(sqerr.mean())
