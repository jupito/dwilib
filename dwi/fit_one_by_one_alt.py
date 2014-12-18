"""Fitting implementation that fits curves simply one by one in serial.

This is an alternative implementation that uses a simpler self-written
fixed-step gradient descent minimization.
"""

import numpy as np

import dwi.minimize

def fit_curves_mi(f, xdata, ydatas, guesses, step, out_pmap, trylimit=1.0e-10):
    """Fit curves to data with multiple initializations.

    Parameters
    ----------
    f : callable
        Cost function used for fitting in form of f(parameters, x).
    xdata : ndarray, shape = [n_bvalues]
        X data points, i.e. b-values
    ydatas : ndarray, shape = [n_curves, n_bvalues]
        Y data points, i.e. signal intensity curves
    guesses : sequence of tuples
        All combinations of parameter initializations, i.e. starting guesses
    step : step size
        Task-specific step size used in minimization
    out_pmap : ndarray, shape = [n_curves, n_parameters+1]
        Output array
    trylimit : minimization parameter try limit
        Task-specific limit for parameter difference testing; decrease to get
        more accurate results

    For each signal intensity curve, the resulting parameters with best fit will
    be placed in the output array, along with an RMSE value (root mean square
    error). In case of error, curve parameters will be set to NaN and RMSE to
    infinite.

    See files fit.py and models.py for more information on usage.
    """
    for i, ydata in enumerate(ydatas):
        d = fit_curve_mi(f, xdata, ydata, guesses, step, trylimit)
        out_pmap[i, -1] = d['y']
        if np.isfinite(d['y']):
            out_pmap[i, :-1] = d['x']
        else:
            out_pmap[i, :-1].fill(np.nan)

def fit_curve_mi(f, xdata, ydata, guesses, step, trylimit=1.0e-10):
    """Fit a curve to data with multiple initializations.

    Try all given combinations of parameter initializations, and return the
    parameters and RMSE of best fit.
    """
    guess = guesses[0]
    best = dict(y=np.inf)
    for guess in guesses:
        d = fit_curve(f, xdata, ydata, guess, step, trylimit)
        if d['y'] < best['y']:
            best = d
    return best

def fit_curve(f, xdata, ydata, guess, step, trylimit=1.0e-10):
    """Fit a curve to data."""
    residual = lambda p, x, y: rmse(f, p, xdata, ydata)
    d = dwi.minimize.gradient_descent(residual, init=guess, step=step,
            args=[xdata, ydata], trylimit=trylimit)
    return d

def rmse(f, p, xdata, ydata):
    """Root-mean-square error."""
    results = np.asarray([f(p, x) for x in xdata])
    sqerr = (results - ydata)**2
    return np.sqrt(sqerr.mean())

