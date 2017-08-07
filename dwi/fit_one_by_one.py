"""Fitting implementation that fits curves simply one by one in serial."""

import numpy as np

from leastsqbound import leastsqbound


def fit_curves_mi(f, xdata, ydatas, guesses, bounds, out_pmap):
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
    bounds : sequence of tuples
        Constraints for parameters, i.e. minimum and maximum values
    out_pmap : ndarray, shape = [n_curves, n_parameters+1]
        Output array

    For each signal intensity curve, the resulting parameters with best fit
    will be placed in the output array, along with an RMSE value (root mean
    square error). In case of error, curve parameters will be set to NaN and
    RMSE to infinite.

    See files fit.py and models.py for more information on usage.
    """
    for i, ydata in enumerate(ydatas):
        params, err = fit_curve_mi(f, xdata, ydata, guesses(ydata[0]), bounds)
        out_pmap[i, -1] = err
        if np.isfinite(err):
            out_pmap[i, :-1] = params
        else:
            out_pmap[i, :-1].fill(np.nan)


def fit_curve_mi(f, xdata, ydata, guesses, bounds):
    """Fit a curve to data with multiple initializations.

    Try all given combinations of parameter initializations, and return the
    parameters and RMSE of best fit.
    """
    if np.any(np.isnan(ydata)):
        return None, np.nan
    best_params = []
    best_err = np.inf
    for guess in guesses:
        params, err = fit_curve(f, xdata, ydata, guess, bounds)
        if err < best_err:
            best_params = params
            best_err = err
    return best_params, best_err


def fit_curve(f, xdata, ydata, guess, bounds):
    """Fit a curve to data."""
    def residual(p, x, y):
        return f(p, x) - y

    params, ier = leastsqbound(residual, guess, args=(xdata, ydata),
                               bounds=bounds)
    if 0 < ier < 5:
        err = rmse(f, params, xdata, ydata)
    else:
        err = np.inf
    return params, err


def rmse(f, p, xdata, ydata):
    """Root-mean-square error."""
    sqerr = (f(p, xdata) - ydata) ** 2
    return np.sqrt(sqerr.mean())
