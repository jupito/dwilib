"""Fitting implementation that fits curves simply one by one in serial."""

import numpy as np
from leastsqbound import leastsqbound

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
