"""Minimization functions."""

import numpy as np
import scipy.optimize

EPSILON = np.sqrt(np.finfo(np.float64).eps)
try:
    irange = xrange
except NameError:
    irange = range


def gradient(f, x, args=[]):
    """Approximate gradient of f at x."""
    return scipy.optimize.approx_fprime(x, f, EPSILON, *args)


def gradient_descent(f, init=[0.0], step=0.5, args=[], maxiter=100):
    """Minimize f by gradient descent.

    The gradient is numerically approximated at each step.
    """
    assert 0 < step < 1
    assert maxiter > 0
    init = np.atleast_1d(np.asarray(init, dtype=np.float64))
    x = init
    i = -1
    for i in irange(maxiter):
        dfx = gradient(f, x, args)
        # x_prev = x
        x = x - dfx*step
    d = dict(x=x, y=f(x, *args), grad=dfx, nit=i+1, init=init, step=step,
             args=args, maxiter=maxiter)
    return d


def gradient_descent_mi(f, inits, **kwargs):
    """Gradient descent with multiple initializations."""
    best = dict(y=np.inf)
    for init in inits:
        d = gradient_descent(f, init=init, **kwargs)
        if d['y'] < best['y']:
            best = d
    return best


def line_search(f, x, args, rho=0.4, c=0.4, alpha0=0.4):
    """Backtracking line search. Nodecal & Wright 1999 pg41."""
    alpha = alpha0
    dfx = gradient(f, x, args)
    p = -dfx/np.linalg.norm(dfx)
    while f(x+alpha*p, *args) > f(x, *args) + c*alpha*np.dot(dfx, p):
        alpha = rho*alpha
    return alpha


def fletcher_reeves(x, x_):
    return np.dot(x, x) / np.dot(x_, x_)


def polak_ribiere(x, x_):
    return np.dot(x, (x-x_)) / np.dot(x_, x_)


def cg(f, x0, args=[], maxiter=10000):
    """Nonlinear conjugate gradient method. Nocedel & Wright 1999 pg120."""
    x0 = np.atleast_1d(np.asarray(x0, dtype=np.float64))
    x = x0
    dfx = gradient(f, x, args)
    p = -dfx
    k = 0
    while dfx.any() and k < maxiter:
        dfx_prev = dfx
        alpha = line_search(f, x, args)
        x = x + alpha*p
        dfx = gradient(f, x, args)
        beta = fletcher_reeves(dfx, dfx_prev)
        # beta = max(0, polak_ribiere(dfx, dfx_prev))
        p = -dfx + beta*p
        k = k+1
    d = dict(x=x, y=f(x, *args), nit=k)
    return d


def cg_old(f, x0, args=[], maxiter=1000):
    x0 = np.atleast_1d(np.asarray(x0, dtype=np.float64))
    x = x0
    dfx = gradient(f, x, args)
    dx = -dfx
    s = dx
    alpha = line_search(f, x, args)
    x = x + alpha*dx
    i = -1
    for i in irange(maxiter):
        dfx = gradient(f, x, args)
        dx_prev = dx
        dx = -dfx
        beta = max(0, polak_ribiere(dx, dx_prev))
        s_prev = s
        s = dx + beta*s_prev
        alpha = line_search(f, x, args)
        x = x + alpha*s
    d = dict(x=x, y=f(x, *args), nit=i+1)
    return d
