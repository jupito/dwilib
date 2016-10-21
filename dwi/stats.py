"""Statistical functionality."""

from __future__ import absolute_import, division, print_function

import numpy as np

import dwi.util


def mean_squared_difference(a1, a2):
    """Return mean squared difference of two arrays."""
    a1 = np.asanyarray(a1)
    a2 = np.asanyarray(a2)
    assert len(a1) == len(a2), 'Array length mismatch'
    n = len(a1)
    ds = a1-a2
    sds = ds**2
    msd = np.sqrt(sum(sds) / (n-1))
    return msd


def repeatability_coeff(a1, a2, avgfun=np.mean):
    """Calculate reproducibility coefficients for two arrays by Bland-Altman
    analysis. Return average, average squared difference, confidence interval,
    within-patient coefficient of variance, coefficient of repeatability."""
    a1 = np.asanyarray(a1)
    a2 = np.asanyarray(a2)
    assert len(a1) == len(a2), 'Array length mismatch'
    n = len(a1)
    a = np.concatenate((a1, a2))
    avg = avgfun(a)
    avg_ci1, avg_ci2 = dwi.util.ci(a)
    msd = mean_squared_difference(a1, a2)
    ci = 1.96*msd / np.sqrt(n)
    wcv = (msd/np.sqrt(2)) / avg
    cor = 1.96*msd
    d = dict(avg=avg, avg_ci1=avg_ci1, avg_ci2=avg_ci2, msd=msd, ci=ci,
             wcv=wcv, cor=cor)
    return d


def icc(baselines):
    """Calculate ICC(3,1) intraclass correlation.

    See Shrout, Fleiss 1979: Intraclass Correlations: Uses in Assessing Rater
    Reliability.
    """
    data = np.array(baselines)
    k, n = data.shape  # Number of raters, targets.
    mpt = np.mean(data, axis=0)  # Mean per target.
    mpr = np.mean(data, axis=1)  # Mean per rater.
    tm = np.mean(data)  # Total mean.
    wss = sum(sum((data-mpt)**2))  # Within-target sum of squares.
    # wms = wss / (n * (k-1))  # Within-target mean of squares.
    rss = sum((mpr-tm)**2) * n  # Between-rater sum of squares.
    # rms = rss / (k-1)  # Between-rater mean of squares.
    bss = sum((mpt-tm)**2) * k  # Between-target sum of squares.
    bms = bss / (n-1)  # Between-target mean of squares.
    ess = wss - rss  # Residual sum of squares.
    ems = ess / ((k-1) * (n-1))  # Residual mean of squares.
    icc31 = (bms - ems) / (bms + (k-1)*ems)
    return icc31


def bootstrap_icc(baselines, nboot=2000):
    """Calculate ICC bootstrapped target-wise. Return mean and confidence
    intervals.
    """
    data = np.array(baselines)
    values = np.zeros((nboot,))
    for i in xrange(nboot):
        sample = dwi.util.resample_bootstrap_single(data.T).T
        values[i] = icc(sample)
    mean = np.mean(values)
    ci1, ci2 = dwi.util.ci(values)
    return mean, ci1, ci2
