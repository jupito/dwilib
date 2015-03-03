#!/usr/bin/env python2

"""Calculate reproducibility coefficients for parametric maps by Bland-Altman
analysis."""

import argparse
import numpy as np

from dwi import patient
from dwi import util

def parse_args():
    """Parse command-line arguments."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('-v', '--verbose', action='count',
            help='be more verbose')
    p.add_argument('-s', '--scans', default='patients.txt',
            help='patients file')
    p.add_argument('-b', '--nboot', type=int, default=2000,
            help='number of bootstraps')
    p.add_argument('-m', '--pmaps', nargs='+', required=True,
            help='pmap files')
    args = p.parse_args()
    return args

def mean_squared_difference(a1, a2):
    """Return mean squared difference of two arrays."""
    assert len(a1) == len(a2)
    n = len(a1)
    ds = a1-a2
    sds = ds**2
    msd = np.sqrt(sum(sds) / (n-1))
    return msd

def coefficients(a1, a2, avgfun=np.mean):
    """Return average, average squared difference, confidence interval,
    within-patient coefficient of variance, coefficient of repeatability."""
    n = len(a1)
    a = np.concatenate((a1, a2))
    avg = avgfun(a)
    avg_ci1, avg_ci2 = util.ci(a)
    msd = mean_squared_difference(a1, a2)
    ci = 1.96*msd / np.sqrt(n)
    wcv = (msd/np.sqrt(2)) / avg
    cor = 1.96*msd
    d = dict(avg=avg, avg_ci1=avg_ci1, avg_ci2=avg_ci2, msd=msd, ci=ci, wcv=wcv,
            cor=cor)
    return d

def icc(baselines):
    """Calculate Shrout & Fleiss ICC(3,1) intraclass correlation."""
    data = np.array(baselines)
    k, n = data.shape # Number of raters, targets.
    mpt = np.mean(data, axis=0) # Mean per target.
    mpr = np.mean(data, axis=1) # Mean per rater.
    tm = np.mean(data) # Total mean.
    wss = sum(sum((data-mpt)**2)) # Within-target sum of squares.
    wms = wss / (n * (k-1)) # Within-target mean of squares.
    rss = sum((mpr-tm)**2) * n # Between-rater sum of squares.
    rms = rss / (k-1) # Between-rater mean of squares.
    bss = sum((mpt-tm)**2) * k # Between-target sum of squares.
    bms = bss / (n-1) # Between-target mean of squares.
    ess = wss - rss # Residual sum of squares.
    ems = ess / ((k-1) * (n-1)) # Residual mean of squares.
    icc31 = (bms - ems) / (bms + (k-1)*ems)
    return icc31

def bootstrap_icc(baselines, nboot=2000):
    """Produce an array of ICC values bootstrapped target-wise."""
    data = np.array(baselines)
    values = np.zeros((nboot))
    for i in xrange(nboot):
        sample = util.resample_bootstrap_single(data.T).T
        values[i] = icc(sample)
    return values


args = parse_args()
patients = patient.read_patients_file(args.scans)
pmaps, numsscans, params = patient.load_files(patients, args.pmaps, pairs=True)

X = pmaps[:,0,:] # Use ROI1 only.

if args.verbose > 1:
    print 'Samples: %i, features: %i'\
            % (X.shape[0], X.shape[1])
    print 'Number of bootstraps: %d' % args.nboot

# Print coefficients for each parameter.
if args.verbose:
    print '# param\tavg[lower-upper]\tmsd/avg\tCI/avg\twCV\tCoR/avg\tICC\tbsICC[lower-upper]'
skipped_params = 'SI0N C RMSE'.split()
for values, param in zip(X.T, params):
    if param in skipped_params:
        continue
    baselines = util.pairs(values)
    d = dict(param=param)
    d.update(coefficients(*baselines, avgfun=np.median))
    d['msdr'] = d['msd']/d['avg']
    d['cir'] = d['ci']/d['avg']
    d['corr'] = d['cor']/d['avg']
    d['icc'] = icc(baselines)
    icc_values = bootstrap_icc(baselines, nboot=args.nboot)
    d['icc_bs'] = np.mean(icc_values)
    d['icc_ci1'], d['icc_ci2'] = util.ci(icc_values)
    s = '{param:7}'\
            '\t{avg:.8f}[{avg_ci1:.8f}-{avg_ci2:.8f}]'\
            '\t{msdr:.4f}\t{cir:.4f}\t{wcv:.4f}\t{corr:.4f}'\
            '\t{icc:.4f}\t{icc_bs:.4f}[{icc_ci1:.4f}-{icc_ci2:.4f}]'
    print s.format(**d)
