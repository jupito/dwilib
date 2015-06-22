#!/usr/bin/env python2

"""Calculate reproducibility coefficients for parametric maps by Bland-Altman
analysis."""

import argparse
import numpy as np

import dwi.files
import dwi.patient
import dwi.util

def parse_args():
    """Parse command-line arguments."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('-v', '--verbose', action='count',
            help='be more verbose')
    p.add_argument('-p', '--patients', default='patients.txt',
            help='patients file')
    p.add_argument('-b', '--nboot', type=int, default=2000,
            help='number of bootstraps')
    p.add_argument('--voxel', default='0',
            help='index of voxel to use, or mean or median')
    p.add_argument('-m', '--pmaps', nargs='+', required=True,
            help='pmap files')
    args = p.parse_args()
    return args

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

def coefficients(a1, a2, avgfun=np.mean):
    """Return average, average squared difference, confidence interval,
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
    d = dict(avg=avg, avg_ci1=avg_ci1, avg_ci2=avg_ci2, msd=msd, ci=ci, wcv=wcv,
            cor=cor)
    return d

def icc(baselines):
    """Calculate ICC(3,1) intraclass correlation.
    
    See Shrout, Fleiss 1979: Intraclass Correlations: Uses in Assessing Rater
    Reliability.
    """
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
    """Calculate ICC bootstrapped target-wise. Return mean and confidence
    intervals."""
    data = np.array(baselines)
    values = np.zeros((nboot,))
    for i in xrange(nboot):
        sample = dwi.util.resample_bootstrap_single(data.T).T
        values[i] = icc(sample)
    mean = np.mean(values)
    ci1, ci2 = dwi.util.ci(values)
    return mean, ci1, ci2


args = parse_args()
patients = dwi.files.read_patients_file(args.patients)
pmaps, numsscans, params = dwi.patient.load_files(patients, args.pmaps,
        pairs=True)

# Select voxel to use.
if args.voxel == 'mean':
    X = np.mean(pmaps, axis=1) # Use mean voxel.
elif args.voxel == 'median':
    X = np.median(pmaps, axis=1) # Use median voxel.
else:
    i = int(args.voxel)
    X = pmaps[:,i,:] # Use single voxel only.

if args.verbose > 1:
    print 'Samples: %i, features: %i, voxel: %s'\
            % (X.shape[0], X.shape[1], args.voxel)
    print 'Number of bootstraps: %d' % args.nboot

# Print coefficients for each parameter.
if args.verbose:
    print '# param    avg[lower-upper]'\
            '    msd/avg    CI/avg    wCV    CoR/avg'\
            '    ICC    bsICC[lower-upper]'
skipped_params = 'SI0N C RMSE'.split()
for values, param in zip(X.T, params):
    if param in skipped_params:
        continue
    baselines = dwi.util.pairs(values)
    d = dict(param=param)
    d.update(coefficients(*baselines, avgfun=np.median))
    d['msdr'] = d['msd']/d['avg']
    d['cir'] = d['ci']/d['avg']
    d['corr'] = d['cor']/d['avg']
    d['icc'] = icc(baselines)
    d['icc_bs'], d['icc_ci1'], d['icc_ci2'] = bootstrap_icc(baselines,
            nboot=args.nboot)
    s = '{param:7}'\
            '    {avg:.8f}[{avg_ci1:.8f}-{avg_ci2:.8f}]'\
            '    {msdr:.4f}    {cir:.4f}    {wcv:.4f}    {corr:.4f}'\
            '    {icc:.4f}    {icc_bs:.4f}[{icc_ci1:.4f}-{icc_ci2:.4f}]'
    print s.format(**d)
