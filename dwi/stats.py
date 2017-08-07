"""Statistical functionality."""

from collections import defaultdict
import random

import numpy as np
from scipy import stats
import sklearn.metrics
import sklearn.preprocessing

import dwi.util


def rmse(a, b):
    """Root mean square error."""
    a = np.asarray(a)
    b = np.asarray(b)
    return np.sqrt(np.mean((a - b)**2))


def stem_and_leaf(values):
    """A quick and dirty text mode stem-and-leaf diagram for non-negative real
    values. Uses integer part as stem and first decimal as leaf.
    """
    stems = defaultdict(list)
    for v in sorted(values):
        stem = int(v)
        leaf = int((v-stem) * 10)
        stems[stem].append(leaf)
    lines = []
    for i in range(min(stems), max(stems)+1):
        leaves = ''.join(str(x) for x in stems[i])
        lines.append('{i:2}|{l}'.format(i=i, l=leaves))
    return lines


def resample_bootstrap_single(a):
    """Get a bootstrap resampled group for single array."""
    indices = [random.randint(0, len(a)-1) for _ in a]
    return a[indices]


def resample_bootstrap(Y, X):
    """Get a bootstrap resampled group without stratification."""
    indices = [random.randint(0, len(Y)-1) for _ in Y]
    return Y[indices], X[indices]


def get_indices(seq, val):
    """Return indices of elements containing given value in a sequence."""
    r = []
    for i, v in enumerate(seq):
        if v == val:
            r.append(i)
    return r


def resample_bootstrap_stratified(Y, X):
    """Get a bootstrap resampled group with stratification.

    Note that as a side-effect the resulting Y array will be sorted, but that
    doesn't matter because X will be randomized accordingly.
    """
    # TODO: Should be rewritten to make it faster.
    uniques = np.unique(Y)
    indices = []
    for u in uniques:
        l = get_indices(Y, u)
        l_rnd = [l[random.randint(0, len(l)-1)] for _ in l]
        for v in l_rnd:
            indices.append(v)
    return Y[indices], X[indices]


def posneg_to_labelsvalues(pos, neg):
    """From two data sequences, positives and negatives, create to ndarrays,
    labels and values, where labels contains True/False labels, and values
    contains all corresponding values.
    """
    values = np.concatenate([pos, neg])
    labels = np.zeros_like(values, dtype=np.bool)
    labels[:len(pos)] = True
    return labels, values


def scale_standard(x):
    """Scale to standard distribution with zero mean, unit variance."""
    return sklearn.preprocessing.scale(x)


def scale_minmax(x):
    """Scale to range."""
    return sklearn.preprocessing.minmax_scale(x)


def calculate_roc_auc(y, x, autoflip=False, scale=True):
    """Calculate ROC and AUC from data points and their classifications.

    By default, the samples are scaled, because sklearn.metrics.roc_curve()
    interprets very close samples as equal.
    """
    y = np.asarray(y)
    x = np.asarray(x)
    if scale:
        x = scale_standard(x)
    fpr, tpr, _ = sklearn.metrics.roc_curve(y, x)
    auc = sklearn.metrics.auc(fpr, tpr)
    if autoflip and auc < 0.5:
        fpr, tpr, auc = calculate_roc_auc(y, -x, autoflip=False, scale=False)
    return fpr, tpr, auc


def bootstrap_aucs(y, x, n=2000):
    """Produce an array of bootstrapped ROC AUCs."""
    aucs = np.zeros(n)
    for i in range(n):
        yb, xb = resample_bootstrap_stratified(y, x)
        _, _, auc = calculate_roc_auc(yb, xb, autoflip=False, scale=False)
        aucs[i] = auc
    return aucs


def roc_auc(labels, values, autoflip=False, nboot=None):
    """Calculate ROC AUC with optional bootstrapping. If autoflip is True,
    results under 0.5 are 'switched' automatically. Returns a dictionary with
    keys: auc: AUC, flipped: whether flipped, ci1, ci2: confidence interval.
    """
    _, _, auc = dwi.stats.calculate_roc_auc(labels, values, autoflip=False,
                                            scale=False)
    if autoflip and auc < 0.5:
        # Must flip here for the bootstrap to work.
        return dict(roc_auc(labels, -values, autoflip=False, nboot=nboot),
                    flipped=True)
    d = dict(auc=auc, flipped=False)
    if nboot:
        # Note: values may now be negated (ROC flipped).
        d['aucs'] = dwi.stats.bootstrap_aucs(labels, values, nboot)
        d['ci1'], d['ci2'] = dwi.stats.conf_int(d['aucs'])
    return d


def compare_aucs(aucs1, aucs2):
    """Compare two arrays of (bootstrapped) ROC AUC values, with the method
    described in pROC software.
    """
    aucs1 = np.asarray(aucs1)
    aucs2 = np.asarray(aucs2)
    D = aucs1 - aucs2
    z = np.mean(D) / np.std(D)
    p = 1.0 - stats.norm.cdf(abs(z))
    return np.mean(D), z, p


def conf_int(x, p=0.05):
    """Confidence interval of a normally distributed array."""
    x = sorted(x)
    l = len(x)
    i1 = int(round((p/2) * l + 0.5))
    i2 = int(round((1-p/2) * l - 0.5))
    ci1 = x[i1]
    ci2 = x[i2]
    return ci1, ci2


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
    avg_ci1, avg_ci2 = conf_int(a)
    msd = mean_squared_difference(a1, a2)
    ci = 1.96*msd / np.sqrt(n)
    wcv = (msd/np.sqrt(2)) / avg
    cor = 1.96*msd
    d = dict(avg=avg, avg_ci1=avg_ci1, avg_ci2=avg_ci2, msd=msd, ci=ci,
             wcv=wcv, cor=cor)
    return d


def icc(baselines):
    """Calculate ICC(3,1) intraclass correlation.

    Parameter baselines is an array of size (k, n) where k is the number of
    raters (repetitions) and n is the number of targets (samples).

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
    for i in range(nboot):
        sample = resample_bootstrap_single(data.T).T
        values[i] = icc(sample)
    mean = np.mean(values)
    ci1, ci2 = conf_int(values)
    return mean, ci1, ci2


def walsh_averages(seq):
    """Calculate Walsh averages."""
    w = np.asmatrix(seq)
    assert len(w) == 1, w.shape
    w = (w + w.T) / 2
    w = w[np.tri(len(w), dtype=np.bool)]  # Take lower half.
    w = np.sort(w.flat)
    return w


def wilcoxon_signed_rank_test(x, conflevel=0.95):
    """Wilcoxon signed rank test, with confidence interval. Requires rpy2."""
    from rpy2.robjects import numpy2ri
    from rpy2.robjects.packages import importr
    numpy2ri.activate()
    stats = importr('stats')
    d = {'conf.int': True, 'conf.level': conflevel}
    r = stats.wilcox_test(np.array(x), **d)
    r = dict(
        statistic=np.asscalar(np.asarray(r.rx('statistic'))),
        pvalue=np.asscalar(np.asarray(r.rx('p.value'))),
        confint=[np.asscalar(x) for x in np.asarray(r.rx('conf.int')).flat],
        estimate=np.asscalar(np.asarray(r.rx('estimate'))),
        )
    return r
