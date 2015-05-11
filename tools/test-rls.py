#!/usr/bin/env python2

import sys
import numpy as np
import scipy as sp
from sklearn import preprocessing
from sklearn import metrics

import dwi.files
from dwi import patient
from dwi import dwimage
from dwi import util

sys.path.append('/home/jussi/src/RLScore')
from rlscore.learner import RLS
from rlscore.measure.cindex_measure import cindex
from rlscore.kernel.linear_kernel import LinearKernel
from rlscore.kernel.gaussian_kernel import GaussianKernel
from rlscore.kernel.polynomial_kernel import PolynomialKernel

def load_data(pmaps, labels, group_ids):
    """Load data indicated by command arguments."""
    assert len(pmaps) == len(labels) == len(group_ids)
    X = []
    Y = []
    G = []
    for pmap, label, group_id in zip(pmaps, labels, group_ids):
        for x in pmap:
            X.append(x)
            Y.append(label)
            G.append(group_id)
    X = np.array(X, dtype=float)
    Y = np.array(Y, dtype=float)
    return X, Y, G

def scale(a, min=0.0, max=1.0):
    """Scale data between given values."""
    # FIXME: min, max parameters don't work
    std = (a - a.min()) / (a.max() - a.min())
    return std / (max - min) + min

def normalize_sis(X):
    for x in X:
        util.normalize_si_curve(x)
    return X

def normalize_params(X):
    X = preprocessing.scale(X)
    #X = preprocessing.normalize(X)
    return X

def normalize_y(Y):
    #Y = preprocessing.scale(Y)
    #Y = preprocessing.scale(Y, with_mean=False)
    #Y = preprocessing.MinMaxScaler().fit_transform(Y)
    Y = scale(Y)
    return Y

def get_group_index_lists(group_ids):
    """Find out which group each sample belongs to. Return a list of lists that
    contain the indices of samples that belong to the same group."""
    groups = list(set(group_ids))
    r = []
    for group in groups:
        l = []
        for i in range(len(group_ids)):
            if group_ids[i] == group:
                l.append(i)
        r.append(l)
    return r

def compute_leave_group_out(rls, Indices):
    """Compute Hold-Out predictions by leaving out samples that belong to the
    same group. Their groups are indicated in the groups list parameter."""
    r = []
    for indices in Indices:
        p = rls.computeHO(indices)
        p = np.array(p)
        p = p.flatten() # Turn label matrix into 1D array.
        r.append(p)
    return r

def compute_performance(Y, P):
    """Compute performance over one group."""
    #perf = cindex(Y, P)
    perf = metrics.roc_auc_score(Y, P)
    #Y = np.array(np.round(Y), dtype=int)
    #P = np.array(np.round(P, 0), dtype=int)
    #perf = metrics.accuracy_score(Y, P)
    return perf

def compute_performance_all(trainY, Indices, Predictions):
    """Compute performance over all groups as one."""
    p_lgo = np.zeros_like(trainY)
    for indices, predictions in zip(Indices, Predictions):
        for i, p in zip(indices, predictions):
            p_lgo[i] = p
            #p_lgo[i] = np.mean(predictions)
    perf = compute_performance(trainY, p_lgo)
    return perf, []

def compute_performance_group(trainY, index_groups, p_groups):
    """Compute performance over each group separately."""
    y_groups = []
    for indices in index_groups:
        a = []
        for i in indices:
            a.append(trainY[i])
        a = np.array(a)
        y_groups.append(a)

    perf_groups = []
    for Y, P in zip(y_groups, p_groups):
        perf = compute_performance(Y, P)
        perf_groups.append(perf)

    return np.mean(perf_groups), perf_groups

def search_rp(rls, trainY, group_ids, rprange):
    """Search best regularization parameter in log space."""
    Indices = get_group_index_lists(group_ids)
    bestperf = -1.
    for logrp in range(*rprange):
        rp = 2. ** logrp
        rls.solve(rp)
        Predictions = compute_leave_group_out(rls, Indices)
        perf, perf_groups = compute_performance_all(trainY, Indices, Predictions)
        #perf, perf_groups = compute_performance_group(trainY, Indices, Predictions)
        if perf > bestperf:
            bestperf = perf
            bestperf_groups = perf_groups
            bestrp = rp
    return bestperf, bestperf_groups, bestrp

def search_params_linear(trainX, trainY, group_ids, rprange):
    """Search best parameters for kernel and regularization."""
    kwargs = {
        'train_features': trainX,
        'train_labels': trainY,
        'kernel_obj': LinearKernel(trainX),
    }
    rls = RLS.createLearner(**kwargs)
    rls.train()
    perf, perf_groups, rp = search_rp(rls, trainY, group_ids, rprange)
    return perf, perf_groups, rp

def search_params_nonlinear(trainX, trainY, group_ids, rprange, gammarange):
    """Search best parameters for kernel and regularization."""
    bestperf = -1.
    for loggamma in range(*gammarange):
        gamma = 2. ** loggamma
        kwargs = {
            'train_features': trainX,
            'train_labels': trainY,
            #'kernel_obj': LinearKernel(trainX),
            'kernel_obj': GaussianKernel(trainX, gamma=gamma),
            #'kernel_obj': PolynomialKernel(trainX, gamma=gamma, coef0=1, degree=2),
        }
        rls = RLS.createLearner(**kwargs)
        rls.train()
        perf, perf_groups, rp = search_rp(rls, trainY, group_ids, rprange)
        if perf > bestperf:
            bestperf = perf
            bestperf_groups = perf_groups
            bestrp = rp
            bestgamma = gamma
    return bestperf, bestperf_groups, bestrp, bestgamma
    

a, filenames = util.get_args(3)
use_roi2, labeltype, patients_file = a
patients = dwi.files.read_patients_file(patients_file)
pmaps, numsscans, params = patient.load_files(patients, filenames, pairs=True)
pmaps, numsscans = util.baseline_mean(pmaps, numsscans)

nums = [n for n, s in numsscans]
pmaps1, pmaps2 = util.split_roi(pmaps)

labels = patient.load_labels(patients, nums, labeltype)
labels_nocancer = [0] * len(labels)
X1, Y1, G1 = load_data(pmaps1, labels, numsscans)
X2, Y2, G2 = load_data(pmaps2, labels_nocancer, numsscans)

if use_roi2 == 't':
    X = np.concatenate((X1, X2))
    Y = np.concatenate((Y1, Y2))
    G = G1 + G2
else:
    X = X1
    Y = Y1
    G = G1

if X.shape[1] < 10:
    print 'Assuming fitted parameters.'
    #X = X[:,:-1] # Cut RMSE parameter.
    X = X[:,0:1]
    X = normalize_params(X)
else:
    print 'Assuming SI data.'
    #X = X[:,1:] # Cut noisy beginning.
    #X = X[:,6:] # Cut features.
    #X = X[:,::3] # Cut features.
    X = normalize_sis(X)
X = util.add_dummy_feature(X)
Y = normalize_y(Y)
print 'Samples: %i, features: %i, labels: %i, type: %s'\
        % (X.shape[0], X.shape[1], len(np.unique(Y)), labeltype)
print 'Labels: %s' % sorted(np.unique(Y))

rprange = (-10, 10)
gammarange = (0, 5)

perf, perf_groups, rp = search_params_linear(X, Y, G, rprange)
print 'Linear performance: %f, rp: %f' % (perf, rp)
if perf_groups:
    print util.fivenum(perf_groups)
    for numscan, label, perf in zip(numsscans, labels, perf_groups):
        num, scan = numscan
        d = dict(n=num, s=scan, l=label, p=perf, pg='x'*int(perf * 10))
        print '{n:02d}  {s:s}  {l:d}  {p:4.0%}  |{pg:10s}|'.format(**d)

perf, perf_groups, rp, gamma = search_params_nonlinear(X, Y, G, rprange, gammarange)
print 'Non-linear performance: %f, rp: %f, gamma: %f' % (perf, rp, gamma)
if perf_groups:
    print util.fivenum(perf_groups)
    for numscan, label, perf in zip(numsscans, labels, perf_groups):
        num, scan = numscan
        d = dict(n=num, s=scan, l=label, p=perf, pg='x'*int(perf * 10))
        print '{n:02d}  {s:s}  {l:d}  {p:4.0%}  |{pg:10s}|'.format(**d)
