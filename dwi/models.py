"""Parametric model definitions used for signal decay curve fitting."""

import numpy as np

from dwi.fit import Parameter, Model
import dwi.util

"""
Im Matlab, we are using lsqnonlin and following initializations:
1. Mono-exponential model:
   ADCm from 0.1 um2/ms to 3.0 um2/ms with step size of 0.01 um2/ms.
2. Stretched exponential model:
   ADCs from 0.1 um2/ms to 3.0 um2/ms with step size of 0.02 um2/ms;
   alpha from 0.1 to 1.0 with step size of 0.05
3. Kurtosis model:
   ADCk from 0.1 um2/ms to 3.0 um2/ms with step size of 0.02 um2/ms;
   K from 0.0 to 2.0 with step size of 0.1.
4. Bi-exponential model:
   Df from 1.0 um2/ms to 9.0 um2/ms with step size of 0.2 um2/ms;
   Ds from 0.0 um2/ms to 4.0 um2/ms with step size of 0.02 um2/ms;
   f from 0.2 to 1.0 with step size of (0.1).

ADCm    0.0001 mm2/ms to 0.003 mm2/ms with step size of 0.00001 mm2/ms.

ADCk    0.0001 mm2/ms to 0.003 mm2/ms with step size of 0.00002 mm2/ms;
K       0.0 to 2.0 with step size of 0.1.

ADCs    0.0001 mm2/ms to 0.003 mm2/ms with step size of 0.00002 mm2/ms;
Alpha   0.1 to 1.0 with step size of 0.05

f       0.2 to 1.0 with step size of (0.1).
Df      0.001 mm2/ms to 0.009 mm2/ms with step size of 0.0002 mm2/ms;
Ds      0.000 mm2/ms to 0.004 mm2/ms with step size of 0.00002 mm2/ms;
"""


def biexp_flip(params):
    """If Df < Ds, flip them."""
    if params[1] < params[2]:
        params[1], params[2] = params[2], params[1]
        params[0] = 1.-params[0]


# Model functions.

def adcm(b, ADCm, C=1):
    """ADC mono: single exponential decay.

    C * exp(-b * ADCm)
    """
    return C * np.exp(-b * ADCm)


def adck(b, ADCk, K, C=1):
    """ADC kurtosis: K reflects deviation from Gaussian shape.

    C * exp(-b * ADCk + 1/6 * b^2 * ADCk^2 * K)
    """
    return C * np.exp(-b * ADCk + 1/6 * b**2 * ADCk**2 * K)


def adcs(b, ADCs, alpha, C=1):
    """ADC stretched.

    C * exp(-(b * ADCs)^alpha)
    """
    return C * np.exp(-(b * ADCs)**alpha)


def biexp(b, Af, Df, Ds, C=1):
    """Bi-exponential.

    C * ((1 - Af) * exp(-b * Ds) + Af * exp(-b * Df))
    """
    return C * ((1-Af) * np.exp(-b*Ds) + Af * np.exp(-b*Df))


def t2(t, T2, C=1):
    """T2, spin-spin relaxation time.

    C * exp(-t / T2)
    """
    return C * np.exp(-t / T2)


# Model definitions.

# General C parameter used in non-normalized models.
# ParamC = Parameter('C', (0.5, 1.25, 0.25), (0, 2), relative=True)
ParamC = Parameter('C', (500, 1250, 250), (0, 1e9))

Models = []

Models.append(Model(
    'Si',
    'Signal intensity values',
    None,
    []))
Models.append(Model(
    'SiN',
    'Normalized signal intensity values',
    None,
    [],
    preproc=dwi.util.normalize_si_curve))

Models.append(Model(
    'Mono',
    'ADC monoexponential',
    lambda p, x: adcm(x, *p),
    [
        Parameter('ADCm', (0.0001, 0.003, 0.00001), (0, 1)),
        ParamC
    ]))
Models.append(Model(
    'MonoN',
    'Normalized ADC monoexponential',
    lambda p, x: adcm(x, *p),
    [
        Parameter('ADCmN', (0.0001, 0.003, 0.00001), (0, 1)),
    ],
    preproc=dwi.util.normalize_si_curve))

Models.append(Model(
    'Kurt',
    'ADC kurtosis',
    lambda p, x: adck(x, *p),
    [
        Parameter('ADCk', (0.0001, 0.003, 0.00002), (0, 1)),
        Parameter('K', (0.0, 2.0, 0.1), (0, 10)),
        ParamC
    ]))
Models.append(Model(
    'KurtN',
    'Normalized ADC kurtosis',
    lambda p, x: adck(x, *p),
    [
        Parameter('ADCkN', (0.0001, 0.003, 0.00002), (0, 1)),
        Parameter('KN', (0.0, 2.0, 0.1), (0, 10)),
    ],
    preproc=dwi.util.normalize_si_curve))

Models.append(Model(
    'Stretched',
    'ADC stretched',
    lambda p, x: adcs(x, *p),
    [
        Parameter('ADCs', (0.0001, 0.003, 0.00002), (0, 1)),
        Parameter('Alpha', (0.1, 1.0, 0.05), (0, 1)),
        ParamC
    ]))
Models.append(Model(
    'StretchedN',
    'Normalized ADC stretched',
    lambda p, x: adcs(x, *p),
    [
        Parameter('ADCsN', (0.0001, 0.003, 0.00002), (0, 1)),
        Parameter('AlphaN', (0.1, 1.0, 0.05), (0, 1)),
    ],
    preproc=dwi.util.normalize_si_curve))

Models.append(Model(
    'Biexp',
    'Bi-exponential',
    lambda p, x: biexp(x, *p),
    [
        Parameter('Af', (0.2, 1.0, 0.1), (0, 1)),
        Parameter('Df', (0.001, 0.009, 0.0002), (0, 1)),
        Parameter('Ds', (0.000, 0.004, 0.00002), (0, 1)),
        ParamC
    ],
    postproc=biexp_flip))
Models.append(Model(
    'BiexpN',
    'Normalized Bi-exponential',
    lambda p, x: biexp(x, *p),
    [
        Parameter('AfN', (0.2, 1.0, 0.1), (0, 1)),
        Parameter('DfN', (0.001, 0.009, 0.0002), (0, 1)),
        Parameter('DsN', (0.000, 0.004, 0.00002), (0, 1)),
    ],
    preproc=dwi.util.normalize_si_curve,
    postproc=biexp_flip))

Models.append(Model(
    'T2',
    'T2 relaxation',
    lambda p, x: t2(x, *p),
    [
        Parameter('T2', (1, 300, 50), (1, 300)),
        Parameter('C', (0.25, 1, 0.5), (0, 1e9), relative=True)
    ]))
