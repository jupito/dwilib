#!/usr/bin/env python
# -*- coding: iso-8859-15 -*-

import numpy as np
import scipy as sp
import scipy.io
import matplotlib as mpl
import matplotlib.pyplot as plt

from pylab import plot, show, ylim, yticks
from numpy import sin, cos, exp, pi, arange

t = arange(0.0, 2.0, 0.01)

s1 = sin(2 * pi * t)
s2 = exp(-t)
s3 = sin(2 * pi * t) * exp(-t)
s4 = sin(2 * pi * t) * cos(4 * pi * t)

t = arange(0.0, 2.0, 0.01)
plot(t, s1, t, s2+1, t, s3+2, t, s4+3, color='k')
ylim(-1, 4)
yticks(arange(4), ['s1', 's2', 's3', 's4'])

show()
