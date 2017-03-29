#!/usr/bin/python3
'''
Test cross-validating the splines with an example.
'''

import numpy
import scipy.stats

import test_common
import prob_spline


npoints = 11


# Get Poisson samples around mu(x).
X = numpy.linspace(test_common.x_min, test_common.x_max, npoints)
Y = scipy.stats.poisson.rvs(test_common.mu(X))