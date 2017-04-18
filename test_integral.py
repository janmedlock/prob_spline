#!/usr/bin/python3
'''
Test the exact method for computing the variation of the smoothing splines.
'''

import numpy
import scipy.integrate
import scipy.stats

import prob_spline
import test_common


npoints = 20

numpy.random.seed(2)

# Get Poisson samples around mu(x).
X_pad = (test_common.x_max - test_common.x_min) / 2 / npoints
X_min = test_common.x_min + X_pad
X_max = test_common.x_max - X_pad
X = numpy.linspace(X_min, X_max, npoints)
Y = scipy.stats.poisson.rvs(test_common.mu(X))

# Fit an interpolating spline.
spline = prob_spline.NormalSpline()
spline.fit(X, Y)


def get_variation_exact(spline, X):
    '''
    Compute the variation using the exact algorithm
    in prob_spline.ProbSpline._objective().
    '''
    dX = numpy.diff(numpy.hstack((X, X[0] + spline.period)))
    deriv1 = numpy.polyder(numpy.ones(spline.degree + 1),
                           m = spline.degree - 2)
    deriv2 = numpy.polyder(numpy.ones(spline.degree + 1),
                           m = spline.degree - 1)
    adjustment_constant = (deriv2[1] / deriv2[0]
                           * (deriv1[1]
                              - deriv1[0] * deriv2[1] / deriv2[0]))
    variation = spline._variation(spline.knots_, spline.coef_,
                                  dX, deriv1, deriv2, adjustment_constant)
    return variation


def get_variation_numint(spline, a, b):
    '''
    Compute the variation using numerical integration
    of the spline's second derivative.
    '''
    def absf2(X):
        return numpy.abs(spline(X, derivative = spline.degree - 1))
    variation, error = scipy.integrate.quad(absf2, a, b,
                                            limit = 1000)
    return variation


variation_exact = get_variation_exact(spline, X)
variation_numint = get_variation_numint(spline,
                                        test_common.x_min,
                                        test_common.x_max)

assert numpy.isclose(variation_exact, variation_numint)
