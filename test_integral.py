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
# Make mu(x) periodic.
X = numpy.linspace(test_common.x_min, test_common.x_max, npoints + 1)
Y = scipy.stats.poisson.rvs(test_common.mu(X[ : -1]))
Y = numpy.hstack((Y, Y[0]))

# Fit an interpolating spline.
spline = prob_spline.NormalSpline()
spline.fit(X, Y)


def get_variation_exact(spline, X):
    '''
    Compute the variation using the exact algorithm
    in prob_spline.ProbSpline._objective().
    '''
    dX = numpy.diff(X)
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


def get_variation_numint(spline, X):
    '''
    Compute the variation using numerical integration
    of the spline's second derivative.
    '''
    def absf2(X):
        # Find which interval the x values are in.
        ix = (numpy.searchsorted(spline.knots_, X) - 1).clip(min = 0)
        # Handle scalar vs vector x.
        if numpy.isscalar(ix):
            ix = [ix]
        # Get the coefficients in those intervals.
        coef = (spline.coef_[(spline.degree + 1) * i :
                             (spline.degree + 1) * (i + 1)]
                for i in ix)
        # Take the (degree - 1)st derivative.
        coef = numpy.column_stack((numpy.polyder(c, m = spline.degree - 1)
                                   for c in coef))
        v = numpy.abs(numpy.polyval(coef, X - spline.knots_[ix]))
        if numpy.isscalar(X):
            v = numpy.asscalar(v)
        return v
    variation, error = scipy.integrate.quad(absf2, X[0], X[-1],
                                            limit = 1000)
    return variation


variation_exact = get_variation_exact(spline, X)
variation_numint = get_variation_numint(spline, X)

assert numpy.isclose(variation_exact, variation_numint)
