#!/usr/bin/python3

import numpy
import scipy.interpolate
import scipy.linalg


def N(x, i, degree, knots):
    '''
    The ith basis spline evaluated at x.

    Note that the 3rd argument is the *degree*,
    while Dierckx uses order = degree + 1.
    '''
    if degree == 0:
        return numpy.where((knots[i] <= x) & (x < knots[i + 1]), 1, 0)
    else:
        if knots[i + degree] > knots[i]:
            coef_left = (x - knots[i]) / (knots[i + degree] - knots[i])
        else:
            coef_left = numpy.where(x == knots[i], 1, 0)
        if knots[i + degree + 1] > knots[i + 1]:
            coef_right = ((knots[i + degree + 1] - x)
                          / (knots[i + degree + 1] - knots[i + 1]))
        else:
            coef_right = numpy.where(x == knots[i + degree + 1], 1, 0)
        return (coef_left * N(x, i, degree - 1, knots)
                + coef_right * N(x, i + 1, degree - 1, knots))


def eval_spline(x, tck):
    knots, coef, degree = tck
    n = len(knots) - degree - 1
    return numpy.sum(coef[i] * N(x, i, degree, knots)
                     for i in range(n))


def get_knots(X, degree = 3, a = None, b = None):
    '''
    Build knots for an interpolating spline from the data points X.
    '''
    if a == None:
        a = X[0]
    if b == None:
        b = X[-1]
    n = len(X)
    knots = numpy.empty(n + 2 * degree)
    # The internal knots.
    if degree % 2 == 1:
        knots[degree : n + degree] = X
    else:
        # degree % 2 == 0
        knots[degree] = a
        knots[degree + 1 : n + degree - 1] = (X[1 : - 1] + X[0 : - 2]) / 2
        knots[n + degree - 1] = b
    # Periodic boundary knots.
    knots[ : degree] = knots[n - 1 : n + degree - 1] - b + a
    knots[n + degree : ] = knots[degree + 1 : 2 * degree + 1] + b - a
    return knots


def get_coef_matrix(X, degree, knots):
    '''
    Build the matrix used to find the coefficients.
    '''
    n = len(X)
    # A will have nonzero entries
    # [[ x x x           ]]
    # [[   x x x         ]]
    # [[     \ \ \       ]]
    # [[       x x x     ]]
    # [[         x x x   ]]
    # [[           x x x ]]
    # where the width of the nonzero band is (degree + 1).
    #
    # We will take the right-most degree columns
    # and add them to the left-most degree columns
    # to get A:
    # [[ x x x       ]]
    # [[   x x x     ]]
    # [[     \ \ \   ]]
    # [[       x x x ]]
    # [[ x       x x ]]
    # [[ x x       x ]]
    A = numpy.column_stack([N(X[ : -1], i, degree, knots)
                            for i in range(n - 1)])
    B = numpy.column_stack([N(X[ : -1], i, degree, knots)
                            for i in range(n - 1, n + degree - 1)])
    A[:, : degree] += B
    return A


def get_coef(X, Y, degree, knots):
    '''
    Find the coefficients for an interpolating spline through
    the points (X, Y).
    '''
    A = get_coef_matrix(X, degree, knots)
    coef = scipy.linalg.solve(A, Y[ : -1])
    coef = numpy.hstack((coef, coef[0 : degree], [0] * (degree + 1)))
    return coef


def get_spline(X, Y, degree = 3, a = None, b = None):
    '''
    Build an interpolating spline through the points (X, Y).

    This returns a tuple (knots, coefficients, degree)
    that is compatible with scipy.interpolate.splev() etc.
    '''
    knots = get_knots(X, degree = degree, a = a, b = b)
    coef = get_coef(X, Y, degree, knots)
    return (knots, coef, degree)


def get_deriv(tck, n = 1):
    '''
    Get the nth derivative of the spline with
    (knots, coefficients, degree) = tck.

    This returns the tuple (knots, coefficients, degree)
    of the derivative.
    '''
    if n == 0:
        return tck
    else:
        knots, coef, degree = tck
        knots_deriv = knots[1 : -1]
        degree_deriv = degree - 1
        coef_deriv = numpy.zeros_like(knots_deriv)
        numer = degree * (coef[1 : - (degree + 1)] - coef[ : - (degree + 2)])
        denom = knots[degree + 1 : - 1] - knots[1 : - (degree + 1)]
        #                 { numer[i] / denom[i]  if denom[i] > 0,
        # coef_deriv[i] = {
        #                 { coef[i]              otherwise.
        quot = numpy.ma.divide(numer, denom)
        coef_deriv[ : - degree] = numpy.where(denom > 0,
                                              quot,
                                              coef[ : - (degree + 2)])
        return get_deriv((knots_deriv, coef_deriv, degree_deriv),
                         n = n - 1)


if __name__ == '__main__':
    # Compare my versions with scipy.interpolate.splrep()
    # for degree = 1, ..., 5.
    X = numpy.sort(numpy.random.uniform(size = 10))
    Y = numpy.random.uniform(size = len(X))
    # Force to be periodic.
    X = numpy.hstack((X, X[0] + 1))
    Y = numpy.hstack((Y, Y[0]))
    for degree in range(1, 5 + 1):
        tck_scipy = scipy.interpolate.splrep(X, Y,
                                             k = degree,
                                             s = 0,
                                             per = True)
        knots = get_knots(X, degree)
        assert all(numpy.isclose(knots, tck_scipy[0]))
        coef = get_coef(X, Y, degree, knots)
        assert all(numpy.isclose(coef, tck_scipy[1]))
        tck = get_spline(X, Y, degree = degree)
        # Compare nth derivatives for n = 1, ..., degree.
        for n in range(1, degree + 1):
            tck_deriv = get_deriv(tck, n = n)
            tck_deriv_scipy = scipy.interpolate.splder(tck_scipy, n = n)
            assert all(numpy.isclose(tck_deriv[0], tck_deriv_scipy[0]))
            assert all(numpy.isclose(tck_deriv[1], tck_deriv_scipy[1]))
            assert tck_deriv[2] == tck_deriv_scipy[2]
