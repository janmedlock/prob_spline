#!/usr/bin/python3

import numpy
import scipy.interpolate
import scipy.linalg
import scipy.sparse
import scipy.sparse.linalg


def N(x, i, degree, knots):
    '''
    The ith basis spline evaluated at x.

    Note that the 3rd argument is the *degree*,
    while Dierckx uses order = degree + 1.
    '''
    if degree == 0:
        cond_left = (knots[i] <= x)
        if i + 1 < len(knots) - 1:
            cond_right = (x < knots[i + 1])
        else:
            # i + 1 == len(knots) - 1
            # Include the right-most endpoint in the interval.
            cond_right = (x <= knots[i + 1])
        return numpy.where(cond_left & cond_right, 1, 0)
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
    assert n >= degree + 1
    knots = numpy.empty(n + degree + 1)
    # (degree + 1) copies of a at the front.
    knots[ : (degree + 1)] = a
    if degree % 2 == 1:
        # Middle is X without the first and last (degree + 1) / 2 entries.
        j = int((degree + 1) / 2)
        knots[degree + 1 : n] = X[j : n - j]
    else:
        # degree % 2 == 0
        # Middle is the average of
        # X without the first degree / 2 and last degree / 2 + 1 entries
        # and X without the first degree / 2 + 1 and last degree / 2 entries.
        j = int(degree / 2)
        knots[degree + 1 : n] = (X[j : n - j - 1]
                                 + X[j + 1 : n - j]) / 2
    # (degree + 1) copies of b at the end.
    knots[n : ] = b
    return knots


def get_coef_matrix(X, degree, knots, sparse = True):
    '''
    Build the matrix used to find the coefficients.
    '''
    n = len(X)
    if not sparse:
        A = numpy.column_stack([N(X, i, degree, knots)
                                for i in range(n)])
    else:
        diag_max = max(degree - 1, 0)
        offsets = range(- diag_max, diag_max + 1)
        diags = [[N(X[j], i + j, degree, knots)
                  for j in range(- min(i, 0), n - max(i, 0))]
                 for i in offsets]
        A = scipy.sparse.diags(diags, offsets)
    return A


def get_coef(X, Y, degree, knots, sparse = True):
    '''
    Find the coefficients for an interpolating spline through
    the points (X, Y).
    '''
    A = get_coef_matrix(X, degree, knots, sparse = sparse)
    if not scipy.sparse.issparse(A):
        coef = scipy.linalg.solve(A, Y)
    else:
        if not (scipy.sparse.isspmatrix_csc(A)
                or scipy.sparse.isspmatrix_csc(A)):
            # Need to convert for spsolve.
            A = scipy.sparse.csc_matrix(A)
        coef = scipy.sparse.linalg.spsolve(A, Y)
    coef = numpy.hstack((coef, [0] * (degree + 1)))
    return coef


def get_spline(X, Y, degree = 3, sparse = True, a = None, b = None):
    '''
    Build an interpolating spline through the points (X, Y).

    This returns a tuple (knots, coefficients, degree)
    that is compatible with scipy.interpolate.splev() etc.
    '''
    knots = get_knots(X, degree = degree, a = a, b = b)
    coef = get_coef(X, Y, degree, knots, sparse = sparse)
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
    for degree in range(1, 5 + 1):
        tck_scipy = scipy.interpolate.splrep(X, Y, k = degree, s = 0)
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
