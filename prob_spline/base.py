import abc
import numbers
import warnings

import numpy
import scipy.optimize
import scipy.sparse
import scipy.sparse.linalg
import sklearn.base
import sklearn.utils.validation


def identity(Y):
    '''
    Null transform.
    '''
    return Y


class ProbSpline(sklearn.base.BaseEstimator, abc.ABC):
    '''
    A smoothing spline using a loglikelihood as the error metric
    instead of the usual squared distance.
    '''
    _parameter_min = - numpy.inf
    _parameter_max = numpy.inf

    _transform = _transform_inverse = staticmethod(identity)

    def __init__(self, degree = 3, sigma = 0):
        assert isinstance(degree, numbers.Integral) and (degree >= 0), \
            'degree must be a nonnegative integer.'
        self.degree = degree
        assert (sigma >= 0), 'sigma must be nonnegative.'
        self.sigma = sigma

    @abc.abstractmethod
    def _loglikelihood(self, Y, mu):
        '''
        This must be defined by subclasses.
        '''

    def fit(self, X, Y, _force_smoothing = False, **options):
        '''
        Fit the spline to (X, Y).

        For testing, with self.sigma = 0, _force_smoothing = True causes
        an interpolating spline to be fit
        *using the code for smoothing splines*.
        '''
        if (self.sigma == 0) and (not _force_smoothing):
            self.knots_, self.coef_ = self._fit_interpolating_spline(X, Y)
        else:
            self.knots_, self.coef_ = self._fit_smoothing_spline(X, Y,
                                                                 **options)
        return self

    def _check_is_fitted(self):
        '''
        Make sure that the spline has been fitted,
        i.e. fit() has been called.
        '''
        # Make sure knots_ and coef_ are defined.
        sklearn.utils.validation.check_is_fitted(self, ('knots_', 'coef_'))

    def _evaluate(self, X, knots, coef):
        '''
        Evaluate the spline defined by (knots, coef) at X.
        '''
        # Find which interval the x values are in.
        ix = (numpy.searchsorted(knots, X) - 1).clip(min = 0)
        # Handle scalar vs vector x.
        if numpy.isscalar(ix):
            ix = [ix]
        # Get the coefficients in those intervals.
        coef = numpy.stack(
            [coef[(self.degree + 1) * i : (self.degree + 1) * (i + 1)]
             for i in ix], axis = -1)
        # Evaluate the polynomials in those intervals at the X values.
        z = numpy.polyval(coef, X - knots[ix])
        mu = self._transform_inverse(z).clip(self._parameter_min,
                                             self._parameter_max)
        if numpy.isscalar(X):
            mu = numpy.squeeze(mu, axis = 0)
            if numpy.ndim(mu) == 0:
                mu = numpy.asscalar(mu)
        return mu

    def predict(self, X):
        '''
        The Y values of the spline at X.
        '''
        self._check_is_fitted()
        return self._evaluate(X, self.knots_, self.coef_)

    __call__ = predict

    def score(self, X, Y):
        '''
        The likelihood, i.e. not the *log*likelihood.
        '''
        mu = self.predict(X)
        return numpy.exp(numpy.sum(self._loglikelihood(Y, mu)))

    def _fit_interpolating_spline(self, X, Y,
                                  continuity_matrix = None):
        '''
        Fit an interpolating spline to (X, Y).
        '''
        if continuity_matrix is None:
            continuity_matrix = self._get_continuity_matrix(X, Y)
        knots = X[0 : -1]
        # The continuity conditions.
        # A_C.dot(coef) = 0.
        A_C = continuity_matrix
        m_C, n_C = A_C.shape
        b_C = numpy.zeros(m_C)
        # The interpolating conditions.
        # A_I.dot(coef) = Y.
        m_I = len(knots) + 1
        n_I = (self.degree + 1) * len(knots)
        A_I = scipy.sparse.lil_matrix((m_I, n_I))
        # At all the X values except the last one,
        # we just have that the constant term in the
        # polynomial is equal to the corresponding
        # (transformed) Y value.
        rows = numpy.arange(len(knots), dtype = int)
        # The constant terms.
        cols = rows * (self.degree + 1) + self.degree
        A_I[rows, cols] = 1
        # The final interpolation condtion is that
        # the last polynomial, from the penultimate X value,
        # evaluated at the final X value,
        # is equal to the final (transformed) Y value.
        exponents = numpy.arange(self.degree, -1, -1)
        dX = X[-1] - X[-2]
        A_I[-1, - (self.degree + 1) : ] = dX ** exponents
        b_I = self._transform(Y)
        # Solve these two sets of conditions together.
        A = scipy.sparse.vstack((A_C, A_I))
        b = numpy.hstack((b_C, b_I))
        coef = scipy.sparse.linalg.spsolve(A.tocsr(), b)
        return (knots, coef)

    def _get_continuity_matrix(self, X, Y):
        '''
        Build the matrix for the continuity conditions
        at the internal knots.
        The product of the resulting matrix and the vector of coeffecients
        is 0 when the conditions are all satisfied.
        '''
        knots = X[0 : -1]
        m = self.degree * (len(knots) - 1) + 2
        n = (self.degree + 1) * len(knots)
        continuity_matrix = scipy.sparse.lil_matrix((m, n))
        dX = numpy.diff(X)
        if self.degree > 0:
            # Natural boundary conditions.
            # The (degree - 1)st derivative is zero at the boundaries.
            constants = numpy.polyder(numpy.ones(self.degree + 1),
                                      m = self.degree - 1)
            exponents = numpy.arange(1, -1, -1)
            # Left boundary.
            continuity_matrix[0, 1] = constants[1]
            # Right boundary.
            cols = slice(- (self.degree + 1), - (self.degree - 1))
            continuity_matrix[-1, cols] = constants * dX[-1] ** exponents
        # At each internal knot,
        # the jth derivatives are continuous
        # for j = 0, 1, ..., degree - 1.
        for j in range(self.degree):
            constants = numpy.polyder(numpy.ones(self.degree + 1),
                                      m = j)
            exponents = numpy.arange(self.degree - j, -1, -1)
            for i in range(len(knots) - 1):
                row = self.degree * i + j + 1
                # The columns of the terms in
                # the jth derivative of the ith polynomial.
                cols = slice((self.degree + 1) * i,
                             (self.degree + 1) * (i + 1) - j)
                continuity_matrix[row, cols] = constants * dX[i] ** exponents
                # The column of the constant term in
                # the jth derivative of the (i + 1)st polynomial.
                col = (self.degree + 1) * (i + 2) - 1 - j
                continuity_matrix[row, col] = - constants[-1]
        return continuity_matrix

    def _fit_smoothing_spline(self, X, Y,
                              tol = 1e-3, maxiter = 1000, **options):
        '''
        Fit a smoothing spline to (X, Y)
        by minimizing
        - loglikelihood + sigma * \int |f''(x)| dx.
        '''
        # Build the options for scipy.optimize.minimize()
        options.update(maxiter = maxiter)
        # The continuity matrix is needed to get the initial
        # guess for the coefficients (_fit_interpolating_spline())
        # and by _continuity_constraints, so build it once now.
        continuity_matrix = self._get_continuity_matrix(X, Y)
        # Get the initial guess for the coefficients
        # from the interpolating spline.
        knots_interp, coef_interp = self._fit_interpolating_spline(
            X, Y, continuity_matrix = continuity_matrix)
        # We could also optimize over the knots.
        initial_guess = coef_interp
        knots = knots_interp
        # Set some constants so that the objective function
        # doesn't recompute them every time it is called.
        dX = numpy.diff(X)
        deriv1 = numpy.polyder(numpy.ones(self.degree + 1),
                               m = self.degree - 2)
        deriv2 = numpy.polyder(numpy.ones(self.degree + 1),
                               m = self.degree - 1)
        adjustment_constant = (deriv2[1] / deriv2[0]
                               * (deriv1[1]
                                  - deriv1[0] * deriv2[1] / deriv2[0]))
        variation_args = (dX, deriv1, deriv2, adjustment_constant)
        objective_args = (knots, X, Y) + variation_args
        constraints = dict(fun = self._continuity_constraints,
                           args = (continuity_matrix, ),
                           type = 'eq')
        result = scipy.optimize.minimize(self._objective,
                                         initial_guess,
                                         constraints = constraints,
                                         args = objective_args,
                                         tol = tol,
                                         options = options)
        if not result.success:
            warnings.warn(result.message, scipy.optimize.OptimizeWarning)
        coef = result.x
        return (knots, coef)

    def _objective(self, coef, knots, X, Y, *variation_args):
        '''
        The objective function for optimizing the smoothing spline,
        - loglikelihood + sigma * \int |f''(x)| dx.
        '''
        mu = self._evaluate(X, knots, coef)
        loglikelihood = numpy.sum(self._loglikelihood(Y, mu))
        if self.sigma == 0:
            return - loglikelihood
        else:
            variation = self._variation(knots, coef, *variation_args)
            return - loglikelihood + self.sigma * variation

    def _variation(self, knots, coef,
                   dX, deriv1, deriv2, adjustment_constant):
        '''
        \int |f^{(k - 1)}(x)| dx
        '''
        # Calculate the variation integral.
        a = coef[0 : : self.degree + 1]
        b = coef[1 : : self.degree + 1]
        # Check if 2nd derivative is 0 inside each interval.
        condition1  = (a * b < 0)
        condition2 = (numpy.abs(deriv2[1] * b)
                      < numpy.abs(deriv2[0] * a * dX))
        haszero = condition1 & condition2
        variations = dX * (deriv1[0] * a * dX + deriv1[1] * b)
        adjustments = numpy.ma.divide(2 * adjustment_constant * b ** 2, a)
        variations[haszero] += adjustments[haszero]
        # Sum over pieces of the spline.
        variation = numpy.sum(numpy.abs(variations))
        return variation

    def _continuity_constraints(self, coef, continuity_matrix):
        '''
        The continuity constraints function for optimizing the smoothing spline.
        '''
        return continuity_matrix.dot(coef)
