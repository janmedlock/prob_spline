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
    A smoothing periodic spline using a loglikelihood as the error metric
    instead of the usual squared distance.
    '''
    _parameter_min = - numpy.inf
    _parameter_max = numpy.inf

    _transform = _transform_inverse = staticmethod(identity)

    def __init__(self, degree = 3, sigma = 0, period = 1):
        assert isinstance(degree, numbers.Integral) and (degree >= 0), \
            'degree must be a nonnegative integer.'
        self.degree = degree
        assert (sigma >= 0), 'sigma must be nonnegative.'
        self.sigma = sigma
        assert (period > 0), 'period must be positive.'
        self.period = period

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

    def _evaluate(self, X, knots, coef, derivative = 0):
        '''
        Evaluate the spline defined by (knots, coef) at X.
        '''
        # Find which interval the x values are in.
        # Shift so that knots[0] is at 0.
        knots_shifted = knots - knots[0]
        X_shifted = (X - knots[0]) % self.period
        ix = numpy.searchsorted(knots_shifted, X_shifted, side = 'right') - 1
        # Handle scalar vs vector x.
        if numpy.isscalar(ix):
            ix = [ix]
        # Get the coefficients in those intervals
        # and stack them for numpy.polyval().
        C = numpy.stack(
            [coef[..., (self.degree + 1) * i : (self.degree + 1) * (i + 1)].T
             for i in ix], axis = -1)
        # Take derivatives or integrals as needed.
        if derivative > 0:
            C = numpy.stack([numpy.polyder(C[..., i], derivative)
                             for i in range(numpy.shape(C)[-1])],
                            axis = -1)
        elif derivative < 0:
            C = numpy.stack([numpy.polyint(C[..., i], - derivative)
                             for i in range(numpy.shape(C)[-1])],
                            axis = -1)
        # Evaluate the polynomials in those intervals at the X values.
        z = numpy.polyval(C, X_shifted - knots_shifted[ix])
        mu = self._transform_inverse(z).clip(self._parameter_min,
                                             self._parameter_max)
        if numpy.isscalar(X):
            mu = numpy.squeeze(mu)
            if numpy.ndim(mu) == 0:
                mu = numpy.asscalar(mu)
        return mu

    def predict(self, X, *args, **kwargs):
        '''
        The Y values of the spline at X.
        '''
        self._check_is_fitted()
        return self._evaluate(X, self.knots_, self.coef_, *args, **kwargs)

    __call__ = predict

    def score(self, X, Y):
        '''
        The likelihood, i.e. not the *log*likelihood.
        '''
        mu = self.predict(X)
        return numpy.exp(numpy.sum(self._loglikelihood(Y, mu)))

    def _get_knots(self, X):
        knots = X
        return knots

    def _fit_interpolating_spline(self, X, Y,
                                  continuity_matrix_vector = None):
        '''
        Fit an interpolating spline to (X, Y).
        '''
        knots = self._get_knots(X)
        # The continuity conditions.
        # A_C.dot(coef) = b_C.
        if continuity_matrix_vector is not None:
            A_C, b_C = continuity_matrix_vector
        else:
            A_C, b_C = self._get_continuity_matrix_vector(X, Y)
        # The interpolating conditions.
        # A_I.dot(coef) = b_I.
        A_I, b_I = self._get_interpolating_matrix_vector(X, Y)
        # Solve these two sets of conditions together.
        A = scipy.sparse.vstack((A_C, A_I))
        b = numpy.hstack((b_C, b_I))
        coef = scipy.sparse.linalg.spsolve(A.tocsr(), b)
        Z = self._transform(Y)
        if numpy.ndim(Z) > 1:
            r = numpy.shape(Z)[0]
            coef = numpy.reshape(coef, (r, -1))
        return (knots, coef)

    def _get_interpolating_matrix_vector(self, X, Y):
        '''
        Build the matrix A_I and vector b_I
        for the interpolating conditions at the knots.
        A_I * coef = b_I
        when the conditions are all satisfied.
        '''
        knots = self._get_knots(X)
        m_I = len(knots)
        n_I = (self.degree + 1) * len(knots)
        # The matrix.
        A_I = scipy.sparse.lil_matrix((m_I, n_I))
        # At all the X values, we just have that
        # the constant term in the polynomial is
        # equal to the corresponding
        # (transformed) Y value.
        rows = numpy.arange(len(knots), dtype = int)
        # The constant terms.
        cols = rows * (self.degree + 1) + self.degree
        A_I[rows, cols] = 1
        # The right-hand side vector.
        b_I = self._transform(Y)
        # If the (transformed) Y is a matrix,
        # built out the matrix and vector
        # by block operations.
        if numpy.ndim(b_I) > 1:
            r = numpy.shape(b_I)[0]
            A_I = scipy.sparse.block_diag([A_I] * r)
            b_I = numpy.hstack(b_I)
        return (A_I, b_I)

    def _get_continuity_matrix_vector(self, X, Y):
        '''
        Build the matrix A_C and vector b_C
        for the continuity conditions at the internal knots.
        A_C * coef = b_C
        when the conditions are all satisfied.
        '''
        knots = self._get_knots(X)
        m = self.degree * len(knots)
        n = (self.degree + 1) * len(knots)
        A_C = scipy.sparse.lil_matrix((m, n))
        # Add periodic point.
        dX = numpy.diff(numpy.hstack((X, X[0] + self.period)))
        # At each knot,
        # the jth derivatives are continuous
        # for j = 0, 1, ..., degree - 1.
        for j in range(self.degree):
            constants = numpy.polyder(numpy.ones(self.degree + 1),
                                      m = j)
            exponents = numpy.arange(self.degree - j, -1, -1)
            for i in range(len(knots)):
                row = self.degree * i + j
                # The columns of the terms in
                # the jth derivative
                # of the ith polynomial.
                cols = slice((self.degree + 1) * i,
                             (self.degree + 1) * (i + 1) - j)
                A_C[row, cols] = constants * dX[i] ** exponents
                # The column of the constant term in
                # the jth derivative of the (i + 1)st polynomial,
                # but wrap around using modular arithmetic
                # to connect the last spline with the first
                # for periodicity.
                col = ((self.degree + 1) * (i + 2) - 1 - j) % n
                A_C[row, col] = - constants[-1]
        # The right-hand side vector is just all zeros.
        b_C = numpy.zeros(m)
        # If the (transformed) Y is a matrix,
        # built out the matrix and vector
        # by block operations.
        Z = self._transform(Y)
        if numpy.ndim(Z) > 1:
            r = numpy.shape(Z)[0]
            A_C = scipy.sparse.block_diag([A_C] * r)
            b_C = numpy.hstack([b_C] * r)
        return (A_C, b_C)

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
        A_C, b_C = self._get_continuity_matrix_vector(X, Y)
        # Get the initial guess for the coefficients
        # from the interpolating spline.
        knots_interp, coef_interp = self._fit_interpolating_spline(
            X, Y, continuity_matrix_vector = (A_C, b_C))
        # We could also optimize over the knots.
        initial_guess = numpy.ravel(coef_interp)
        knots = knots_interp
        # Set some constants so that the objective function
        # doesn't recompute them every time it is called.
        # Add periodic point.
        dX = numpy.diff(numpy.hstack((X, X[0] + self.period)))
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
                           args = (A_C, b_C),
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
        coef = numpy.reshape(coef, numpy.shape(coef_interp))
        return (knots, coef)

    def _objective(self, coef, knots, X, Y, *variation_args):
        '''
        The objective function for optimizing the smoothing spline,
        - loglikelihood + sigma * \int |f''(x)| dx.
        '''
        r = int(len(coef) / len(knots) / (self.degree + 1))
        if r > 1:
            coef = numpy.reshape(coef, (r, -1))
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
        a = coef[..., 0 : : self.degree + 1]
        b = coef[..., 1 : : self.degree + 1]
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

    def _continuity_constraints(self, coef, A_C, b_C):
        '''
        The continuity constraints function for optimizing the smoothing spline.
        '''
        return A_C.dot(coef) - b_C
