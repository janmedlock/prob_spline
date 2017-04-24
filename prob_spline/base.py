import abc
import numbers
import warnings

import numpy
import scipy.optimize
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

    def _fit_interpolating_spline(self, X, Y):
        '''
        Fit an interpolating spline to (X, Y).
        '''
        knots = self._get_knots(X)
        # Find the coefficients for the constant term.
        Y_transform = self._transform(Y)
        coef_constants = Y_transform
        # Find the rest of the coefficients.
        continuity_matrix = self._build_continuity_matrix(X, Y)
        coef = coef_constants @ continuity_matrix
        return (knots, coef)

    def _build_continuity_matrix(self, X, Y):
        '''
        Build the matrices used by _get_coef().
        '''
        m = len(X)
        n = (self.degree + 1) * m
        A = numpy.zeros((n - m, n))
        # Add periodic point.
        dX = numpy.diff(numpy.hstack((X, X[0] + self.period)))
        # At each knot,
        # the jth derivatives are continuous
        # for j = 0, 1, ..., degree - 1.
        for j in range(self.degree):
            constants = numpy.polyder(numpy.ones(self.degree + 1),
                                      m = j)
            exponents = numpy.arange(self.degree - j, -1, -1)
            for i in range(len(X)):
                row = self.degree * i + j
                # The columns of the terms in
                # the jth derivative
                # of the ith polynomial.
                cols = slice((self.degree + 1) * i,
                             (self.degree + 1) * (i + 1) - j)
                A[row, cols] = constants * dX[i] ** exponents
                # The column of the constant term in
                # the jth derivative of the (i + 1)st polynomial,
                # but wrap around using modular arithmetic
                # to connect the last spline with the first
                # for periodicity.
                col = ((self.degree + 1) * (i + 2) - 1 - j) % n
                A[row, col] = - constants[-1]
        # Pick off the columns for the constant terms
        ix_constants = (numpy.arange(n) % (self.degree + 1)
                          == self.degree)
        A_constants = A[:,  ix_constants]
        # and the columns for the non-constant terms.
        A_rest = A[:, ~ix_constants]
        # Build the continuity matrix.
        continuity_matrix = numpy.zeros((n, m))
        # Keep the constant terms unchanged.
        continuity_matrix[ix_constants] = numpy.eye(m)
        # Compute the non-constant terms from the constant terms.
        continuity_matrix[~ix_constants] = (
            - numpy.linalg.inv(A_rest) @ A_constants)
        # Now transpose this (sorry) to handle matrix coefficients.
        return continuity_matrix.T

    def _fit_smoothing_spline(self, X, Y,
                              method = 'Nelder-Mead', tol = 1e-3,
                              options = dict(maxiter = 100000,
                                             maxfev = 100000)):
        '''
        Fit a smoothing spline to (X, Y)
        by minimizing
        - loglikelihood + sigma * \int |f''(x)| dx.
        '''
        # Get the initial guess for the coefficients
        # from the interpolating spline.
        knots_interp, coef_interp = self._fit_interpolating_spline(X, Y)
        # Pick off the columns for the constant terms
        ix_constants = (
            numpy.arange((self.degree + 1) * len(X)) % (self.degree + 1)
            == self.degree)
        coef_constants_interp = coef_interp[..., ix_constants]
        coef_constants_shape = numpy.shape(coef_constants_interp)
        # We could also optimize over the knots.
        # Flatten for optimizer.
        initial_guess = numpy.ravel(coef_constants_interp)
        knots = knots_interp
        # Set some constants so that the objective function
        # doesn't recompute them every time it is called.
        continuity_matrix = self._build_continuity_matrix(X, Y)
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
        objective_args = ((knots, X, Y, coef_constants_shape, continuity_matrix)
                          + variation_args)
        result = scipy.optimize.minimize(self._objective,
                                         initial_guess,
                                         args = objective_args,
                                         method = method,
                                         tol = tol,
                                         options = options)
        if not result.success:
            warnings.warn(result.message,
                          category = scipy.optimize.OptimizeWarning)
        coef_constants_flat = result.x
        coef_constants = numpy.reshape(coef_constants_flat,
                                       coef_constants_shape)
        coef = coef_constants @ continuity_matrix
        return (knots, coef)

    def _objective(self, coef_constants_flat, knots, X, Y,
                   coef_constants_shape, continuity_matrix,
                   *variation_args):
        '''
        The objective function for optimizing the smoothing spline,
        - loglikelihood + sigma * \int |f''(x)| dx.
        '''
        coef_constants = numpy.reshape(coef_constants_flat,
                                       coef_constants_shape)
        coef = coef_constants @ continuity_matrix
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
