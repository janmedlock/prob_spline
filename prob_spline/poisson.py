import warnings

import numpy
import scipy.stats

from . import base


class PoissonSpline(base.ProbSpline):
    # Poisson parameter is nonnegative.
    _parameter_min = 0

    _alpha = 1e-8

    @staticmethod
    def _loglikelihood(Y, mu):
        if numpy.isscalar(mu):
            mu = numpy.array([mu])
        # Handle mu = +inf gracefully.
        isposinf = numpy.isposinf(mu)
        # Silence warnings.
        mu[isposinf] = 0
        V = scipy.stats.poisson.logpmf(Y, mu)
        V[isposinf] = -numpy.inf
        if numpy.isscalar(Y):
            V = numpy.asscalar(V)
        return V

    @classmethod
    def _transform(cls, Y):
        return numpy.log(Y + cls._alpha)

    @classmethod
    def _transform_inverse(cls, Z):
        # Silence warnings.
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore',
                                    category = RuntimeWarning,
                                    message = 'overflow encountered in exp')
            return numpy.exp(Z) - cls._alpha
