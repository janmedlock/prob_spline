import numpy
import scipy.stats

from . import base


class PoissonSpline(base.ProbSpline):
    # Poisson parameter is nonnegative.
    _parameter_min = 0

    _alpha = 1e-8

    @staticmethod
    def _loglikelihood(Y, mu):
        return scipy.stats.poisson.logpmf(Y, mu)

    @classmethod
    def _transform(cls, Y):
        return numpy.log(Y + cls._alpha)

    @classmethod
    def _transform_inverse(cls, Z):
        return numpy.exp(Z) - cls._alpha
