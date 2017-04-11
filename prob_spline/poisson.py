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
        with numpy.errstate(invalid = 'ignore'):
            V = numpy.where(numpy.isposinf(mu),
                            - numpy.inf,
                            scipy.stats.poisson.logpmf(Y, mu))
        if numpy.isscalar(Y):
            V = numpy.squeeze(V, axis = 0)
            if numpy.ndim(V) == 0:
                V = numpy.asscalar(V)
        return V

    @classmethod
    def _transform(cls, Y):
        return numpy.log(Y + cls._alpha)

    @classmethod
    def _transform_inverse(cls, Z):
        # Silence warnings.
        with numpy.errstate(over = 'ignore'):
            return numpy.exp(Z) - cls._alpha
