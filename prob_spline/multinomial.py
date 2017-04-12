import numpy
import scipy.stats

from . import base


class MultinomialSpline(base.ProbSpline):
    # Multinomrial parameters are in [0, 1].
    _parameter_min = 0
    _parameter_max = 1

    _alpha = 1e-8

    @staticmethod
    def _loglikelihood(Y, mu):
        V = scipy.stats.multinomial.logpmf(Y.T,
                                           numpy.sum(Y, axis=0),
                                           mu.T)
        if numpy.isscalar(Y):
            V = numpy.squeeze(V, axis = -1)
        return V

    @classmethod
    def _transform(cls, Y):
        p = Y / numpy.sum(Y, axis = 0)
        with numpy.errstate(divide = 'ignore'):
            q = (numpy.log(p[0 : -1] + cls._alpha)
                 - numpy.log(p[-1] + cls._alpha))
        # Fix up NaNs.
        q[numpy.isnan(q)] = - numpy.inf
        return q

    @classmethod
    def _transform_inverse(cls, q):
        # First dimension is one larger than q.
        q_shape = numpy.shape(q)
        p_shape = (q_shape[0] + 1, ) + q_shape[1 : ]
        p = numpy.zeros(p_shape)
        # Silence warnings.
        with numpy.errstate(over = 'ignore'):
            for j in range(len(q)):
                p[j] = (1 / (numpy.exp(- q[j])
                             + numpy.sum(numpy.exp(q - q[j]), axis=0))
                        - cls._alpha)
            p[-1] = (1 / (1 + numpy.sum(numpy.exp(q), axis = 0))
                     - cls._alpha)
        # Fix up NaNs.
        p[ : -1][numpy.isposinf(q)] = 1
        p[ : -1][numpy.isneginf(q)] = 0
        p[numpy.isnan(p)] = 0
        return p
