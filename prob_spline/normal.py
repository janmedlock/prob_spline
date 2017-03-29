import scipy.stats

from . import base


class NormalSpline(base.ProbSpline):
    @staticmethod
    def _loglikelihood(Y, mu):
        '''
        This is the squared l_2 distance,
        so we get a traditional spline.
        '''
        return scipy.stats.norm.logpdf(Y, mu)
