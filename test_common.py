import numpy
import scipy.stats


def mu(x):
    '''
    An example parameter function to test the splines.
    '''
    return (12 * scipy.stats.norm.pdf(x, loc = 2.5, scale = 0.5)
            + 50 * scipy.stats.norm.pdf(x, loc = 5.5, scale = 1))

# Useful limits for this function.
x_min = 0
x_max = 10
