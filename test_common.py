import numpy
import scipy.stats


def mu(x):
    '''
    An example parameter function to test the splines.
    '''
    return (12 * scipy.stats.norm.pdf(x, loc = 0.25, scale = 0.05)
            + 50 * scipy.stats.norm.pdf(x, loc = 0.55, scale = 0.1))

# Useful limits for this function.
x_min = 0
x_max = 1
