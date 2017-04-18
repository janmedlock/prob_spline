#!/usr/bin/python3
'''
Test the module with an example.
'''

from matplotlib import pyplot
import numpy
import scipy.stats
import seaborn

import prob_spline
import test_common


npoints = 20

numpy.random.seed(2)

# Plot mu
x = numpy.linspace(test_common.x_min, test_common.x_max, 1001)
handles = []  # To control the order in the legend.
l = pyplot.plot(x, test_common.mu(x),
                color = 'black', linestyle = 'dotted',
                label = '$\mu(x)$')
handles.append(l[0])

# Get Poisson samples around mu(x) and plot.
X_pad = (test_common.x_max - test_common.x_min) / 2 / npoints
X_min = test_common.x_min + X_pad
X_max = test_common.x_max - X_pad
X = numpy.linspace(X_min, X_max, npoints)
Y = scipy.stats.poisson.rvs(test_common.mu(X))
s = pyplot.scatter(X, Y,
                   s = 30, color = 'black', zorder = 3,
                   label = 'Poisson($\mu(x)$) samples')
handles.append(s)

# Build a spline using the normal loglikelihood.
normal_spline = prob_spline.NormalSpline(sigma = 0)
normal_spline.fit(X, Y)
l = pyplot.plot(x, normal_spline(x),
                label = 'Fitted NormalSpline($\sigma =$ {:g})'.format(
                    normal_spline.sigma))
handles.append(l[0])

# Build a spline using the Poisson loglikelihood.
poisson_spline = prob_spline.PoissonSpline(sigma = 0)
poisson_spline.fit(X, Y)
l = pyplot.plot(x, poisson_spline(x),
                label = 'Fitted PoissonSpline($\sigma =$ {:g})'.format(
                    poisson_spline.sigma))
handles.append(l[0])

# Add decorations to plot.
pyplot.xlabel('$x$')
pyplot.legend(handles, [h.get_label() for h in handles])
pyplot.show()
