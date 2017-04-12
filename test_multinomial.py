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
nsamples = 10

numpy.random.seed(2)

x_min = 0
x_max = 1

def mu(x):
    v = numpy.row_stack((numpy.cos(2 * numpy.pi * x) ** 4,
                         numpy.sin(2 * numpy.pi * x) ** 4))
    v = numpy.row_stack((v, 1 - numpy.sum(v, axis = 0)))
    if numpy.isscalar(x):
        v = numpy.squeeze(v, axis = -1)
    return v


# Plot mus
x = numpy.linspace(x_min, x_max, 1001)
mu_ = mu(x)
fig, axes = pyplot.subplots(mu_.shape[0], 1, sharex = 'col')
handles = []  # To control the order in the legend.
for (axes_i, mu_i) in zip(axes, mu_):
    l = axes_i.plot(x, mu_i,
                    color = 'black', linestyle = 'dotted',
                    label = '$\mu(x)$')
handles.append(l[0])

# Get Poisson samples around mu(x) and plot.
X = numpy.linspace(x_min, x_max, npoints + 1)
# Y = scipy.stats.multinomial.rvs(nsamples, mu(X))
Y = numpy.column_stack([numpy.random.multinomial(nsamples, mu(x))
                        for x in X])
for (axes_i, Y_i) in zip(axes, Y):
    s = axes_i.scatter(X, Y_i / nsamples,
                       s = 30, color = 'black', zorder = 3,
                       label = 'Multinomial(${}$, $\mu(x)$) samples'.format(
                           nsamples))
handles.append(s)

# Build a spline using the multinomial loglikelihood.
spline = prob_spline.MultinomialSpline(sigma = 0)
spline.fit(X, Y)
y = spline(x)
for (axes_i, y_i) in zip(axes, y):
    l = axes_i.plot(x, y_i,
                    label = 'Fitted MultinomialSpline($\sigma =$ {:g})'.format(
                        spline.sigma))
handles.append(l[0])

# Add decorations to plot.
axes[-1].set_xlabel('$x$')
axes[0].legend(handles, [h.get_label() for h in handles])
pyplot.show()
