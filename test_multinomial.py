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

def mu(x):
    v = numpy.row_stack((numpy.cos(numpy.pi * x) ** 4,
                         numpy.sin(numpy.pi * x) ** 4))
    v = numpy.row_stack((v, 1 - numpy.sum(v, axis = 0)))
    if numpy.isscalar(x):
        v = numpy.squeeze(v, axis = -1)
    return v


# Plot mus
x = numpy.linspace(0, 1, 1001)
mu_ = mu(x)
fig, axes = pyplot.subplots(mu_.shape[0], 1, sharex = 'col')
handles = []  # To control the order in the legend.
for (axes_i, mu_i) in zip(axes, mu_):
    l = axes_i.plot(x, mu_i,
                    color = 'black', linestyle = 'dotted',
                    label = '$\mu(x)$')
handles.append(l[0])

# Get Poisson samples around mu(x) and plot.
X_pad = 1 / 2 / npoints
X = numpy.linspace(X_pad, 1 - X_pad, npoints)
# Y = scipy.stats.multinomial.rvs(nsamples, mu(X))
Y = numpy.column_stack([numpy.random.multinomial(nsamples, mu(x))
                        for x in X])
for (axes_i, Y_i) in zip(axes, Y):
    s = axes_i.scatter(X, Y_i / nsamples,
                       s = 30, color = 'black', zorder = 3,
                       label = 'Multinomial(${}$, $\mu(x)$) samples'.format(
                           nsamples))
handles.append(s)

# Build an interpolating spline using the multinomial loglikelihood.
spline_I = prob_spline.MultinomialSpline(sigma = 0)
spline_I.fit(X, Y)
y = spline_I(x)
for (axes_i, y_i) in zip(axes, y):
    l = axes_i.plot(x, y_i,
                    label = 'Fitted MultinomialSpline($\sigma =$ {:g})'.format(
                        spline_I.sigma))
handles.append(l[0])

# Build a smoothing spline using the multinomial loglikelihood.
spline_S = prob_spline.MultinomialSpline(sigma = 1)
spline_S.fit(X, Y)
y = spline_S(x)
for (axes_i, y_i) in zip(axes, y):
    l = axes_i.plot(x, y_i,
                    label = 'Fitted MultinomialSpline($\sigma =$ {:g})'.format(
                        spline_S.sigma))
handles.append(l[0])

# Add decorations to plot.
axes[-1].set_xlabel('$x$')
axes[0].legend(handles, [h.get_label() for h in handles])
pyplot.show()
