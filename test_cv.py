#!/usr/bin/python3
'''
Test cross-validating the splines with an example.
'''

from matplotlib import pyplot
import numpy
import scipy.stats
import seaborn
import sklearn.model_selection

import prob_spline
import test_common


npoints = 21


# Plot mu
x = numpy.linspace(test_common.x_min, test_common.x_max, 1001)
handles = []  # To control the order in the legend.
l = pyplot.plot(x, test_common.mu(x), color = 'black', linestyle = 'dotted',
                label = '$\mu(x)$')
handles.append(l[0])

# Get Poisson samples around mu(x).
X = numpy.linspace(test_common.x_min, test_common.x_max, npoints)
Y = scipy.stats.poisson.rvs(test_common.mu(X))
s = pyplot.scatter(X, Y, color = 'black',
                   label = 'Poisson($\mu(x)$) samples')
handles.append(s)

spline = prob_spline.PoissonSpline()
# Find the best sigma value by K-fold cross-validation.
kfold = sklearn.model_selection.KFold(n_splits = 5, shuffle = True)
param_grid = dict(sigma = numpy.logspace(-6, 6, 25))
gridsearch = sklearn.model_selection.GridSearchCV(spline,
                                                  param_grid,
                                                  cv = kfold,
                                                  error_score = 0)
gridsearch.fit(X, Y)
spline = gridsearch.best_estimator_
l = pyplot.plot(x, spline(x),
                label = 'Fitted {}($\sigma =$ {:g})'.format(
                    spline.__class__.__name__,
                    spline.sigma))
handles.append(l[0])

# Add decorations to plot.
pyplot.xlabel('$x$')
pyplot.legend(handles, [h.get_label() for h in handles])
pyplot.show()
