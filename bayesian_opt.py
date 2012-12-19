# -*- coding: utf-8 -*-
from numpy import *
from numpy.linalg import *
from math import *
from random import *
from numpy.random import *
import copy
import numpy
from matplotlib.pyplot import *

from predictive_dist import *
from utility import *
from particle_filter import *

def objective(x, pf, i):
    mu, sigma = pf.predict_new_point_value(x)
    #print sigma
    return sigma + mu

# the number of particles
P = 10
tau = 2.0
tau2 = 1.0
tau3 = 0.1
tau4 = 500.0

alpha0 = 2.0
d = 3

pf = ParticleFilter(P, tau, tau2, tau3, tau4, alpha0, d)

pts = arange(-3.0, 4.0, 0.1)

originl_pts = pts
y = 10*numpy.cos(pts*4)
#, pt**2, sqrt(5.0**4-pt**4)
pts = [mat([1, pt, sqrt(5.0**2-pt**2)]).T for pt in pts]

#y = 5*pts*pts*pts - 10*pts*pts

sample_pts = []
sample_value = []

for i in range(5):
    values = [objective(pt, pf, i) for pt in pts]
    pt_index = values.index(max(values))
    print pts[pt_index][1], y[pt_index]
    sample_pts.append(float(pts[pt_index][1]))
    sample_value.append(y[pt_index])
    pf.sample_new_point(pts[pt_index])
    pf.resample(pts[pt_index], y[pt_index]) # Resample

test_range = arange(-3.0, 4.2, 0.1)
#, pt**2, sqrt(5.0**4-pt**4)
test_range_2 = [mat([1, pt, sqrt(5.0**2-pt**2)]).T for pt in test_range]

predictions = []
upper = []
lower = []
for pt in test_range_2:
    #mu, sigma = predict(pf, pt)
    mu, sigma = pf.predict_new_point_value(pt)
    #print sigma
    predictions.append(mu)
    upper.append(mu+2*sqrt(sigma))
    lower.append(mu-2*sqrt(sigma))

# Plotting
line1 = plot(originl_pts, y)
line2 = plot(test_range, predictions, '-r*')
line3 = plot(test_range, upper, '--')
line4 = plot(test_range, lower, '-.k')
line5 = plot(sample_pts, sample_value, 'bo')

figlegend( (line1, line2, line3, line4, line5),
           ('True Function', 'Predictions (After Opt.)', 'Upper Confidence Bound', 'Lower Confidence Bound', 'Sampled points (During Opt.)'),
           'lower right' )
show()