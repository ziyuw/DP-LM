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

d = 1

# the number of particles
P = 10
tau = 500.0
tau2 = 0.01
tau4 = 0.05

vu = float(d) + 2.0
k = 0.00001 # how to tune this parameter?

alpha0 = 10.0

pf = ParticleFilter(P, tau, tau2, tau4, vu, k, alpha0, d)

pts = arange(2.0, 8.0, 0.1)
y = 10*numpy.cos(pts*4)
#y = 5*pts*pts*pts - 10*pts*pts

sample_pts = []
sample_value = []

for i in range(10):
    values = [objective(pt, pf, i) for pt in pts]
    pt_index = values.index(max(values))
    print pts[pt_index], y[pt_index]
    sample_pts.append(pts[pt_index])
    sample_value.append(y[pt_index])
    pf.sample_new_point(pts[pt_index])
    pf.resample(pts[pt_index], y[pt_index]) # Resample

test_range = arange(2.0, 8.2, 0.1)

predictions = []
upper = []
lower = []
for pt in test_range:
    #mu, sigma = predict(pf, pt)
    mu, sigma = pf.predict_new_point_value(pt)
    #print sigma
    predictions.append(mu)
    upper.append(mu+2*sqrt(sigma))
    lower.append(mu-2*sqrt(sigma))

# Plotting
line1 = plot(pts, y)
line2 = plot(test_range, predictions, '-r*')
line3 = plot(test_range, upper, '--')
line4 = plot(test_range, lower, '-.k')
line5 = plot(sample_pts, sample_value, 'bo')

figlegend( (line1, line2, line3, line4, line5),
           ('True Function', 'Predictions (After Opt.)', 'Upper Confidence Bound', 'Lower Confidence Bound', 'Sampled points (During Opt.)'),
           'lower right' )
show()