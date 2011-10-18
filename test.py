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

d = 1

# the number of particles
P = 10
tau = 500.0
tau2 = 0.0001
tau4 = 0.01

vu = float(d) + 2.0
k = 0.0001 # how to tune this parameter?

alpha0 = 10.0

pf = ParticleFilter(P, tau, tau2, tau4, vu, k, alpha0, d)

seq = arange(2.0, 7.0, 0.1)
#y_value = -(0.5*seq*seq*seq - 0.1*seq*seq)
y_value = 10*numpy.cos(seq*4)

pts = (seq.tolist())
#shuffle(pts)
pts = array(pts)

print pts

#y = -(0.5*pts*pts*pts - 0.10*pts*pts)
y = 10*numpy.cos(pts*4)

for i in range(size(pts)):
    # Sample new points
    pf.sample_new_point(pts[i])
    # Resample
    pf.resample(pts[i], y[i])

print "Training finished."

# Testing
test_range = arange(2.0, 6.0, 0.1)
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


line1 = plot(seq, y_value)
line2 = plot(test_range, predictions, '-ro')
line3 = plot(test_range, upper)
plot(test_range, lower)
figlegend( (line1, line2, line3),
           ('label1', 'label2', 'label3'),
           'upper right' )
show()