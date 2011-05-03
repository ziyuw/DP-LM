# -*- coding: utf-8 -*-
from numpy import *
from numpy.linalg import *
from math import *


def mvnpdf(x, mu, sigma):
    # mu is an array
    # sigma is a matrix or a square array
    # x is an array
    return exp(-dot(dot( mat(x-mu).T, inv(mat(sigma)) ), mat(x-mu))/2)/sqrt(det(2*pi*sigma))

def CRP(x, x_n_plus_1, alpha0):
    # Returns the probability returned by the CRP
    # x is a row vector

    l = float(x.shape[0])
    size_contained = float(nonzero(x == x_n_plus_1)[0].shape[0])

    if size_contained == 0:
	size_contained = alpha0
	#print x_n_plus_1
	#print size_contained/(l+alpha0)

    prob = size_contained/(l+alpha0)

    return prob

def normpdf(x, mu, sigma):
    # mu is number
    # sigma is a positive number
    # x is a number
    return exp(-pow((x-mu),2)/(2*sigma))/sqrt(2*pi*sigma)