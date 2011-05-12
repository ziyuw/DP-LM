# -*- coding: utf-8 -*-
from numpy import *
from numpy.linalg import *
from math import *
import scipy.special
from numpy.linalg import *


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
    
def multitpdf(x, df, mu, sigma, d):
    const = divide(scipy.special.gamma(float(df + d)/2), scipy.special.gamma(float(df)/2))
    return const*power(det(sigma)*power(df*pi, d), -0.5)*power(1 + 1.0/df*(x-mu).H*inv(sigma)*(x-mu), -float(df+d)/2)
    
def multigamma(p, a):
    val = power(pi, float(p*(p-1))/4)
    for i in range(p):
	val = val * scipy.special.gamma(a + float(1-i-1)/2)
    return val
    