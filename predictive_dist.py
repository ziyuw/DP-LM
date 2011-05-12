# -*- coding: utf-8 -*-
from numpy import *
from numpy.linalg import *
from math import *
import scipy.special
from matplotlib.pyplot import *

from utility import *

#function probs = predictive_dist_x(x, z, z_n_plus_1, tau3, tau4, alpha0)

def predictive_dist_x(x, particle_dict, z_n_plus_1, k, vu, tau4, alpha0, d):
    # returns an array of probabilites
    # x is row vector of size N that is to say x does not include x_n_plus_1
    # z is the design matrix. Each row of z is a input point.
    # z_n_plus_1 is a column vector of dimension (d, 1)
    # particle_dict contains key, value pairs where keys are table numbers values are:
    #   number of customers, z_sum, Zc'*Zc, and Zc'*Yc defined as a tuple
    #   Where z_sum and Zc'*Yc are column vectors, and Zc'*Zc is a (d, d) matrix
    
    # MUST MAKE THINGS MATRIX!!!!
    I = mat(eye(d))
    
    #print z_n_plus_1, 'z_n_plus_1'
    
    z_n_plus_1 = mat(z_n_plus_1).reshape(d, 1)
    
    
    if size(x) == 0:
	# table number is zero indexed
	next = array([0])
    else:
	next = array([x.max()]) + 1
    
    # all the possible tables to sit at
    possible_values = concatenate((unique(x), next), 2)
    
    # returned probabilites in ROW vector
    probs = zeros((possible_values.shape[0]))
    
    #print (possible_values.shape[0])
    
    counter = 0
    for x_n_plus_1 in possible_values:
	
	if x_n_plus_1 in particle_dict:
	    n = float(particle_dict[x_n_plus_1][0])
	else:
	    n = 0.0

	predictive_density = 0
	
	if n == 0.0:
	    n = 1.0
	    k_n = k + 1.0
	    vu_n = vu + 1.0
	    mean = z_n_plus_1
	    
	    delta_0 = mat(eye((d)))*tau4

	    #delta_n = vu*delta_0 + z_n_plus_1*z_n_plus_1.H  - divide(m_n*m_n.H, k_n)
	    
	    delta_n = z_n_plus_1*z_n_plus_1.H + delta_0 - mean*mean.H*n + float(k*n)*mean*mean.H/k_n
	    
	    #upper = scipy.special.multigammaln([vu_n/2], d)
	    #lower = scipy.special.multigammaln([vu/2], d)
	    
	    upper = multigamma(d, vu_n/2)
	    lower = multigamma(d, vu/2)
	    
	    #print exp(upper - lower)
	    #print pow(det(delta_n/k_n), vu_n/2), pow(det(delta_0/vu), vu/2)
	    
	    predictive_density = pow( divide(1.0, pi)*divide(k, k_n), float(d)/2 ) * divide(upper, lower) * divide(pow(det(delta_0), vu/2), pow(det(delta_n), vu_n/2))

	    #print predictive_density, x_n_plus_1, z_n_plus_1

	else:
	    k_n = k + n
	    
	    vu_n = vu + n
	    
	    mean = divide(mat(particle_dict[x_n_plus_1][1]), n)
	    
	    delta_0 = mat(eye((d)))*tau4
	    delta_n = delta_0 + particle_dict[x_n_plus_1][2] + (float(k*n)/k_n-n)*mean*mean.H
	    delta = delta_n*(k_n + 1)/(k_n*(vu_n - d + 1))
	    
	    predictive_density = multitpdf(z_n_plus_1, vu_n - d + 1, mean*n/k_n, delta, d)
	    
	    
	    #print delta, delta_n, particle_dict[x_n_plus_1][2], mean*mean.H*n, float(k*n)*mean*mean.H/k_n
	    #print x
	    #print delta_0
	    #if n > 10:
		#print delta, particle_dict[x_n_plus_1][2], mean*mean.H*n, float(k*n)*mean*mean.H/k_n
		#print x, "here"
		#print x_n_plus_1
		#myrange = arange(-1.0, 7.0, 0.2)
		#ys = [float(multitpdf(myx, vu_n - d + 1, mean, delta, d)) for myx in myrange]
		#figure1 = plot(myrange, ys)
		#show()

	#print CRP(x, x_n_plus_1, alpha0), x_n_plus_1
	probs[counter] = CRP(x, x_n_plus_1, alpha0)*predictive_density
	counter = counter + 1
    #print probs/sum(probs)
    return probs


def predictive_dist_y(x_n_plus_1, z_n_plus_1, particle_dict, tau, tau2, d):
    # x is row vector of size N
    # y is a column vector of size N
    # z is the design matrix of size (N, d). Each row of z is a input point.
    # z_n_plus_1 is a column vector of dimension (d, 1)
    
    # This method returns the mean and variance of the predictive distribution of y
    
    I = mat(eye(d)) # identity matrix
    
    if x_n_plus_1 in particle_dict:
	n = particle_dict[x_n_plus_1][0] # the number of customers
    else:
	n = 0
    
    if n == 0:
	mu = 0;
	sigma = 1/tau + divide(dot(z_n_plus_1.T, z_n_plus_1), tau2)
	
	#print sigma, z_n_plus_1, "UPPER"
	
	#if sigma > 5:
	    #print sigma, 1/tau, 'look here 3'
	
	return mu, sigma
    else:
	vc = inv(tau2*I + tau*mat(particle_dict[x_n_plus_1][2])) # posterior variance
	wc = tau*dot(vc, particle_dict[x_n_plus_1][3]) # posterior mean
	
	#print particle_dict[x_n_plus_1][3], "here"
	#print vc

	mu = dot(z_n_plus_1.T, wc) # predictive mean
	sigma = 1/tau + dot(dot(z_n_plus_1.T, vc), z_n_plus_1) # predictive variance

	return mu, sigma
