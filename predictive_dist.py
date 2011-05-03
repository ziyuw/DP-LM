# -*- coding: utf-8 -*-
from numpy import *
from numpy.linalg import *
from math import *

from utility import *

#function probs = predictive_dist_x(x, z, z_n_plus_1, tau3, tau4, alpha0)

def predictive_dist_x(x, particle_dict, z_n_plus_1, tau3, tau4, alpha0, d):
    # returns an array of probabilites
    # x is row vector of size N that is to say x does not include x_n_plus_1
    # z is the design matrix. Each row of z is a input point.
    # z_n_plus_1 is a column vector of dimension (d, 1)
    # particle_dict contains key, value pairs where keys are table numbers values are:
    #   number of customers, z_sum, Zc'*Zc,and Zc'*Yc defined as a tuple
    #   Where z_sum and Zc'*Yc are column vectors, and Zc'*Zc is a (d, d) matrix
    
    # MUST MAKE THINGS MATRIX!!!!

    I = mat(eye(d))
    
    if size(x) == 0:
	# table number is zero indexed
	next = array([0])
    else:
	next = array([x.max()]) + 1
    
    possible_values = concatenate((unique(x), next), 2)
    
    # returned probabilites in ROW vector
    probs = zeros((possible_values.shape[0]))
    
    counter = 0
    for x_n_plus_1 in possible_values:
	
	if x_n_plus_1 in particle_dict:
	    n = float(particle_dict[x_n_plus_1][0])
	else:
	    n = 0.0

	predictive_density = 0
	
	if n == 0.0:
	    arb_theta = zeros((d, 1))
	    mu = arb_theta
	    
	    # likelihood * prior / posterior
	    likelihood = mvnpdf(z_n_plus_1, arb_theta, I/tau4) # likelihood
	    prior = mvnpdf(arb_theta, mu, I/tau3) # prior
	
	    vc = linalg.inv(tau3*I + tau4*I); wc = dot(tau4*vc, z_n_plus_1) # posterior mean and variance
	    
	    posterior = mvnpdf(arb_theta, wc, vc); # posterior
	    
	    if posterior == 0.0 and likelihood == 0.0:
		predictive_density = prior
	    else:
		predictive_density = likelihood*prior/posterior
	    
	    #print predictive_density
	else:
	    zc_sum = particle_dict[x_n_plus_1][1]
	
	    vc = inv(tau3*I + n*tau4*I) # posterior variance
	    wc = dot(tau4*vc, zc_sum) # posterior (predictive) mean
	    vc_p = vc + I/tau4 # variance of the predictive dist.
	
	    predictive_density = mvnpdf(z_n_plus_1, wc, vc_p);
	    
	    #print vc, vc_p, z_n_plus_1, 'mean:',wc, predictive_density, "Look here"
	    #print predictive_density, "lower"
	
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
	sigma = 1/tau + 1/tau2*dot(z_n_plus_1.T, z_n_plus_1)
	#print sigma, z_n_plus_1, "UPPER"
	return mu, sigma
    else:
	
	vc = inv(tau2*I + tau*mat(particle_dict[x_n_plus_1][2])) # posterior variance
	wc = tau*dot(vc, particle_dict[x_n_plus_1][3]) # posterior mean

	mu = dot(z_n_plus_1.T, wc) # predictive mean
	sigma = 1/tau + dot(dot(z_n_plus_1.T, vc), z_n_plus_1) # predictive variance

	#print float(sigma), z_n_plus_1, "LOWER"

	return mu, sigma
