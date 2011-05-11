# -*- coding: utf-8 -*-
from numpy import *
from numpy.linalg import *
from math import *
from random import *
from numpy.random import *
from copy import *

from predictive_dist import *
from utility import *


class ParticleFilter:
    def __init__(self, P, tau, tau2, tau4, vu, k, alpha0, d):
	# P is the number of particles
	# d is dimension of the input points
	
	self.num_particles = P
        self.particles = []
        self.tau = tau
        self.tau2 = tau2
        self.vu = vu
        self.k = k
        self.tau4 = tau4
        self.alpha0 = alpha0
        self.d = d
        
        self.init_particles() # creates N particles
        
    def init_particles(self):
	for i in range(self.num_particles):
	    self.particles.append(Particle(self.num_particles))
	
    def sample_new_point(self, z_n_plus_1):
	# Sample x_n_plus_1 according to proposal distribution
	
	for particle in self.particles:
	    
	    #print particle.x
	    
	    prob_vec = predictive_dist_x(particle.x, particle.particle_dict, z_n_plus_1, self.k, self.vu, self.tau4, self.alpha0, self.d)
	    particle.z_n_plus_1_prob = sum(prob_vec)
	    
	    p = prob_vec/particle.z_n_plus_1_prob
	    
	    #print p, sum(p)
	    
	    if particle.z_n_plus_1_prob == 0.0:
		prob_vec = array([1 for p in prob_vec])
		p = prob_vec/sum(prob_vec)
	    
	    particle.x_n_plus_1 = nonzero(multinomial(1, p) == 1)[0][0]
    
    def predict_new_point_value(self, z_n_plus_1):
	# Sample x_n_plus_1 according to proposal distribution
	
	total_mu = 0; total_var = 0; Mus = []
	
	for particle in self.particles:
	    prob_vec = predictive_dist_x(particle.x, particle.particle_dict, z_n_plus_1, self.k, self.vu, self.tau4, self.alpha0, self.d)
	    z_n_plus_1_prob = sum(prob_vec)
	    
	    if z_n_plus_1_prob == 0.0:
		prob_vec = array([1 for p in prob_vec])
		z_n_plus_1_prob = sum(prob_vec)
		
	    prob_vec = prob_vec/z_n_plus_1_prob
	    
	    #print prob_vec, z_n_plus_1
	    
	    mean_mu = 0
	    
	    
	    for x_n_plus_1 in range(size(prob_vec)):
		(mu, sigma) = predictive_dist_y(x_n_plus_1, z_n_plus_1, particle.particle_dict, self.tau, self.tau2, self.d)
		#if abs(z_n_plus_1) <= 0.0001:
			#print mu, sigma, "Look here"
		mean_mu = mean_mu + particle.w*prob_vec[x_n_plus_1]*float(mu)
		total_var = total_var + particle.w*prob_vec[x_n_plus_1]*float(sigma)
		#print float(sigma), particle.w, prob_vec[x_n_plus_1]
	    total_mu = total_mu + mean_mu
	    Mus.append(float(mean_mu)*self.num_particles)
	
	#print 'total variance:', total_var, total_mu
	#print Mus
	temp_var = 0
	for mu in Mus:
	    temp_var = temp_var + pow((mu-total_mu), 2)
    
	total_var = total_var + temp_var/size(Mus)
	return total_mu, total_var
    
    def calculate_Us(self):
	# Calculate Us according to Systematic Resampling
	# Returns a list (Python not numpy) of Us
	
	Us = []
	u_1 = random.random()/self.num_particles
	Us.append(u_1)
	
	for i in range(1, self.num_particles):
	    Us.append(u_1 + float(i)/self.num_particles)
	
	return Us
	
    def calculate_Wk(self):
	# Returns a list of Ws
	Ws = []
	
	total_weight = 0;
	for particle in self.particles:
	    #print particle.w
	    total_weight = total_weight + particle.w
	    Ws.append(total_weight)
	
	if total_weight == 0.0:
	    Ws = [i+1/self.num_particles for i in range(self.num_particles)]
	else:
	    Ws = [w/total_weight for w in Ws]
	
	return Ws

    def resample(self, z_n_plus_1, y_n_plus_1):
	# Systematic Resampling is used.
	# returns void
	
	# update and reweight particles
	for particle in self.particles:
	    particle.update_particle(z_n_plus_1, y_n_plus_1, self.tau, self.tau2, self.d)

	Ws = self.calculate_Wk()
	Us = self.calculate_Us()
	
	Ns = []; index = 0; N_i = 0; i = 0
	
	#for i in range(size(Us)):
	    #print Ws[i], Us[i]
	
	i  = 0
	while i in range(0, size(Us)):
	    #print i, Us[i], Ws[index]
	    if Us[i] <= Ws[index]:
		N_i = N_i + 1
		i = i + 1
		
		if i == size(Us):
		    Ns.append(N_i)
	    else:
		Ns.append(N_i)

		index = index + 1
		N_i = 0

	
	Ns.extend([0 for i in range(self.num_particles - size(Ns))])
	
	zero_entries = [i for i in range(size(Ns)) if Ns[i] == 0]
	
	# Reassign particles sampled many times to empty spots
	index = 0
	for i in range(size(self.particles)):
	    if Ns[i] > 1:
		N_i = Ns[i] - 1
		for j in range(N_i):
		    self.particles[zero_entries[index]] = Particle(self.num_particles, self.particles[i]) # deep copy particles
		    index = index + 1
	
	# assign 1/P weight to all new particles
	for particle in self.particles:
	    particle.w = 1.0/self.num_particles
	

class Particle:
    # particle_dict contains key, value pairs where keys are table numbers values are:
    #   number of customers, z_sum, Zc'*Zc,and Zc'*Yc defined as a tuple
    #   Where z_sum and Zc'*Yc are column vectors, and Zc'*Zc is a (d, d) matrix
    
    def __init__(self, num_particles, particle = None):
	
	self.num_particles = num_particles
	if particle == None:
	    self.x = array([])
	    self.x_n_plus_1 = -1 # remember x_n_plus_1 temporarily
	    self.z_n_plus_1_prob = -1
	    self.w = 1.0/self.num_particles
	    self.particle_dict = {}
	else:
	    # Deep copy particle
	    self.x = copy(particle.x)
	    self.x_n_plus_1 = particle.x_n_plus_1 # remember x_n_plus_1 temporarily
	    self.w = particle.w
	    self.particle_dict = deepcopy(particle.particle_dict)
	
    def update_particle(self, z_n_plus_1, y_n_plus_1, tau, tau2, d):
	# z_n_plus_1 is a column vector of dimension (d, 1)

	# Update x
	self.x = concatenate((self.x, array([self.x_n_plus_1]) ), 2)
	
	# Update w
	(mu, sigma) = predictive_dist_y(self.x_n_plus_1, z_n_plus_1, self.particle_dict, tau, tau2, d)
	alpha_n = normpdf(y_n_plus_1, mu, sigma)*self.z_n_plus_1_prob
	
	#print mu, sigma, normpdf(y_n_plus_1, mu, sigma), self.z_n_plus_1_prob
	
	#print alpha_n	
	self.w = self.w*alpha_n
	
	#print normpdf(y_n_plus_1, mu, sigma), self.z_n_plus_1_prob, sigma, y_n_plus_1, mu, self.w
	
	# Update dictionary
	if self.x_n_plus_1 not in self.particle_dict:
	    num_customer = 1
	    z_sum = z_n_plus_1
	    zcpzc = dot(mat(z_n_plus_1), mat(z_n_plus_1).T)
	    zcyc = y_n_plus_1*z_n_plus_1
	    self.particle_dict[self.x_n_plus_1] = [num_customer, z_sum, zcpzc, zcyc]
	else:
	    self.particle_dict[self.x_n_plus_1][0] = self.particle_dict[self.x_n_plus_1][0] + 1
	    self.particle_dict[self.x_n_plus_1][1] = self.particle_dict[self.x_n_plus_1][1] + z_n_plus_1
	    self.particle_dict[self.x_n_plus_1][2] = self.particle_dict[self.x_n_plus_1][2] + dot(mat(z_n_plus_1), mat(z_n_plus_1).T)
	    self.particle_dict[self.x_n_plus_1][3] = self.particle_dict[self.x_n_plus_1][3] + y_n_plus_1*z_n_plus_1
