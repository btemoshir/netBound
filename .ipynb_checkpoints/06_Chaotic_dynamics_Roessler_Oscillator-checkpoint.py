import numpy as np
import scipy as sc
#import matplotlib.pyplot as plt
from scipy.integrate import odeint
import sys
import os
sys.path.append('packages/')
from utilities import *
#from importlib import reload
#from utilities reload *
import utilities
import networkx as nx
#import sdeint

#Network Parameters:
num_species = 100
degree      = 5
num_subnw   = 90
num_bulk    = 10 #Convention: in the vector of conc. the subnetwork species appear first
parameters  = np.ones(num_species)*1

nw = nx.gnm_random_graph(num_species,num_species*degree,directed=True)
#IMPORT a networkx graph here instead!

boundary_list = find_boundaries(nw,np.arange(num_subnw,num_species))

print('boundary size = ', len(boundary_list))

in_degree_list = np.array([nw.in_degree[i] for i in range(num_species)])

in_edges_list = []
for i in range(num_species):
    in_edges_list.append([[p for p in nw.in_edges(i)][n][0] for n in range(len([m for m in nw.in_edges(i)]))])

def dynamcical_system_RO_fast(t,x,in_degree_list,in_edges_list,Ω,num_osc=3):
    
    #Defines the Roessler oscillator system, num_osc defines the number of oscillators at each node!
    #Omega is the vector with alpha,beta and gamma
    
    x = x.reshape(num_species,num_osc)
    
    α = Ω[0] #+ ((np.random.rand(num_species)*0.2)-0.1)

    β = Ω[1] #+ 10.*((np.random.rand(num_species)*0.2)-0.1)
    
    γ = Ω[2]
    #Introduce a scaling for the strength of the interaction
    δ = Ω[3] #+ np.zeros(num_species)
    
    #y = np.zeros((nw.number_of_nodes(),num_osc))

    y = np.zeros(np.shape(x))
    
    y[:,0] = -x[:,1] - x[:,2]
    
    y[:,1] = x[:,0] + (α - δ*in_degree_list)*x[:,1]
    #y[:,1] = x[:,0] + α*x[:,1]

    y[:,2] = β + (x[:,0] - γ)*x[:,2]

    
    for i in range(len(y)):
        #in_edges = [[p for p in nw.in_edges(i)][n][0] for n in range(len([m for m in nw.in_edges(i)]))]
        #print(in_edges)
        for j in in_edges_list[i]:
            #y[i,1] += c[i][j]*x[j,1]
            y[i,1] += δ[j]*x[j,1]
        in_edges = []
    
    return np.ravel(y)

def dynamcical_system_RO(t,x,nw,c,Ω,num_osc=3):
    
    #Defines the Roessler oscillator system, num_osc defines the number of oscillators at each node!
    #Omega is the vector with alpha,beta and gamma
    
    x = x.reshape(num_species,num_osc)
    
    α = Ω[0] #+ ((np.random.rand(num_species)*0.2)-0.1)

    β = Ω[1] #+ 10.*((np.random.rand(num_species)*0.2)-0.1)
    
    γ = Ω[2]
    
    #Introduce a scaling for the strength of the interaction
    δ = Ω[3] 
    
    y = np.zeros((nw.number_of_nodes(),num_osc))
    
    y[:,0] = -x[:,1] - x[:,2]
    
    y[:,1] = x[:,0] + (α - δ*np.array([nw.in_degree[i] for i in range(num_species)]))*x[:,1]

    y[:,2] = β + (x[:,0] - γ)*x[:,2]
    
    for i in range(len(y)):
        in_edges = [[p for p in nw.in_edges(i)][n][0] for n in range(len([m for m in nw.in_edges(i)]))]
        #print(in_edges)
        for j in in_edges:
            #y[i,1] += c[i][j]*x[j,1]
            y[i,1] += δ[j]*x[j,1]
        in_edges = []
    
    return np.ravel(y)

δ = 0.01*np.ones(num_species) #Defines the coupling strength!!
α = 0.2*np.ones(num_species)

Omega_chaotic_attractor = [α,0.2,5.7,δ]
init_chaotic_attractor  = [-6, 0., 0.17]
init_cond = init_chaotic_attractor

#Simulation parameters
Num_osc     = 3
deltaT      = 0.1
T           = 10**6.
Num_runs    = 1

parameters = np.ones((num_species,num_species))

#obs       = np.zeros([int(T/deltaT)*Num_runs,num_species*Num_osc])
soln      = np.zeros([int(T/deltaT)*Num_runs,num_species*Num_osc])
obs_noise = np.zeros([int(T/deltaT)*Num_runs,num_species*Num_osc])

for i in range(Num_runs):

    #init_cond = np.ravel(init_chaotic_attractor*np.ones((num_species,3))) + 10.*np.ravel((np.random.rand(num_species,3)*0.2)-0.1)

    init_cond = 10.*np.ravel((np.random.rand(num_species,3)*0.2)-0.1)
    
    #soln_temp = (sc.integrate.solve_ivp(dynamcical_system_RO,(0,T),init_cond,t_eval=np.arange(0,T,deltaT),args=(nw,parameters, Omega_chaotic_attractor,))).y.T

    soln_temp = (sc.integrate.solve_ivp(dynamcical_system_RO_fast,(0,T),init_cond,\
                                            t_eval=np.arange(0,T,deltaT),args=(in_degree_list,in_edges_list,Omega_chaotic_attractor,))).y.T

    soln[int(T/deltaT)*i:int(T/deltaT)*(i+1),:]    = soln_temp #- soln_ref

    for j in range(num_subnw*Num_osc):
        obs_noise[int(T/deltaT)*i:int(T/deltaT)*(i+1),j]   = np.gradient((soln_temp[:,j]),deltaT)

        #obs[int(T/deltaT)*i:int(T/deltaT)*(i+1),:,s] = [dynamcical_system_RO(0,soln_temp[tt,:],\
        #                                 nw1,parameters,Omega_chaotic_attractor) for tt in range(int(T/deltaT))]


#####

from sklearn import linear_model

alpha_opt        = np.zeros((num_subnw,Num_osc))
beta_opt         = np.zeros((num_subnw,Num_osc))
ordered_beta_log = np.zeros((num_subnw,Num_osc))
#log_likhd_d2    = []

ridge_model = linear_model.BayesianRidge(alpha_init=1.,lambda_init=0.1,max_iter=100,fit_intercept=False,tol=1e-5,compute_score=False)

for i in range(num_subnw):
    
    d2 = utilities.design_matrix_linear(np.append(np.expand_dims(soln[:,0+Num_osc*i],axis=1),soln[:,1:num_subnw*Num_osc:3],axis=1))
    
    #ridge_model.append(linear_model.BayesianRidge(alpha_init=1.,lambda_init=0.1,max_iter=100,fit_intercept=False,tol=1e-5,compute_score=True))

    ridge_model.fit(X=d2,y=obs_noise[:,(Num_osc*i)+1])
    alpha_opt[i,1] = 1./ridge_model.lambda_
    beta_opt[i,1]  = ridge_model.alpha_
    
    #log_likhd_d2.append(ridge_model_d2[-1].scores_)
    #print(ridge_model_d2[-1].scores_)
    
    print("optimised node = ", i)

ordered_beta_log[:,1] = np.log(np.append(beta_opt[boundary_list,1],beta_opt[[i for i in range(num_subnw) if i not in boundary_list],1]))

temp_list = np.copy(ordered_beta_log[:,1])

temp_list = np.expand_dims(temp_list,axis=0)

auc_list = utilities.auc_roc(temp_list, beta_error=np.zeros(np.shape(temp_list)), num_threshold=200, num_bdry=len(boundary_list), method='single', plot=False)

saving = {}
saving["auc_list"]   = auc_list
saving["T"]          = T
saving["deltaT"]     = deltaT
saving["beta"]       = beta_opt
saving["ordered_beta_log"] = ordered_beta_log
saving["alpha"]      = alpha_opt
saving["network"]    = nw
saving["boundary_list"]    = boundary_list
saving["coupling_strength"]    = δ

np.save("data/Roessler_long_trajectoryT1000000",saving)
    
    
        