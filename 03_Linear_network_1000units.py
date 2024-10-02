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
num_species = 1000
degree      = 5
num_subnw   = 900
num_bulk    = 100 #Convention: in the vector of conc. the subnetwork species appear first
parameters  = np.ones(num_species)*1

nw = nx.gnm_random_graph(num_species,num_species*degree,directed=True)
#IMPORT a networkx graph here instead!

boundary_list = find_boundaries(nw,np.arange(num_subnw,num_species))

print('boundary size = ', len(boundary_list))

def dynamcical_system_linear(t,x):
    y    = np.zeros(nw.number_of_nodes())
    
    for i in range(len(y)):
        in_edges = [[p for p in nw.in_edges(i)][n][0] for n in range(len([m for m in nw.in_edges(i)]))]
        for j in in_edges:
            y[i] += parameters[i]*(x[j] - x[i])
        in_edges = []
    return y

#Simulation parameters
Num_runs = 1000
deltaT      = 0.01
T           = 1.

soln      = np.zeros([int(T/deltaT)*Num_runs,num_species])
obs_noise = np.zeros([int(T/deltaT)*Num_runs,num_species])

for i in range(Num_runs):
    
    init_cond = 10*((np.random.rand(num_species)*0.2) - 0.1)

    soln_temp = sc.integrate.solve_ivp(dynamcical_system_linear,(0,T),init_cond,t_eval=np.arange(0,T,deltaT)).y.T

    soln[int(T/deltaT)*i:int(T/deltaT)*(i+1),:]    = soln_temp

    for j in range(num_subnw):        
        obs_noise[int(T/deltaT)*i:int(T/deltaT)*(i+1),j]   = np.gradient((soln_temp[:,j]),deltaT)
    
    print("Num run = ", i)


from sklearn import linear_model

#auc_tol = 0.05

alpha_opt            = np.zeros(num_subnw)
beta_opt             = np.zeros(num_subnw)

alpha_error          = np.zeros(num_subnw)
beta_error           = np.zeros(num_subnw)

ordered_beta         = np.zeros(num_subnw)
ordered_beta_error   = np.zeros(num_subnw)
#auc_list             = []
    
d1  = utilities.design_matrix_linear(soln[:int(T/deltaT)*Num_runs,0:num_subnw])

ridge_model = linear_model.BayesianRidge(alpha_init=1.,lambda_init=0.1,max_iter=100,fit_intercept=False,tol=1e-5,compute_score=True)

for i in range(num_subnw):

    ridge_model.fit(X=d1,y=obs_noise[:int(T/deltaT)*Num_runs,i])
    alpha_opt[i] = 1./ridge_model.lambda_
    beta_opt[i]  = ridge_model.alpha_

    temp           = second_der_alpha_beta([alpha_opt[i],beta_opt[i]],d1,num_subnw,obs_noise[:,i])*len(obs_noise[:,i])
    
    alpha_error[i] = np.abs(temp[1]/(temp[0]*temp[1]-temp[2]**2))**0.5
    beta_error[i]  = np.abs(temp[0]/(temp[0]*temp[1]-temp[2]**2))**0.5

    print("optimised node = ", i)

#ordered_beta_log[:] = np.log(np.append(beta_opt[boundary_list],beta_opt[[i for i in range(num_subnw) if i not in boundary_list]]))

ordered_beta[:] = np.append(beta_opt[boundary_list],beta_opt[[i for i in range(num_subnw) if i not in boundary_list]])
ordered_beta_error = np.append(beta_error[boundary_list],beta_error[[i for i in range(num_subnw) if i not in boundary_list]])

temp_list = np.copy(ordered_beta[:])
temp_list = np.expand_dims(temp_list,axis=0)

ordered_beta_error = np.expand_dims(ordered_beta_error,axis=0)

auc_list = utilities.auc_roc(temp_list, beta_error = ordered_beta_error, num_threshold=200, num_bdry=len(boundary_list), method='single', plot=False,threshold_scale='log')

print("auc = ", auc_list)

saving = {}
saving["auc_list"]   = auc_list
saving["T"]          = T
saving["deltaT"]     = deltaT
saving["beta"]       = beta_opt
saving["ordered_beta"] = ordered_beta
saving["ordered_beta_error"] = ordered_beta_error
saving["alpha"]      = alpha_opt
saving["network"]    = nw
saving["boundary_list"]    = boundary_list
np.save("data/linear_network_1000units_2",saving)
    
    
        