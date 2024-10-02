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

def dynamcical_system_linear(t,x):
    y    = np.zeros(nw.number_of_nodes())
    
    for i in range(len(y)):
        in_edges = [[p for p in nw.in_edges(i)][n][0] for n in range(len([m for m in nw.in_edges(i)]))]
        for j in in_edges:
            y[i] += parameters[i]*(x[j] - x[i])
        in_edges = []
    return y

#Simulation parameters
#dt_list = np.round(np.logspace(-4,-1,10),4)
dt_list = np.round(np.logspace(-3,-1,10),4)
print('dt_list = ', dt_list)

num_repeats_list = np.int32(np.ceil(np.logspace(1,2,10)))
print('num_repeats_list = ', num_repeats_list)

deltaT = 0.05
#T = 0.2
#T = 1
T  = 4
Num_runs = num_repeats_list[-1]

soln = []
obs_noise = []

for n in range(len(dt_list)):

    deltaT = dt_list[n]

    soln.append(np.zeros([np.int32(np.ceil((T/deltaT))*Num_runs),num_species]))
    obs_noise.append(np.zeros([np.int32(np.ceil((T/deltaT))*Num_runs),num_species]))

    for i in range(Num_runs):    
        init_cond = 10*((np.random.rand(num_species)*0.2) - 0.1)

        soln_temp = sc.integrate.solve_ivp(dynamcical_system_linear,(0,T),init_cond,t_eval=np.arange(0,T,deltaT)).y.T

        #soln_temp = (sdeint.itoEuler(f=dynamcical_system_linear,G=wiener_linear,y0=init_cond,tspan=np.arange(0,T,deltaT)))

        #soln[n][np.int32(np.ceil(T/deltaT))*i:np.int32(np.ceil(T/deltaT))*(i+1),:]    = soln_temp
        soln[n][len(soln_temp)*i:len(soln_temp)*(i+1),:]    = soln_temp

        for j in range(num_subnw):        
            obs_noise[n][len(soln_temp)*i:len(soln_temp)*(i+1),j]   = np.gradient((soln_temp[:,j]),deltaT)
    print(n)

from sklearn import linear_model

alpha_opt            = np.zeros((len(num_repeats_list),len(dt_list),num_subnw))
beta_opt             = np.zeros((len(num_repeats_list),len(dt_list),num_subnw))
ordered_beta         = np.zeros((len(num_repeats_list),len(dt_list),num_subnw))
ordered_beta_error   = np.zeros((len(num_repeats_list),len(dt_list),num_subnw))
auc_list             = np.zeros((len(num_repeats_list),len(dt_list)))

alpha_error = np.zeros((len(num_repeats_list),len(dt_list),num_subnw))
beta_error  = np.zeros((len(num_repeats_list),len(dt_list),num_subnw))

for n in range(len(dt_list)):

    deltaT = dt_list[n]

    for n_reps in range(len(num_repeats_list)):
    
        d1  = utilities.design_matrix_linear(soln[n][:int(T/deltaT)*num_repeats_list[n_reps],0:num_subnw])
        
        ridge_model = linear_model.BayesianRidge(alpha_init=1.,lambda_init=0.1,max_iter=100,fit_intercept=False,\
                                                       tol=1e-5,compute_score=True)
        
        for i in range(num_subnw):
        
            ridge_model.fit(X=d1,y=obs_noise[n][:int(T/deltaT)*num_repeats_list[n_reps],i])
            alpha_opt[n_reps,n,i] = 1./ridge_model.lambda_
            beta_opt[n_reps,n,i]  = ridge_model.alpha_

            temp           = second_der_alpha_beta([alpha_opt[n_reps,n,i],beta_opt[n_reps,n,i]],d1,\
                           num_subnw,obs_noise[n][:int(T/deltaT)*num_repeats_list[n_reps],i])*len(obs_noise[n][:int(T/deltaT)*num_repeats_list[n_reps],i])

            alpha_error[n_reps,n,i] = np.abs(temp[1]/(temp[0]*temp[1]-temp[2]**2))**0.5
            beta_error[n_reps,n,i]  = np.abs(temp[0]/(temp[0]*temp[1]-temp[2]**2))**0.5
        
        #ordered_beta_log[n_reps,n,:] = np.log(np.append(beta_opt[n_reps,n,boundary_list],beta_opt[n_reps,n,[i for i in range(num_subnw) if i not in boundary_list]]))

        ordered_beta[n_reps,n,:] = (np.append(beta_opt[n_reps,n,boundary_list],beta_opt[n_reps,n,[i for i in range(num_subnw) if i not in boundary_list]]))

        ordered_beta_error[n_reps,n,:] = (np.append(beta_error[n_reps,n,boundary_list],beta_error[n_reps,n,[i for i in range(num_subnw) if i not in boundary_list]]))
        
        temp_list = np.copy(ordered_beta[n_reps,n,:])
        temp_list = np.expand_dims(temp_list,axis=0)

        temp_list_error = np.copy(ordered_beta_error[n_reps,n,:])
        temp_list_error = np.expand_dims(temp_list_error,axis=0)
        
        
        auc_list[n_reps,n] = utilities.auc_roc(temp_list,beta_error=temp_list_error\
                        ,num_threshold=200,num_bdry=len(boundary_list),method='single',plot=False,threshold_scale='log')
        
        print(n,n_reps)

t1_data = {}
t1_data["dt_list"] = dt_list
t1_data["num_reps"] = num_repeats_list
t1_data["network"] = nw
t1_data["network"].nodes()
t1_data["alpha_opt"] = alpha_opt
t1_data["beta_opt"] = beta_opt
t1_data["ordered_beta"] = ordered_beta
t1_data["ordered_beta_error"] = ordered_beta_error
t1_data["auc_list"] = auc_list
#t1_data["soln"] = soln
#t1_data["obs_noise"] = obs_noise

#np.save("data/linear_network_dt_N_withError_T0point2",t1_data)
#np.save("data/linear_network_dt_N_withError_T1",t1_data)
np.save("data/linear_network_dt_N_withError_T4",t1_data)
    
    
if False:
    #Num_runs = 1000
    #deltaT      = 0.01
    #T           = 1.
    
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
    ordered_beta_log     = np.zeros(num_subnw)
    #auc_list             = []
        
    d1  = utilities.design_matrix_linear(soln[:int(T/deltaT)*Num_runs,0:num_subnw])
    
    ridge_model = linear_model.BayesianRidge(alpha_init=1.,lambda_init=0.1,max_iter=100,fit_intercept=False,tol=1e-5,compute_score=True)
    
    for i in range(num_subnw):
    
        ridge_model.fit(X=d1,y=obs_noise[:int(T/deltaT)*Num_runs,i])
        alpha_opt[i] = 1./ridge_model.lambda_
        beta_opt[i]  = ridge_model.alpha_        
    
        print("optimised node = ", i)
    
    ordered_beta_log[:] = np.log(np.append(beta_opt[boundary_list],beta_opt[[i for i in range(num_subnw) if i not in boundary_list]]))
    
    temp_list = np.copy(ordered_beta_log[:])
    
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
    np.save("data/linear_network_1000units_1",saving)
    
    
        