import numpy as np
import scipy as sc
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import sys
import os
from sklearn import linear_model
import networkx as nx
import sdeint
import matplotlib as mpl

sys.path.append('../packages/')
sys.path.append('packages/')
import utilities
from utilities import *
plt.rcParams.update({'font.size': 16,'font.family':'arial'})
from matplotlib import colors


#Parameters for the network:
num_species = 3
degree      = 1 #Defines an average degree (useful for random graph generation)
num_subnw   = 2 #Num of observed species
num_bulk    = 1 #Num of bulk species; Convention: in the vector of conc. the subnetwork species appear first

nw1 = nx.DiGraph()
nw1.add_nodes_from([0,1,2])
nw1.add_edges_from([(0, 1),(1, 0),(0,2),(2,0)])

boundary_list = find_boundaries(nw1,np.arange(num_subnw,num_species))

def dynamcical_system_linear(t,x):
    y    = np.zeros(nw.number_of_nodes())
    
    for i in range(len(y)):
        in_edges = [[p for p in nw.in_edges(i)][n][0] for n in range(len([m for m in nw.in_edges(i)]))]
        for j in in_edges:
            y[i] += c[i]*(x[j] - x[i])
        in_edges = []
    return y

#Dynamical parameters
deltaT      = 0.01 #Time step
T           = 1.   # Total time for each run
Num_runs    = 10 #Num of exp, $\tilde{N}$
parameters  = np.ones(num_species)*1 # The network couplings
nw,c        = nw1,parameters # Just refedinition of variables
num_repeats = 1000

#Network dynamics
soln      = np.zeros([num_repeats,int(T/deltaT)*Num_runs,num_species]) # Dyamics solution 
obs_noise = np.zeros([num_repeats,int(T/deltaT)*Num_runs,num_species]) # Observation/Time derivatives

for n in range(num_repeats):
    
    for i in range(Num_runs):    
        
        init_cond = 10*((np.random.rand(num_species)*0.2) - 0.1)

        soln_temp = sc.integrate.solve_ivp(dynamcical_system_linear,(0,T),init_cond,t_eval=np.arange(0,T,deltaT)).y.T

        soln[n,int(T/deltaT)*i:int(T/deltaT)*(i+1),:]    = soln_temp

        for j in range(num_subnw):        
            obs_noise[n,int(T/deltaT)*i:int(T/deltaT)*(i+1),j]   = np.gradient((soln_temp[:,j]),deltaT)
    

alpha_opt            = np.zeros((num_repeats,num_subnw))
beta_opt             = np.zeros((num_repeats,num_subnw))
ordered_beta_log     = np.zeros((num_repeats,num_subnw))
auc_list             = np.zeros(num_repeats)
ordered_beta_error   = np.zeros((num_repeats,num_subnw))
alpha_error          = np.zeros((num_repeats,num_subnw))
beta_error           = np.zeros((num_repeats,num_subnw))

for n in range(num_repeats):
    
    d1  = utilities.design_matrix_linear(soln[n,:int(T/deltaT)*Num_runs,0:num_subnw])
    
    ridge_model = linear_model.BayesianRidge(alpha_init=1.,lambda_init=0.1,max_iter=100,fit_intercept=False,tol=1e-5,compute_score=True)
    
    for i in range(num_subnw):
    
        ridge_model.fit(X=d1,y=obs_noise[n,:int(T/deltaT)*Num_runs,i])
        alpha_opt[n,i] = 1./ridge_model.lambda_
        beta_opt[n,i]  = ridge_model.alpha_

        temp           = second_der_alpha_beta([alpha_opt[n,i],beta_opt[n,i]],d1,\
                           num_subnw,obs_noise[n][:int(T/deltaT)*Num_runs,i])*len(obs_noise[n][:int(T/deltaT)*Num_runs,i])

        alpha_error[n,i] = np.abs(temp[1]/(temp[0]*temp[1]-temp[2]**2))**0.5
        beta_error[n,i]  = np.abs(temp[0]/(temp[0]*temp[1]-temp[2]**2))**0.5
    
    ordered_beta_log[n,:] = np.log(np.append(beta_opt[n,boundary_list],beta_opt[n,[i for i in range(num_subnw) if i not in boundary_list]]))
    ordered_beta_error[n,:] = np.log(np.append(beta_error[n,boundary_list],beta_error[n,[i for i in range(num_subnw) if i not in boundary_list]]))
    
    temp_list       = np.expand_dims(np.copy(ordered_beta_log[n,:]),axis=0)
    temp_list_error = np.expand_dims(np.copy(ordered_beta_error[n,:]),axis=0)
    
    auc_list[n] = utilities.auc_roc(temp_list,beta_error=temp_list_error,num_threshold=200,num_bdry=len(boundary_list),method='single',plot=False)
    
print(alpha_opt,beta_opt)
    
    