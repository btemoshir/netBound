#!/usr/bin/env python2
# -*- coding: utf-8 -*-




"""
Utility fucntions to make inference of boundary species from network dynamics
Author: Moshir Harsh, btemoshir@gmail.com

Associated paper: "Physics-inspired machine learning detects “unknown unknowns” in networks: discovering network boundaries from observable dynamics"

"""
import pandas as pd
import numpy as np
import scipy as sc
import matplotlib.pyplot as plt
import sympy as sp


### Defining Constants ###

EGFR_M_YVALS_RUBIN14_PATH = 'parameters/EGFR/yvals.txt'
path_dir_script = 'packages'

## EGFR NETWORK with Michaelis-Menten Dynamics without enzyme units
EGFR_M_S_UNITS_PATH =path_dir_script + '/parameters/EGFR/egfr_m_s_units.csv'

## Defining the paths for the EGFR network with Michaelis-Menten Dynamics
M_DYNB_PATH =path_dir_script + '/parameters/EGFR/bdyn.txt'
M_DYNH_PATH= path_dir_script + '/parameters/EGFR/hdyn.txt'
M_DYNU_PATH= path_dir_script + '/parameters/EGFR/udyn.txt'
M_DYNM_PATH= path_dir_script + '/parameters/EGFR/mdyn.txt'




def find_boundaries(nw,bulk_species):

        """
        Outputs the boundary list (species which have INCOMING conenctions from the bulk species) from a networkx graph nw given the list of bulk_species
        
        Input:        
        nw            = networkx graph nw
        bulk_species  = list of bulk species
    
        Output:    
        Sorted list of boundary species (species which have INCOMING conenctions from the bulk species)
        
        """

        boundary_list = []
        
        for sp in bulk_species:
            recieving_nodes = [[i for i in nw.out_edges(sp)][j][1] for j in range(len([i for i in nw.out_edges(sp)]))]
            for j in recieving_nodes:
                if j not in boundary_list and j not in bulk_species:
                    boundary_list.append(j)
                
        return(np.sort(boundary_list))

def find_boundaries_out(nw,bulk_species):

        """
    Outputs the boundary list (species which have OUTGOING conenctions from the bulk species) from a networkx graph nw given the list of bulk_species

        Input:        
        nw            = networkx graph nw
        bulk_species  = list of bulk species
    
        Output:    
        Sorted list of boundary species (species which have OUTGOING conenctions from the bulk species)
    
        """
        boundary_list = []
        
        for sp in bulk_species:
            recieving_nodes = [[i for i in nw.in_edges(sp)][j][0] for j in range(len([i for i in nw.in_edges(sp)]))]
            for j in recieving_nodes:
                if j not in boundary_list and j not in bulk_species:
                    boundary_list.append(j)
                
        return(np.sort(boundary_list))

def design_matrix_linear(x):

        """
        Ouputs the design matrix with all linear basis functions. The first basis function is the constant function.

        Input:
        x       = array with numbers of all species at all time points. dim:(time_points, num_species)

        Output:
        des_mat = design matrix with all linear basis functions. dim:(time_points,num_species + 1)

        """

        time_points,num_species = np.shape(x)
        num_basis               = num_species + 1
        des_mat                 = np.zeros([int(time_points),int(num_basis)])
        des_mat[:,0] = 1.
        des_mat[:,1:num_species+1] = x

        #for i in range(time_points):
        #    for p in range(num_species-1):
        #        des_mat[i,1+num_species*(p+1):1+num_species*(p+2)] = x[i]*np.roll(x[i],p)   

        return des_mat

def design_matrix_KM(x):

        """
        Ouputs the design matrix with all linear basis functions for the Kuramoto oscillator examples! The first basis function is the constant function.

        Input:
        x       = array with numbers of all species at all time points. dim:(time_points, num_species)

        Output:
        des_mat = design matrix with all linear basis functions. dim:(time_points,num_species + 1)

        """

        time_points,num_species = np.shape(x)
        num_basis               = num_species + 1
        des_mat                 = np.zeros([int(time_points),int(num_basis)])
        des_mat[:,0] = 1.
        des_mat[:,1:num_species+1] = x

        #for i in range(time_points):
        #    for p in range(num_species-1):
        #        des_mat[i,1+num_species*(p+1):1+num_species*(p+2)] = x[i]*np.roll(x[i],p)   

        return des_mat
    
    
def basis_KM(x):

        """
        Outputs all the basis functions at one time point which are linear and quadratic combinations.

        Input:
        x = array with all species concentrations at a time point.

        Output:
        basis = the basis. dim: num_species*(num_species+3)/2 + 1 

        """

        num_species  = len(x)
        num_basis    = num_species + 1
        basis        = np.zeros(int(num_basis))
        basis[0]     = 1.
        basis[1:num_species+1] = x

        return basis



def design_matrix(x):
    
    """
    Ouputs the design matrix with all linear and quadratic basis functions evaluated at all time points in the input.
    
    Input:
    x       = array with numbers of all species at all time points. dim:(time_points, num_species)
    
    Output:
    des_mat = design matrix with all linear and quadratic basis functions. dim:(time_points,num_species*(num_species+3)/2 + 1)
    
    """
    
    time_points,num_species = np.shape(x)
    num_basis               = num_species*(num_species+3)/2 + 1
    des_mat                 = np.zeros([int(time_points),int(num_basis)])
    des_mat[:,0] = 1.
    des_mat[:,1:num_species+1] = x
    
    idnx = num_species+1
    for p in range(num_species):
        for l in range(p,num_species):
            des_mat[:,idnx] = x[:,p]*x[:,l] 
            idnx += 1
            
    #print(idnx)
    
    #for i in range(time_points):
    #    for p in range(num_species-1):            
    #        des_mat[i,1+num_species*(p+1):1+num_species*(p+2)] = x[i]*np.roll(x[i],p)
    
    return des_mat


def basis(x):
    
    """
    Outputs all the basis functions at one time point which are linear and quadratic combinations.
    
    Input:
    x = array with all species concentrations at a time point.
    
    Output:
    basis = the basis. dim: num_species*(num_species+3)/2 + 1 
    
    """
    
    num_species  = len(x)
    num_basis    = num_species*(num_species+3)/2 + 1
    basis        = np.zeros(int(num_basis))
    basis[0]     = 1.
    basis[1:num_species+1] = x

    for p in range(num_species-1):
        basis[1+num_species*(p+1):1+num_species*(p+2)] = x*np.roll(x,p)   
    
    return basis

def log_probability(par,d1,num_subnw,obs):
    
    """
    Returns the log-likelihood per number of data points, of the training data coming from a Gaussian process with weight correlation alpha and random noise beta
    
    Input:
    par       = par[0]: alpha value, par[1]: beta value
    d1        = design matrix
    num_subnw = num of species in the subnetwork
    obs       = the observations, that is the time derivatives of the concentrations at all time points.
    
    Output:
    
    log_likelihood per number of training points at the alpha and beta. Scalar
    
  
    """
    
    alpha     = par[0]
    beta      = par[1]
    num_basis = np.shape(d1)[1]
    num_obs   = len(obs)
    phiT      = np.matmul(d1.T,obs)
    A         = np.diag(alpha*np.ones(num_basis)) + beta*np.matmul(d1.T,d1)
    A_inv     = np.linalg.inv(np.diag(alpha*np.ones(num_basis)) + beta*np.matmul(d1.T,d1))
    
    mn = beta*np.matmul(A_inv,np.matmul(d1.T,obs))
    
    return  (num_obs*np.log(beta)/2 + num_basis*np.log(alpha)/2 - num_obs*np.log(2*np.pi)/2 -\
            np.log(np.linalg.det(A))/2 - beta*np.matmul((obs-np.matmul(d1,mn)).T,(obs-np.matmul(d1,mn)))/2 -\
            alpha*np.matmul(mn.T,mn)/2)/num_obs

    
def gradient_alpha_beta(par,d1,num_subnw,obs,method='linear'):
    
    """
    Returns the derivative of the log-likelihood per number of data points, of the training data coming from a Gaussian process with weight correlation alpha and random noise beta
    
    Input:
    par       = par[0]: alpha value, par[1]: beta value
    d1        = design matrix
    num_subnw = num of species in the subnetwork
    obs       = the observations, that is the time derivatives of the concentrations at all time points.
    method    = if log, then it return the gradient wrt log alpha and log beta
    
    Output:
    
    The Gradient of log_likelihood per number of training points at given alpha and beta. dim = 2.
    """
    
    alpha = par[0]
    beta  = par[1]
    
    num_basis = np.shape(d1)[1]
    num_obs   = len(obs)
    A_inv     = np.linalg.inv(np.diag(alpha*np.ones(num_basis)) + beta*np.matmul(d1.T,d1))
    A         = np.linalg.inv(A_inv)
    phiT      = np.matmul(d1.T,obs)
    mn        = beta*np.matmul(A_inv,np.matmul(d1.T,obs))
    
    g_alpha   = 0.5*num_basis/alpha -0.5*np.matmul(mn.T,mn) - 0.5*np.trace(A_inv)
    #g_beta   = 0.5*num_obs/beta + 0.5*np.trace(np.identity(len(A))-alpha*A)/beta -\
    #            0.5*np.matmul(obs.T,obs) -0.5*alpha*beta*np.matmul(phiT,phiT.T) + 1.5*np.matmul(mn,np.matmul(A,mn.T).T)/beta
    g_beta    =  0.5*num_obs/beta - np.matmul((obs-np.matmul(d1,mn)).T,(obs-np.matmul(d1,mn)))/2 - 0.5*np.trace(np.matmul(A_inv,np.matmul(d1.T,d1)))
    
    if method == 'log':
        return np.array([alpha*g_alpha,beta*g_beta])/num_obs
    
    else:
        return np.array([g_alpha,g_beta])/num_obs
    
    
def second_der_alpha_beta(par,d1,num_subnw,obs,method='linear'):
    
    """
    Returns the second derivative of the log-likelihood per number of data points, of the training data coming from a Gaussian process with weight correlation alpha and random noise beta
    
    Input:
    par       = par[0]: alpha value, par[1]: beta value
    d1        = design matrix
    num_subnw = num of species in the subnetwork
    obs       = the observations, that is the time derivatives of the concentrations at all time points.
    method    = if log, then it return the gradient wrt log alpha and log beta
    
    Output:
    
    The second derivative of log_likelihood per number of training points at given alpha and beta. dim = 3: wrt alpha, beta and mixed term.
    
  
    """
    
    alpha = par[0]
    beta  = par[1]
    
    #num_basis = int(num_subnw*(num_subnw+3)/2 + 1)
    num_basis = np.shape(d1)[1]
    num_obs   = len(obs)
    A_inv     = np.linalg.inv(np.diag(alpha*np.ones(num_basis)) + beta*np.matmul(d1.T,d1))
    A         = np.linalg.inv(A_inv)
    phiT      = np.matmul(d1.T,obs)
    mn        = beta*np.matmul(A_inv,np.matmul(d1.T,obs))
    
    g2_alpha      = -0.5*num_basis/alpha**2 + np.matmul(mn.T,np.matmul(A_inv,mn)) + 0.5*np.trace(np.linalg.matrix_power(A_inv,2))
    g2_beta       = -0.5*num_obs/beta**2 - np.matmul((obs-np.matmul(d1,mn)).T,(np.matmul(d1,np.matmul(A_inv,np.matmul(d1.T,np.matmul(d1,mn))))- np.matmul(d1,mn)/beta))\
                    +0.5*np.trace(np.linalg.matrix_power(np.matmul(A_inv,np.matmul(d1.T,d1)),2))
    g2_alpha_beta = -np.matmul((obs - np.matmul(d1,mn)).T,(np.matmul(d1,np.matmul(A_inv,mn)))) +\
                    0.5*np.trace(np.matmul(np.linalg.matrix_power(A_inv,2),np.matmul(d1.T,d1)))
    
    if method == 'log':
        return np.array([alpha**2*g2_alpha,beta**2*g2_beta,alpha*beta*g2_alpha_beta])/num_obs #this is only valid at the point where first derivative is zero
    
    else:
        return np.array([g2_alpha,g2_beta,g2_alpha_beta])/num_obs  


def discrimination(beta,error):
    
    """
    Returns the discriminant for all the subnetwork species which allows us to determine how well we can differentiate boundary species from the subnetwork.
    
    Input:
    beta  = the optimal beta value for all the subnetwork species
    error = the corressponding error in beta determination
    
    Output:
    discrim = the discriminant value for all the subnw species in the same order
    
    """
    
    discrim = np.zeros(len(beta))
    for i in range(len(beta)):
        discrim[i] = np.sum(((beta[i] - beta)**2)*np.heaviside(beta-beta[i],0)/(error[i]**2 + error**2))
        
    return discrim

def numerical_hessian(x,spacing):
    
    """
    Calculate the hessian matrix with finite differences
    Parameters:
       - x      : ndarray
       - spacing: the spacing between values of x matrix as an narray
    Returns:
       an array of shape (x.dim, x.ndim) + x.shape
       where the array[i, j, ...] corresponds to the second derivative x_ij
    """
    #print(spacing)
    
    x_grad = np.gradient(x,*spacing) 
    hessian = np.empty((x.ndim, x.ndim) + x.shape, dtype=x.dtype)
    
    for k, grad_k in enumerate(x_grad):
        # iterate over dimensions
        # apply gradient again to every component of the first derivative.
        tmp_grad = np.gradient(grad_k,*spacing) 
        for l, grad_kl in enumerate(tmp_grad):
            hessian[k, l, :, :] = grad_kl
    
    return hessian

def auc_roc(beta,beta_error,num_threshold,num_bdry=1,method='total',plot=False,threshold_scale='linear'):
    
    """
    Calculates the Area under the curve for the ROC curve for a multiple data samples of the experiment with the given threshold and optionally plots them.
    
    Inputs:
    -beta         : beta values for repeated runs for all the subnetwork species as an np.array
    -beta_error   : corressponding standard deviation for the betas in the same form
    -num_threshold: the number of threshold values to be used, this determines the resolution of the ROC curve
    -num_bdry     : the number of boundary species in the network. We keep to the convention that the boundary speices are always first in any vector such as beta. = 1 by default
    -method       : 'total' = draw the ROC curve over different runs of the data first and then calculate the AUC 
                    'single'= draw the ROC for each run and calculate the AUC and then take the average
    -plot         : plot the ROC or not
    
    Output:
    - auc: the area under the curve
    
    """

    if threshold_scale == 'linear':
        threshold   = np.linspace(np.min(beta)-1,np.max(beta)+1,num_threshold)
    elif threshold_scale == 'log':
        threshold   = np.logspace(np.log10(np.min(beta))-1,np.log10(np.max(beta))+1,num_threshold)
        
    Num_reps    = len(beta)
    tp,fp,fn,tn = np.zeros([Num_reps,len(threshold)]),np.zeros([Num_reps,len(threshold)]),np.zeros([Num_reps,len(threshold)]),np.zeros([Num_reps,len(threshold)])
    auc         = np.zeros(Num_reps)
    num_int     = len(beta[0]) - num_bdry #Number of internal species
    #num_int     = len(beta) - num_bdry #Number of internal species

    for i in range(Num_reps):

        for j in range(len(threshold)):

            t = 0.5*(1+ sc.special.erf((threshold[j] - beta[i,:])/(np.sqrt(2)*beta_error[i,:])))

            tp[i,j] += np.sum(t[0:num_bdry])
            fn[i,j] += num_bdry-np.sum(t[0:num_bdry])
            fp[i,j] += np.sum(t[num_bdry:])
            tn[i,j] += num_int - np.sum(t[num_bdry:]) 

    if method == 'total':
        if plot:
            plt.figure()
            plt.scatter(np.sum(fp,0)/(np.sum(fp,0)+np.sum(tn,0)),np.sum(tp,0)/(np.sum(tp,0)+np.sum(fn,0)),marker='.')
            plt.xlabel('False Positive fraction')
            plt.ylabel('True Positive fraction')
            plt.plot(np.sum(fp,0)/(np.sum(fp,0)+np.sum(tn,0)),np.sum(fp,0)/(np.sum(fp,0)+np.sum(tn,0)),'r--')

        auc = np.trapz(y=np.sum(tp,0)/(np.sum(tp,0)+np.sum(fn,0)),x=np.sum(fp,0)/(np.sum(fp,0)+np.sum(tn,0)))
        return auc   

    elif method == 'single':

        if plot:
            plt.figure()
            plt.scatter(np.mean(fp/(fp+tn),0),np.mean(tp/(tp+fn),0),marker='.')
            plt.xlabel('False Positive fraction')
            plt.ylabel('True Positive fraction')
            plt.plot(np.mean(fp/(fp+tn),0),np.mean(fp/(fp+tn),0),'r--')

        for i in range(Num_reps):
            auc[i] = np.trapz(y=tp[i]/(tp[i]+fn[i]),x=fp[i]/(fp[i]+tn[i]))

        return np.mean(auc)
    

def get_stoch_term(epsilon, S, flux):
    g = lambda t, stat: np.sqrt(epsilon) * S @ np.diag(np.sqrt(flux(t, stat))) 
    
    return g

def second_der_alpha_beta_2(alpha, beta, d1, x):
    
    """
    Returns the second derivative of the log-likelihood per number of data points, of the training data coming from a Gaussian process with weight correlation alpha and random noise beta
    
    Input:
    par       = par[0]: alpha value, par[1]: beta value
    d1        = design matrix
    num_subnw = num of species in the subnetwork
    obs       = the observations, that is the time derivatives of the concentrations at all time points.
    method    = if log, then it return the gradient wrt log alpha and log beta
    
    Output:
    
    The second derivative of log_likelihood per number of training points at given alpha and beta. dim = 3: wrt alpha, beta and mixed term.
    
  
    """

    
    num_basis = d1.shape[1] #int(num_subnw*(num_subnw+3)/2 + 1)
    num_obs   = len(x)
    A_inv     = np.linalg.inv(np.diag(alpha*np.ones(num_basis)) + beta*np.matmul(d1.T,d1))
    mn        = beta*np.matmul(A_inv,np.matmul(d1.T,x))
    
    g2_alpha      = -0.5*num_basis/alpha**2 + np.matmul(mn.T,np.matmul(A_inv,mn)) + 0.5*np.trace(np.linalg.matrix_power(A_inv,2))
    g2_beta       = -0.5*num_obs/beta**2 - np.matmul((x-np.matmul(d1,mn)).T,(np.matmul(d1,np.matmul(A_inv,np.matmul(d1.T,np.matmul(d1,mn))))- np.matmul(d1,mn)/beta))\
                    +0.5*np.trace(np.linalg.matrix_power(np.matmul(A_inv,np.matmul(d1.T,d1)),2))
    g2_alpha_beta = -np.matmul((x - np.matmul(d1,mn)).T,(np.matmul(d1,np.matmul(A_inv,mn)))) +\
                    0.5*np.trace(np.matmul(np.linalg.matrix_power(A_inv,2),np.matmul(d1.T,d1)))

    
    return np.array([g2_alpha,g2_beta,g2_alpha_beta])/num_obs  



def auc_roc2(beta,beta_error,num_threshold,num_bdry=1,method='total',plot=False,threshold_scale='linear'):
    
    """
    Calculates the Area under the curve for the ROC curve for a multiple data samples of the experiment with the given threshold and optionally plots them.
    
    Inputs:
    -beta         : beta values for repeated runs for all the subnetwork species as an np.array
    -beta_error   : corressponding standard deviation for the betas in the same form
    -num_threshold: the number of threshold values to be used, this determines the resolution of the ROC curve
    -num_bdry     : the number of boundary species in the network. We keep to the convention that the boundary speices are always first in any vector such as beta. = 1 by default
    -method       : 'total' = draw the ROC curve over different runs of the data first and then calculate the AUC 
                    'single'= draw the ROC for each run and calculate the AUC and then take the average
    -plot         : plot the ROC or not
    
    Output:
    - auc: the area under the curve
    
    """

    if threshold_scale == 'linear':
        threshold   = np.linspace(np.min(beta)-1,np.max(beta)+1,num_threshold)
    elif threshold_scale == 'log':
        threshold   = np.logspace(np.log10(np.min(beta))-1,np.log10(np.max(beta))+1,num_threshold)
        
    Num_reps    = len(beta)
    tp,fp,fn,tn = np.zeros([Num_reps,len(threshold)]),np.zeros([Num_reps,len(threshold)]),np.zeros([Num_reps,len(threshold)]),np.zeros([Num_reps,len(threshold)])
    auc         = np.zeros(Num_reps)
    num_int     = len(beta[0]) - num_bdry #Number of internal species
    #num_int     = len(beta) - num_bdry #Number of internal species

    for i in range(Num_reps):

        for j in range(len(threshold)):

            t = 0.5*(1+ sc.special.erf((threshold[j] - beta[i,:])/(np.sqrt(2)*beta_error[i,:])))

            tp[i,j] += np.sum(t[0:num_bdry])
            fn[i,j] += num_bdry-np.sum(t[0:num_bdry])
            fp[i,j] += np.sum(t[num_bdry:])
            tn[i,j] += num_int - np.sum(t[num_bdry:]) 

    if method == 'total':
        if plot:
            plt.figure()
            plt.scatter(np.sum(fp,0)/(np.sum(fp,0)+np.sum(tn,0)),np.sum(tp,0)/(np.sum(tp,0)+np.sum(fn,0)),marker='.')
            plt.xlabel('False Positive fraction')
            plt.ylabel('True Positive fraction')
            plt.plot(np.sum(fp,0)/(np.sum(fp,0)+np.sum(tn,0)),np.sum(fp,0)/(np.sum(fp,0)+np.sum(tn,0)),'r--')

        auc = np.trapz(y=np.sum(tp,0)/(np.sum(tp,0)+np.sum(fn,0)),x=np.sum(fp,0)/(np.sum(fp,0)+np.sum(tn,0)))
        return auc   

    elif method == 'single':

        if plot:
            plt.figure()
            plt.scatter(np.mean(fp/(fp+tn),0),np.mean(tp/(tp+fn),0),marker='.')
            plt.xlabel('False Positive fraction')
            plt.ylabel('True Positive fraction')
            plt.plot(np.mean(fp/(fp+tn),0),np.mean(fp/(fp+tn),0),'r--')

        for i in range(Num_reps):
            auc[i] = np.trapz(y=tp[i]/(tp[i]+fn[i]),x=fp[i]/(fp[i]+tn[i]))

        return np.mean(auc)

 ### differential equation functions #####
def load_all_deq_pars(save_dir, units): 
    '''
    Loading all parameters for the reactions
    '''

    a_dfb1 = np.load(save_dir + '/a_dfb1.npy', allow_pickle=True)   # binary reactions
    a_dfb2 = np.load(save_dir + '/a_dfb2.npy', allow_pickle=True)   # binary reactions
    a_dfh1 = np.load(save_dir + '/a_dfh1.npy', allow_pickle=True)   # heterodimer reactions
    a_dfh2 = np.load(save_dir + '/a_dfh2.npy', allow_pickle=True)   # heterodimer reactions
    a_dfu1 = np.load(save_dir + '/a_dfu1.npy', allow_pickle=True)   # unary reactions
    a_dfu2 = np.load(save_dir + '/a_dfu2.npy', allow_pickle=True)   # unary reactions
    a_dfm1 = np.zeros((29))                                         # Michaelis-Menten reactions
    a_dfm2 = np.zeros((29))                                         # Michaelis-Menten reactions


    if os.path.exists(save_dir + '/a_dfm1.npy') and os.path.exists(save_dir + '/a_dfm2.npy'): 
        a_dfm1 = np.load(save_dir + '/a_dfm1.npy', allow_pickle=True)
        a_dfm2 = np.load(save_dir + '/a_dfm2.npy', allow_pickle=True)

    deq_pars =  {'a_dfb1': a_dfb1,'a_dfb2': a_dfb2, 'a_dfh1': a_dfh1, 'a_dfh2': a_dfh2, 'a_dfu1': a_dfu1, \
        'a_dfu2': a_dfu2, 'a_dfm1':a_dfm1, 'a_dfm2':a_dfm2, 'units': units}
    return deq_pars

def make_deq(S, f): 
    ''' 
    Returns the deq for S, f 
    '''
    def deq_func(t, stat): 
        return S@f(t, stat)
    
    return deq_func

def get_dynamics(network='', dynb_path=M_DYNB_PATH, dynh_path=M_DYNH_PATH, dynu_path=M_DYNU_PATH, dynm_path=M_DYNM_PATH, units_path= EGFR_M_S_UNITS_PATH) -> tuple:


    dynb = pd.read_csv(dynb_path, comment='#')
    dynh = pd.read_csv(dynh_path, comment='#')
    dynu = pd.read_csv(dynu_path, comment='#')
    dynm = pd.DataFrame([], columns=['index', 'type', 'subs', 'product1', 'Vmax', 'Km'])

    dynb.sort_values(by='product1', inplace=True)
    dynh.sort_values(by='product1', inplace=True)
    dynu.sort_values(by='product1', inplace=True)

    dynb.reset_index(inplace=True)
    dynh.reset_index(inplace=True)
    dynu.reset_index(inplace=True)
    
    if dynm_path != '': 
        # print('dynm_path not \'\' --> loading Michaelis-Menten equations ')
        dynm = pd.read_csv(dynm_path, comment='#')
        dynm.sort_values(by='product1', inplace=True)
        dynm.reset_index(inplace=True)

    units = pd.read_csv(units_path, index_col=0).values.flatten()


    # loading rates
    rates_par = {'dynb':dynb, 'dynh':dynh, 'dynu':dynu, 'dynm':dynm, 'units':units}


    # network dynamics    
    S, R = make_S(**rates_par)
    f = make_flux(**rates_par)
    dyn_func = make_deq(S, f)

    return dyn_func, S, f 


def make_S(dynb, dynh, dynu, dynm, units) -> tuple: 
    '''
    Constructs the stochiometry matrix
    '''

    # nz stands for not zeros; f, b for forward & backward reaction;
    # b,h,u,m for reaction type

    R = 0
    S = None 

    for idx_u, unit in enumerate(units): 
        func_b_f = lambda unit, e1, e2, prod: -1 if unit in [e1, e2] else (+1 if unit == prod else 0)
        func_b_b = lambda unit, e1, e2, prod: +1 if unit in [e1, e2] else (-1 if unit == prod else 0)

        func_h_f = lambda unit, e1, e2, prod: -2 if unit in [e1, e2] else (+1 if unit == prod else 0)
        func_h_b = lambda unit, e1, e2, prod: +2 if unit in [e1, e2] else (-1 if unit == prod else 0)

        func_u_f = lambda unit, e1, prod: -1 if unit == e1 else (+1 if unit == prod else 0)
        func_u_b = lambda unit, e1, prod: +1 if unit == e1 else (-1 if unit == prod else 0)

        func_m_f = lambda unit, subs, prod: -1 if unit == subs else (+1 if unit == prod else 0)
        # func_m_b = lambda unit, subs, prod: +1 if unit == subs else (-1 if unit == prod else 0)

        a_b_f = [func_b_f(unit, e1, e2, prod) for e1, e2, prod, ratef in dynb[['educt1', 'educt2', 'product1', 'ratef']].to_numpy() if ratef != 0]
        a_b_b = [func_b_b(unit, e1, e2, prod) for e1, e2, prod, rateb in dynb[['educt1', 'educt2', 'product1', 'rateb']].to_numpy() if rateb != 0]

        a_h_f = [func_h_f(unit, e1, e2, prod) for e1, e2, prod, ratef in dynh[['educt1', 'educt2', 'product1', 'ratef']].to_numpy() if ratef != 0]
        a_h_b = [func_h_b(unit, e1, e2, prod) for e1, e2, prod, rateb in dynh[['educt1', 'educt2', 'product1', 'rateb']].to_numpy() if rateb != 0]

        a_u_f = [func_u_f(unit, e1, prod) for e1, prod, ratef in dynu[['educt', 'product1', 'ratef']].to_numpy() if ratef != 0]
        a_u_b = [func_u_b(unit, e1, prod) for e1, prod, rateb in dynu[['educt', 'product1', 'rateb']].to_numpy() if rateb != 0]

        a_m_f = [func_m_f(unit, subs, prod) for subs, prod, Vmax in dynm[['subs', 'product1', 'Vmax']].to_numpy() if Vmax != 0]
        # a_m_b = [func_m_b(unit, subs, prod) for subs, prod, Vmax in dynm[['subs', 'product1', 'Vmax']].to_numpy() if Vp != 0]
        
        row = [*a_b_f, *a_b_b, *a_h_f, *a_h_b, *a_u_f, *a_u_b, *a_m_f]#, *a_m_b]
        if S is None: 
            R = len(row)
            S = np.zeros((len(units), R))

        S[idx_u] = row

    return S, R

def make_flux(dynb, dynh, dynu, dynm, units): 

   '''
   Constructs the flux function
   '''
   
   
   u_id =  {unit:u_id for u_id, unit in enumerate(units)}
   u_id_f = lambda unit: int(u_id[unit])
   _dynb = dynb.copy()
   _dynh = dynh.copy()
   _dynu = dynu.copy()
   _dynm = dynm.copy()

   _dynb[['educt1', 'educt2', 'product1']] = _dynb[['educt1', 'educt2', 'product1']].map(u_id_f)
   _dynh[['educt1', 'educt2', 'product1']] = _dynh[['educt1', 'educt2', 'product1']].map(u_id_f)
   _dynu[['educt', 'product1']] = _dynu[['educt', 'product1']].map(u_id_f)
   _dynm[['subs', 'product1']] = _dynm[['subs', 'product1']].map(u_id_f)


   a_b_f = np.array([[e1, e2, ratef] for e1, e2, ratef in _dynb[['educt1', 'educt2', 'ratef']].to_numpy() if ratef != 0])
   a_b_b = np.array([[prod, rateb] for prod, rateb in _dynb[['product1', 'rateb']].to_numpy() if rateb != 0])

   a_h_f = np.array([[e1, e2, ratef] for e1, e2, ratef in _dynh[['educt1', 'educt2', 'ratef']].to_numpy() if ratef != 0])
   a_h_b = np.array([[prod, rateb] for prod, rateb in _dynh[['product1', 'rateb']].to_numpy() if rateb != 0])

   a_u_f = np.array([[e1, ratef] for e1, ratef in _dynu[['educt', 'ratef']].to_numpy() if ratef != 0])
   a_u_b = np.array([[prod, rateb] for prod, rateb in _dynu[['product1', 'rateb']].to_numpy() if rateb != 0])
   
   a_m_f = np.array([[subs, Vmax, Km] for subs, Vmax, Km in _dynm[['subs', 'Vmax', 'Km']].to_numpy() if Vmax != 0])
   

   func_b_f =  lambda e1, e2, ratef: lambda t, st: st[int(e1)]*st[int(e2)]*ratef 
   func_b_b =  lambda prod, rateb:   lambda t, st: st[int(prod)]*rateb 
   func_h_f =  lambda e1, e2, ratef: lambda t, st: st[int(e1)]*st[int(e2)]*ratef 
   func_h_b =  lambda prod, rateb:   lambda t, st: st[int(prod)]*rateb 
   func_u_f =  lambda e1, ratef:     lambda t, st: st[int(e1)]*ratef 
   func_u_b =  lambda prod, rateb:   lambda t, st: st[int(prod)]*rateb 
   func_m_f =  lambda subs, Vmax, Km:lambda t, st: st[int(subs)]*Vmax/(Km + st[int(subs)])
   
   func_dat_pairs = [(func_b_f, a_b_f.T), (func_b_b, a_b_b.T), (func_h_f, a_h_f.T), (func_h_b, a_h_b.T),\
      (func_u_f, a_u_f.T), (func_u_b, a_u_b.T), (func_m_f, a_m_f.T)]

   _flux = [list(map(func, *lst)) for func, lst in func_dat_pairs if len(lst) > 0]

   flux_con = np.concatenate(_flux, axis=0)

   def flux(t, stat): 
       return np.array([flux_con[i](t, stat) for i in range(len(flux_con))])    
   
   return  flux
   

# Symbolic states 
def make_states_sp(units, latex_style=True): 
    '''
    Helper function that creates sympy symbols for unit states
    Inputs: 
        - units: units of network 
        - latex_style: wether or not to use latex style i.e. subscript of concentration
    Returns: 
        - states_sp: sympy symbolic states
    '''

    states_sp = np.empty((len(units)), dtype=object)
    for i, unit in enumerate(units):
        if latex_style: 
            states_sp[i] = sp.Symbol('x_{{{}}}'.format(str(unit)))
        else: 
            states_sp[i] = sp.Symbol('{}'.format(str(unit)))
    return states_sp





def load_rubin14_MM_yvals(y_vals_file = '../data/parameters/EGFR/yvals.txt'): 
    y_vals = np.loadtxt( y_vals_file)

    return y_vals




def make_par(network): 
    '''
    Intern function that creates provisional parameter dicts for specified network
    Inputs:     
        - network: desired_network
    Returns: 
        - netw_par        
    '''

    netw_par = {}

    if network =='egfr': 
        netw_par = {     
            'units_subnw':  ['EGF','GRb','GS','PLCg','PLCgP','PLCgPI','R','R2','RG','RGS','RP','RPL','RPLP','Ra','SOS'],            
            'idx_boundary_rel': [1, 2, 10, 14],
            'network': 'EGFR',                         
            'units':  ['EGF', 'GRb', 'GS', 'PLCg', 'PLCgP', 'PLCgPI', 'R', 'R2', 'RG', 'RGS', 'RP', 'RPL', 'RPLP', 'RSh', 'RShG', 'RShGS', 'RShP', 'Ra','SOS', 'ShG', 'ShGS', 'ShP', 'Shc']
        }

    return netw_par

def find_neighbors(netw:dict): 
    '''
    Finds neighbors of each unit using reaction parameters

    Inputs: 
        - netw: (dict) network dict
    Returns: 
        - units_neighbors: units that are neighbors, excluding the starting unit
        - idx_neighbors: index list of units_neighbors
    '''
    units_neighbors = [[unit for var in expr.free_symbols if (unit:=str(var).split('_')[1][1:-1]) != netw['units'][i]] for i, expr in enumerate(netw['dynamics'](0, netw['states'])) ]
    idx_neighbors = [[neigh_idx for neighbor in units_neighbors[i] if (neigh_idx:=list(netw['units']).index(neighbor)) != i] for i in range(len(netw['units']))]

    return units_neighbors, idx_neighbors


def refresh_netw_par(netw_par_:dict) -> tuple: 
    '''
    Calculates all topological keywords like boundary units, bulk units ... if provided with 'units' and 'units_subnw'
    as keys

    Inputs: 
        - netw_par_: (dict) network parameter dict
        - reg_par_: (dict) regression parameter dict
        - sim_par_: (dict) simulation parameter dict 

    Returns:    
        - netw_par: (dict) network parameter dict
        - reg_par: (dict) regression parameter dict
        - sim_par: (dict) simulation parameter dict

    '''
    netw_par = netw_par_.copy()
    netw_par['states'] = make_states_sp(netw_par['units'])
    
    # Required keys that need to be given in order to work
    units = np.array(netw_par['units'])
    units_subnw = np.array(netw_par['units_subnw'])

    # calculated
    idx_subnw = np.array([list(units).index(sub) for sub in units_subnw])

    # finding boundary units

    units_neighbors, idx_neighbors = find_neighbors(netw_par)
    boundary_mask = np.array([np.any([neighbor not in units_subnw for neighbor in units_neighbors[idx]]) \
        for idx in idx_subnw])


    units_boundary = units_subnw[boundary_mask]
    units_bulk = [unit for unit in units if unit not in units_subnw]

    # adjacency matrix 
    # adjacency = get_adjacency_of_netw(netw_par)

    # index recaluclation 
    idx_boundary = [list(units).index(bound) for bound in units_boundary]
    idx_boundary_rel = [list(units_subnw).index(bound) for bound in units_boundary]
    idx_bulk = [list(units).index(bulk) for bulk in units_bulk]

    # index inner and inner relative
    units_inner = [unit_s for unit_s in units_subnw if unit_s not in units_boundary]
    idx_inner = [list(units).index(unit_i) for unit_i in units_inner]
    idx_inner_rel = [list(units_subnw).index(unit_i) for unit_i in units_inner]

    # counting units
    num_subnw = len(units_subnw)
    num_bulk = len(units_bulk)
    num_boundary = len(units_boundary)


    # saving values
    
    netw_par['units_boundary'] = units_boundary
    netw_par['units_bulk'] = units_bulk
    netw_par['idx_subnw'] = idx_subnw
    netw_par['idx_boundary'] = idx_boundary
    netw_par['idx_boundary_rel'] = idx_boundary_rel
    netw_par['idx_bulk'] = idx_bulk
    netw_par['num_subnw'] = num_subnw
    netw_par['num_bulk'] = num_bulk
    netw_par['num_boundary'] = num_boundary
    netw_par['num_species'] = len(netw_par['units'])
    # netw_par['adjacency'] = adjacency

    # inner units
    netw_par['units_inner'] =  units_inner
    netw_par['idx_inner'] =  idx_inner
    netw_par['idx_inner_rel'] =  idx_inner_rel





    return netw_par



def int_ito(f, g, y0, t_eval, max_reject=100, n_lower_dt=10):
    
    n_steps = len(t_eval)
    N_w = g(0, y0).shape[1]


    y = np.zeros((len(y0), n_steps))
    y[:, 0] = y0
    dt = t_eval[1] - t_eval[0]
    for i in range(1, n_steps):
        val = -1

        counter = 0

        while np.any(val< 0): 
            if counter > max_reject : 
                raise RuntimeError('Max reject reached')                
            elif counter > n_lower_dt: 
                print('lower dt')    
                _t_eval = np.linspace(t_eval[i-1], t_eval[i], n_lower_dt)
                val = int_ito(f, g, y[:, i-1], _t_eval, max_reject=n_lower_dt)[:, -1]
            else:
                val = y[:, i-1] + f(t_eval[i-1], y[:, i-1]) * dt + g(t_eval[i-1], y[:, i-1]) @ np.random.normal(0, 1, N_w) * np.sqrt(dt)

            y[:, i] = val
            counter += 1

    return y





def get_true_weights(active_units, dynb, dynh, dynu, dynm, units): 
    '''
    Returns the parameters for a set of active_units as the parameter of the get
    Inputs: 
        - active_units: units that are observed -> are included in the design matrix
        - reg: regression dict
        - **deq_pars: **deq_pars from get_all_deq_data function 
    
    Returns: 
        - weights: (len(active_units), num_basis) parameters of design matrix
        - basis_funcs_sym: (num_basis) corresponding basis functions as sympy expressions
    '''
    

    
    u_id =  {unit:u_id for u_id, unit in enumerate(units)}
    u_id_f = lambda unit: int(u_id[unit])
    _dynb = dynb.copy()
    _dynh = dynh.copy()
    _dynu = dynu.copy()
    _dynm = dynm.copy()

    a_b_f = np.array([[e1, e2, prod, float(ratef), float(rateb)] for e1, e2,  prod, ratef, rateb in _dynb[['educt1', 'educt2', 'product1', 'ratef', 'rateb']].to_numpy()])
    a_b_b = np.array([[e1, e2, prod,  float(ratef), float(rateb)] for e1, e2,  prod, ratef, rateb in _dynb[['educt1', 'educt2', 'product1', 'ratef', 'rateb']].to_numpy()])

    a_h_f = np.array([[e1, e2, prod,  float(ratef), float(rateb)] for e1, e2, prod,ratef,  rateb in _dynh[['educt1', 'educt2', 'product1', 'ratef', 'rateb']].to_numpy()])
    a_h_b = np.array([[e1, e2, prod,  float(ratef), float(rateb)] for e1, e2, prod,ratef,  rateb in _dynh[['educt1', 'educt2', 'product1', 'ratef', 'rateb']].to_numpy()])

    a_u_f = np.array([[e1, prod, float(ratef), float(rateb)] for e1, prod, ratef, rateb in _dynu[['educt', 'product1' , 'ratef', 'rateb']].to_numpy()])
    a_u_b = np.array([[e1, prod, float(ratef), float(rateb)] for e1, prod, ratef, rateb in _dynu[['educt', 'product1' , 'ratef', 'rateb']].to_numpy()])
    
    a_m_f = np.array([[subs, Vmax, Km] for subs, Vmax, Km in _dynm[['subs', 'Vmax', 'Km']].to_numpy()])
    

    active_units_sp = make_states_sp(active_units)
    basis_funcs = design_matrix_symbolic(active_units_sp.reshape(1,-1), symbolic=True)[0].astype(str)
    
    
    weights = np.zeros((len(active_units), len(basis_funcs)))
    
    for idx_u, unit in enumerate(active_units_sp.astype(str)): 
        for u_e1, u_e2, u_p, ratef, rateb in zip(*a_b_f.T): 
            
            u_p = make_states_sp([u_p]).astype(str)[0]
            tple = make_states_sp(sorted((u_e1, u_e2))).astype(str)
            
            if unit in tple:     
                tple_str = str(f'{tple[0]}*{tple[1]}')
                
                hit_q = np.where(tple_str == basis_funcs) 
                hit_l = np.where(u_p == basis_funcs)

                weights[idx_u, hit_q]  += -float(ratef)
                weights[idx_u, hit_l]  += float(rateb)


        for u_e1, u_e2, u_p, ratef, rateb in zip(*a_b_b.T): 
            tple = make_states_sp(sorted((u_e1, u_e2))).astype(str)
            u_p = make_states_sp([u_p]).astype(str)[0]
            if unit == u_p: 
                tple_str = str(f'{tple[0]}*{tple[1]}')
                
                hit_q = np.where(tple_str == basis_funcs) 
                hit_l = np.where(u_p == basis_funcs)

                weights[idx_u, hit_q]  += float(ratef)
                weights[idx_u, hit_l]  += -float(rateb)

        for u_e1, u_e2, u_p, ratef, rateb in zip(*a_h_f.T): 
            tple = make_states_sp(sorted((u_e1, u_e2))).astype(str)
            u_p = make_states_sp([u_p]).astype(str)[0]
            if unit in tple: 

                tple_str = str(f'{tple[0]}*{tple[1]}')
                hit_q = np.where(tple_str == basis_funcs) 
                hit_l = np.where(u_p == basis_funcs)

                weights[idx_u, hit_q]  += -2*float(ratef )
                weights[idx_u, hit_l]  += 2*float(rateb)

        for u_e1, u_e2, u_p, ratef, rateb in zip(*a_h_b.T):
            tple = make_states_sp(sorted((u_e1, u_e2))).astype(str)
            u_p = make_states_sp([u_p]).astype(str)[0]
            if unit == u_p:                 
                tple_str = str(f'{tple[0]}*{tple[1]}')
                hit_q = np.where(tple_str == basis_funcs) 
                hit_l = np.where(u_p == basis_funcs)

                weights[idx_u, hit_q]  += float(ratef )
                weights[idx_u, hit_l]  += -float(rateb)

        for u_e1,  u_p, ratef, rateb in zip(*a_u_f.T):
            u_p = make_states_sp([u_p]).astype(str)[0]
            u_e1 = make_states_sp([u_e1]).astype(str)[0]
            if unit == u_e1: 
                hit_le = np.where(u_e1 == basis_funcs)
                hit_lp = np.where(u_p == basis_funcs)

                weights[idx_u, hit_le] += -float(ratef)
                weights[idx_u, hit_lp] += float(rateb)

        for u_e1, u_p, ratef, rateb in zip(*a_u_b.T): 
            u_p = make_states_sp([u_p]).astype(str)[0]
            u_e1 = make_states_sp([u_e1]).astype(str)[0]
            if unit == u_p: 
                hit_le = np.where(u_e1 == basis_funcs)
                hit_lp = np.where(u_p == basis_funcs)

                weights[idx_u, hit_le] += +float(ratef )
                weights[idx_u, hit_lp] += -float(rateb)

    
    basis_funcs_sym = design_matrix_symbolic(active_units_sp[np.newaxis], symbolic=True)[0]
    return weights, basis_funcs_sym


def design_matrix_symbolic(X, get_num_basis=False, params=None, symbolic=False):
    
    """
    Outputs all the basis functions at one time point which are linear and quadratic combinations.
    
    Input:
    X = array with all species concentrations at several time points.
    
    Output:
    basis_func = the basis_funcs. dim: num_species*(num_species+3)/2 + 1 
    
    """
    
    if params is not None: 
        n_subnw = params['num_subnw'] 
        n_basis_func = int(n_subnw * (n_subnw + 3) / 2 + 1)
        if get_num_basis: 
            return n_basis_func
        if X.shape[1] != n_subnw: 
            X = X[:, params['idx_subnw']]

    
    # print('X: ', X, 'X.shape', X.shape)
    n_samples, n_subnetwork = X.shape
    n_basis_func = int(n_subnetwork * (n_subnetwork + 3) / 2 + 1)
    
    if symbolic: 
        _dtype = object
    else: 
        _dtype = np.float64
    
    d1 = np.zeros(shape=(n_samples, n_basis_func), dtype=_dtype)
    

    temp = np.concatenate([[X[:, unit] * X[:, unit:][:, i] for i in range(n_subnetwork - unit)] for unit in
            range(n_subnetwork)]).T
    

    d1[:, :len(temp[0])] = temp[:, :]
    d1[:,  len(temp[0]): -1] = X[:, :]
    d1[:, -1] = 1

    # print('design matr: \tsamples: ', d1.shape[0], ' basis_funcs: ', d1.shape[1])
    return d1




def get_rates_par(network='', dynb_path=M_DYNB_PATH, dynh_path=M_DYNH_PATH, dynu_path=M_DYNU_PATH, dynm_path=M_DYNM_PATH, units_path= EGFR_M_S_UNITS_PATH): 
        


    dynb = pd.read_csv(dynb_path, comment='#')
    dynh = pd.read_csv(dynh_path, comment='#')
    dynu = pd.read_csv(dynu_path, comment='#')
    dynm = pd.DataFrame([], columns=['index', 'type', 'subs', 'product1', 'Vmax', 'Km'])

    dynb.sort_values(by='product1', inplace=True)
    dynh.sort_values(by='product1', inplace=True)
    dynu.sort_values(by='product1', inplace=True)

    dynb.reset_index(inplace=True)
    dynh.reset_index(inplace=True)
    dynu.reset_index(inplace=True)
    
    if dynm_path != '': 
        # print('dynm_path not \'\' --> loading Michaelis-Menten equations ')
        dynm = pd.read_csv(dynm_path, comment='#')
        dynm.sort_values(by='product1', inplace=True)
        dynm.reset_index(inplace=True)

    units = pd.read_csv(units_path, index_col=0).values.flatten()

    # loading rates
    rates_par = {'dynb':dynb, 'dynh':dynh, 'dynu':dynu, 'dynm':dynm, 'units':units}
    
   
    return rates_par