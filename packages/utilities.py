#!/usr/bin/env python2
# -*- coding: utf-8 -*-


"""
Utility fucntions to make inference of boundary species from network dynamics
Author: Moshir Harsh, btemoshir@gmail.com

Associated paper: "Physics-inspired machine learning detects “unknown unknowns” in networks: discovering network boundaries from observable dynamics"

"""

import numpy as np
import scipy as sc
import matplotlib.pyplot as plt

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
            auc[i] = np.trapezoid(y=tp[i]/(tp[i]+fn[i]),x=fp[i]/(fp[i]+tn[i]))

        return np.mean(auc)
    