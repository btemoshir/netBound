from tqdm import tqdm
import sys
import os
import numpy as np 
from scipy.integrate import solve_ivp
from sklearn.linear_model import BayesianRidge



from packages.utilities import *
path_dir_script = 'packages'

## EGFR NETWORK with Michaelis-Menten Dynamics without enzyme units
EGFR_M_S_UNITS_PATH =path_dir_script + '/parameters/EGFR/egfr_m_s_units.csv'

## Defining the paths for the EGFR network with Michaelis-Menten Dynamics
M_DYNB_PATH =path_dir_script + '/parameters/EGFR/bdyn.txt'
M_DYNH_PATH= path_dir_script + '/parameters/EGFR/hdyn.txt'
M_DYNU_PATH= path_dir_script + '/parameters/EGFR/udyn.txt'
M_DYNM_PATH= path_dir_script + '/parameters/EGFR/mdyn.txt'



n_epsilon = 1
epsilon_list = [1e-5]
dt_list = [1e-2]
n_runs = 100
n_avg = 50
T = 10
stdv = .2


n_dt = len(dt_list)
t_eval = [np.linspace(0, T, int(np.ceil(T / dt))) for dt in dt_list]
par = make_par('egfr')
dyn, S, f = get_dynamics()
par['dynamics'] = dyn
par = refresh_netw_par(par) 

# Loading steady state values for units 
y_val = load_rubin14_MM_yvals(y_vals_file='packages/parameters/EGFR/yvals.txt')


n_sub = len(par['units_subnw'])

np.random.seed(0)
def func(): 
    # Generating Data 
    init_list = [[np.abs(y_val * ( 1 +  np.random.normal(0, stdv, len(y_val)))) for i in range(n_runs)] for _ in range(n_avg)]



    alpha_list = np.empty((n_epsilon, n_dt, n_avg, n_sub ))
    beta_list  = np.empty((n_epsilon, n_dt, n_avg, n_sub ))
    weights_list = np.empty((n_epsilon, n_dt, n_avg, n_sub ), dtype=object)
    S_list = np.empty((n_epsilon, n_dt, n_avg, n_sub ), dtype=object)
    alpha_error_list = np.empty((n_epsilon, n_dt, n_avg, n_sub ))
    beta_error_list = np.empty((n_epsilon, n_dt, n_avg, n_sub ))
    emp_var_list = np.empty((n_epsilon, n_dt, n_avg, n_sub))


    bar_e = tqdm(enumerate(epsilon_list), total=len(epsilon_list))
    for i_e, epsilon in bar_e:
        bar_e.set_description(f'epsilon: {epsilon:.1e}')    
        det_dyn, S, flux = get_dynamics()
        stoch_dyn = get_stoch_term(epsilon, S, flux)
        
        bar_a = tqdm(range(n_avg), total=n_avg, leave=False)
        for i_a in bar_a:        
            
            d1 = [None]*n_dt
            der = [None]*n_dt
            der_det = [None]*n_dt



            bar_runs = tqdm(range(n_runs), total=n_runs, leave=False)
            for r in bar_runs:
                init = init_list[i_a][r]

                sol_det = solve_ivp(det_dyn,t_span = (0, T),  t_eval=t_eval[0], y0= init, method='BDF')  # solve the system of ODEs
                sol_stoch = int_ito(det_dyn, stoch_dyn, init, t_eval=t_eval[0])

                x_det = sol_det.y[par['idx_subnw'], :]  # extract the concentrations of the observed units 
                x_stoch = sol_stoch[par['idx_subnw'], :]  # extract the concentrations of the observed units 
                
            
                for i_dt, _ in enumerate(dt_list): 
                    x_s = x_stoch[:, ::int(np.ceil(dt_list[i_dt] / dt_list[0]))]
                    x_d = x_det[:, ::int(np.ceil(dt_list[i_dt] / dt_list[0]))]

                    if (d1[i_dt] is not None): 
                        d1[i_dt] = np.append(d1[i_dt], design_matrix(x_s.T), axis=0)
                        der[i_dt] = np.append(der[i_dt], np.gradient(x_s,t_eval[i_dt],  axis=1), axis=1)                                        
                        der_det[i_dt] = np.append(der_det[i_dt], np.gradient(x_d, t_eval[i_dt],  axis=1), axis=1)                    
                                            
                                        
                    else:     
                        d1[i_dt] = design_matrix(x_s.T)
                        der[i_dt] = np.gradient(x_s,t_eval[i_dt],  axis=1)    
                        der_det[i_dt] = np.gradient(x_d,t_eval[i_dt],  axis=1)  
                    
            for i_dt in range(n_dt):
                
                emp_var_list[i_e, i_dt, i_a] = np.var(der[i_dt] - der_det[i_dt], axis=1) 

            
            for i_dt in range(n_dt):
                for i in range(len(par['units_subnw'])): 
                    clf = BayesianRidge(lambda_1=0, lambda_2=0, alpha_1=0, alpha_2=0, copy_X = True, tol=1e-4, fit_intercept=False)
                    clf.fit(d1[i_dt], der[i_dt][i])

                    alpha = clf.lambda_
                    beta = clf.alpha_            

                    temp = second_der_alpha_beta_2(alpha, beta, d1[i_dt], der[i_dt][i])

                    alpha_error_list[i_e, i_dt, i_a, i] = np.abs(temp[1]/(temp[0]*temp[1]-temp[2]**2))**0.5
                    beta_error_list[i_e,i_dt, i_a, i]  = np.abs(temp[0]/(temp[0]*temp[1]-temp[2]**2))**0.5                
                    
                    alpha_list[i_e, i_dt,  i_a, i] = clf.lambda_
                    beta_list[i_e,i_dt,  i_a, i] = clf.alpha_
                    weights_list[i_e,i_dt, i_a, i] = clf.coef_
                    S_list[i_e, i_dt, i_a, i] = clf.sigma_

                                
                    
    t1_data = {'alpha_list': alpha_list, 'beta_list': beta_list, 'weights_list': weights_list, 'S_list': S_list, 'alpha_error_list': alpha_error_list, 'beta_error_list': beta_error_list, 'emp_var_list': emp_var_list, 'epsilon_list': epsilon_list, 'dt_list': dt_list, 'n_runs': n_runs, 'n_avg': n_avg, 'T': T, 'stdv': stdv}

    np.save('data/10_EGFR_MM_hist', t1_data)

if __name__ == '__main__':

    func()
