'''
MACS 40000 Rick Evans
PSET2 Driver
Author: Fiona Fan
'''
import SS as ss
import TPI as tpi
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator


#Household parameters
beta_annual=0.96
beta=beta_annual**20
sigma=3
nvec=np.array([1.0,1.0,0.2])
L=nvec.sum()

#Firm parameters
A=1.0
alpha=0.35
delta_annual=0.05
delta=1-((1-delta_annual)**20)



'''
Problem 1: Checking if feasible () works properly
'''

b_guess_1=np.array([1.0,1.2])
b_guess_2=np.array([0.06,-0.001])
b_guess_3=np.array([0.1,0.1])
f_params=(nvec,A,alpha,delta)
results1=np.array(ss.feasible(f_params,b_guess_1))
results2=np.array(ss.feasible(f_params,b_guess_2))
results3=np.array(ss.feasible(f_params,b_guess_3))
print (results1)
print (results2)
print (results3)

'''
Problem 2: Solving steady state
'''
# SS parameters
SS_tol = 1e-13
SS_graphs = False

bvec_guess = np.array([0.1, 0.1])
f_params = (nvec, A, alpha, delta)
b_cnstr, c_cnstr, K_cnstr = ss.feasible(f_params, bvec_guess)
ss_params = (beta, sigma, nvec, L, A, alpha, delta, SS_tol)
ss_output = ss.get_SS(ss_params, bvec_guess,SS_graphs)

ss_params2 = (0.55, sigma, nvec, L, A, alpha, delta, SS_tol)
ss_output2=ss.get_SS(ss_params2, bvec_guess,SS_graphs)



b_ss = ss_output['b_ss']
K_ss = ss_output['K_ss']
w_ss = ss_output['w_ss']
r_ss = ss_output['r_ss']
c_ss = ss_output['c_ss']



'''
Problem 3: Solving TPI
'''
# TPI params
T = 30
max_iter = 300
mindist_TPI = 1e-9
xi = 0.5
b1vec = np.array([0.8 * b_ss[0], 1.1 * b_ss[1]])
TPI_graph=False

Kpath, r_path, w_path=tpi.get_TPI((T, beta, sigma, nvec, L, A, alpha, delta, b_ss, K_ss,
     max_iter, mindist_TPI, xi),b1vec)

if TPI_graph:
    plt.plot(1 + np.arange(T + 5), np.append(Kpath[:T-2], K_ss * np.ones(7)), 'go--')
    plt.grid(b=True, which='major', color='0.65', linestyle='-')
    plt.title('Time path of aggregate capital', fontsize=20)
    plt.xlabel(r'$t$')
    plt.ylabel(r'$K$')
    plt.show()

    plt.plot(1 + np.arange(T + 5), np.append(w_path[:T-2], w_ss * np.ones(7)), 'go--')
    plt.grid(b=True, which='major', color='0.65', linestyle='-')
    plt.title('Time path of wage', fontsize=20)
    plt.xlabel(r'$t$')
    plt.ylabel(r'$w$')
    plt.show()

    plt.plot(1 + np.arange(T + 5), np.append(r_path[:T-2], r_ss * np.ones(7)), 'go--')
    plt.grid(b=True, which='major', color='0.65', linestyle='-')
    plt.title('Time path of interest rate', fontsize=20)
    plt.xlabel(r'$t$')
    plt.ylabel(r'$r$')
    plt.show()


# weizhi=np.where( Kpath-K_ss < 0.00001 )
# k_first=np.zeros_like(Kpath)
# k_first = [1 for k in Kpath if abs(k - K_ss) < 0.00001]
# print(k_first)
k_first = [k for k in Kpath if abs(k - K_ss) < 0.00001][0]
print(k_first)
T1 = np.where(Kpath == k_first)[0][0]
print(T1+1)

k_last = [k for k in Kpath[:-3] if abs(k- K_ss) > 0.00001][-1]
print(k_last)
T2 = np.where(Kpath == k_last)[0][0]
print(T2+1)