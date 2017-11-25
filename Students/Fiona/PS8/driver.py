import demographics as demog
import SS as ss
import TPI as tpi
import time
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from mpl_toolkits.mplot3d import Axes3D
import sys
import os

#time parameters
E= 20
S =80
tot_per= E+S
T = 320
#population parameters
fert_graph = False
mort_graph = False
imm_graph = False
omega_graph = False
# df_mort = pd.read_csv('mort_rates2011.csv')
# df_pop = pd.read_csv('pop_data.csv')
fert_rates = demog.get_fert(tot_per, fert_graph) [E:]
mort_rates0, infant_mort = demog.get_mort(tot_per,  mort_graph)
# print (mort_rates)
mort_rates = mort_rates0 [E:]
imm_rates = demog.get_imm_resid(tot_per, imm_graph) [E:]
omega_SS0, om_mat, omega_hat_path0, g_vec0, imm_path = demog.get_omega(E, S, T, omega_graph)
omega_SS = omega_SS0 [E:]
omega_hat_path = omega_hat_path0 [E:,:]
g_vec = g_vec0 [E:]
print (f'mort:{mort_rates.shape}, omega_SS:{omega_SS.shape}')
#HH parameters
beta_annual = .96
beta = beta_annual ** (80 / S)
sigma = 2.2
nvec = np.ones(S)
cut = round(2 * S / 3)
nvec[cut: ] = 0.2
chi_vec_bq = np.ones(S)

# Firm Parameters
alpha = 0.35
A = 1.0
delta_annual = 0.05
delta = 1- ((1 - delta_annual) ** (80 / S))

# Economic Growth
g_y = 0.03

# SS parameters
SS_solve = True
SS_tol = 1e-13
SS_graphs = True
SS_EulDiff = True

# TPI parameters
TPI_solve = False
TPI_tol = 1e-13
maxiter_TPI = 200
mindist_TPI = 1e-13
xi_TPI = 0.20
TPI_graphs = True
TPI_EulDiff = True
xi=0.99

'''
Problem 2: Solving steady state
'''

bvec_guess = np.array([-0.01, 0.1, 0.2, 0.23, 0.25, 0.23, 0.2, 0.1,
           -0.01, 0.1, 0.2, 0.23, 0.25, 0.23, 0.2, 0.1,
           -0.01, 0.1, 0.2, 0.23, 0.25, 0.23, 0.2, 0.1,
           -0.01, 0.1, 0.2, 0.23, 0.25, 0.23, 0.2, 0.1,
           -0.01, 0.1, 0.2, 0.23, 0.25, 0.23, 0.2, 0.1,
           -0.01, 0.1, 0.2, 0.23, 0.25, 0.23, 0.2, 0.1,
           -0.01, 0.1, 0.2, 0.23, 0.25, 0.23, 0.2, 0.1,
           0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1,
           0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1,
           0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
print (f'bvecguess:{bvec_guess.shape}')

# f_params = (nvec, A, alpha, delta, bq_distr, beta)
# b_cnstr, c_cnstr, K_cnstr = ss.feasible(f_params, bvec_guess)
#beta, sigma, nvec, A, alpha, delta, SS_tol, chi, fert_rates, mort_rates, imm_rates, omega_SS, gn_SS, g_y
ss_params = (beta, sigma, nvec, A, alpha, delta, SS_tol,  fert_rates, mort_rates, imm_rates, omega_SS, g_vec[-1], g_y)
ss_output = ss.get_SS(ss_params, bvec_guess, SS_graphs)



b_ss = np.append([0],ss_output['b_ss'])
K_ss = ss_output['K_ss']
w_ss = ss_output['w_ss']
r_ss = ss_output['r_ss']
c_ss = ss_output['c_ss']
C_ss = ss_output['C_ss']
Y_ss = ss_output['Y_ss']
EulErr_ss = np.append([0],ss_output['EulErr_ss'])
RCerr_ss = ss_output['RCerr_ss']
BQ_ss = ss_output['BQ_ss']


results = np.zeros((S,10))


results[:, 0] = b_ss.T
results[:, 1] = c_ss.T
results[:, 2] = EulErr_ss.T
results[:, 3] = RCerr_ss.T
results[:, 4] = w_ss.T
results[:, 5] = r_ss.T
results[:, 6] = K_ss.T
results[:, 7] = Y_ss.T
results[:, 8] = C_ss.T
results[:, 9] = BQ_ss.T


np.savetxt("ss.csv", results, delimiter=",", header="b_ss,c_ss,EulErr_ss,RCerr_ss,w_ss,r_ss,K_ss,Y_ss,C_ss,BQ_ss")

'''
Problem 3: Solving TPI
'''
# TPI params
max_iter = 300
mindist_TPI = 1e-9
ages=np.arange(1,81)
x_s = (1.5-0.87)/78*(ages-2)+0.87
# b1vec: s=2 ~ S+1
b1vec = b_ss[1:] * x_s[1:]
print (b1vec.shape)

TPI_graph=True

#
TPI_output=tpi.get_TPI((S, T, beta, sigma, nvec, A, alpha, delta, b_ss, K_ss, C_ss, BQ_ss, omega_SS, \
    maxiter_TPI, mindist_TPI, xi, TPI_tol, fert_rates, mort_rates, imm_rates, g_y, omega_hat_path, g_vec), b1vec,TPI_graph)

Kpath=TPI_output['Kpath']
Ypath=TPI_output['Ypath']
Cpath=TPI_output['Cpath']
w_path=TPI_output['wpath']
r_path=TPI_output['rpath']
bpath=TPI_output['bpath']
# print(bpath.shape)
cpath=TPI_output['cpath']
# print(cpath.shape)

if TPI_graph:

    # Create directory if images directory does not already exist
    cur_path = os.path.split(os.path.abspath(__file__))[0]
    output_fldr = "images"
    output_dir = os.path.join(cur_path, output_fldr)
    if not os.access(output_dir, os.F_OK):
        os.makedirs(output_dir)

    # Plot time path of aggregate capital stock
    tvec = np.linspace(1, T + 5, T + 5)
    minorLocator = MultipleLocator(1)
    fig, ax = plt.subplots()
    plt.plot(tvec, Kpath[:T + 5], marker='D')
    # for the minor ticks, use no labels; default NullFormatter
    ax.xaxis.set_minor_locator(minorLocator)
    plt.grid(b=True, which='major', color='0.65', linestyle='-')
    plt.title('Time path for aggregate capital stock K')
    plt.xlabel(r'Period $t$')
    plt.ylabel(r'Aggregate capital $K_{t}$')
    output_path = os.path.join(output_dir, "Kpath")
    plt.savefig(output_path)
    # plt.show()

    # Plot time path of aggregate output (GDP)
    fig, ax = plt.subplots()
    plt.plot(tvec, Ypath[:T + 5], marker='D')
    # for the minor ticks, use no labels; default NullFormatter
    ax.xaxis.set_minor_locator(minorLocator)
    plt.grid(b=True, which='major', color='0.65', linestyle='-')
    plt.title('Time path for aggregate output (GDP) Y')
    plt.xlabel(r'Period $t$')
    plt.ylabel(r'Aggregate output $Y_{t}$')
    output_path = os.path.join(output_dir, "Ypath")
    plt.savefig(output_path)
    # plt.show()

    # Plot time path of aggregate consumption
    fig, ax = plt.subplots()
    plt.plot(tvec, Cpath[:T + 5], marker='D')
    # for the minor ticks, use no labels; default NullFormatter
    ax.xaxis.set_minor_locator(minorLocator)
    plt.grid(b=True, which='major', color='0.65', linestyle='-')
    plt.title('Time path for aggregate consumption C')
    plt.xlabel(r'Period $t$')
    plt.ylabel(r'Aggregate consumption $C_{t}$')
    output_path = os.path.join(output_dir, "C_aggr_path")
    plt.savefig(output_path)
    # plt.show()

    # Plot time path of real wage
    fig, ax = plt.subplots()
    plt.plot(tvec, w_path[:T + 5], marker='D')
    # for the minor ticks, use no labels; default NullFormatter
    ax.xaxis.set_minor_locator(minorLocator)
    plt.grid(b=True, which='major', color='0.65', linestyle='-')
    plt.title('Time path for real wage w')
    plt.xlabel(r'Period $t$')
    plt.ylabel(r'Real wage $w_{t}$')
    output_path = os.path.join(output_dir, "wpath")
    plt.savefig(output_path)
    # plt.show()

    # Plot time path of real interest rate
    fig, ax = plt.subplots()
    plt.plot(tvec, r_path[:T + 5], marker='D')
    # for the minor ticks, use no labels; default NullFormatter
    ax.xaxis.set_minor_locator(minorLocator)
    plt.grid(b=True, which='major', color='0.65', linestyle='-')
    plt.title('Time path for real interest rate r')
    plt.xlabel(r'Period $t$')
    plt.ylabel(r'Real interest rate $r_{t}$')
    output_path = os.path.join(output_dir, "rpath")
    plt.savefig(output_path)
    # plt.show()

    # Plot time path of individual savings distribution
    tgridT = np.linspace(1, T + 5, T + 5)
    sgrid2 = np.linspace(2, S, S - 1)
    tmatb, smatb = np.meshgrid(tgridT, sgrid2)
    cmap_bp = matplotlib.cm.get_cmap('summer')
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.set_xlabel(r'period-$t$')
    ax.set_ylabel(r'age-$s$')
    ax.set_zlabel(r'individual savings $b_{s,t}$')
    strideval = max(int(1), int(round(S / 10)))
    ax.plot_surface(tmatb, smatb, bpath[:, :T + 5], rstride=strideval,
                    cstride=strideval, cmap=cmap_bp)
    output_path = os.path.join(output_dir, "bpath")
    plt.savefig(output_path)
    # plt.show()

    # Plot time path of individual consumption distribution
    tgridTm1 = np.linspace(1, T + 4, T + 4)
    sgrid = np.linspace(1, S, S)
    tmatc, smatc = np.meshgrid(tgridTm1, sgrid)
    cmap_cp = matplotlib.cm.get_cmap('summer')
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.set_xlabel(r'period-$t$')
    ax.set_ylabel(r'age-$s$')
    ax.set_zlabel(r'individual consumption $c_{s,t}$')
    strideval = max(int(1), int(round(S / 10)))
    ax.plot_surface(tmatc, smatc, cpath[:, :T + 4],
                    rstride=strideval, cstride=strideval,
                    cmap=cmap_cp)
    output_path = os.path.join(output_dir, "cpath")
    plt.savefig(output_path)

    print('T', T)
    # Plot time path of b_15
    fig, ax = plt.subplots()
    plt.plot(tvec, bpath[15, :T + 5], marker='D')
    # for the minor ticks, use no labels; default NullFormatter
    ax.xaxis.set_minor_locator(minorLocator)
    plt.grid(b=True, which='major', color='0.65', linestyle='-')
    plt.title('Time path for savings of 15-year-olds')
    plt.xlabel(r'Period $t$')
    plt.ylabel(r'Savings for $b_{15,t}$')
    output_path = os.path.join(output_dir, "b15path")
    plt.savefig(output_path)
    # plt.show()
