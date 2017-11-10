'''
MACS 40000 Rick Evans
PSET6 Driver
Author: Fiona Fan
'''
import SS as ss
import TPI as tpi
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from mpl_toolkits.mplot3d import Axes3D
import sys
import os
import utils
# import script as util


# Household parameters
S = int(80)
beta_annual = 0.96
beta = beta_annual ** (80 / S)
sigma = 2.5
ncutper = round((2/3) * (S))
nvec = np.ones(S)
nvec[ncutper:] = 0.2
L=utils.get_L(nvec)
chi = np.ones(S)
# Firm parameters
A = 1.0
alpha = 0.35
delta_annual = 0.05
delta = 1 - ((1 - delta_annual) ** (80 / S))
# SS parameters
SS_solve = True
SS_tol = 1e-13
SS_graphs = True
SS_EulDiff = True
# TPI parameters
T = 320
TPI_solve = False
TPI_tol = 1e-13
maxiter_TPI = 200
mindist_TPI = 1e-13
xi_TPI = 0.20
TPI_graphs = True
TPI_EulDiff = True
xi=0.99
bq_distr= np.array([0.0, 0.00030482046238934716, 0.0, 0.0014992543498555824, 0.0, 0.041294651801817704, 0.0, 0.0, 0.0,
          0.00082980416811082199, 0.0013446395242258677, 0.0, 0.0, 0.0035335988976101301, 0.0037669631543512144,
          0.003893904016709956, 0.0, 0.0, 0.0016025350014449526, 3.9231710670095225e-06, 0.019958190815940594, 0.0,
          0.0027496784934702544, 0.00061493799132572504, 0.0, 0.0013484396071612496, 0.00067264117347193655,
          0.046217502605662694, 0.040118151035299997, 0.0, 0.058808416913566093, 0.014665109951514056,
          0.033668771198552189, 0.023917789912783313, 0.024729005611190624, 0.12417602338385574, 0.04371546929375824,
          0.20102421303552584, 0.032855438761641577, 0.0051497534735665358, 0.018306562653512277, 0.008127003141876676,
          0.079889427113757405, 0.024923819607154479, 0.00087284956444492155, 0.010343049511578765,
          0.0094987853694684897, 0.0010195349467771559, 0.00720069368022938, 0.0, 0.11009982506218106,
          0.0001729770901235561, 0.0, 0.05147303229903729, 0.0, 0.0052391449707809698, 0.0, 0.0, 0.0037838269233418102,
          0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
print (bq_distr.shape)


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
           0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
print (bvec_guess.shape)

f_params = (nvec, A, alpha, delta, bq_distr, beta)
# b_cnstr, c_cnstr, K_cnstr = ss.feasible(f_params, bvec_guess)
ss_params = (beta, sigma, nvec, L, A, alpha, delta, SS_tol, bq_distr, chi)
ss_output = ss.get_SS(ss_params, bvec_guess, SS_graphs)



b_ss = ss_output['b_ss']
K_ss = ss_output['K_ss']
w_ss = ss_output['w_ss']
r_ss = ss_output['r_ss']
c_ss = ss_output['c_ss']
C_ss = ss_output['C_ss']
Y_ss = ss_output['Y_ss']
EulErr_ss = ss_output['EulErr_ss']
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


np.savetxt("problem2.csv", results, delimiter=",", header="b_ss,c_ss,EulErr_ss,RCerr_ss,w_ss,r_ss,K_ss,Y_ss,C_ss,BQ_ss")
#
'''
Problem 3: Solving TPI
'''
# TPI params
max_iter = 300
mindist_TPI = 1e-9
ages=np.arange(1,81)
x_s = (1.5-0.87)/78*(ages-2)+0.87
# b1vec: s=2 ~ S+1
b1vec = b_ss * x_s

TPI_graph=True
#
TPI_output=tpi.get_TPI((S, T, beta, sigma, nvec, L, A, alpha, delta, b_ss, K_ss, C_ss, BQ_ss,
                        maxiter_TPI, mindist_TPI, xi, TPI_tol, bq_distr, chi), b1vec,TPI_graph)

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


# weizhi=np.where( Kpath-K_ss < 0.00001 )
# k_first=np.zeros_like(Kpath)
# k_first = [1 for k in Kpath if abs(k - K_ss) < 0.00001]
# print(k_first)
# k_first = [k for k in Kpath if abs(k - K_ss) < 0.00001][0]
# print(k_first)
# T1 = np.where(Kpath == k_first)[0][0]
# print(T1+1)
#
# k_last = [k for k in Kpath[:-3] if abs(k- K_ss) > 0.00001][-1]
# print(k_last)
# T2 = np.where(Kpath == k_last)[0][0]
# print(T2+1)

