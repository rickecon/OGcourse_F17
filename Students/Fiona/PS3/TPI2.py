'''
MACS 40000 Rick Evans
PSET 3
Solving TPI for S-period model
Author: Fiona Fan
'''
import time
import numpy as np
import scipy.optimize as opt
import SS as ss
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from mpl_toolkits.mplot3d import Axes3D
import sys
import os

def getPath(x1,xT,T):
    return np.linspace(int(x1), int(xT), int(T))


def EulerSys(bvec, *args):
    beta, sigma, beg_wealth, nvec, rpath, wpath= args
    bvec2 = np.append(beg_wealth, bvec)
    # print(nvec.shape)
    cvec, c_cnstr = get_cvec(rpath, wpath, nvec, bvec2)
    b_err_params = (beta, sigma)
    b_err_vec = ss.get_b_errors(b_err_params, rpath[1:], cvec,
                                c_cnstr)
    return b_err_vec

def get_cvec(rpath, wpath, nvec, bvec):

    b_s = bvec
    b_sp1 = np.append(bvec[1:], [0])
    cvec = (1 + rpath) * b_s + wpath * nvec - b_sp1
    c_cnstr = cvec <= 0
    return cvec, c_cnstr

def solver(params, beg_age, beg_wealth, nvec, rpath, wpath, b_init):

    S, beta, sigma, TPI_tol = params
    p = int(S - beg_age + 1)
    b_guess = 1.01 * b_init
    eullf_objs = (beta, sigma, beg_wealth, nvec, rpath, wpath)
    bpath = opt.root(EulerSys, b_guess, args=(eullf_objs)).x
    bvec=np.append(beg_wealth, bpath)
    cpath, c_cnstr = get_cvec(rpath, wpath, nvec, bvec)
    b_err_params = (beta, sigma)
    b_err_vec = ss.get_b_errors(b_err_params, rpath[1:], cpath, c_cnstr)
    return bpath, cpath, b_err_vec


def get_cbepath(params,rpath,wpath):
    S, T, beta, sigma, nvec, bvec1, b_ss, TPI_tol = params
    cpath = np.zeros((S, T + S - 2))
    bpath=np.zeros((S - 1, T + S - 2))
    EulErrPath = np.zeros((S - 1, T + S - 2))
    bpath[:, 0] = bvec1
    cpath[S - 1, 0] = ((1 + rpath[0]) * bvec1[S - 2] + wpath[0] * nvec[S - 1])
    sol_params = (S, beta, sigma, TPI_tol)

    #solve remaining lifetime decisions
    for p in range(2, S):
        b_guess = np.diagonal(bpath[S - p:, :p - 1])
        bveclf, cveclf, b_err_veclf = solver(
            sol_params, S - p + 1, bvec1[S - p - 1], nvec[-p:],
            rpath[:p], wpath[:p], b_guess)

        DiagMaskb = np.eye(p - 1, dtype=bool)
        DiagMaskc = np.eye(p, dtype=bool)
        bpath[S - p:, 1:p] = DiagMaskb * bveclf + bpath[S - p:, 1:p]
        cpath[S - p:, :p] = DiagMaskc * cveclf + cpath[S - p:, :p]
        EulErrPath[S - p:, 1:p] = (DiagMaskb * b_err_veclf +
                                   EulErrPath[S - p:, 1:p])

    # solve complete lifetime decisions
    DiagMaskb = np.eye(S - 1, dtype=bool)
    DiagMaskc = np.eye(S, dtype=bool)
    for t in range(1, T):  # Go from periods 1 to T-1
        b_guess = np.diagonal(bpath[:, t - 1:t + S - 2])
        bveclf, cveclf, b_err_veclf = solver(
            sol_params, 1, 0, nvec, rpath[t - 1:t + S - 1],
            wpath[t - 1:t + S - 1], b_guess)
        bpath[:, t:t + S - 1] = (DiagMaskb * bveclf +
                                 bpath[:, t:t + S - 1])
        cpath[:, t - 1:t + S - 1] = (DiagMaskc * cveclf +
                                     cpath[:, t - 1:t + S - 1])
        EulErrPath[:, t:t + S - 1] = (DiagMaskb * b_err_veclf +
                                      EulErrPath[:, t:t + S - 1])

    return cpath, bpath, EulErrPath

def get_TPI(params,bvec,graphs):
    start_time = time.clock()
    (S,T, beta, sigma, nvec, L, A, alpha, delta, b_ss, K_ss,C_ss,
     maxiter_TPI, mindist_TPI, xi,TPI_tol) = params

    abs2 = 1
    tpi_iter = 0
    cbe_params = (S, T, beta, sigma, nvec, bvec, b_ss, TPI_tol)
    L = ss.get_L(nvec)
    K1 = ss.get_K(bvec)[0]
    Kpath_old = np.zeros(T + S-2)
    Kpath_old[:T] = getPath(K1, K_ss, T)
    Kpath_old[T:] = K_ss
    Kpath_new=Kpath_old.copy()

    while abs2 > mindist_TPI and tpi_iter < maxiter_TPI:
        tpi_iter += 1
        w_path = ss.get_w((A, alpha), Kpath_old, L)
        r_path = ss.get_r((A, alpha, delta), Kpath_old, L)
        Kpath_old = xi * Kpath_new + (1 - xi) * Kpath_old
        cpath, bpath, EulErrPath = get_cbepath(cbe_params, r_path, w_path)
        #print(bvec)
        Kpath_new = np.zeros(T + S - 2)
        Kpath_new[:T] = ss.get_K(bpath[:, :T])[0]
        Kpath_new[T:] = K_ss
        abs2 = (((Kpath_new[1:T] - Kpath_old[1:T])/Kpath_old[1:T]*100) ** 2).sum()
        print('iteration:', tpi_iter, 'squared pct deviation sum: ', abs2)

    tpi_time = time.clock() - start_time
    print(f'It took {tpi_time} seconds to run.')

    Kpath = Kpath_new
    Ypath = ss.get_Y((A, alpha), Kpath, L)
    Cpath = np.zeros(T + S - 2)
    Cpath[:T - 1] = ss.get_C(cpath[:, :T - 1])
    Cpath[T - 1:] = C_ss * np.ones(S - 1)
    RCerrPath = (Ypath[:-1] - Cpath[:-1] - Kpath[1:] +
                 (1 - delta) * Kpath[:-1])
    tpi_output = {
        'bpath': bpath, 'cpath': cpath, 'wpath': w_path, 'rpath': r_path,
        'Kpath': Kpath_new, 'Ypath': Ypath, 'Cpath': Cpath,
        'EulErrPath': EulErrPath, 'RCerrPath': RCerrPath,
        'tpi_time': tpi_time}

    return tpi_output