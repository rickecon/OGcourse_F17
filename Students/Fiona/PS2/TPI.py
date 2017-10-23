'''
MACS 40000 Rick Evans
PSET 2
Solving TPI for 3-period model
Author: Fiona Fan
'''
import time
import numpy as np
import scipy.optimize as opt
import SS as ss
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator


def getPath(x1,xT,T):
    return np.linspace(int(x1), int(xT), int(T))


def b3_error(b3, *args):

    nvec, beta, sigma, b2, w_path, r_path = args
    n2, n3 = nvec
    w1, w2 = w_path
    r1, r2 = r_path
    c2 = (1 + r1) * b2 + w1 * n2 - b3
    c3 = (1 + r2) * b3 + w2 * n3
    muc2, muc3 = ss.get_MUc(np.array([c2, c3]), sigma)
    error = muc2 - beta * (1 + r2) * muc3
    return error
def zero_func(bvec,*args):
    nvec, beta, sigma, w_path, r_path = args
    b = np.append([0], bvec)
    b1 = np.append(bvec, [0])
    cvec = (1 + np.append([0], r_path)) * b + w_path * nvec - b1

    muc = cvec ** (-sigma)
    errors = muc[:-1] - beta * (1 + r_path) * muc[1:]

    return errors


# def get_cbepath(params,rpath,wpath):
#     T, beta, sigma, nvec, bvec1, b_ss, TPI_tol, EulDiff = params


def get_TPI(params,bvec):
    start_time = time.clock()
    (T, beta, sigma, nvec, L, A, alpha, delta, b_ss, K_ss,
     maxiter_TPI, mindist_TPI, xi) = params

    abs2 = 1
    tpi_iter = 0
    L = ss.get_L(nvec)
    K1 = ss.get_K(bvec)[0]
    Kpath_old = np.zeros(T + 1)
    Kpath_old[:-1] = getPath(K1, K_ss, T)
    Kpath_old[-1] = K_ss

    while abs2 > mindist_TPI and tpi_iter < maxiter_TPI:
        tpi_iter += 1
        w_path = ss.get_w((A, alpha), Kpath_old, L)
        r_path = ss.get_r((A, alpha, delta), Kpath_old, L)
        b = np.zeros([2,T+1])
        #print(bvec)
        b[:,0] = bvec

        b32 = opt.root(b3_error, b[1,0], args=(nvec[1:], beta, sigma, b[0, 0], w_path[:2], r_path[:2]))
        b[1, 1] = b32.x

        for t in range(T - 1):
            bvec_guess = np.array([b[0,t], b[1,t + 1]])
            bt = opt.root(zero_func, bvec_guess, (nvec, beta, sigma, w_path[t: t + 3], r_path[t + 1: t + 3]))
            bgrid=np.eye(2,dtype=bool)
            b[:,t+1:t+3]=bgrid*bt.x+b[:,t+1:t+3]

        # Calculate the implied capital stock from conjecture and the error
        Kpath_new = b.sum(axis=0)
        abs2 = ((Kpath_old[:] - Kpath_new[:]) ** 2).sum()
        # Update guess
        Kpath_old = xi * Kpath_new + (1 - xi) * Kpath_old
        print('iteration:', tpi_iter, ' squared distance: ', abs2)

    tpi_time = time.clock() - start_time

    return Kpath_old, r_path, w_path