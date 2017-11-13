import numpy as np
import scipy.optimize as opt
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import os



'''
Stiching functions for U(c) and g(b)
'''
def MU_c_stitch(cvec, sigma):
    epsilon = 1e-4
    c_cnstr = cvec < epsilon
    muc = cvec ** (-sigma)
    # print (f"cvec:{cvec.shape}")
    m1 = (-sigma) * epsilon ** (-sigma - 1)
    m2 = epsilon ** (-sigma) - m1 * epsilon
    muc[c_cnstr] = m1 * cvec[c_cnstr] + m2

    return muc

'''
Firm utilities:
calculate wage and interest rate
'''

def get_w(K, L, params):
    A, alpha = params
    w = (1 - alpha) * A * (K/L) **alpha

    return w

def get_r(K, L, params):
    A, alpha, delta = params
    r = alpha * A * ((L/K) ** (1 - alpha)) - delta
    return r

'''
Household utilities
'''
# #Calculate the remaning consumptions in one's lifetime.


def get_cvec_ss(r, w, bvec, nvec, bq_distr):
    #bvec=b2 ~ bS+1
    #cvec:c1 ~ cS+1
    #nvec:n1 ~ nS+1
    b=np.append([0],bvec[:-1])
    b1=bvec

    bq = (1 + r) * bvec[-1]
    # print(f"bvec shape is: {bvec.shape}, b shape is: {b.shape}, bq {bq}, bq_distr {bq_distr.shape}, w {w}, nvec {nvec.shape}, r {r}")

    c_vec = (1 + r) * b + w * nvec - b1 + bq_distr * bq
    c_cnstr = c_vec <= 0
    # print(f"cvec shape is: {c_vec.shape}")
    return c_vec, c_cnstr

#
def get_cvec_tpi(rpath, wpath, bvec, nvec, bq, bq_distr):
    # bq_distr, beta = params
    #bvec=b1 ~ bS+1
    #cvec:c1 ~ cS
    #nvec:n1 ~ nS
    b=bvec[:-1]
    b1=bvec[1:]
    # print(f"bvec shape is: {bvec.shape}, b shape is: {b.shape}, bq {bq}, bq_distr {bq_distr.shape}, w {w}, nvec {nvec.shape}, r {r}") c_vec: {c_vec.shape},
    # print (f'b: {b.shape}, rpath{rpath.shape}, wpath:{wpath.shape}, bq_distr{bq_distr.shape}, bq{bq.shape}, nvec{nvec.shape}')
    c_vec = (1 + rpath) * b + wpath * nvec - b1 + bq * bq_distr
    c_cnstr = c_vec <= 0
    # print(f"cvec shape is: {c_vec.shape}")
    return c_vec, c_cnstr


######## Euler Equations


def EulerSys_ss(vec, *args):
    beta, sigma, nvec, A, alpha, delta, bq_distr, chi=args
    K= get_K(vec)[0]
    L=get_L(nvec)
    r=get_r(K,L, (A, alpha, delta))
    w=get_w(K,L, (A, alpha))
    # cvec:c1 ~ cS
    cvec,c_cnstr=get_cvec_ss(r, w, vec, nvec, bq_distr)
    errors=np.zeros(np.shape(vec))
    # MU_c=MU_c_stitch(cvec,sigma)
    errors[:-1] = beta*(1+r)*MU_c_stitch(cvec[1:],sigma)-MU_c_stitch(cvec[:-1],sigma)
    errors[-1] = chi[-1] * vec[-1] ** (-sigma)- cvec[-1] ** (-sigma)
    return errors

# def EulerSys_tpi(bvec, *args):
#     #bvec: 2 ~ S+1
#     beta, sigma, beg_wealth, nvec, rpath, wpath, bq_distr, chi, bq = args
#     #bvec2: 1 ~ S+1
#     bvec2 = np.append(beg_wealth, bvec)
#     #cvec: 1 ~ S
#     cvec, c_cnstr = get_cvec_tpi(rpath, wpath, nvec, bvec2, bq)
#     errors = np.zeros(np.shape(bvec))
#     #errors: 1 ~ S
#     errors[:-1] = beta * (1 + rpath[1:]) * MU_c_stitch(cvec[1:], sigma) - MU_c_stitch(cvec[:-1], sigma)
#     errors[-1] = chi[-1] * bvec2[-1] ** (-sigma) - cvec[-1] ** (-sigma)
#     return errors

def EulerSys_tpi(bvec, *args):
    beg_wealth, nvec, beta, sigma, wpath, rpath, BQpath, chi, bq_distr = args
    b = np.append(beg_wealth, bvec)
    # print (f'b: {b.shape}, BQ:{BQpath.shape}, chi:{chi.shape}, bq_distr:{bq_distr.shape}, bvec: {bvec.shape}',rpath.shape)
    # cvec = get_cvec_ss(rpath, wpath, bvec, nvec, bq_distr)
    cvec = (1 + rpath) * b[:-1] + wpath * nvec + bq_distr * BQpath - bvec
    muc = MU_c_stitch(cvec, sigma)
    errors = np.zeros(np.shape(cvec))
    errors[:-1] = muc[:-1] - beta * (1 + rpath[1:]) * muc[1:]
    errors[-1] = chi[-1] * (b[-1]) ** (-sigma)- ((1 + rpath[-1]) * b[-2] + wpath[-1] * nvec[-1] + bq_distr[-1] *
                                                 BQpath[-1] - b[-1]) ** (-sigma)

    return errors


######## Calculate aggregates

#from s=1 to S
def get_L(narr): # function for aggregate labor supply
    epsilon = 0.01
    a = epsilon / np.exp(1)
    b = 1 / epsilon
    if narr.ndim == 1:  # This is the steady-state case
        L = narr.sum()
        L_cstr = L < epsilon
        if L_cstr:
            print('get_L() warning: Distribution of savings and/or ' +
                  'parameters created L < epsilon')
            # Force K >= eps by stitching a * exp(b * K) for K < eps
            L = a * np.exp(b * L)

    elif narr.ndim == 2:  # This is the time path case
        L = narr.sum(axis=0)
        L_cstr = L < epsilon
        if L.min() < epsilon:
            print('get_L() warning: Aggregate capital constraint is ' +
                  'violated (L < epsilon) for some period in time ' +
                  'path.')
            L[L_cstr] = a * np.exp(b * L[L_cstr])
    return L

# from s=2 to S+1
def get_K(barr): # function for aggregate capital supply
    epsilon = 0.01
    a = epsilon / np.exp(1)
    b = 1 / epsilon
    if barr.ndim == 1:  # This is the steady-state case
        K = barr.sum()
        K_cstr = K < epsilon
        if K_cstr:
            print('get_K() warning: Distribution of savings and/or ' +
                  'parameters created K < epsilon')
            # Force K >= eps by stitching a * exp(b * K) for K < eps
            K = a * np.exp(b * K)

    elif barr.ndim == 2:  # This is the time path case
        K = barr.sum(axis=0)
        K_cstr = K < epsilon
        if K.min() < epsilon:
            print('get_K() warning: Aggregate capital constraint is ' +
                  'violated (K < epsilon) for some period in time ' +
                  'path.')
            K[K_cstr] = a * np.exp(b * K[K_cstr])

    return K, K_cstr

def get_Y(K, L, params): # function for aggregate output
    A, alpha = params
    Y = A * (K ** alpha) * (L ** (1 - alpha))
    return Y

def get_C(carr):
    if carr.ndim == 1:
        C = carr.sum()
    elif carr.ndim == 2:
        C = carr.sum(axis=0)

    return C
