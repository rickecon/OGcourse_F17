import numpy as np
import scipy.optimize as opt
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import os


'''
Ellip filling for disutility of labor
'''
def fit_ellip(elast_Frisch, l_tilde):
    b_i=0.1
    upsi_i=0.3
    l_sup=np.linspace(0.05,0.95, 1000)
    fargs=(elast_Frisch,l_tilde,l_sup)
    bnds=((1e-10, None), (1e-10, None))
    params_init=np.array([b_i,upsi_i])
    result=opt.minimize(MU_err,params_init,args=(fargs),bounds=bnds)
    b_ellip, upsilon=result.x
    sum_err=result.fun
    return b_ellip, upsilon


def MU_err (params,*args):
    bi, ui = params
    theta, l_tilde, l_sup=args
    MU_cfe= l_sup**(1/theta)
    MU_elp=1/l_sup*bi*((l_sup/l_tilde)**ui)*(1-(l_sup/l_tilde)**ui)**(1/ui-1)
    sum_sqerr=((MU_cfe-MU_elp)**2).sum()
    return sum_sqerr

'''
Stiching functions for U(c) and g(b)
'''
def MU_c_stitch(cvec, sigma):
    epsilon = 1e-4
    c_cnstr = cvec < epsilon
    muc = cvec ** (-sigma)
    m1 = (-sigma) * epsilon ** (-sigma - 1)
    m2 = epsilon ** (-sigma) - m1 * epsilon
    muc[c_cnstr] = m1 * cvec[c_cnstr] + m2

    return muc

def MU_n_stitch(nvec, params):
    l_tilde, b, upsilon = params
    epsilon_lb = 1e-6
    epsilon_ub = l_tilde - epsilon_lb
    nl_cstr = nvec < epsilon_lb
    nu_cstr = nvec > epsilon_ub

    mun = ((b / l_tilde) * ((nvec / l_tilde) ** (upsilon - 1)) * (1 - ((nvec / l_tilde) ** upsilon)) **\
           ((1 - upsilon) / upsilon))
    m1 = (b * (l_tilde ** (-upsilon)) * (upsilon - 1) * (epsilon_lb ** (upsilon - 2)) * \
         ((1 - ((epsilon_lb / l_tilde) ** upsilon)) ** ((1 - upsilon) / upsilon)) * \
         (1 + ((epsilon_lb / l_tilde) ** upsilon) * ((1 - ((epsilon_lb / l_tilde) ** upsilon)) ** (-1))))
    m2 = ((b / l_tilde) * ((epsilon_lb / l_tilde) ** (upsilon - 1)) * \
         ((1 - ((epsilon_lb / l_tilde) ** upsilon)) ** ((1 - upsilon) / upsilon)) - (m1 * epsilon_lb))
    q1 = (b * (l_tilde ** (-upsilon)) * (upsilon - 1) * (epsilon_ub ** (upsilon - 2)) * \
         ((1 - ((epsilon_ub / l_tilde) ** upsilon)) ** ((1 - upsilon) / upsilon)) * \
         (1 + ((epsilon_ub / l_tilde) ** upsilon) * ((1 - ((epsilon_ub / l_tilde) ** upsilon)) ** (-1))))
    q2 = ((b / l_tilde) * ((epsilon_ub / l_tilde) ** (upsilon - 1)) * \
         ((1 - ((epsilon_ub / l_tilde) ** upsilon)) ** ((1 - upsilon) / upsilon)) - (q1 * epsilon_ub))
    mun[nl_cstr] = m1 * nvec[nl_cstr] + m2
    mun[nu_cstr] = q1 * nvec[nu_cstr] + q2
    return mun

'''
Firm utilities:
calculate wage and interest rate
'''

def get_w(r, params):
    A, alpha, delta = params
    w = (1 - alpha) * A * (((alpha * A) / (r + delta)) ** (alpha / (1 - alpha)))
    return w

def get_r(K, L, params):
    A, alpha, delta = params
    r = alpha * A * ((L / K) ** (1 - alpha)) - delta
    return r

'''
Household utilities
'''
#Calculate the remaning consumptions in one's lifetime.
def get_c(c1, r, beta, sigma, p):
    cvec = np.zeros(p)
    cvec[0] = c1
    cs = c1
    for s in range (p - 1):
        cvec[s + 1] = cs * (beta * (1 + r[s + 1])) ** (1 / sigma)
        cs = cvec[s + 1]
    return cvec

def get_b(cvec, nvec, r, w, p, bs = 0.0): # function for calculating lifetime savings, given consumption and labor decisions
    bvec = np.zeros(p)
    bvec[0] = bs
    for s in range (p-1):
        bvec[s + 1] = (1 + r[s]) * bs + w[s] * nvec[s] - cvec[s]
        bs = bvec[s + 1]
        s += 1
    return bvec

def get_b_last(c1, *args): # function for last-period savings, given intial guess c1
    r, w, beta, sigma, l_ub, b, upsilon, p, bs, chi = args
    cvec = get_c(c1, r, beta, sigma, p)
    nvec = get_n(cvec, sigma, l_ub, b, upsilon, w, p, chi)
    bvec = get_b(cvec, nvec, r, w, p, bs)
    b_last = (1 + r[-1]) * bvec[-1] + w[-1] * nvec[-1] - cvec[-1]
    return b_last

######## Euler Equations

def get_n_errors(nvec, *args): # function for calculating intratemporal euler error
    cvec, sigma, l_ub, b, upsilon, w, chi = args
    muc = MU_c_stitch(cvec, sigma)
    mun = MU_n_stitch(nvec, (l_ub, b, upsilon))
    n_errors = w * muc - chi*mun
    return n_errors

def get_n(cvec, sigma, l_ub, b, upsilon, w, p, chi): # function for labor supply, calculated from intratemporal euler, given path of lifetime consumption
    n_args = (cvec, sigma, l_ub, b, upsilon, w, chi)
    n_guess = 0.5 * l_ub * np.ones(p)
    result = opt.root(get_n_errors, n_guess, args = (n_args), method = 'lm')
    if result.success:
        nvec = result.x
    else:
        raise ValueError("failed to find an appropriate labor decision")
    return nvec

def get_b_errors(cvec, r, *args): # function for calculating intertemporal euler error
    beta, sigma = args
    mu_c = MU_c_stitch(cvec[:-1], sigma)
    mu_c1 = MU_c_stitch(cvec[1:], sigma)

    b_errors = beta * (1 + r) * mu_c1 - mu_c

    return b_errors


######## Calculate aggregates

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
    return L, L_cstr

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
