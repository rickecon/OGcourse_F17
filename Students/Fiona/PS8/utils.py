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

def get_w(K, L, alpha, A):
    return (1-alpha) * A * ((K / L)**alpha)

def get_r(K,L, alpha, delta, A):
    return alpha * A* ((L/ K)**(1-alpha)) - delta

'''
Household utilities
'''
# #Calculate the remaning consumptions in one's lifetime.

def get_BQ(bvec, r, g_n, omhat, mort_rate):
    # print (f'mort:{mort_rate.shape}, omht:{omhat.shape}, bvec:{bvec.shape}')
    return ((1 + r) / (1 + g_n)) * (mort_rate[:-1] * omhat[:-1] * bvec).sum()


def get_cvec_ss(r, w, bvec, nvec, g_n, g_y, omhat, mort_rate):
    #bvec=b2 ~ bS+1
    #cvec:c1 ~ cS+1
    #nvec:n1 ~ nS+1
    b=np.append([0],bvec)
    b1=np.append(bvec, [0])
    # print ( f'In getting BQ in cvec, bvec:{bvec.shape}, r:{r}, gn:{g_n}, omhat:{omhat.shape}, mort_rate:{mort_rate.shape}')
    bq = get_BQ(bvec, r, g_n, omhat, mort_rate)
    # print(f"In getting cvec, bvec shape is: {bvec.shape}, b shape is: {b.shape}, bq {bq}, w {w}, nvec {nvec.shape}, r {r}")

    c_vec = (1 + r) * b + w * nvec - b1*np.exp(g_y) + bq/get_N(omhat)
    c_cnstr = c_vec <= 0
    # print(f"cvec shape is: {c_vec.shape}")
    return c_vec, c_cnstr

#
def get_cvec_tpi(rpath, wpath, bvec, nvec,  g_y, bq, omhat):
    # bq_distr, beta = params
    #bvec=b1 ~ bS+1TypeError: fsolve: there is a mismatch between the input and output shape of the 'func' argument 'EulerSys_ss'.Shape should be (79,) but it is (78,).

    #cvec:c1 ~ cS
    #nvec:n1 ~ nS
    b=np.append([0],bvec)
    # b1=np.append(bvec,[0])
    # print(f"bvec shape is: {bvec.shape}, b shape is: {b.shape}, bq {bq}, bq_distr {bq_distr.shape}, w {w}, nvec {nvec.shape}, r {r}") c_vec: {c_vec.shape},
    # print (f'b: {b.shape}, rpath{rpath.shape}, wpath:{wpath.shape}, bq_distr{bq_distr.shape}, bq{bq.shape}, nvec{nvec.shape}')
    # print (f'bq: {bq.shape}, b:{b.shape}, nvec:{nvec.shape},wpath:{wpath.shape},rpath:{rpath.shape}', bvec.shape)
    c_vec = (1 + rpath[:-1]) * b[:-1] + wpath[:-1] * nvec[:-1] - bvec*np.exp(g_y) + bq[:-1] /get_N(omhat)
    c_cnstr = c_vec <= 0
    # print(f"cvec shape is: {c_vec.shape}")
    return c_vec, c_cnstr


######## Euler Equations


def EulerSys_ss(vec, *args):
    beta, sigma, nvec, A, alpha, delta, g_n,g_y,omhat,mort_rate, imm_rates=args
    K= get_K(vec, omhat, g_n, imm_rates)
    L=get_L(nvec,omhat)
    r=get_r(K, L, alpha, delta, A)
    w=get_w(K, L, alpha, A)
    # cvec:c1 ~ cS
    cvec,c_cnstr=get_cvec_ss(r, w, vec, nvec, g_n, g_y, omhat, mort_rate)
    # print (f'ss eur cvec:{cvec.shape}')
    # errors=np.zeros(np.shape(vec))
    # MU_c=MU_c_stitch(cvec,sigma)
    errors = beta*(1+r)*MU_c_stitch(cvec[1:],sigma)*np.exp(-sigma*g_y)-MU_c_stitch(cvec[:-1],sigma)
    return errors


def EulerSys_tpi(bvec, *args):
    beg_wealth, nvec, beta, sigma, wpath, rpath, BQpath, omhat,g_y = args
    # b = np.append(beg_wealth, bvec)
    b1 = np.append(bvec,[0])
    b0 = np.append(beg_wealth,bvec)
    # print (f'b:{b0.shape,b1.shape}, wpath:{wpath.shape}, nvec:{nvec.shape}, {beg_wealth}')
    cvec = (1 + rpath) * b0+ wpath * nvec + BQpath/get_N(omhat) - b1 * np.exp(g_y)
    muc = MU_c_stitch(cvec, sigma)
    # errors = np.zeros(np.shape(cvec))
    errors = muc[:-1] - beta * (1 + rpath[1:]) * muc[1:] *np.exp((-sigma)*g_y)

    return errors


######## Calculate aggregates

#from s=1 to S
def get_L(nvec, omhat): # function for aggregate labor supply
    return (nvec * omhat).sum()

# from s=2 to S
def get_K(bhat_vec, omhat, g_n,imm_rates): # function for aggregate capital supply
    # print (f'omhat{omhat[:-1].shape}, bhat_vec:{bhat_vec.shape}')
    # print (f'In getting K, the shape is {bhat_vec.shape} ')
    return (omhat[:-1]*bhat_vec +  imm_rates[1:] * omhat[1:] * bhat_vec)[1:].sum()/(1+g_n)

def get_Y(K, L, params): # function for aggregate output
    A, alpha = params
    Y = A * (K ** alpha) * (L ** (1 - alpha))
    return Y

def get_C(carr, omhat):
    if carr.ndim == 1:
        C = (omhat*carr).sum()
    elif carr.ndim == 2:
        C = (omhat*carr).sum(axis=0)

    return C

def get_N (omhat):
    return omhat.sum()