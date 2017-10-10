'''
MACS 40000 Rick Evans
PSET2 problem1
Author: Fiona Fan
'''
import numpy as np
import time
import scipy.optimize as opt

#Household parameters
S= int(3)
beta_annual=0.96
beta=beta_annual**20
sigma=2.2
nvec=np.array([1.0,1.0,0.2])
L=nvec.sum()
#Firm parameters
A=1.0
alpha=0.35
delta_annual=0.05
delta=1-((1-delta_annual)**20)

def feasible(f_params, bvec_guess):
    b_cnstr=np.array([False,False])
    nvec, A, alpha, delta=f_params
    #n1,n2,n3=nvec
    K,K_cnstr=get_K(bvec_guess)
    L=get_L(nvec)
    r=get_r(np.array([A, alpha, delta]),L,K)
    w=get_w(np.array([A, alpha]),L,K)
    c_vec,c_cnstr=get_cvec(r,w,bvec_guess,nvec)
    
    if c_cnstr[0]:
        b_cnstr[0]=True
    if c_cnstr[1]:
        b_cnstr[0]=True
        b_cnstr[1]=True
    if c_cnstr[2]:
        b_cnstr[1]=True
    return b_cnstr, c_cnstr, K_cnstr

def get_r(params,L,K):
    A, alpha, delta=params
    return alpha*A*(L/K)**(1-alpha)-delta
def get_w(params,L, K):
    A, alpha   = params
    return (1-alpha)*A*(K/L)**alpha
def get_cvec(r, w, bvec, nvec):
    b2,b3=bvec
    n1,n2,n3=nvec
    
    c1=w*n1-b2
    c2=w*n2+b2*(1+r)-b3
    c3=w*n3+b3*(1+r)
    c_vec=np.array([c1,c2,c3])
    
    c_cnstr=(c_vec<=0)

    return c_vec, c_cnstr
    
    
def get_K(b_vec):
    return sum(b_vec), sum(b_vec)<=0
def get_L(nvec):
    return sum(nvec)
def get_SS(params, bvec_guess, SS_graphs):
    start_time = time.clock()
    beta, sigma, nvec, L, A, alpha, delta, SS_tol = params
    f_params = (nvec, A, alpha, delta)
    b1_cnstr, c1_cnstr, K1_cnstr = feasible(f_params, bvec_guess)
    try:
        if b1_cnstr.max() or c1_cnstr.max() or K1_cnstr.max():
            raise cstError
            break
    except cstError:
        pass
            
    
    
    ss_time = time.clock() - start_time
    ss_output={'b_ss': b_ss, 'c_ss': c_ss, 'w_ss': w_ss, 'r_ss': r_ss,
    'K_ss': K_ss, 'Y_ss': Y_ss, 'C_ss': C_ss,
    'EulErr_ss': EulErr_ss, 'RCerr_ss': RCerr_ss,
    'ss_time': ss_time}

    return ss_output

def EulerSys(bvec, params):

    beta, sigma, nvec, L, A, alpha, delta = params
    K, K_cnstr = get_K(bvec)
    try:
        if K_cnstr:
            raise cstError
            break
        
        else:
            r_params =np.array( (A, alpha, delta))
            r = get_r(r_params, K, L)
            w_params = np.array((A, alpha))
            w = get_w(w_params, K, L)
            cvec, c_cnstr = get_cvec(r, w, bvec, nvec)
            b_err_params = (beta, sigma)
            b_err_vec = get_b_errors(b_err_params, r, cvec, c_cnstr)
    except cstError:
        pass

    return b_err_vec

def get_b_errors(params, r, cvec, c_cnstr):
    beta, sigma = params
    try:
        if c_cnstr.max():
            raise cstError
            break

    return b_errors

class cstError (Exception):
    print 'Did not pass the feasible test' 
#testing
b2=0.06
b3=-0.001
b_guess=np.array([b2,b3])
f_params=(nvec,A,alpha,delta)
b_cnstr, c_cnstr,K_cnstr=feasible(f_params,b_guess)
print b_cnstr
print c_cnstr
print K_cnstr