'''
MACS 40000 Rick Evans
PSET 3
Solving steady state for S-period model
Author: Fiona Fan
'''
import numpy as np
import time
import scipy.optimize as opt
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator



def feasible(f_params, bvec_guess):
    b_cnstr=np.zeros(bvec_guess.shape,dtype=bool)
    nvec, A, alpha, delta=f_params
    K,K_cnstr=get_K(bvec_guess)
    L=get_L(nvec)
    r=get_r((A, alpha, delta),L,K)
    w=get_w((A, alpha),L,K)
    c_vec,c_cnstr=get_cvec(r,w,bvec_guess,nvec)

    b_cnstr[0]=c_cnstr[0]
    b_cnstr[-1]=c_cnstr[-1]
    for k in range(1,len(c_cnstr)-1):
        b_cnstr[k]=c_cnstr[k]
        b_cnstr[k-1]=b_cnstr[k]
    return b_cnstr, c_cnstr, K_cnstr

def get_r(params,K,L):
    A, alpha, delta=params
    return alpha*A*((L/K)**(1-alpha))-delta
def get_w(params,K,L):
    A, alpha   = params
    return (1-alpha)*A*((K/L)**alpha)
def get_cvec(r, w, bvec, nvec):
    b=np.append([0],bvec)
    b1=np.append(bvec,[0])
    # print(f"nvec shape is: {nvec.shape}")
    c_vec = (1 + r) * b + w * nvec - b1
    c_cnstr = c_vec <= 0
    # print(f"cvec shape is: {c_vec.shape}")
    return c_vec, c_cnstr
    
    
def get_K(b_vec):
    return sum(b_vec), sum(b_vec)<=0
def get_L(nvec):
    return sum(nvec)

def get_Y(params, K, L):

    A, alpha = params
    Y = A * (K ** alpha) * (L ** (1 - alpha))

    return Y

def get_C(cvec):

    if cvec.ndim == 1:
        C = cvec.sum()
    elif cvec.ndim == 2:
        C = cvec.sum(axis=0)
    return C

def zero_func(vec,*args):
    beta, sigma, nvec, L, A, alpha, delta=args
    K= get_K(vec)[0]
    L=get_L(nvec)
    r=get_r((A, alpha, delta),K,L)
    w=get_w((A, alpha),K,L)
    cvec,c_cnstr=get_cvec(r, w, vec, nvec)
    MU_c=get_MUc(cvec,sigma)
    errors=beta*(1+r)*MU_c[1:]-MU_c[:-1]
    return errors

def get_SS(params, bvec_guess, SS_graphs):
    start_time = time.clock()
    beta, sigma, nvec, L, A, alpha, delta, SS_tol = params
    f_params = (nvec, A, alpha, delta)
    b1_cnstr, c1_cnstr, K1_cnstr = feasible(f_params, bvec_guess)
    try:
        if b1_cnstr.max() or c1_cnstr.max() or K1_cnstr.max():
            raise cstError
            
        else:
            # errors=zero_func(bvec_guess,beta, sigma, nvec, L, A, alpha, delta)
            b = opt.root(zero_func, bvec_guess,args=(beta, sigma, nvec, L, A, alpha, delta), tol=SS_tol)
    except cstError:
        print ('Did not pass the feasible test')
    if b.success:
        b_ss = b.x
        # iterations=b.nit

    K_ss, K_cnstr = get_K(b_ss)
    L=get_L(nvec)
    w_ss = get_w((A, alpha),K_ss,L)
    r_ss = get_r((A, alpha, delta),K_ss, L)
    Y_ss = get_Y( (A, alpha),K_ss, L)
    c_ss, c_cnstr = get_cvec(r_ss, w_ss, b_ss, nvec)
    EulErr_ss = get_b_errors((beta, sigma), r_ss, c_ss, c_cnstr)
    C_ss=get_C(c_ss)
    RCerr_ss = Y_ss - C_ss - delta * K_ss

    ss_time = time.clock() - start_time
    ss_output={'b_ss': b_ss, 'c_ss': c_ss, 'w_ss': w_ss, 'r_ss': r_ss,
    'K_ss': K_ss, 'Y_ss': Y_ss, 'C_ss': C_ss,
    'EulErr_ss': EulErr_ss, 'RCerr_ss': RCerr_ss,
    'ss_time': ss_time}

    print('\n Savings: \t\t\t {} \n Capital and Labor: \t\t {} \n Wage and Interest rate: \t {} \n Consumption: \t\t\t {}'.format(
            b_ss, np.array([K_ss, L]), np.array([w_ss, r_ss]), c_ss))

    print('Euler errors: ', EulErr_ss)
    print('Resource Constraint error: ', RCerr_ss)
    print('Time needed: ', ss_time)
    # print ('It took {iterations} iterations to get the solution.')
    if SS_graphs:
        age = np.arange(1, 81)
        fig, ax = plt.subplots()
        plt.plot(age, c_ss, marker='D', label='Consumption')
        plt.plot(age, np.append([0], b_ss), marker='D', label='Savings')

        minorLocator = MultipleLocator(1)
        ax.xaxis.set_minor_locator(minorLocator)
        plt.grid(b=True, which='major', color='0.65', linestyle='-')
        plt.title('Steady-state consumption and savings')
        plt.xlabel('Age')
        plt.ylabel('Consumption units')
        plt.legend()
        plt.show()


    return ss_output

def EulerSys(bvec, params):

    beta, sigma, nvec, L, A, alpha, delta = params
    K, K_cnstr = get_K(bvec)
    try:
        if K_cnstr:
            raise cstError
            
        
        else:
            r_params =np.array( [A, alpha, delta])
            r = get_r(r_params, K, L)
            w_params = np.array([A, alpha])
            w = get_w(w_params, K, L)
            cvec, c_cnstr = get_cvec(r, w, bvec, nvec)
            b_err_params = np.array([beta, sigma])
            b_err_vec = get_b_errors(b_err_params, r, cvec, c_cnstr)
    except cstError:
        print ('Did not pass the feasible test')

    return b_err_vec

def get_b_errors(params, r, cvec, c_cnstr):
    beta, sigma = params
    # try:
    #     if c_cnstr.max():
    #         raise cstError
    #
    #     else:
    MU_c12=get_MUc(cvec[:-1],sigma)
    MU_c23=get_MUc(cvec[1:],sigma)
    b_errors = (beta * (1 + r) * MU_c23) - MU_c12
    # except cstError:
    #     print ('Did not pass the feasible test')
    return b_errors



def get_MUc(cvec,sigma):
    epsilon=0.0001
    cvec_neg=cvec<=0
    MU_c=np.zeros_like(cvec)
    MU_c[cvec_neg]=epsilon**(-sigma)
    MU_c[~cvec_neg]=cvec[~cvec_neg]**(-sigma)
    return MU_c


class cstError (Exception):
    pass 

