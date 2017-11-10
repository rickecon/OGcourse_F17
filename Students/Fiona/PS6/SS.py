'''
MACS 40000 Rick Evans
PSET 6
Solving steady state for S-period model
Author: Fiona Fan
'''
import numpy as np
import time
import scipy.optimize as opt
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import utils
import os


def feasible(f_params, bvec_guess):
    b_cnstr=np.zeros(bvec_guess.shape,dtype=bool)
    nvec, A, alpha, delta, bq_distr, beta=f_params
    K,K_cnstr=utils.get_K(bvec_guess)
    L=utils.get_L(nvec)
    r=utils.get_r(K,L,(A, alpha, delta))
    w=utils.get_w(K,L,(A, alpha))
    c_vec,c_cnstr=utils.get_cvec_ss(r, w, bvec_guess, nvec, bq_distr)
    b_cnstr[0]=c_cnstr[0]
    b_cnstr[-1]=c_cnstr[-1]
    for k in range(1,len(c_cnstr)-1):
        b_cnstr[k]=c_cnstr[k]
        b_cnstr[k-1]=b_cnstr[k]
    return b_cnstr, c_cnstr, K_cnstr





def get_SS(params, bvec_guess, SS_graphs):
    start_time = time.clock()
    beta, sigma, nvec, L, A, alpha, delta, SS_tol, bq_distr, chi = params
    f_params = (nvec, A, alpha, delta, bq_distr, beta)
    b1_cnstr, c1_cnstr, K1_cnstr = feasible(f_params, bvec_guess)
    try:
        if b1_cnstr.max() or c1_cnstr.max() or K1_cnstr.max():
            raise cstError
            
        else:
            # errors=zero_func(bvec_guess,beta, sigma, nvec, L, A, alpha, delta)
            b = opt.root(utils.EulerSys_ss, bvec_guess, args=(beta, sigma, nvec, A, alpha, delta, bq_distr,chi), tol=SS_tol)
    except cstError:
        print ('Did not pass the feasible test')
    if b.success:
        b_ss = b.x
        # iterations=b.nit

    K_ss, K_cnstr = utils.get_K(b_ss)
    L=utils.get_L(nvec)
    w_ss = utils.get_w(K_ss,L, (A, alpha))
    r_ss = utils.get_r(K_ss, L, (A, alpha, delta))
    Y_ss = utils.get_Y(K_ss, L, (A, alpha))
    c_ss, c_cnstr = utils.get_cvec_ss(r_ss, w_ss, b_ss, nvec, bq_distr)
    EulErr_ss = utils.EulerSys_ss(b_ss, beta, sigma, nvec, A, alpha, delta, bq_distr, chi)
    C_ss=utils.get_C(c_ss)
    RCerr_ss = Y_ss - C_ss - delta * K_ss
    BQ_ss = b_ss[-1]

    ss_time = time.clock() - start_time
    ss_output={'b_ss': b_ss, 'c_ss': c_ss, 'w_ss': w_ss, 'r_ss': r_ss, 'K_ss': K_ss, 'Y_ss': Y_ss, 'C_ss': C_ss,
               'EulErr_ss': EulErr_ss, 'RCerr_ss': RCerr_ss, 'BQ_ss': BQ_ss, 'ss_time': ss_time}

    # print('\n Savings: \t\t\t {} \n Capital and Labor: \t\t {} \n Wage and Interest rate: \t {} \n Consumption: \t\t\t {}'.format(
    #         b_ss, np.array([K_ss, L]), np.array([w_ss, r_ss]), c_ss))
    #
    # print('Euler errors: ', EulErr_ss)
    # print('Resource Constraint error: ', RCerr_ss)
    # print('Time needed: ', ss_time)
    # print ('It took {iterations} iterations to get the solution.')
    if SS_graphs:
        cur_path = os.path.split(os.path.abspath(__file__))[0]
        output_fldr = "images"
        output_dir = os.path.join(cur_path, output_fldr)
        if not os.access(output_dir, os.F_OK):
            os.makedirs(output_dir)

        age = np.arange(21, 101)
        fig, ax = plt.subplots()
        plt.plot(age, c_ss, marker='D', label='Consumption')
        plt.plot(age, b_ss, marker='D', label='Savings')
        minorLocator = MultipleLocator(1)
        ax.xaxis.set_minor_locator(minorLocator)
        plt.grid(b=True, which='major', color='0.65', linestyle='-')
        plt.title('Steady-state consumption and savings')
        plt.xlabel('Age')
        plt.ylabel('Consumption units')
        plt.legend()
        output_path = os.path.join(output_dir, 'ss_bc')
        plt.savefig(output_path)
        # plt.show()
        plt.close()

    return ss_output

# def EulerSys(bvec, params):
#
#     beta, sigma, nvec, L, A, alpha, delta = params
#     K, K_cnstr = get_K(bvec)
#     try:
#         if K_cnstr:
#             raise cstError
#
#
#         else:
#             r_params =np.array( [A, alpha, delta])
#             r = get_r(r_params, K, L)
#             w_params = np.array([A, alpha])
#             w = get_w(w_params, K, L)
#             cvec, c_cnstr = get_cvec(r, w, bvec, nvec)
#             b_err_params = np.array([beta, sigma])
#             b_err_vec = get_b_errors(b_err_params, r, cvec, c_cnstr)
#     except cstError:
#         print ('Did not pass the feasible test')
#
#     return b_err_vec

# def get_b_errors(params, r, cvec, c_cnstr):
#     beta, sigma = params
#     # try:
#     #     if c_cnstr.max():
#     #         raise cstError
#     #
#     #     else:
#     MU_c12=get_MUc(cvec[:-1],sigma)
#     MU_c23=get_MUc(cvec[1:],sigma)
#     b_errors = (beta * (1 + r) * MU_c23) - MU_c12
#     # except cstError:
#     #     print ('Did not pass the feasible test')
#     return b_errors


#
# def get_MUc(cvec,sigma):
#     epsilon = 1e-4
#     c_cnstr = cvec < epsilon
#     muc = cvec ** (-sigma)
#     m1 = (-sigma) * epsilon ** (-sigma - 1)
#     m2 = epsilon ** (-sigma) - m1 * epsilon
#     muc[c_cnstr] = m1 * cvec[c_cnstr] + m2
#
#     return muc

# def MU_n_stitch(nvec, params):
#     l_tilde, b, upsilon = params
#     epsilon_lb = 1e-6
#     epsilon_ub = l_tilde - epsilon_lb
#     nl_cstr = nvec < epsilon_lb
#     nu_cstr = nvec > epsilon_ub
#
#     mun = ((b / l_tilde) * ((nvec / l_tilde) ** (upsilon - 1)) * (1 - ((nvec / l_tilde) ** upsilon)) **\
#            ((1 - upsilon) / upsilon))
#     m1 = (b * (l_tilde ** (-upsilon)) * (upsilon - 1) * (epsilon_lb ** (upsilon - 2)) * \
#          ((1 - ((epsilon_lb / l_tilde) ** upsilon)) ** ((1 - upsilon) / upsilon)) * \
#          (1 + ((epsilon_lb / l_tilde) ** upsilon) * ((1 - ((epsilon_lb / l_tilde) ** upsilon)) ** (-1))))
#     m2 = ((b / l_tilde) * ((epsilon_lb / l_tilde) ** (upsilon - 1)) * \
#          ((1 - ((epsilon_lb / l_tilde) ** upsilon)) ** ((1 - upsilon) / upsilon)) - (m1 * epsilon_lb))
#     q1 = (b * (l_tilde ** (-upsilon)) * (upsilon - 1) * (epsilon_ub ** (upsilon - 2)) * \
#          ((1 - ((epsilon_ub / l_tilde) ** upsilon)) ** ((1 - upsilon) / upsilon)) * \
#          (1 + ((epsilon_ub / l_tilde) ** upsilon) * ((1 - ((epsilon_ub / l_tilde) ** upsilon)) ** (-1))))
#     q2 = ((b / l_tilde) * ((epsilon_ub / l_tilde) ** (upsilon - 1)) * \
#          ((1 - ((epsilon_ub / l_tilde) ** upsilon)) ** ((1 - upsilon) / upsilon)) - (q1 * epsilon_ub))
#     mun[nl_cstr] = m1 * nvec[nl_cstr] + m2
#     mun[nu_cstr] = q1 * nvec[nu_cstr] + q2
#     return mun

class cstError (Exception):
    pass 

