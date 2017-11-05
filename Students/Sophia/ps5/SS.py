import time
import numpy as np
import scipy.optimize as opt
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import os
import Households as hh
import firms
import aggregates as aggr

def get_SS(c1_guess, r_old, params):
    beta, sigma, p, ltilde, b, upsilon, chi_n_vec, A, alpha, delta, ss_max_iter, ss_tol, xi_ss = params
    abs_ss = 1
    ss_iter = 0
    while abs_ss > ss_tol and ss_iter < ss_max_iter:
        ss_iter += 1
        r_old = r_old * np.ones(p)
        w_old = firms.get_w(r_old, (A, alpha, delta)) * np.ones(p)
        # Calculate household decisions that make last-period savings zero
        c1_args = (r_old, w_old, beta, sigma, ltilde, b, upsilon, chi_n_vec, p, 0.0)
        result_c1 = opt.root(hh.get_b_last, c1_guess, args = (c1_args))
        if result_c1.success:
            c1 = result_c1.x
        else:
            raise ValueError("failed to find an appropriate initial consumption")
        # Calculate aggregate supplies for capital and labor
        cvec = hh.get_c(c1, r_old, beta, sigma, p)
        nvec = hh.get_n(cvec, sigma, ltilde, b, upsilon, chi_n_vec, w_old, p)
        bvec = hh.get_b(cvec, nvec, r_old, w_old, p)
        K = aggr.get_K(bvec)[0]
        L = aggr.get_L(nvec)[0]
        C = aggr.get_C(cvec)
        Y = aggr.get_Y(K, L, (A, alpha))
        b_err = abs(hh.get_b_errors(cvec, r_old[0], beta, sigma)).max()
        n_err = abs(hh.get_n_errors(nvec, cvec, sigma, ltilde, b, upsilon, chi_n_vec, w_old[0])).max()
        b_last = hh.get_b_last(cvec[0], r_old, w_old, beta, sigma, ltilde, b, upsilon, chi_n_vec, p, 0.0)
        r_new = firms.get_r(K, L, (A, alpha, delta))
        # Check market clearing
        abs_ss = ((r_new - r_old) ** 2).max()
        # Update guess
        r_old = xi_ss * r_new + (1 - xi_ss) * r_old
        print('iteration:', ss_iter, ' squared distance: ', abs_ss)
    return r_old[0], w_old[0], cvec, nvec, bvec, K, L, C, Y, b_err, n_err, b_last

def create_graphs(c_ss, b_ss, n_ss):
    cur_path = os.path.split(os.path.abspath(__file__))[0]
    output_fldr = "images"
    output_dir = os.path.join(cur_path, output_fldr)
    if not os.access(output_dir, os.F_OK):
        os.makedirs(output_dir)
    age = np.arange(1, c_ss.shape[0] + 1)
    plt.plot(age, b_ss, 'go--', color = 'green', label = 'savings')
    plt.grid(b=True, which='major', color='0.65', linestyle='-')
    plt.plot (age, c_ss, 'go--', color = 'blue', label = 'consumption')
    plt.grid(b=True, which='major', color='0.65', linestyle='-')
    plt.title('Steady-State distribution of Consumption and Savings', fontsize=10)
    plt.xlabel('age')
    plt.ylabel('units of consumption')
    plt.legend()
    output_path1 = os.path.join(output_dir, 'ss_bc')
    plt.savefig(output_path1)
    plt.close()

    plt.plot(age, n_ss, 'go--', label = 'labor supply')
    plt.grid(b=True, which='major', color='0.65', linestyle='-')
    plt.title('Steady-State distribution of Labor Supply', fontsize=20)
    plt.xlabel('age')
    plt.ylabel('labor supply')
    plt.legend()
    output_path2 = os.path.join(output_dir, 'ss_n')
    plt.savefig(output_path2)
    plt.close()
