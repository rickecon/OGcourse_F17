import numpy as np
import scipy.optimize as opt
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import os
import utils
import pandas as pd



def get_SS(c1_guess, r_old, params):
    S, beta, p, sigma, l_tilde, b, upsilon, chi, A, alpha, delta, ss_max_iter, ss_tol, xi_ss = params
    abs_ss = 1
    ss_iter = 0
    while abs_ss > ss_tol and ss_iter < ss_max_iter:
        ss_iter += 1
        r_old = r_old * np.ones(p)
        w_old = utils.get_w(r_old, (A, alpha, delta)) * np.ones(p)
        # Calculate household decisions that make last-period savings zero
        c1_args = (r_old, w_old, beta, sigma, l_tilde, b, upsilon, p, 0.0, chi)
        result_c1 = opt.root(utils.get_b_last, c1_guess, args=(c1_args))
        if result_c1.success:
            c1 = result_c1.x
        else:
            raise ValueError("failed to find an appropriate initial consumption")
        # Calculate aggregate supplies for capital and labor
        cvec = utils.get_c(c1, r_old, beta, sigma, p)
        nvec = utils.get_n(cvec, sigma, l_tilde, b, upsilon, w_old, p, chi)
        bvec = utils.get_b(cvec, nvec, r_old, w_old, p)
        K = utils.get_K(bvec)[0]
        L = utils.get_L(nvec)[0]
        r_new = utils.get_r(K, L, (A, alpha, delta))
        # Check market clearing
        abs_ss = ((r_new - r_old) ** 2).max()
        # Update guess
        r_old = xi_ss * r_new + (1 - xi_ss) * r_old
        print('iteration:', ss_iter, ' squared distance: ', abs_ss)

    b_err = abs(utils.get_b_errors(cvec, r_old[0], beta, sigma)).max()
    n_err = abs(utils.get_n_errors(nvec, cvec, sigma, l_tilde, b, upsilon, w_old[0],chi)).max()
    b_last = utils.get_b_last(cvec[0], r_old, w_old, beta, sigma, l_tilde, b, upsilon, p, 0.0,chi)
    return r_old[0], w_old[0], cvec, nvec, bvec, K, L, b_err, n_err,b_last

def create_graphs(c_ss, b_ss, n_ss):
    cur_path = os.path.split(os.path.abspath(__file__))[0]
    output_fldr = "images_SS"
    output_dir = os.path.join(cur_path, output_fldr)
    if not os.access(output_dir, os.F_OK):
        os.makedirs(output_dir)

    plt.plot (np.arange(21, 80, 6), b_ss, 'go--', color = 'green', label = 'savings')
    plt.grid(b=True, which='major', color='0.65', linestyle='-')
    plt.plot (np.arange(21, 80, 6), c_ss, 'go--', color = 'blue', label = 'consumption')
    plt.grid(b=True, which='major', color='0.65', linestyle='-')
    plt.title('Steady-State distribution of Consumption and Savings', fontsize=10)
    plt.xlabel('age')
    plt.ylabel('units of consumption')
    plt.legend()
    output_path1 = os.path.join(output_dir, 'ss_bc')
    plt.savefig(output_path1)
    plt.close()

    plt.plot (np.arange(21, 80, 6), n_ss, 'go--', label = 'labor supply')
    plt.grid(b=True, which='major', color='0.65', linestyle='-')
    plt.title('Steady-State distribution of Labor Supply', fontsize=20)
    plt.xlabel('age')
    plt.ylabel('labor supply')
    plt.legend()
    output_path2 = os.path.join(output_dir, 'ss_n')
    plt.savefig(output_path2)
    plt.close()

def write_csv(r_ss, w_ss, K_ss, L_ss, Y_ss, C_ss, b_err, n_err, b_last, cnt_err):
    output={'r_ss':r_ss, 'w_ss':w_ss, 'K_ss':K_ss, 'L_ss':L_ss, 'Y_ss':Y_ss, 'C_ss':C_ss, 'b_err':b_err, 'n_err':n_err,
            'b_last':b_last, 'cnt_err:':cnt_err}
    pd.DataFrame.from_dict(data=output, orient='index').to_csv('output_ss.csv', header=True)