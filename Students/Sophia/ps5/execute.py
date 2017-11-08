import numpy as np
import os
import SS as ss
import TPI as tpi
import aggregates as aggr

# Household Parameters
yrs_live = 80
S = 10
beta_annual = .96
beta = beta_annual ** (yrs_live / S)
sigma = 2.2
ltilde = 1.0
b = 0.50
upsilon = 1.5
chi_n_vec = 1.0 * np.ones(S)

# Firms Parameters
alpha = 0.35
A = 1.0
delta_annual = 0.05
delta = 1.0 - ((1.0 - delta_annual) ** (yrs_live / S))

# SS Parameters
ss_solve = True
ss_max_iter = 400
ss_tol = 1e-13
xi_ss = 0.1

if ss_solve:
    c1_guess = 0.5
    r_old = 0.5
    params = (beta, sigma, S, ltilde, b, upsilon, chi_n_vec, A, alpha, delta, ss_max_iter, ss_tol, xi_ss)
    r_ss, w_ss, c_ss, n_ss, b_ss, K_ss, L_ss, C_ss, Y_ss, b_err, n_err, b_last = ss.get_SS(c1_guess, r_old, params)
    ss.create_graphs(c_ss, b_ss, n_ss)
    print(r_ss, w_ss, K_ss, L_ss, Y_ss, C_ss)
    print("savings euler error is {}".format(b_err))
    print("labor supply euler error is {}".format(n_err))
    print("final period saving is {}".format(b_last))
    print("resource constraint error is {}".format(Y_ss - C_ss - delta * K_ss))

# TPI Patameters
tpi_solve = True
T1 = 60
T2 = 90
tpi_max_iter = 500
tpi_tol = 1e-12
xi_tpi = 0.3
b1vec = 1.08 * b_ss

if tpi_solve:
    ss_params = r_ss, w_ss, c_ss, n_ss, b_ss, K_ss, L_ss
    params = (beta, sigma, S, ltilde, b, upsilon, chi_n_vec, A, alpha, delta, tpi_max_iter, tpi_tol, xi_tpi, T1, T2)
    cmat, nmat, bmat, r, w, K, L, Y, C, b_err, n_err, b_last, Rc_err, T1 = tpi.get_TPI(b1vec, ss_params, params)
    tpi.create_graphs(r, w, K, L, Y, C, cmat, nmat, bmat, T2, S)
