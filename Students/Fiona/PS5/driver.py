import SS as ss
import TPI as tpi
import utils
import numpy as np

# Household Parameters
yrs_live = 80
S = 10
beta_annual = .96
beta = beta_annual ** (yrs_live / S)
sigma = 2.5
l_ub = 1.0
b = 0.5
upsilon = 1.5
chi=np.ones(S)

# Firms Parameters
alpha = 0.35
A = 1.0
delta_annual = 0.05
delta = 1.0 - ((1.0 - delta_annual) ** (yrs_live / S))


# SS Parameters
ss_solve = True
ss_max_iter = 400
ss_tol = 1e-9
xi_ss = 0.1

if ss_solve:
    c1_guess = 0.5
    r_old = 0.5
    params = (S, beta, S, sigma, l_ub, b, upsilon, chi, A, alpha, delta, ss_max_iter, ss_tol, xi_ss)
    r_ss, w_ss, c_ss, n_ss, b_ss, K_ss, L_ss,b_err, n_err, b_last = ss.get_SS(c1_guess, r_old, params)
    C_ss=utils.get_C(c_ss)
    Y_ss=utils.get_Y(K_ss,L_ss,(A, alpha))
    cnt_err= Y_ss - C_ss - delta * K_ss
    # ss.create_graphs(c_ss, b_ss, n_ss)
    # ss.write_csv(r_ss, w_ss, K_ss, L_ss, Y_ss, C_ss,b_err, n_err, b_last, cnt_err)


# TPI Patameters
tpi_solve = True
T1 = 60
T2 = 90
tpi_max_iter = 500
tpi_tol = 1e-12
xi_tpi = 0.3
b1vec = 1.08 * b_ss

if tpi_solve:
    c1_guess = 1
    ss_params = r_ss, w_ss, c_ss, n_ss, b_ss, K_ss, L_ss
    params = (beta, sigma, S, l_ub, b, upsilon, chi, A, alpha, delta, tpi_max_iter, tpi_tol, xi_tpi, T1, T2)
    r, w, K, L,bmat, nmat, cmat, b_last, b_err, n_err, cnt_err, T_1 = tpi.get_TPI(b1vec, c1_guess, ss_params, params)
    Y = A * (K ** alpha) * (L ** (1 - alpha))

    C = utils.get_C(cmat)

    tpi.create_graphs(T2, S, r, w, K, L, C, Y, bmat, cmat, nmat)
    tpi.write_csv(b_last, b_err, n_err, cnt_err,T_1)
    # T, S, rpath, wpath, Kpath, Lpath, Cpath, Ypath, bpath, cpath, npath
