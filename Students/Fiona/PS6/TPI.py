
import time
import numpy as np
import scipy.optimize as opt
import SS as ss
import utils
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from mpl_toolkits.mplot3d import Axes3D
import sys
import os

'''
------------------------------------------------------------------------
    Functions
------------------------------------------------------------------------
'''


def get_path(x1, xT, T):
    return np.linspace(x1, xT, T)




# def get_cvec(rpath, wpath, nvec, bvec):
#     b_s = bvec
#     b_sp1 = np.append(bvec[1:], [0])
#     cvec = (1 + rpath) * b_s + wpath * nvec - b_sp1
#     c_cnstr = cvec <= 0
#     return cvec, c_cnstr


# def solver(params, beg_wealth, nvec, rpath, wpath, b_init, bq_distr, chi):
#
#     S, beta, sigma, TPI_tol, A, alpha, delta = params
#     # p = int(S - beg_age + 1)
#
#     b_guess = 1.01 * b_init
#     bq = b_guess [-1]
#     eullf_objs = (beg_wealth, nvec, beta, sigma, wpath, rpath,bq, chi, bq_distr)
#     #beg_wealth, nvec, beta, sigma, wpath, rpath, BQpath, chi, bq_distr
#     bpath = opt.root(utils.EulerSys_tpi, b_guess, args=(eullf_objs)).x
#     cpath, c_cnstr = utils.get_cvec_tpi(rpath, wpath,
#                                  np.append(beg_wealth, bpath),nvec, bq, bq_distr)
#
#     b_err_vec = utils.EulerSys_ss(bpath, beta, sigma, nvec, A, alpha, delta, bq_distr, chi)
#     return bpath, cpath, b_err_vec
#
#
# def get_cbepath(params, rpath, wpath, bvec1):
#
#     S, T, beta, sigma, nvec, b_ss, TPI_tol, A, alpha, delta, bq_distr, chi = params
#     cpath = np.zeros((S, T + S - 2))
#     #b: s=2 ~ S+1
#     bpath = np.append(bvec1.reshape((S, 1)), np.zeros((S, T + S - 3)), axis=1)
#     #err: s = 1 ~ S
#     EulErrPath = np.zeros((S, T + S - 2))
#     # Solve the incomplete remaining lifetime decisions of agents alive
#     # in period t=1 but not born in period t=1
#     cpath[S-1, 0] = ((1 + rpath[0]) * bvec1[S-1] + wpath[0] * nvec[-1])
#     pl_params = (S, beta, sigma, TPI_tol, A, alpha, delta, bq_distr, chi)
#     for p in range(2, S):
#         b_guess = np.diagonal(bpath[S - p:, :p - 1])
#
#
#
#         # Insert the vector lifetime solutions diagonally (twist donut)
#         # into the cpath, bpath, and EulErrPath matrices
#         DiagMaskb = np.eye(p, dtype=bool)
#         DiagMaskc = np.eye(p, dtype=bool)
#         bpath[S - p:, :p] = DiagMaskb * bveclf + bpath[S - p:, :p]
#         cpath[S - p:, :p] = DiagMaskc * cveclf + cpath[S - p:, :p]
#         EulErrPath[S - p:, :p] = (DiagMaskb * b_err_veclf +
#                                    EulErrPath[S - p:, :p])
#     # Solve for complete lifetime decisions of agents born in periods
#     # 1 to T and insert the vector lifetime solutions diagonally (twist
#     # donut) into the cpath, bpath, and EulErrPath matrices
#     DiagMaskb = np.eye(S, dtype=bool)
#     DiagMaskc = np.eye(S, dtype=bool)
#     for t in range(1, T):  # Go from periods 1 to T-1
#         b_guess = np.diagonal(bpath[:, t - 1:t + S - 1])
#         bveclf, cveclf, b_err_veclf = solver(pl_params, 1, 0, nvec, rpath[t - 1:t + S - 1],
#                                              wpath[t - 1:t + S - 1], b_guess)
#         # Insert the vector lifetime solutions diagonally (twist donut)
#         # into the cpath, bpath, and EulErrPath matrices
#         bpath[:, t-1:t + S - 1] = (DiagMaskb * bveclf +
#                                  bpath[:, t-1:t + S - 1])
#         cpath[:, t - 1:t + S - 1] = (DiagMaskc * cveclf +
#                                      cpath[:, t - 1:t + S - 1])
#         EulErrPath[:, t-1:t + S - 1] = (DiagMaskb * b_err_veclf +
#                                       EulErrPath[:, t-1:t + S - 1])
#
#     return cpath, bpath, EulErrPath


def get_TPI(params, b1vec, graphs):

    start_time = time.clock()
    S, T, beta, sigma, nvec, L, A, alpha, delta, b_ss, K_ss, C_ss, BQ_ss, \
    maxiter_TPI, mindist_TPI, xi, TPI_tol, bq_distr, chi = params


    K1 = utils.get_K(b1vec)[0]
    # K: s=1 ~ S, sum of b: s=2 ~ S+1
    Kpath_old = np.zeros(T + S )
    # print (f'K1: {K1}, K_ss: {K_ss}')
    Kpath_old[:T] = np.linspace(K1, K_ss, T)  # Until reaching steady state
    Kpath_old[T:] = K_ss
    r = utils.get_r(Kpath_old[0], utils.get_L(nvec), (A, alpha, delta))
    BQ1 = (1 + r) * b1vec[-1]
    BQpath_old = np.zeros(T + S )
    BQpath_old[:T] = np.linspace(BQ1, BQ_ss, T)  # Until reaching steady state
    BQpath_old[T:] = BQ_ss
    L = np.sum(nvec)

    iter_TPI = int(0)
    abs2 = 10.
    Kpath_new = Kpath_old.copy()
    r_params = (A, alpha, delta)
    w_params = (A, alpha)
    # cbe_params = (S, T, beta, sigma, nvec, b_ss, TPI_tol, A, alpha, delta, bq_distr, chi)

    while (iter_TPI < maxiter_TPI) and (abs2 >= mindist_TPI):
        iter_TPI += 1
        # Kpath_init = xi * Kpath_new + (1 - xi) * Kpath_old
        rpath = utils.get_r(Kpath_old, L, r_params)
        wpath = utils.get_w(Kpath_old, L, w_params)
        # cpath, bpath, EulErrPath = get_cbepath(cbe_params, rpath, wpath, b1vec)
        # b: 2~S+1
        bpath = np.append(b1vec.reshape(S,1), np.zeros((S, T+S-2)), axis=1)
        cpath = np.zeros((S, T + S-1 ))
        EulErrPath = np.zeros((S, T + S-1 ))
        cpath[S - 1, 0] = ((1 + rpath[0]) * b1vec[S - 2] + wpath[0] * nvec[S - 1])
        for p in range (1, S):
            bvec_guess = np.diagonal(bpath[S - p:, :p])
            beg_wealth = bpath[S - p - 1, 0]
            #beg_wealth, nvec, beta, sigma, wpath, rpath, BQpath, chi, bq_distr
            args_sol= (beg_wealth, nvec[-p:], beta, sigma, wpath[:p], rpath[:p], BQpath_old[:p], chi[-p:], bq_distr[-p:])
            b_sol = opt.root(utils.EulerSys_tpi, bvec_guess, args=(args_sol)).x
            #rpath, wpath, bvec, nvec, bq, bq_distr
            cp, c_cnstr_p = utils.get_cvec_tpi(rpath[:p], wpath[:p], np.append(beg_wealth, b_sol), nvec[-p:],
                                               BQpath_old[:p],bq_distr[-p:])
            # rpath, wpath, bvec, nvec, bq, bq_distr
            b_err_p = utils.EulerSys_tpi(b_sol, beg_wealth, nvec[-p:], beta, sigma, wpath[:p], rpath[:p], BQpath_old[:p], chi[-p:], bq_distr[-p:])

            # Insert the vector lifetime solutions diagonally (twist donut)
            bp_path = np.eye(p) * b_sol
            cp_path = np.eye(p) * cp
            ep_path = np.eye(p) * b_err_p
            bpath[S - p:, 1:p+1] += bp_path
            cpath[S - p:, 1:p+1] += cp_path
            EulErrPath[S - p:, 1:p+1] += ep_path

        for t in range(1, T):
            bvec_guess_t = np.diagonal(bpath[:, t - 1:S + t - 1])
            args_bt = (0, nvec, beta, sigma, wpath[t - 1: S + t - 1], rpath[t - 1: S + t - 1], BQpath_old[t - 1: S + t - 1],
            chi, bq_distr)
            bt = opt.root(utils.EulerSys_tpi, bvec_guess_t, args=(args_bt)).x
            # print (f'bt.shape: {bt.shape}, ')
            #np.append(nvec,[0.2])
            # rpath, wpath, bvec, nvec, bq, bq_distr
            ct, c_cnstr_t = utils.get_cvec_tpi(rpath[t - 1: S + t - 1], wpath[t - 1: S + t - 1],np.append([0], bt),
                                               nvec, BQpath_old[t - 1: S + t - 1], bq_distr)
            b_err_t = utils.EulerSys_tpi(bt, 0, nvec, beta, sigma, wpath[t - 1: S + t - 1], rpath[t - 1: S + t - 1], BQpath_old[t - 1: S + t - 1],
                                        chi, bq_distr)

            # DiagMask = np.eye(S)

            bt_path = np.eye(p+1) * bt
            # print(f'bt: {bt.shape}, btpath:{bt_path.shape}, t:{t}, bpath:{bpath.shape}')
            ct_path = np.eye(p+1) * ct
            et_path = np.eye(p+1) * b_err_t
            bpath[:, t: S + t] += bt_path
            cpath[:, t: S + t] += ct_path
            EulErrPath[:, t: S + t] += et_path

        Kpath_new = np.zeros(T + S)
        Kpath_new[:T], Kpath_cnstr = utils.get_K(bpath[:, :T])
        Kpath_new[T:] = K_ss * np.ones(S)
        Kpath_cnstr = np.append(Kpath_cnstr,
                                np.zeros(S, dtype=bool))
        Kpath_new[Kpath_cnstr] = 0.1

        BQpath_new = np.zeros(T + S)
        # print (f'Here: bpath {bpath[S, :T].shape}')
        #, BQ:{BQpath_new.shape}, rpath:{rpath[:T].shape}
        BQpath_new[:T] = (1 + rpath[:T]) * bpath [-1, :T]
        BQpath_new[T:] = BQ_ss * np.ones(S)

        abs2 = (((Kpath_old[:T] - Kpath_new[:T]) / Kpath_old[:T] * 100) ** 2).sum() + \
               (((BQpath_old[:T] - BQpath_new[:T]) / BQpath_old[:T] * 100) ** 2).sum()


        Kpath_old[:T] = xi * Kpath_new[:T] + (1 - xi) * Kpath_old[:T]
        BQpath_old[:T] = xi * BQpath_new[:T] + (1 - xi) * BQpath_old[:T]
        print('iter: ', iter_TPI, ', squared pct deviation sum: ', abs2,
              ',max Eul err: ', np.absolute(EulErrPath).max())

    BQ_path = BQpath_old
    Kpath = Kpath_old
    Ypath = utils.get_Y(Kpath,L, (A, alpha))
    Cpath = np.zeros(Kpath.shape)
    Cpath[:T - 1] = utils.get_C(cpath[:, :T - 1])
    Cpath[T - 1:] = C_ss * np.ones(S + 1)
    RCerrPath = (Ypath[:-1] - Cpath[:-1] - Kpath[1:] + (1 - delta) * Kpath[:-1])
    tpi_time = time.clock() - start_time

    tpi_output = {
        'bpath': bpath, 'cpath': cpath, 'wpath': wpath, 'rpath': rpath,
        'Kpath': Kpath_new, 'Ypath': Ypath, 'Cpath': Cpath,
        'EulErrPath': EulErrPath, 'RCerrPath': RCerrPath,
        'tpi_time': tpi_time, 'BQ_path': BQ_path}

    # Print TPI computation time
    print(f'It took {tpi_time} seconds to run.')
    if graphs:
        '''
        ----------------------------------------------------------------
        cur_path    = string, path name of current directory
        output_fldr = string, folder in current path to save files
        output_dir  = string, total path of images folder
        output_path = string, path of file name of figure to be saved
        tvec        = (T+S-2,) vector, time period vector
        tgridTm1    = (T-1,) vector, time period vector to T-1
        tgridT      = (T,) vector, time period vector to T-1
        sgrid       = (S,) vector, all ages from 1 to S
        sgrid2      = (S-1,) vector, all ages from 2 to S
        tmatb       = (2, 18) matrix, time periods for all savings
                      decisions ages (S-1) and time periods (T)
        smatb       = (2, 18) matrix, ages for all savings decision ages
                      (S-1) and time periods (T)
        tmatc       = (3, 17) matrix, time periods for all consumption
                      decisions ages (S) and time periods (T-1)
        smatc       = (3, 17) matrix, ages for all consumption decisions
                      ages (S) and time periods (T-1)
        ----------------------------------------------------------------
        '''
        # Create directory if images directory does not already exist
        cur_path = os.path.split(os.path.abspath(__file__))[0]
        output_fldr = "images"
        output_dir = os.path.join(cur_path, output_fldr)
        if not os.access(output_dir, os.F_OK):
            os.makedirs(output_dir)

        # Plot time path of aggregate capital stock
        tvec = np.linspace(1, T + 5, T + 5)
        minorLocator = MultipleLocator(1)
        fig, ax = plt.subplots()
        plt.plot(tvec, Kpath[:T+5], marker='D')
        # for the minor ticks, use no labels; default NullFormatter
        ax.xaxis.set_minor_locator(minorLocator)
        plt.grid(b=True, which='major', color='0.65', linestyle='-')
        plt.title('Time path for aggregate capital stock K')
        plt.xlabel(r'Period $t$')
        plt.ylabel(r'Aggregate capital $K_{t}$')
        output_path = os.path.join(output_dir, "Kpath")
        plt.savefig(output_path)
        # plt.show()

        # Plot time path of aggregate capital stock
        tvec = np.linspace(1, T + 5, T + 5)
        minorLocator = MultipleLocator(1)
        fig, ax = plt.subplots()
        plt.plot(tvec, BQ_path[:T + 5], marker='D')
        # for the minor ticks, use no labels; default NullFormatter
        ax.xaxis.set_minor_locator(minorLocator)
        plt.grid(b=True, which='major', color='0.65', linestyle='-')
        plt.title('Time path for bequest BQ')
        plt.xlabel(r'Period $t$')
        plt.ylabel(r'Bequest $BQ$')
        output_path = os.path.join(output_dir, "BQpath")
        plt.savefig(output_path)
        # plt.show()

        # Plot time path of aggregate output (GDP)
        fig, ax = plt.subplots()
        plt.plot(tvec, Ypath[:T+5], marker='D')
        # for the minor ticks, use no labels; default NullFormatter
        ax.xaxis.set_minor_locator(minorLocator)
        plt.grid(b=True, which='major', color='0.65', linestyle='-')
        plt.title('Time path for aggregate output (GDP) Y')
        plt.xlabel(r'Period $t$')
        plt.ylabel(r'Aggregate output $Y_{t}$')
        output_path = os.path.join(output_dir, "Ypath")
        plt.savefig(output_path)
        # plt.show()

        # Plot time path of aggregate consumption
        fig, ax = plt.subplots()
        plt.plot(tvec, Cpath[:T+5], marker='D')
        # for the minor ticks, use no labels; default NullFormatter
        ax.xaxis.set_minor_locator(minorLocator)
        plt.grid(b=True, which='major', color='0.65', linestyle='-')
        plt.title('Time path for aggregate consumption C')
        plt.xlabel(r'Period $t$')
        plt.ylabel(r'Aggregate consumption $C_{t}$')
        output_path = os.path.join(output_dir, "C_aggr_path")
        plt.savefig(output_path)
        # plt.show()

        # Plot time path of real wage
        fig, ax = plt.subplots()
        plt.plot(tvec, wpath[:T+5], marker='D')
        # for the minor ticks, use no labels; default NullFormatter
        ax.xaxis.set_minor_locator(minorLocator)
        plt.grid(b=True, which='major', color='0.65', linestyle='-')
        plt.title('Time path for real wage w')
        plt.xlabel(r'Period $t$')
        plt.ylabel(r'Real wage $w_{t}$')
        output_path = os.path.join(output_dir, "wpath")
        plt.savefig(output_path)
        # plt.show()

        # Plot time path of real interest rate
        fig, ax = plt.subplots()
        plt.plot(tvec, rpath[:T+5], marker='D')
        # for the minor ticks, use no labels; default NullFormatter
        ax.xaxis.set_minor_locator(minorLocator)
        plt.grid(b=True, which='major', color='0.65', linestyle='-')
        plt.title('Time path for real interest rate r')
        plt.xlabel(r'Period $t$')
        plt.ylabel(r'Real interest rate $r_{t}$')
        output_path = os.path.join(output_dir, "rpath")
        plt.savefig(output_path)
        # plt.show()


        # Plot time path of real wage
        fig, ax = plt.subplots()
        plt.plot(tvec, bpath[24, :T+5], marker='D')
        # for the minor ticks, use no labels; default NullFormatter
        ax.xaxis.set_minor_locator(minorLocator)
        plt.grid(b=True, which='major', color='0.65', linestyle='-')
        plt.title('Time path for savings by 25-year-olds $b_{25}$')
        plt.xlabel(r'Period $t$')
        plt.ylabel(r'$b_{25}$')
        output_path = os.path.join(output_dir, "b25path")
        plt.savefig(output_path)
        # plt.show()

        

    return tpi_output
