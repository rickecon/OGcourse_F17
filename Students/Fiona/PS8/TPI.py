
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


def get_TPI(params, b1vec, graphs):

    start_time = time.clock()
    S, T, beta, sigma, nvec, A, alpha, delta, b_ss, K_ss, C_ss, BQ_ss, omega_SS, \
    maxiter_TPI, mindist_TPI, xi, TPI_tol, fert_rates, mort_rates, imm_rates, g_y, omega_path, gvec = params
    #b1vec 80, s=2~S+1
    # print (f"omega_path: {omega_path.shape}, {nvec.shape},{b1vec.shape}")
    K1 = utils.get_K(b1vec, omega_path[:,0], gvec[0], imm_rates)

    # K: s=1 ~ S, sum of b: s=2 ~ S+1
    Kpath_old = np.zeros(T + S-1)
    # print (f'K1: {K1}, K_ss: {K_ss}')
    Kpath_old[:T] = np.linspace(K1, K_ss, T)  # Until reaching steady state
    Kpath_old[T:] = K_ss
    r = utils.get_r(Kpath_old, utils.get_L(nvec, omega_path[:, 0]), alpha, delta, A)
    BQ1 = utils.get_BQ(b1vec,r[0], gvec[0], omega_path[:,0], mort_rates)
    BQpath_old = np.zeros(T + S-1)
    BQpath_old[:T] = np.linspace(BQ1, BQ_ss, T)  # Until reaching steady state
    BQpath_old[T:] = BQ_ss
    L = utils.get_L(nvec, omega_path[:,0])

    iter_TPI = int(0)
    abs2 = 10.
    Kpath_new = Kpath_old.copy()

    # cbe_params = (S, T, beta, sigma, nvec, b_ss, TPI_tol, A, alpha, delta, bq_distr, chi)

    while (iter_TPI < maxiter_TPI) and (abs2 >= mindist_TPI):
        Kpath_old[T:] = K_ss
        BQpath_old[T:] = BQ_ss
        iter_TPI += 1
        # Kpath_init = xi * Kpath_new + (1 - xi) * Kpath_old
        rpath = utils.get_r(Kpath_old, L, alpha, delta, A)
        wpath = utils.get_w(Kpath_old, L, alpha, A)
        # print (rpath.shape, wpath.shape)
        # cpath, bpath, EulErrPath = get_cbepath(cbe_params, rpath, wpath, b1vec)
        # b: 2~S+1
        bpath = np.zeros((S, T + S - 1))
        bpath[:, 0] = np.append(0,b1vec)

        cpath = np.zeros((S, T + S-1 ))
        EulErrPath = np.zeros((S, T + S-1))
        cpath[S - 1, 0] = ((1 + rpath[0]) * b1vec[S - 2] + wpath[0] * nvec[S - 1])
        for p in range (2, S):
            bvec_guess = np.diagonal(bpath[S - p:, :p-1])
            beg_wealth = bpath[S - p - 1, 0]
            #beg_wealth, nvec, beta, sigma, wpath, rpath, BQpath, omhat,g_y
            args_sol= (beg_wealth, nvec[-p:], beta, sigma, wpath[:p], rpath[:p], BQpath_old[:p], omega_path[-p:,0],g_y)
            b_sol = opt.root(utils.EulerSys_tpi, bvec_guess, args=(args_sol)).x
            #rpath, wpath, bvec, nvec,  g_y, bq, omhat
            # print (rpath[:p+1].shape, wpath[:p].shape,bvec_guess.shape, nvec[-p:].shape,BQpath_old[:p].shape,omega_path[-p:,0].shape)
            cp, c_cnstr_p = utils.get_cvec_tpi(rpath[:p], wpath[:p], b_sol, nvec[-p:], g_y,
                                               BQpath_old[:p],omega_path[-p:,0])
            #beg_wealth, nvec, beta, sigma, wpath, rpath, BQpath, omhat,g_y
            b_err_p = utils.EulerSys_tpi(b_sol, beg_wealth,  nvec[-p:], beta, sigma, wpath[:p], rpath[:p], BQpath_old[:p],omega_path[-p:,0],g_y)

            # Insert the vector lifetime solutions diagonally (twist donut)
            bp_path = np.eye(p-1) * b_sol
            cp_path = np.eye(p-1) * cp
            ep_path = np.eye(p-1) * b_err_p
            # print(f' ep-path:{b_err_p}, cp_path:{cp}')
            # print (S - p+1,p)
            bpath[S - p+1:, 1:p] += bp_path
            cpath[S - p+1:, 1:p] += cp_path
            # print (f'eulErrPathL: {EulErrPath[S - p:, 1:p+1].shape}, ep-path:{ep_path.shape}')
            # if ep_path!=[]:
            EulErrPath[S - p+1:, 1:p] += ep_path

        for t in range(1, T):
            bvec_guess_t = np.diagonal(bpath[:, t - 1:S + t - 2])
            #beg_wealth, nvec, beta, sigma, wpath, rpath, BQpath, omhat,g_y
            args_bt = ([0], nvec, beta, sigma, wpath[t - 1: S + t - 1], rpath[t - 1: S + t - 1], BQpath_old[t - 1: S + t - 1],
                       omega_path[:, t], g_y)
            bt = opt.root(utils.EulerSys_tpi, bvec_guess_t, args=(args_bt)).x
            # print (f'bt.shape: {bt.shape}, ')
            #np.append(nvec,[0.2])
            # rpath, wpath, bvec, nvec, bq, bq_distr
            ct, c_cnstr_t = utils.get_cvec_tpi(rpath[t - 1: S + t - 1], wpath[t - 1: S + t - 1],bt,
                                               nvec, g_y, BQpath_old[t - 1: S + t - 1], omega_path[:, t])
            b_err_t = utils.EulerSys_tpi(bt,[0], nvec, beta, sigma, wpath[t - 1: S + t - 1], rpath[t - 1: S + t - 1], BQpath_old[t - 1: S + t - 1],
                                         omega_path[:, t], g_y)

            # DiagMask = np.eye(S)

            bt_path = np.eye(S-1) * bt
            # print(f'bt: {bt.shape}, btpath:{bt_path.shape}, t:{t}, bpath:{bpath.shape}')
            ct_path = np.eye(S-1) * ct
            et_path = np.eye(S-1) * b_err_t
            bpath[:-1, t: S + t-1] += bt_path
            cpath[:-1, t: S + t-1] += ct_path
            EulErrPath[:-1, t: S + t-1] += et_path

        #(omhat[:-1]*bhat_vec +  imm_rates[1:] * omhat[1:] * bhat_vec)[1:].sum()/(1+g_n)
        Kpath_new = np.zeros(T + S-2)
        # omega_path1 = np.hstack((omega_path,omega_path[:,-1:].reshape(S,1)))
        # imm_rates1 = np.repeat(imm_rates.reshape(S,1),400,axis=1)
        # gvec1 = np.hstack((gvec, np.repeat(gvec[-1],22)))
        print (bpath.shape, omega_path.shape, Kpath_new.shape, Kpath_old.shape)
        # Kpath_new = (1 / (1 + gvec1[1:])) * ((omega_path1[:-1, :] * bpath +
        #                                     imm_rates1[1:,1:] * omega_path1[1:, :] * bpath).sum(axis=0))
        Kpath_new[:T]= utils.get_K(bpath[:-1, T],omega_path[:,T],gvec[T],imm_rates)
        Kpath_cnstr = Kpath_new[:T]<=0
        Kpath_new[T:] = K_ss * np.ones(Kpath_new[T:].shape)
        # Kpath_cnstr = np.append(Kpath_cnstr,
        #                         np.zeros(S, dtype=bool))
        # Kpath_new[Kpath_cnstr] = 0.1
        Kpath_new[0] = K1
        BQpath_new = np.zeros(T + S-1)
        # print (f'Here: bpath {rpath:{rpath[:T].shape}')
        #, BQ:{BQpath_new.shape}, rpath:{rpath[:T].shape}
        # BQpath_new =((1 + rpath) / (1 + gvec1[1:])) * (( omega_path1[:-1, :] * bpath).sum(axis=0))
        BQpath_new[:T] = utils.get_BQ(bpath[:-1,T],rpath[:T],gvec[-1],omega_path[:,T],mort_rates)
        BQpath_new[T:] = BQ_ss * np.ones(BQpath_new[T:].shape)
        BQpath_new[0] = BQ1
        # print (Kpath_new.shape, Kpath_old.shape)
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
    Cpath[:T - 1] = utils.get_C(cpath[:, :T - 1], omega_path[:,:T-1])
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
