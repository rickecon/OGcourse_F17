import time
import numpy as np
import numpy.matlib
import aggregates as aggr
import firms
import Households as hh
import scipy.optimize as opt
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import os
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

def get_path(x1, xT, T, spec):
    if spec == "linear":
        xpath = np.linspace(x1, xT, T)
    elif spec == "quadratic":
        cc = x1
        bb = 2 * (xT - x1) / (T - 1)
        aa = (x1 - xT) / ((T - 1) ** 2)
        xpath = (aa * (np.arange(0, T) ** 2) + (bb * np.arange(0, T)) +
                 cc)
    return xpath

def get_TPI(b1vec, ss_params, params):
    # r_guess: guess of interest rate path from period 1 to T1
    beta, sigma, S, ltilde, b, upsilon, chi_n_vec, A, alpha, delta, tpi_max_iter, tpi_tol, xi_tpi, T1, T2 = params
    r_ss, w_ss, c_ss, n_ss, b_ss, K_ss, L_ss = ss_params
    abs_tpi = 1
    tpi_iter = 0
    rpath_old = np.zeros(T2 + S - 1)
    rpath_old[:T1] = get_path(r_ss, r_ss, T1, 'quadratic')
    rpath_old[T1:] = r_ss
    while abs_tpi > tpi_tol and tpi_iter < tpi_max_iter:
        c1_guess = 1.0
        tpi_iter += 1
        wpath_old = firms.get_w(rpath_old, (A, alpha, delta))
        bmat = np.zeros((S, T2 + S - 1))
        bmat[:, 0] = b1vec
        bmat[:, T2:] = np.matlib.repmat(b_ss, S - 1, 1).T
        nmat = np.zeros((S, T2 + S - 1))
        nmat[:, T2:] = np.matlib.repmat(n_ss, S - 1, 1).T
        cmat = np.zeros((S, T2 + S - 1))
        cmat[:, T2:] = np.matlib.repmat(c_ss, S - 1, 1).T
        # Solve the incomplete remaining lifetime decisions of agents alive
        # in period t=1 but not born in period t=1
        for p in range(S): # p is remaining periods of life
            c1_args = (rpath_old[:p + 1], wpath_old[:p + 1], beta, sigma, ltilde, b, upsilon, chi_n_vec[S - p - 1:], p + 1, b1vec[S - p - 1])
            result_c1 = opt.root(hh.get_b_last, c1_guess, args = (c1_args))
            if result_c1.success:
                c1 = result_c1.x
            else:
                raise ValueError("failed to find an appropriate initial consumption")
            # Calculate aggregate supplies for capital and labor
            cvec = hh.get_c(c1, rpath_old[:p + 1], beta, sigma, p + 1)
            nvec = hh.get_n(cvec, sigma, ltilde, b, upsilon, chi_n_vec[S - p - 1: ], wpath_old[:p + 1], p + 1)
            bvec = hh.get_b(cvec, nvec, rpath_old[:p + 1], wpath_old[:p + 1], p + 1, bs = b1vec[S - p - 1])[1:]
            # Insert the vector lifetime solutions diagonally (twist donut)
            DiagMaskbp = np.eye(p)
            bp_path = DiagMaskbp * bvec
            bmat[S - p:, 1:p + 1] += bp_path

            DiagMasknp = np.eye(p + 1)
            np_path = DiagMasknp * nvec
            nmat[S - p - 1:, :p + 1] += np_path

            DiagMaskcp = np.eye(p + 1)
            cp_path = DiagMaskcp * cvec
            cmat[S - p - 1:, :p + 1] += cp_path
        # Solve for complete lifetime decisions of agents born in periods
        # 1 to T2 and insert the vector lifetime solutions diagonally (twist
        # donut) into the cpath, bpath, and EulErrPath matrices
        for t in range(1, T2):
            c1_args = (rpath_old[t: S + t], wpath_old[t: S + t], beta, sigma, ltilde, b, upsilon, chi_n_vec, S, 0.0)
            result_c1 = opt.root(hh.get_b_last, c1_guess, args = (c1_args))
            if result_c1.success:
                c1 = result_c1.x
            else:
                raise ValueError("failed to find an appropriate initial consumption")
            # Calculate aggregate supplies for capital and labor
            cvec = hh.get_c(c1, rpath_old[t : S + t], beta, sigma, S)
            nvec = hh.get_n(cvec, sigma, ltilde, b, upsilon, chi_n_vec, wpath_old[t: S + t], S)
            bvec = hh.get_b(cvec, nvec, rpath_old[t: S + t], wpath_old[t: S + t], S)
            DiagMaskbt = np.eye(S)
            bt_path = DiagMaskbt * bvec
            bmat[:, t: t + S] += bt_path

            DiagMasknt = np.eye(S)
            nt_path = DiagMasknt * nvec
            nmat[:, t: t + S] += nt_path

            DiagMaskct = np.eye(S)
            ct_path = DiagMaskct * cvec
            cmat[:, t: t + S] += ct_path

        bmat[:, T2:] = np.matlib.repmat(b_ss, S - 1, 1).T
        nmat[:, T2:] = np.matlib.repmat(n_ss, S - 1, 1).T
        cmat[:, T2:] = np.matlib.repmat(c_ss, S - 1, 1).T

        K = aggr.get_K(bmat)[0]
        L = aggr.get_L(nmat)[0]
        Y = aggr.get_Y(K, L, (A, alpha))
        C = aggr.get_C(cmat)
        rpath_new = firms.get_r(K, L, (A, alpha, delta))

        # Calculate the implied capital stock from conjecture and the error
        abs_tpi = ((rpath_old[:T2] - rpath_new[:T2]) ** 2).sum()
        # Update guess
        rpath_old[:T2] = xi_tpi * rpath_new[:T2] + (1 - xi_tpi) * rpath_old[:T2]
        b_err = np.zeros(T2 + S - 1)
        n_err = np.zeros(T2 + S - 1)
        b_last = bmat[S-1, :]
        for i in range(T2 + S - 1):
            b_err[i] = abs(hh.get_b_errors(cmat[:, i], rpath_old[i], beta, sigma)).max()
            n_err[i] = abs(hh.get_n_errors(nmat[:, i], cmat[:, i], sigma, ltilde, b, upsilon, chi_n_vec, wpath_old[i])).max()
        Rc_err = Y[:-1] - C[:-1] - K[1:] + (1 - delta) * K[:-1]
        print('iteration:', tpi_iter, ' squared distance: ', abs_tpi)

        k_first = [k for k in K if abs(k - K_ss) < 0.00001][0]
        T1 = np.where(K == k_first)[0][0]

    return cmat, nmat, bmat, rpath_old, wpath_old, K, L, Y, C, b_err, n_err, b_last, Rc_err, T1

def create_graphs(r, w, K, L, Y, C, cmat, nmat, bmat, T2, S):
    cur_path = os.path.split(os.path.abspath(__file__))[0]
    output_fldr = "images"
    output_dir = os.path.join(cur_path, output_fldr)
    if not os.access(output_dir, os.F_OK):
        os.makedirs(output_dir)
    plt.plot
    plt.plot (1 + np.arange(r.shape[0]), r, 'go--', label = 'interest rate')
    plt.grid(b=True, which='major', color='0.65', linestyle='-')
    plt.title('Time Path of Interest Rate', fontsize=20)
    plt.xlabel('period')
    plt.ylabel('interest rate')
    plt.legend()
    output_path = os.path.join(output_dir, 'tpi_r')
    plt.savefig(output_path)
    plt.close()

    plt.plot (1 + np.arange(w.shape[0]), w, 'go--', label = 'wage')
    plt.grid(b=True, which='major', color='0.65', linestyle='-')
    plt.title('Time Path of Wage', fontsize=20)
    plt.xlabel('period')
    plt.ylabel('wage')
    plt.legend()
    output_path = os.path.join(output_dir, 'tpi_w')
    plt.savefig(output_path)
    plt.close()

    plt.plot (1 + np.arange(K.shape[0]), K, 'go--', label = 'capital')
    plt.grid(b=True, which='major', color='0.65', linestyle='-')
    plt.title('Time Path of Aggregate Capital', fontsize=20)
    plt.xlabel('period')
    plt.ylabel('capital')
    plt.legend()
    output_path = os.path.join(output_dir, 'tpi_K')
    plt.savefig(output_path)
    plt.close()

    plt.plot (1 + np.arange(L.shape[0]), L, 'go--', label = 'labor supply')
    plt.grid(b=True, which='major', color='0.65', linestyle='-')
    plt.title('Time Path of Aggregate Labor', fontsize=20)
    plt.xlabel('period')
    plt.ylabel('labor')
    plt.legend()
    output_path = os.path.join(output_dir, 'tpi_L')
    plt.savefig(output_path)
    plt.close()

    plt.plot (1 + np.arange(Y.shape[0]), Y, 'go--', label = 'output')
    plt.grid(b=True, which='major', color='0.65', linestyle='-')
    plt.title('Time Path of Aggregate Output', fontsize=20)
    plt.xlabel('period')
    plt.ylabel('output')
    plt.legend()
    output_path = os.path.join(output_dir, 'tpi_Y')
    plt.savefig(output_path)
    plt.close()

    plt.plot (1 + np.arange(C.shape[0]), C, 'go--', label = 'consumption')
    plt.grid(b=True, which='major', color='0.65', linestyle='-')
    plt.title('Time Path of Aggregate consumption', fontsize=20)
    plt.xlabel('period')
    plt.ylabel('consumption')
    plt.legend()
    output_path = os.path.join(output_dir, 'tpi_C')
    plt.savefig(output_path)
    plt.close()

    tgrid = np.linspace(1, T2, T2)
    sgrid = np.linspace(1, S, S)
    tmat, smat = np.meshgrid(tgrid, sgrid)
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.set_xlabel(r'period-$t$')
    ax.set_ylabel(r'age-$s$')
    ax.set_zlabel(r'individual consumption $c_{s,t}$')
    strideval = max(int(1), int(round(S / 10)))
    ax.plot_surface(tmat, smat, cmat[:, :T2], rstride=strideval,
                    cstride=strideval, cmap=cm.Blues)
    output_path = os.path.join(output_dir, 'cpath')
    plt.savefig(output_path)
    plt.close()

    tgrid = np.linspace(1, T2, T2)
    sgrid = np.linspace(1, S, S)
    tmat, smat = np.meshgrid(tgrid, sgrid)
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.set_xlabel(r'period-$t$')
    ax.set_ylabel(r'age-$s$')
    ax.set_zlabel(r'individual consumption $n_{s,t}$')
    strideval = max(int(1), int(round(S / 10)))
    ax.plot_surface(tmat, smat, nmat[:, :T2], rstride=strideval,
                    cstride=strideval, cmap=cm.Blues)
    output_path = os.path.join(output_dir, 'npath')
    plt.savefig(output_path)
    plt.close()

    tgrid = np.linspace(1, T2, T2)
    sgrid = np.linspace(1, S, S)
    tmat, smat = np.meshgrid(tgrid, sgrid)
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.set_xlabel(r'period-$t$')
    ax.set_ylabel(r'age-$s$')
    ax.set_zlabel(r'individual consumption $b_{s,t}$')
    strideval = max(int(1), int(round(S / 10)))
    ax.plot_surface(tmat, smat, bmat[:, :T2], rstride=strideval,
                    cstride=strideval, cmap=cm.Blues)
    output_path = os.path.join(output_dir, 'bpath')
    plt.savefig(output_path)
    plt.close()
