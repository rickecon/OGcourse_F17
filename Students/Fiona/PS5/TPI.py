import numpy as np
import numpy.matlib
import scipy.optimize as opt
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import matplotlib
import os
import utils
import pandas as pd
import functools
import time


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

# @functools.lru_cache(maxsize=128)
def get_TPI(b1vec, c1_guess, ss_params, params):
    t1=time.time()

    # r_guess: guess of interest rate path from period 1 to T1
    beta, sigma, S, l_ub, b, upsilon, chi, A, alpha, delta, tpi_max_iter, tpi_tol, xi_tpi, T1, T2 = params
    r_ss, w_ss, c_ss, n_ss, b_ss, K_ss, L_ss = ss_params
    abs_tpi = 1
    tpi_iter = 0
    rpath_old = np.zeros(T2 + S - 1)
    rpath_old[:T1] = get_path(r_ss, r_ss, T1, 'quadratic')
    rpath_old[T1:] = r_ss
    while abs_tpi > tpi_tol and tpi_iter < tpi_max_iter:
        tpi_iter += 1
        wpath_old = utils.get_w(rpath_old, (A, alpha, delta))
        bmat = np.zeros((S, T2 + S - 1))
        bmat[:, 0] = b1vec
        bmat[:, T2:] = numpy.matlib.repmat(b_ss, S - 1, 1).T
        nmat = np.zeros((S, T2 + S - 1))
        nmat[:, T2:] = numpy.matlib.repmat(n_ss, S - 1, 1).T
        cmat = np.zeros((S, T2 + S - 1))
        cmat[:, T2:] = numpy.matlib.repmat(c_ss, S-1, 1).T
        # Solve the incomplete remaining lifetime decisions of agents alive
        # in period t=1 but not born in period t=1
        for p in range(S): # p is remaining periods of life
            c1_args = (rpath_old[:p + 1], wpath_old[:p + 1], beta, sigma, l_ub, b, upsilon, p + 1, b1vec[S - p - 1], chi[S-p-1:])
            result_c1 = opt.root(utils.get_b_last, c1_guess, args = (c1_args))
            if result_c1.success:
                c1 = result_c1.x
            else:
                raise ValueError("failed to find an appropriate initial consumption")
            # Calculate aggregate supplies for capital and labor
            cvec = utils.get_c(c1, rpath_old[:p + 1], beta, sigma, p + 1)
            # print (np.shape(cvec))
            nvec = utils.get_n(cvec, sigma, l_ub, b, upsilon, wpath_old[:p + 1], p + 1,chi[S-p-1:])
            bvec = utils.get_b(cvec, nvec, rpath_old[:p + 1], wpath_old[:p + 1], p + 1, bs = b1vec[S - p - 1])[1:]
            # Insert the vector lifetime solutions diagonally (twist donut)
            DiagMaskbp = np.eye(p)
            bp_path = DiagMaskbp * bvec
            bmat[S - p:, 1:p + 1] += bp_path

            DiagMasknp = np.eye(p + 1)
            np_path = DiagMasknp * nvec
            nmat[S - p - 1:, :p + 1] += np_path

            DiagMasknp = np.eye(p + 1)
            c_path = DiagMasknp * cvec
            cmat[S - p - 1:, :p + 1] += c_path

        # Solve for complete lifetime decisions of agents born in periods
        # 1 to T2 and insert the vector lifetime solutions diagonally (twist
        # donut) into the cpath, bpath, and EulErrPath matrices
        for t in range(1, T2):
            c1_args = (rpath_old[t: S + t], wpath_old[t: S + t], beta, sigma, l_ub, b, upsilon, S, 0.0, chi)
            result_c1 = opt.root(utils.get_b_last, c1_guess, args = (c1_args))
            if result_c1.success:
                c1 = result_c1.x
            else:
                raise ValueError("failed to find an appropriate initial consumption")
            # Calculate aggregate supplies for capital and labor
            cvec = utils.get_c(c1, rpath_old[t : S + t], beta, sigma, S)
            nvec = utils.get_n(cvec, sigma, l_ub, b, upsilon, wpath_old[t: S + t], S, chi)
            bvec = utils.get_b(cvec, nvec, rpath_old[t: S + t], wpath_old[t: S + t], S)
            # print ("nvec,cvec,bvec: {}".format(np.shape(nvec),np.shape(cvec),np.shape(bvec)))
            DiagMaskbt = np.eye(S)
            bt_path = DiagMaskbt * bvec
            bmat[:, t: t + S] += bt_path

            DiagMasknp = np.eye(S)
            np_path = DiagMasknp * nvec
            nmat[:, t: t + S] += np_path

            DiagMasknp = np.eye(S)
            c_path = DiagMasknp * cvec
            cmat[:, t: t + S] += c_path

        bmat[:, T2:] = np.matlib.repmat(b_ss, S - 1, 1).T
        nmat[:, T2:] = np.matlib.repmat(n_ss, S - 1, 1).T
        cmat[:, T2:] = np.matlib.repmat(c_ss, S - 1, 1).T

        K = utils.get_K(bmat)[0]
        L = utils.get_L(nmat)[0]
        Y = utils.get_Y(K, L, (A, alpha))
        C = utils.get_C(cmat)

        rpath_new = utils.get_r(K, L, (A, alpha, delta))

        # Calculate the implied capital stock from conjecture and the error
        abs_tpi = ((rpath_old[:T2] - rpath_new[:T2]) ** 2).sum()
        # Update guess
        rpath_old[:T2] = xi_tpi * rpath_new[:T2] + (1 - xi_tpi) * rpath_old[:T2]

        print('iteration:', tpi_iter, ' squared distance: ', abs_tpi)
    b_last = abs(bmat[S - 1, :]).max()
    b_err = abs(utils.get_b_errors(cmat[:, :T2 + S - 1], rpath_old[:T2 + S - 1], beta, sigma)).max()
    n_err = abs(utils.get_n_errors(nmat[:, :T2 + S - 1], cmat[:, :T2 + S - 1], sigma, l_ub, b,
                                   upsilon, wpath_old[:T2 + S - 1], chi[0] * np.ones(T2 + S - 1))).max()
    cnt_err = abs(Y[:-1] - C[:-1] - K[1:] + (1 - delta) * K[:-1]).max()
    t2=time.time()
    t=t2-t1
    print (f'It took {t} seconds to solve TPI.')

    k_first = [k for k in K if abs(k - K_ss) < 0.0001][0]
    T_1 = np.where(K == k_first)[0][0]

    return rpath_old, wpath_old, K, L, bmat, nmat, cmat, b_last, b_err, n_err, cnt_err, T_1


# def solve_life_decs(p,params,rpath,wpath,bvec_init,nvec_init):
#     beta, sigma, b, upsilon, l_tilde, chi=params
#     nbargs=(beta,sigma, b, upsilon, l_tilde,chi, rpath,wpath)
#     nbvec_init=np.append(nvec_init,bvec_init)




def create_graphs(T, S, r, w, K, L, Y, C, cmat, nmat, bmat):
    cur_path = os.path.split(os.path.abspath(__file__))[0]
    output_fldr = "images_TPI"
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

    tgrid = np.linspace(1, T, T)
    sgrid = np.arange(21, 80, 6)
    tmat, smat = np.meshgrid(tgrid, sgrid)
    fig = plt.figure()
    plt.title('TPI consumption $c_{s,t}$', fontsize=20)
    ax = fig.gca(projection='3d')
    ax.set_xlabel(r'period-$t$')
    ax.set_ylabel(r'age-$s$')
    ax.set_zlabel(r'individual consumption $c_{s,t}$')
    strideval = max(int(1), int(round(S / 10)))
    ax.plot_surface(tmat, smat, cmat[:, :T], rstride=strideval,
                    cstride=strideval, cmap=cm.coolwarm)
    output_path = os.path.join(output_dir, 'cpath')
    plt.savefig(output_path)
    plt.close()

    tgrid = np.linspace(1, T, T)
    sgrid = np.arange(21, 80, 6)
    tmat, smat = np.meshgrid(tgrid, sgrid)
    fig = plt.figure()
    plt.title('TPI labor $n_{s,t}$', fontsize=20)
    ax = fig.gca(projection='3d')
    ax.set_xlabel(r'period-$t$')
    ax.set_ylabel(r'age-$s$')
    ax.set_zlabel(r'individual labor $n_{s,t}$')
    strideval = max(int(1), int(round(S / 10)))
    ax.plot_surface(tmat, smat, nmat[:, :T], rstride=strideval,
                    cstride=strideval, cmap=cm.coolwarm)
    output_path = os.path.join(output_dir, 'npath')
    plt.savefig(output_path)
    plt.close()

    tgrid = np.linspace(1, T, T)
    sgrid = np.arange(21, 80, 6)
    tmat, smat = np.meshgrid(tgrid, sgrid)
    fig = plt.figure()
    plt.title('TPI savings $b_{s,t}$', fontsize=20)
    ax = fig.gca(projection='3d')
    ax.set_xlabel(r'period-$t$')
    ax.set_ylabel(r'age-$s$')
    ax.set_zlabel(r'individual savings $b_{s,t}$')
    strideval = max(int(1), int(round(S / 10)))
    ax.plot_surface(tmat, smat, bmat[:, :T], rstride=strideval,
                    cstride=strideval, cmap=cm.coolwarm)
    output_path = os.path.join(output_dir, 'bpath')
    plt.savefig(output_path)
    plt.close()

def write_csv(b_last, b_err, n_err, cnt_err,T_1):
    output={'b_last':b_last, 'b_err':b_err, 'n_err':n_err, 'cnt_err': cnt_err, 'T_1': T_1}
    pd.DataFrame.from_dict(data=output, orient='index').to_csv('output_tpi.csv', header=True)
    # return output