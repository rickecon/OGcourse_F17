import numpy as np
import scipy.optimize as opt
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import os

def MU_c_stitch(cvec, sigma, graph = False):
    epsilon = 1e-4
    muc = cvec ** (-sigma)
    m1 = (-sigma) * epsilon ** (-sigma - 1)
    m2 = epsilon ** (-sigma) - m1 * epsilon
    c_cnstr = cvec < epsilon
    muc[c_cnstr] = m1 * cvec[c_cnstr] + m2

    if graph:
        # Create directory if images directory does not already exist
        cur_path = os.path.split(os.path.abspath(__file__))[0]
        output_fldr = "images"
        output_dir = os.path.join(cur_path, output_fldr)
        if not os.access(output_dir, os.F_OK):
            os.makedirs(output_dir)

        # Plot steady-state consumption and savings distributions
        cvec_CRRA = np.linspace(epsilon / 2, epsilon * 3, 1000)
        MU_CRRA = cvec_CRRA ** (-sigma)
        cvec_stitch = np.linspace(-0.0001, epsilon, 500)
        MU_stitch = m1 * cvec_stitch + m2
        fig, ax = plt.subplots()
        plt.plot(cvec_CRRA, MU_CRRA, ls='solid', label='$u\'(c)$: CRRA')
        plt.plot(cvec_stitch, MU_stitch, ls='dashed', color='red',
                 label='$g\'(c)$: stitched')
        # for the minor ticks, use no labels; default NullFormatter
        minorLocator = MultipleLocator(1)
        ax.xaxis.set_minor_locator(minorLocator)
        plt.grid(b=True, which='major', color='0.65', linestyle='-')
        plt.title('Marginal utility of consumption with stitched function', fontsize=15)
        plt.xlabel(r'Consumption $c$')
        plt.ylabel(r'Marginal utility $u\'(c)$')
        plt.xlim((-0.0001, epsilon * 3))
        plt.legend(loc='upper right')
        output_path = os.path.join(output_dir, "MU_c_stitched")
        plt.savefig(output_path)
        plt.close()

    return muc

cvec = np.array([-0.01, -0.004, 0.5, 2.6])
sigma = 2.2
print(MU_c_stitch(cvec, sigma, graph = True))

def MDU_n_stitch(nvec, ltide, b, upsilon, graph = False):

    epsilon_lb = 1e-6
    epsilon_ub = ltilde - epsilon_lb

    mdun = ((b / ltilde) * ((nvec / ltilde) ** (upsilon - 1)) * (1 - ((nvec / ltilde) ** upsilon)) **\
           ((1 - upsilon) / upsilon))

    m1 = (b * (ltilde ** (-upsilon)) * (upsilon - 1) * (epsilon_lb ** (upsilon - 2)) * \
         ((1 - ((epsilon_lb / ltilde) ** upsilon)) ** ((1 - upsilon) / upsilon)) * \
         (1 + ((epsilon_lb / ltilde) ** upsilon) * ((1 - ((epsilon_lb / ltilde) ** upsilon)) ** (-1))))
    m2 = ((b / ltilde) * ((epsilon_lb / ltilde) ** (upsilon - 1)) * \
         ((1 - ((epsilon_lb / ltilde) ** upsilon)) ** ((1 - upsilon) / upsilon)) - (m1 * epsilon_lb))

    q1 = (b * (ltilde ** (-upsilon)) * (upsilon - 1) * (epsilon_ub ** (upsilon - 2)) * \
         ((1 - ((epsilon_ub / ltilde) ** upsilon)) ** ((1 - upsilon) / upsilon)) * \
         (1 + ((epsilon_ub / ltilde) ** upsilon) * ((1 - ((epsilon_ub / ltilde) ** upsilon)) ** (-1))))

    q2 = ((b / ltilde) * ((epsilon_ub / ltilde) ** (upsilon - 1)) * \
         ((1 - ((epsilon_ub / ltilde) ** upsilon)) ** ((1 - upsilon) / upsilon)) - (q1 * epsilon_ub))

    nl_cstr = nvec < epsilon_lb
    nu_cstr = nvec > epsilon_ub

    mdun[nl_cstr] = m1 * nvec[nl_cstr] + m2
    mdun[nu_cstr] = q1 * nvec[nu_cstr] + q2

    if graph:
        cur_path = os.path.split(os.path.abspath(__file__))[0]
        output_fldr = "images"
        output_dir = os.path.join(cur_path, output_fldr)
        if not os.access(output_dir, os.F_OK):
            os.makedirs(output_dir)

        # Plot steady-state consumption and savings distributions
        nvec_ellip = np.linspace(epsilon_lb / 2, epsilon_ub +
                                 ((ltilde - epsilon_ub) / 5), 1000)
        MU_ellip = ((b / ltilde) * ((nvec_ellip / ltilde) ** (upsilon - 1)) * \
                    ((1 - ((nvec_ellip / ltilde) ** upsilon)) ** ((1 - upsilon) / upsilon)))
        n_stitch_low = np.linspace(-0.1, epsilon_lb, 500)
        MU_stitch_low = m1 * n_stitch_low + m2
        n_stitch_high = np.linspace(epsilon_ub, ltilde + 0.00005, 500)
        MU_stitch_high = q1 * n_stitch_high + q2

        fig, ax = plt.subplots()
        plt.plot(nvec_ellip, MU_ellip, ls='solid', color='black', label='$v\'(n)$: Elliptical')
        plt.plot(n_stitch_low, MU_stitch_low, ls='dashed', color='red', label='$g\'(n)$: low stitched')
        plt.plot(n_stitch_high, MU_stitch_high, ls='dotted', color='blue', label='$g\'(n)$: high stitched')
        # for the minor ticks, use no labels; default NullFormatter
        minorLocator = MultipleLocator(1)
        ax.xaxis.set_minor_locator(minorLocator)
        plt.grid(b=True, which='major', color='0.65', linestyle='-')
        plt.title('Marginal disutility of labor with stitched function', fontsize=15)
        plt.xlabel(r'Labor $n$')
        plt.ylabel(r'Marginal disutility $v\'(n)$')
        plt.xlim((-0.1, ltilde + 0.00005))
        # plt.ylim((-1.0, 1.15 * (b_ss.max())))
        plt.legend(loc='upper left')
        output_path = os.path.join(output_dir, "MU_n_stitched")
        plt.savefig(output_path)
        plt.close()
        
    return mdun

nvec = np.array([-0.013, -0.002, 0.42, 1.007, 1.011])
ltilde = 1.0
b_ellip = 0.5
upsilon = 1.5
print(MDU_n_stitch(nvec, ltilde, b_ellip, upsilon, graph = True))
