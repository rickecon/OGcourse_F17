import numpy as np
import scipy.optimize as opt
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from matplotlib.patches import Ellipse
import os

def get_sumsq(ellip_params, *args):
    b, upsilon = ellip_params
    elast_Frisch, nvec, ltilde = args
    mu_cfe = nvec ** (1 / elast_Frisch)
    mu_ellip = ((b / ltilde) * ((nvec / ltilde) ** (upsilon - 1)) * ((1 - ((nvec / ltilde) ** upsilon)) ** \
               ((1 - upsilon) / upsilon)))
    sumsq = ((mu_cfe - mu_ellip) ** 2).sum()

    return sumsq

def fit_ellip(ellip_init, elast_Frisch, ltilde, graph = False):
    nvec = np.linspace(0.05, 0.95, 1000)
    args = (elast_Frisch, nvec, ltilde)
    bnds_elp = ((1e-12, None), (1 + 1e-12, None))
    ellip_params = opt.minimize(
        get_sumsq, ellip_init, args=(args), method='L-BFGS-B',
        bounds = bnds_elp)
    if ellip_params.success:
        b_ellip, upsilon = ellip_params.x
    else:
        raise ValueError("Failed to minimize sum of squares")
    if graph:
        cur_path = os.path.split(os.path.abspath(__file__))[0]
        output_fldr = "images"
        output_dir = os.path.join(cur_path, output_fldr)
        if not os.access(output_dir, os.F_OK):
            os.makedirs(output_dir)

        # Plot steady-state consumption and savings distributions
        MU_ellip = \
            ((b_ellip / ltilde) *
             ((nvec / ltilde) ** (upsilon - 1)) *
             ((1 - ((nvec / ltilde) ** upsilon)) **
             ((1 - upsilon) / upsilon)))
        MU_CFE = nvec ** (1 / elast_Frisch)
        fig, ax = plt.subplots()
        plt.plot(nvec, MU_ellip, label='Elliptical')
        plt.plot(nvec, MU_CFE, label='CFE')
        # for the minor ticks, use no labels; default NullFormatter
        minorLocator = MultipleLocator(1)
        ax.xaxis.set_minor_locator(minorLocator)
        plt.grid(b=True, which='major', color='0.65', linestyle='-')
        plt.title('CFE marginal utility and elliptical utility',
                  fontsize=20)
        plt.xlabel(r'Labor supply $n_{s,t}$')
        plt.ylabel(r'Marginal disutility')
        plt.xlim((0, ltilde))
        # plt.ylim((-1.0, 1.15 * (b_ss.max())))
        plt.legend(loc='upper left')
        output_path = os.path.join(output_dir,
                                   'Marginal_Disutility_ellip')
        plt.savefig(output_path)
        # plt.show()
        plt.close()

    return b_ellip, upsilon

elast_Frisch = 0.9
ltilde = 1.0

print(fit_ellip((1, 1), elast_Frisch, ltilde, graph = True))
