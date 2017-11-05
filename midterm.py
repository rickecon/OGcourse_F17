import numpy as np
import scipy.optimize as opt
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from matplotlib.patches import Ellipse
import os

# Problem 3
def Euler_error(n, *args):
    w, r, b_3, b_4, sigma, chi = args
    c = (1 + r) * b_3 + w * n - b_4
    error = w * (c ** (-sigma)) - chi * (1 - n) ** (-sigma - 0.1)

    return error


w = 1.0
r = 0.1
b_3 = 1.0
b_4 = 1.1
sigma = 2.2
chi = 2.0
args = (w, r, b_3, b_4, sigma, chi)
n_guess = 0.3

result = opt.root(Euler_error, n_guess, args = (args))
print(result)
n = result.x
print('Labor supply is: ', n)
c = (1 + r) * b_3 + w * n - b_4
error = w * (c ** (-sigma)) - chi * (1 - n) ** (-sigma - 0.1)
print('Euler error is ', error)

# Problem 4
def sumsq(elp_params, *args):
    b, upsilon = elp_params
    nvec, sigma = args

    mu_crra = (1 - nvec) ** (-sigma)
    mu_ellip = b * nvec ** (upsilon - 1) * (1 - nvec ** upsilon) ** ((1 - upsilon) / upsilon)
    sumsq = ((mu_crra - mu_ellip) ** 2).sum()

    return sumsq

def fit_ellip(ellip_init, sigma, graph = False):
    nvec = np.linspace(0.05, 0.95, 1000)
    args = (nvec, sigma)
    bnds_elp = ((1e-12, None), (1 + 1e-12, None))
    ellip_params = opt.minimize(
        sumsq, ellip_init, args=(args), method='L-BFGS-B',
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
        mu_crra = (1 - nvec) ** (-sigma)
        mu_elp = b_ellip * nvec ** (upsilon - 1) * (1 - nvec ** upsilon) ** ((1 - upsilon) / upsilon)
        fig, ax = plt.subplots()
        plt.plot(nvec, mu_elp, label='Elliptical')
        plt.plot(nvec, mu_crra, label='CRRA')
        # for the minor ticks, use no labels; default NullFormatter
        minorLocator = MultipleLocator(1)
        ax.xaxis.set_minor_locator(minorLocator)
        plt.grid(b=True, which='major', color='0.65', linestyle='-')
        plt.title('CRRA marginal utility and elliptical utility',
                  fontsize=20)
        plt.xlabel(r'Labor supply $n_{s,t}$')
        plt.ylabel(r'Marginal disutility')
        plt.xlim((0, 1))
        plt.legend(loc='upper left')
        output_path = os.path.join(output_dir,
                                   'Marginal_Disutility_ellip')
        plt.savefig(output_path)
        plt.close()

    return b_ellip, upsilon

sigma = 2.2

print(fit_ellip((1, 1), sigma, graph = True))

# Problem 5
c1 = np.linspace(0.3, 3.0, 100)
c2 = np.linspace(-0.1, 0.5, 100)
muc1 = c1 ** (-2.2)
muc2 = -20.217 * c2 + 14.7033

cur_path = os.path.split(os.path.abspath(__file__))[0]
output_fldr = "images"
output_dir = os.path.join(cur_path, output_fldr)
if not os.access(output_dir, os.F_OK):
    os.makedirs(output_dir)
fig, ax = plt.subplots()
plt.plot(c1, muc1, label='CRRA')
plt.plot(c2, muc2, label='Linear')
# for the minor ticks, use no labels; default NullFormatter
minorLocator = MultipleLocator(1)
ax.xaxis.set_minor_locator(minorLocator)
plt.grid(b=True, which='major', color='0.65', linestyle='-')
plt.title('CRRA and linear marginal utility',
          fontsize=20)
plt.xlabel(r'Consumption $c$')
plt.ylabel(r'Marginal utility')
plt.xlim((-0.1, 3.0))
plt.legend(loc='upper left')
output_path = os.path.join(output_dir,
                           'Marginal_Utility_consump')
plt.savefig(output_path)
plt.close()
