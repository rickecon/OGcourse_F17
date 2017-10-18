# import packages
import scipy.optimize as opt
import numpy as np
import time
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import os

# Household Parameters
yrs_live = 80
S = 80
beta_annual = .96
beta = beta_annual ** (yrs_live / S)
sigma = 3.0
nvec = np.ones(S)
cut = round(2 * S / 3)
nvec[cut: ] = 0.2

# Firm Parameters
alpha = 0.35
A = 1.0
delta_annual = 0.05
delta = 1- ((1 - delta_annual) ** (yrs_live / S))

# Define functions for calculating L, K, w, r, Y

def get_L(nvec): # function for aggregate labor
    L = nvec.sum()
    return L

def get_K(bvec): # function for aggregate capital
    K = bvec.sum()
    return K

def get_w(K, L, params): # function for wage
    A, alpha = params
    w = (1 - alpha) * A * ((K / L) ** alpha)
    return w

def get_r(K, L, params): # function for interest rate
    A, alpha, delta = params
    r = alpha * A * ((L / K) ** (1 - alpha)) - delta
    return r

def get_Y(K, L, params): # function for output
    A, alpha = params
    Y = A * (K ** alpha) * (L ** (1 - alpha))
    return Y

# define function that checks feasibility

def feasible(f_params, bvec_guess):
    nvec, A, alpha, delta = f_params

    K = get_K(bvec_guess)
    K_cnstr = K <= 0
    L = get_L(nvec)

    if not K_cnstr:
        w = get_w(K, L, (A, alpha))
        r = get_r(K, L, (A, alpha, delta))

        b = np.append([0], bvec_guess)
        b1 = np.append(bvec_guess, [0])
        cvec = (1 + r) * b + w * nvec - b1

        c_cnstr = cvec <= 0
        b_cnstr = c_cnstr[:-1] + c_cnstr[1:]

    else:
        c_cnstr = np.ones(cvec.shape[0], dtype = bool)
        b_cnstr = np.ones(cvec.shape[0] - 1, dtype = bool)

    return b_cnstr, c_cnstr, K_cnstr

f_params = (nvec, A, alpha, delta)
bvec_guess1 = np.ones(S-1)

bvec_guess2 = \
np.array([-0.01, 0.1, 0.2, 0.23, 0.25, 0.23, 0.2, 0.1,
-0.01, 0.1, 0.2, 0.23, 0.25, 0.23, 0.2, 0.1,
-0.01, 0.1, 0.2, 0.23, 0.25, 0.23, 0.2, 0.1,
-0.01, 0.1, 0.2, 0.23, 0.25, 0.23, 0.2, 0.1,
-0.01, 0.1, 0.2, 0.23, 0.25, 0.23, 0.2, 0.1,
-0.01, 0.1, 0.2, 0.23, 0.25, 0.23, 0.2, 0.1,
-0.01, 0.1, 0.2, 0.23, 0.25, 0.23, 0.2, 0.1,
-0.01, 0.1, 0.2, 0.23, 0.25, 0.23, 0.2, 0.1,
-0.01, 0.1, 0.2, 0.23, 0.25, 0.23, 0.2, 0.1,
-0.01, 0.1, 0.2, 0.23, 0.25, 0.23, 0.2])

bvec_guess3 = \
np.array([-0.01, 0.1, 0.2, 0.23, 0.25, 0.23, 0.2, 0.1,
-0.01, 0.1, 0.2, 0.23, 0.25, 0.23, 0.2, 0.1,
-0.01, 0.1, 0.2, 0.23, 0.25, 0.23, 0.2, 0.1,
-0.01, 0.1, 0.2, 0.23, 0.25, 0.23, 0.2, 0.1,
-0.01, 0.1, 0.2, 0.23, 0.25, 0.23, 0.2, 0.1,
-0.01, 0.1, 0.2, 0.23, 0.25, 0.23, 0.2, 0.1,
-0.01, 0.1, 0.2, 0.23, 0.25, 0.23, 0.2, 0.1,
0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1,
0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1,
0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])

print(np.where(feasible(f_params, bvec_guess1)[0]==True)[0]+1)
print(np.where(feasible(f_params, bvec_guess1)[1]==True)[0]+1)

print(np.where(feasible(f_params, bvec_guess2)[0]==True)[0]+1)
print(np.where(feasible(f_params, bvec_guess2)[1]==True)[0]+1)

print(np.where(feasible(f_params, bvec_guess3)[0]==True)[0]+1)
print(np.where(feasible(f_params, bvec_guess3)[1]==True)[0]+1)

def errors(bvec, *args):

    A, alpha, delta, nvec, beta, sigma = args
    L = get_L(nvec)
    K = get_K(bvec)
    w = get_w(K, L, (A, alpha))
    r = get_r(K, L, (A, alpha, delta))

    b = np.append([0], bvec)
    b1 = np.append(bvec, [0])
    cvec = (1 + r) * b + w * nvec - b1

    muc = cvec ** (-sigma)

    EulErr_ss = muc[:-1] - beta * (1 + r) * muc[1:]

    return EulErr_ss

def get_SS(params, bvec_guess, SS_graphs = False):

    start_time = time.clock()

    beta, sigma, nvec, L, A, alpha, delta, SS_tol = params

    b = opt.root(errors, bvec_guess, args = (A, alpha, delta, nvec, beta, sigma), tol = SS_tol)
    if b.success:
        b_ss = b.x
    else:
        raise ValueError("Failed to find b_ss")

    K_ss = get_K(b_ss)
    w_ss = get_w(K_ss, L, (A, alpha))
    r_ss = get_r(K_ss, L, (A, alpha, delta))
    Y_ss = get_Y(K_ss, L, (A, alpha))

    b = np.append([0], b_ss)
    b1 = np.append(b_ss, [0])
    c_ss = (1 + r_ss) * b + w_ss * nvec - b1
    C_ss = c_ss.sum()

    muc = c_ss ** (-sigma)
    EulErr_ss = muc[:-1] - beta * (1 + r_ss) * muc[1:]

    RCerr_ss = Y_ss - C_ss - delta * K_ss

    ss_time = time.clock() - start_time

    ss_output = {
        'b_ss': b_ss, 'c_ss': c_ss, 'w_ss': w_ss, 'r_ss': r_ss,
        'K_ss': K_ss, 'Y_ss': Y_ss, 'C_ss': C_ss,
        'EulErr_ss': EulErr_ss, 'RCerr_ss': RCerr_ss,
        'ss_time': ss_time}

    b = [float(i) for i in ["%.2f" % v for v in b_ss]]
    K = float("{0:.2f}".format(K_ss))
    w = float("{0:.2f}".format(w_ss))
    r = float("{0:.2f}".format(r_ss))
    c = [float(i) for i in ["%.2f" % v for v in c_ss]]

    print('\n Savings: \t\t\t {} \n Capital and Labor: \t\t {} \n Wage and Interest rate: \t {} \n Consumption: \t\t\t {}'.format(b, np.array([K, L]), np.array([w, r]), c))

    print('Euler errors: ', EulErr_ss)
    print('Resource Constraint error: ', RCerr_ss)
    print('Time needed: ', ss_time)

    if SS_graphs:
        cur_path = os.path.split(os.path.abspath(__file__))[0]
        output_fldr = "images"
        output_dir = os.path.join(cur_path, output_fldr)
        if not os.access(output_dir, os.F_OK):
            os.makedirs(output_dir)

        age = np.arange(1, S + 1)
        fig, ax = plt.subplots()
        plt.plot(age, c_ss, marker='D',label='Consumption')
        plt.plot(age, np.append([0], b_ss), marker = 'D', label='Savings')
        minorLocator = MultipleLocator(1)
        ax.xaxis.set_minor_locator(minorLocator)
        plt.grid(b=True, which='major', color='0.65', linestyle='-')
        plt.title('Steady-state consumption and savings')
        plt.xlabel('Age')
        plt.ylabel('Consumption units')
        plt.legend()
        output_path = os.path.join(output_dir, 'ss_bc')
        plt.savefig(output_path)
        plt.close()

    return ss_output

L_ss = get_L(nvec)
SS_tol = 1e-9

params = (beta, sigma, nvec, L_ss, A, alpha, delta, SS_tol)
bvec_guess = np.ones(S - 1) * 0.1
SS = get_SS(params, bvec_guess, SS_graphs = True)

# TPI params
T = 320
max_iter = 300
tol = 1e-9
xi = 0.2
b_ss = SS['b_ss']
K_ss = SS['K_ss']
w_ss = SS['w_ss']
r_ss = SS['r_ss']


# Initial guess for capital stock
weights = ((1.5 - 0.87) / 78) * np.arange(S - 1) + 0.87
b1vec = weights * b_ss
K1 = get_K(b1vec)
Kpath_old = np.zeros(T + S - 1)
Kpath_old[:T] = np.linspace(K1, K_ss, T) # Until reaching steady state
Kpath_old[T:] = K_ss

# Euler function error
'''
Calculate lifetime euler function error. Remaining lifetime can be of varying length p.
bvec is of length p-1.

'''

def get_errors(bvec, *args):
    beg_wealth, nvec, beta, sigma, w_path, r_path = args
    b1 = np.append(beg_wealth, bvec)
    b2 = np.append(bvec, 0)
    c = (1 + r_path) * b1 + w_path * nvec - b2
    muc = c ** (-sigma)
    errors = muc[:-1] - beta * (1 + r_path[1:]) * muc[1:]
    return errors


# Begin TPI
abs2 = 1
tpi_iter = 0

while abs2 > tol and tpi_iter < max_iter:
    tpi_iter = tpi_iter + 1
    w_path = get_w(Kpath_old, L_ss, (A, alpha))
    r_path = get_r(Kpath_old, L_ss, (A, alpha, delta))
    # Initialize savings matrix
    b = np.zeros((S - 1, T + S - 1))
    b[:, 0] = b1vec

    # Solve the incomplete remaining lifetime decisions of agents alive
    # in period t=1 but not born in period t=1
    for p in range(2, S):
        bvec_guess = np.diagonal(b[S - p:, :p - 1]) # Initial guess of the lifetime savings path for individual with p periods to live
        beg_wealth = b[S - p - 1, 0]
        args_bp = (beg_wealth, nvec[-p:], beta, sigma, w_path[:p], r_path[:p])
        bp = opt.root(get_errors, bvec_guess, args = (args_bp)).x
    # Insert the vector lifetime solutions diagonally (twist donut)
        DiagMaskbp = np.eye(p - 1)
        bp_path = DiagMaskbp * bp
        b[S - p:, 1:p] += bp_path

    # Solve for complete lifetime decisions of agents born in periods
    # 1 to T
    for t in range(1, T + 1):
        bvec_guess = np.diagonal(b[:, t - 1:S + t - 2])
        args_bt = (0, nvec, beta, sigma, w_path[t - 1 : S + t - 1], r_path[t - 1 : S + t - 1])
        bt = opt.root(get_errors, bvec_guess, args = (args_bt)).x
        DiagMaskbt = np.eye(S - 1)
        bt_path = DiagMaskbt * bt
        b[:, t: S + t - 1] += bt_path

    # Calculate the implied capital stock from conjecture and the error
    Kpath_new = b.sum(axis = 0)
    abs2 = (((Kpath_old[:T] - Kpath_new[:T])/Kpath_old[:T] * 100) ** 2).sum()
    # Update guess
    Kpath_old[:T] = xi * Kpath_new[:T] + (1 - xi) * Kpath_old[:T]
    print('iteration:', tpi_iter, ' squared distance: ', abs2)


cur_path = os.path.split(os.path.abspath(__file__))[0]
output_fldr = "images"
output_dir = os.path.join(cur_path, output_fldr)
if not os.access(output_dir, os.F_OK):
    os.makedirs(output_dir)

plt.plot(1 + np.arange(T + 5), np.append(Kpath_old[:T-2], K_ss * np.ones(7)), 'go--')
plt.grid(b=True, which='major', color='0.65', linestyle='-')
plt.title('Time path of aggregate capital', fontsize=20)
plt.xlabel(r'$t$')
plt.ylabel(r'$K$')
output_path = os.path.join(output_dir, 'kplot')
plt.savefig(output_path)
plt.close()

w_path = get_w(Kpath_old, L_ss, (A, alpha))
r_path = get_r(Kpath_old, L_ss, (A, alpha, delta))

plt.plot(1 + np.arange(T + 5), np.append(w_path[:T-2], w_ss * np.ones(7)), 'go--')
plt.grid(b=True, which='major', color='0.65', linestyle='-')
plt.title('Time path of wage', fontsize=20)
plt.xlabel(r'$t$')
plt.ylabel(r'$w$')
output_path = os.path.join(output_dir, 'wplot')
plt.savefig(output_path)
plt.close()

plt.plot(1 + np.arange(T + 5), np.append(r_path[:T-2], r_ss * np.ones(7)), 'go--')
plt.grid(b=True, which='major', color='0.65', linestyle='-')
plt.title('Time path of interest rate', fontsize=20)
plt.xlabel(r'$t$')
plt.ylabel(r'$r$')
output_path = os.path.join(output_dir, 'rplot')
plt.savefig(output_path)
plt.close()

b15 = b[14, :]
plt.plot(1 + np.arange(T + 5), np.append(b15[:T-2], b_ss[14] * np.ones(7)), 'go--')
plt.grid(b=True, which='major', color='0.65', linestyle='-')
plt.title('Time path of savings for age-15 individual', fontsize=20)
plt.xlabel(r'$t$')
plt.ylabel(r'$b_{15}$')
output_path = os.path.join(output_dir, 'bplot')
plt.savefig(output_path)
plt.close()

k_first = [k for k in Kpath_old if abs(k - K_ss) < 0.00001][0]
print(k_first)
T1 = np.where(Kpath_old == k_first)[0][0]
print(T1 + 1)

k_last = [k for k in Kpath_old[:-3] if abs(k- K_ss) > 0.00001][-1]
print(k_last)
T2 = np.where(Kpath_old == k_last)[0][0]
print(T2 + 1)
