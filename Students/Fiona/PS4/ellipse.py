import numpy as np
import scipy.optimize as opt
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import os

def fit_ellip(elast_Frisch, l_tilde):
    b_i=0.1
    upsi_i=0.3
    l_sup=np.linspace(0.05,0.95, 1000)
    fargs=(elast_Frisch,l_tilde,l_sup)
    bnds=((1e-10, None), (1e-10, None))
    params_init=np.array([b_i,upsi_i])
    result=opt.minimize(MU_err,params_init,args=(fargs),bounds=bnds)
    b_ellip, upsilon=result.x
    sum_err=result.fun
    return b_ellip, upsilon


def MU_err (params,*args):
    bi, ui = params
    theta, l_tilde, l_sup=args
    MU_cfe= l_sup**(1/theta)
    MU_elp=1/l_sup*bi*((l_sup/l_tilde)**ui)*(1-(l_sup/l_tilde)**ui)**(1/ui-1)
    sum_sqerr=((MU_cfe-MU_elp)**2).sum()
    return sum_sqerr

theta=0.9
l_tilde=1
b,upsilon=fit_ellip(theta,l_tilde)
l_sup=np.linspace(0.05,0.95, 1000)
MU_cfe=l_sup**(1/theta)
MU_elp=1/l_sup*b*((l_sup/l_tilde)**upsilon)*(1-(l_sup/l_tilde)**upsilon)**(1/upsilon-1)
fig, ax = plt.subplots()
plt.plot(l_sup, MU_elp, label='Ellipse MU')
plt.plot(l_sup, MU_cfe, label='CFE MU')
minorLocator = MultipleLocator(1)
ax.xaxis.set_minor_locator(minorLocator)
plt.grid(b=True, which='major', color='0.65', linestyle='-')
plt.title('CFE marginal utility versus fitted Ellipse',
          fontsize=20)
plt.xlabel(r'Labor supply $n_{s,t}$')
plt.ylabel(r'Marginal disutility')
plt.xlim((0, 1))
plt.legend(loc='upper left')
plt.savefig('EllipseVSCFE_MU')
plt.show()
plt.close()
