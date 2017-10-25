import numpy as np

def MU_c_stitch(cvec, sigma):
    epsilon = 1e-4
    c_cnstr = cvec < epsilon
    muc = cvec ** (-sigma)
    m1 = (-sigma) * epsilon ** (-sigma - 1)
    m2 = epsilon ** (-sigma) - m1 * epsilon
    muc[c_cnstr] = m1 * cvec[c_cnstr] + m2

    return muc

def MU_n_stitch(nvec, params):
    l_tilde, b, upsilon = params
    epsilon_lb = 1e-6
    epsilon_ub = l_tilde - epsilon_lb
    nl_cstr = nvec < epsilon_lb
    nu_cstr = nvec > epsilon_ub

    mun = ((b / l_tilde) * ((nvec / l_tilde) ** (upsilon - 1)) * (1 - ((nvec / l_tilde) ** upsilon)) **\
           ((1 - upsilon) / upsilon))
    m1 = (b * (l_tilde ** (-upsilon)) * (upsilon - 1) * (epsilon_lb ** (upsilon - 2)) * \
         ((1 - ((epsilon_lb / l_tilde) ** upsilon)) ** ((1 - upsilon) / upsilon)) * \
         (1 + ((epsilon_lb / l_tilde) ** upsilon) * ((1 - ((epsilon_lb / l_tilde) ** upsilon)) ** (-1))))
    m2 = ((b / l_tilde) * ((epsilon_lb / l_tilde) ** (upsilon - 1)) * \
         ((1 - ((epsilon_lb / l_tilde) ** upsilon)) ** ((1 - upsilon) / upsilon)) - (m1 * epsilon_lb))
    q1 = (b * (l_tilde ** (-upsilon)) * (upsilon - 1) * (epsilon_ub ** (upsilon - 2)) * \
         ((1 - ((epsilon_ub / l_tilde) ** upsilon)) ** ((1 - upsilon) / upsilon)) * \
         (1 + ((epsilon_ub / l_tilde) ** upsilon) * ((1 - ((epsilon_ub / l_tilde) ** upsilon)) ** (-1))))
    q2 = ((b / l_tilde) * ((epsilon_ub / l_tilde) ** (upsilon - 1)) * \
         ((1 - ((epsilon_ub / l_tilde) ** upsilon)) ** ((1 - upsilon) / upsilon)) - (q1 * epsilon_ub))
    mun[nl_cstr] = m1 * nvec[nl_cstr] + m2
    mun[nu_cstr] = q1 * nvec[nu_cstr] + q2
    return mun

cvec=np.array([-0.01,-0.004, 0.5, 2.6])
sigma=2.2
print (MU_c_stitch(cvec,sigma))

nvec=np.array([-0.013, -0.002, 0.42, 1.007, 1.011])
l_tilde=1
b=0.5
upsilon=1.5
print (MU_n_stitch(nvec,(l_tilde,b,upsilon)))