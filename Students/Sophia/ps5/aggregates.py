import numpy as np

def get_L(narr): # function for aggregate labor supply
    epsilon = 0.01
    a = epsilon / np.exp(1)
    b = 1 / epsilon
    if narr.ndim == 1:  # This is the steady-state case
        L = narr.sum()
        L_cstr = L < epsilon
        if L_cstr:
            print('get_L() warning: Distribution of savings and/or ' +
                  'parameters created L < epsilon')
            # Force K >= eps by stitching a * exp(b * K) for K < eps
            L = a * np.exp(b * L)

    elif narr.ndim == 2:  # This is the time path case
        L = narr.sum(axis=0)
        L_cstr = L < epsilon
        if L.min() < epsilon:
            print('get_L() warning: Aggregate capital constraint is ' +
                  'violated (L < epsilon) for some period in time ' +
                  'path.')
            L[L_cstr] = a * np.exp(b * L[L_cstr])
    return L, L_cstr

def get_K(barr): # function for aggregate capital supply
    epsilon = 0.01
    a = epsilon / np.exp(1)
    b = 1 / epsilon
    if barr.ndim == 1:  # This is the steady-state case
        K = barr.sum()
        K_cstr = K < epsilon
        if K_cstr:
            print('get_K() warning: Distribution of savings and/or ' +
                  'parameters created K < epsilon')
            # Force K >= eps by stitching a * exp(b * K) for K < eps
            K = a * np.exp(b * K)

    elif barr.ndim == 2:  # This is the time path case
        K = barr.sum(axis=0)
        K_cstr = K < epsilon
        if K.min() < epsilon:
            print('get_K() warning: Aggregate capital constraint is ' +
                  'violated (K < epsilon) for some period in time ' +
                  'path.')
            K[K_cstr] = a * np.exp(b * K[K_cstr])

    return K, K_cstr

def get_Y(K, L, params): # function for aggregate output
    A, alpha = params
    Y = A * (K ** alpha) * (L ** (1 - alpha))
    return Y

def get_C(carr):
    if carr.ndim == 1:
        C = carr.sum()
    elif carr.ndim == 2:
        C = carr.sum(axis=0)

    return C
