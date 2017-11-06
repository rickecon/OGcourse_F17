def get_w(r, params): # function for wage given interest rate
    A, alpha, delta = params
    w = (1 - alpha) * A * (((alpha * A) / (r + delta)) ** (alpha / (1 - alpha)))
    return w

def get_r(K, L, params): # function for interest rate given aggregate capital and labor
    A, alpha, delta = params
    r = alpha * A * ((L / K) ** (1 - alpha)) - delta
    return r
