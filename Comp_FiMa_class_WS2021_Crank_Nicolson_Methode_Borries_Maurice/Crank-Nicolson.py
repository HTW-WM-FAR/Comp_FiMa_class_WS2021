import numpy as np
from matplotlib import pyplot


def Ami_Put_Op(K, r, sigma, T, Smax, M, N):
    Delta_t = T/N
    Delta_S = Smax/M
    t = np.arange(1, M, 1)
    V = np.zeros((N+1, M+1))
    S = np.arange(0, Smax + Delta_S, Delta_S)

    for j in range(len(M+1)):
        V = np.zeros((N+1, M+1))
        V[:, :] = np.nan
        S = np.arange(0, Smax + Delta_S, Delta_S)
        for j in range(M+1):
            V[N, j] = np.max((K-S[j], 0))
        V[:, 0] = 50
        V[:, -1] = 0
        a = np.multiply((0.5 * r * Delta_t), t) - 0.5 * \
            (sigma ** 2) * (t ** 2) * Delta_t
        b = 1 + (sigma ** 2) * (t ** 2) * Delta_t + r * Delta_t
        c = -0.5 * r * t * Delta_t - 0.5 * (sigma ** 2) * (t ** 2) * Delta_t
        A = np.zeros((M-1, M-1))
        for j in range(M-1):
            for i in range(M-1):
                if i == j:
                    A[j, i] = b[j]
                if i == j+1:
                    A[j, i] = c[j]
                if i == j-1:
                    A[j, i] = a[j]


K = 200
r = 0.03
sigma = 0.58
T = 5
Smax = 150
M = 30
N = 20
