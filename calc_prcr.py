import scipy.integrate as integral
from math import sqrt, exp, erf, pi, erf
import numpy as np
import cubature
from numba import jit
import matplotlib.pyplot as plt
import scipy.io as sio


def calc_prcr(guy1, guy2, k, ro):
    mu1 = sum(guy1[i][k] for i in range(len(guy1))) / len(guy1)
    mu2 = sum(guy2[i][k] for i in range(len(guy2))) / len(guy2)

    sigma1 = sqrt(sum((guy1[i][k] - mu1) ** 2 for i in range(len(guy1))) / len(guy1))
    sigma2 = sqrt(sum((guy2[i][k] - mu2) ** 2 for i in range(len(guy2))) / len(guy2))

    mumax = max(mu1, mu2)
    sigmax = max(sigma1, sigma2)
    tmax = mumax + 5 * sigmax

    def func(x):
        return exp(ro[k] * ((x - mu1) / mu1)) * \
            0.5 * (1 + erf((x - mu2) / (sigma2 * sqrt(2)))) * \
            (1 / (sigma1 * sqrt(2 * pi))) * exp(-((x - mu1) ** 2) / (2 * sigma1 ** 2))

    # result = integral.quad(func, -tmax, tmax)
    result = integral.quad(func, -tmax, tmax)
    # x = sp.symbols('x')
    # result = sp.integrate(func, (x, 0, 1))

    return result


def calc_pr_cr_v2(guy1, guy2, k, ro):  # this was using
    guy1_mat = sio.loadmat("guy1_for_pui.mat")["guy1"][0, 0]
    guy2_mat = sio.loadmat("guy2_for_pui.mat")["guy2"][0, 0]
    ro_mat = sio.loadmat("ro_for_pui.mat")["ro"]
    k_mat = sio.loadmat("k_for_pui.mat")["k"]

    mu1 = guy1["mean"][k]
    mu1 = 3.877838952509796e+11
    mu2 = guy2["mean"][k]
    mu2 = 3.873567928013255e+11
    sigma1 = sqrt(guy1["variance"][k])
    sigma1 = 2.753642907896908e+09
    sigma2 = sqrt(guy2["variance"][k])
    sigma2 = 2.762280109418097e+09
    mumax = np.maximum(mu1, mu2)
    sigmax = np.maximum(sigma1, sigma2)
    tmax = mumax + 5 * sigmax

    # tmax = tmax.astype(int)
    # def sadelesmis_denklem(x):
    #     exp_term = np.exp(ro[k] * ((x - mu1) / mu1))
    #     erf_term = 0.5 * (1 + erf((x - mu2) / (sigma2 * np.sqrt(2))))
    #     gaussian_term = np.exp(- ((x - mu1) ** 2) / (2 * sigma1 ** 2)) / (sigma1 * np.sqrt(2 * np.pi))
    #     return exp_term * erf_term * gaussian_term
    #
    # @jit
    # def integrand(x):
    #     return (np.exp(ro[k] * ((x - mu1) / mu1)) * 0.5 * (1 + erf((x - mu2) / (sigma2 * np.sqrt(2)))) *
    #             (1 / (sigma1 * np.sqrt(2 * np.pi))) * np.exp(- ((x - mu1) ** 2) / (2 * sigma1 ** 2)))

    # Pr, _ = integral.quad(gfg, -tmax, tmax)  # integrand to gfg # initialy quad, nquad is tried but failed
    # Pr, _ = integral.quad(integrand, -tmax, tmax)
    # x = sp.symbols('x')
    # Pr = sp.integrate(gfg, (x, 0, 1))

    def f(x):
        # return x * np.sin(1 / x)
        return (np.exp(ro[k] * ((x - mu1) / mu1)) * 0.5 * (1 + erf((x - mu2) / (sigma2 * np.sqrt(2)))) *
                (1 / (sigma1 * np.sqrt(2 * np.pi))) * np.exp(- ((x - mu1) ** 2) / (2 * sigma1 ** 2)))

    Pr, _ = integral.quad(f, -tmax, tmax)
    # print(Pr)
    # X = np.arange(-10, 10, 0.001)
    # DF = [integral.quad(f, a, b)[0] for a, b in zip(X[:-1], X[1:])]
    # F = np.cumsum(DF)
    # plt.plot(X[1:], F)
    # plt.show()

    return Pr


def calc_pr_cr_v3(guy1, guy2, k, ro):
    mu1 = guy1["mean"][k]
    mu2 = guy2["mean"][k]
    sigma1 = sqrt(guy1["variance"][k])
    sigma2 = sqrt(guy2["variance"][k])
    mumax = np.maximum(mu1, mu2)
    sigmax = np.maximum(sigma1, sigma2)
    tmax = mumax + 5 * sigmax

    def integrand(x):
        return (np.exp(ro[k] * ((x - mu1) / mu1)) * 0.5 * (1 + np.erf((x - mu2) / (sigma2 * np.sqrt(2)))) *
                (1 / (sigma1 * np.sqrt(2 * np.pi))) * np.exp(- ((x - mu1) ** 2) / (2 * sigma1 ** 2)))

        Pr, _ = integral.quad(integrand, -tmax, tmax)

        return Pr


def calc_pr_discrete(guy1, guy2, k, ro):
    guy1["variance"] = np.array(guy1["variance"])
    guy1["mean"] = np.array(guy1["mean"])
    n_p = len(guy1["variance"][:])
    pr = 0

    for i in range(n_p):
    # pr += np.sum(guy1["variance"][i, k] > guy2["variance"][:, k]) * \
    #       np.exp(ro[k] * ((guy1["variance"][i, k] - guy1["mean"][k]) / guy1["mean"][k]))
        debug_sum1 = np.sum(guy1["variance"][k])
        debug_big = guy1["variance"][i, k] > guy2["variance"][:, k]
        debug_exp = np.exp(ro[k] * ((guy1["variance"][i, k] - guy1["mean"][k]) / guy1["mean"][k]))
        pr += np.sum(guy1["variance"][i, k] > guy2["variance"][:, k]) * \
              np.exp(ro[k] * ((guy1["variance"][i, k] - guy1["mean"][k]) / guy1["mean"][k]))
        debug_test = 0

    pr = pr / (n_p ** 2)
    # pr = pr.astype(float)
    return pr
