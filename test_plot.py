import numpy as np
from scipy import integrate
import matplotlib.pyplot as plt
from scipy.integrate import quad
from scipy.special import erf

# def f(x):
#     return x * np.sin(1 / x)
#
# pul_equ = (np.exp(ro[k] * ((x - mu1) / mu1)) * 0.5 * (1 + np.erf((x - mu2) / (sigma2 * np.sqrt(2)))) *
#                 (1 / (sigma1 * np.sqrt(2 * np.pi))) * np.exp(- ((x - mu1) ** 2) / (2 * sigma1 ** 2)))
#
# X = np.arange(-0.5, 0.5, 0.001)
# DF = [integrate.quad(f, a, b)[0] for a, b in zip(X[:-1], X[1:])]
# F = np.cumsum(DF)
# plt.plot(X[1:], F)
# plt.show()

ro = [0.1, 0.2, 0.3]  # Example ro array
mu1 = 1.0
mu2 = 2.0
sigma1 = 1.0
sigma2 = 2.0


def integrand(x, k):
    term1 = np.exp(ro[k] * ((x - mu1) / mu1))
    term2 = 0.5 * (1 + erf((x - mu2) / (sigma2 * np.sqrt(2))))
    term3 = 1 / (sigma1 * np.sqrt(2 * np.pi))
    term4 = np.exp(- ((x - mu1) ** 2) / (2 * sigma1 ** 2))
    return term1 * term2 * term3 * term4

a = -np.inf  # Lower limit
b = np.inf   # Upper limit

# Evaluate the integral for each k
for k in range(len(ro)):
    result, error = quad(integrand, a, b, args=(k,))
    print(f"Integral result for k={k}: {result}")
    print(f"Estimated error for k={k}: {error}")

