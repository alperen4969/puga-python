import numpy as np
import puga
import import_excel
import pandas as pd
import debug
import population
import scipy.io as sio
from scipy.stats import multivariate_normal, norm, gamma, weibull_min, uniform

pop_size = None

data = import_excel.import_excel()

options = {
    'K': [10, 10],
    # 'initial_population': population.init_population(pop_size),
    'max_gen': 25,
    'pop_size': 28,
    'num_obj': 2,
    'num_cons': 10,
    # 'obj_fun': objective.gep_objective_function(data),
    'crossover_fraction': 0.005,  # düşük olmalı, olasılık fazla
    'mutation_fraction': 0.35,  # yüksek olmalı, olasılık fazla
    'ro': [0, 0],
    'repair': 0,
    'in_repair': 0,
    'rep_gen': 5,
    'pre_fon': 1,
    'var_type': np.zeros(352),
    "num_obj": 2,
    "num_cons": 4,
    "num_var": 352,
    "repair": 0,
    "repgen": 5,
    "coeff_mean_1": 0,
    "coeff_mean_2": 0,
    "coeff_var_1": 0,
    "coeff_var_2": 0,
    "less": 0,
    "greater": 0,
    "less_var": 0,
    "greater_var": 0,
    "ro": [0, 0],
    "alpha": 0,
    "vartype": 2,
    "refWeight": 0,
    "refPoints": 0,
    "nbhds": [],
    "poolsize": 0,
    "refEpsilon": 0,
    "inrepair": 0,
    "lb": data["lb"],  # np.array([0] * 352),
    "ub": data["ub"]  # np.array(pd.read_excel('upperbound.xlsx').values.tolist()[0])
}

file = 'epsilon_output.xlsx'
n = 1000  # Discrete points
Rho = pd.read_excel('input.xls', sheet_name='Coef_matrix', usecols='C:J', skiprows=1, nrows=8).values
Z = multivariate_normal.rvs(mean=np.zeros(8), cov=Rho, size=n)
U = norm.cdf(Z, 0, 1)
X = np.column_stack([
    gamma.ppf(U[:, 0], 51, scale=0.6),
    gamma.ppf(U[:, 1], 6.275, scale=0.23),
    norm.ppf(U[:, 2], 75, 3.6),
    weibull_min.ppf(U[:, 3], 0.83, scale=2.5),
    uniform.ppf(U[:, 4], 55, 65),
    uniform.ppf(U[:, 5], 0.3, 0.6),
    norm.ppf(U[:, 6], 90, 0.6),
    np.zeros(n)
])

# test_c = sio.loadmat("export_c.mat")["temp_c"][0, 0]
test_c = sio.loadmat("temp_c_eps.mat")["temp_c"]
# test_e = sio.loadmat("export_e.mat")["temp_e"][0, 0]
test_e = sio.loadmat("temp_e_eps.mat")["temp_e"]
options["C"] = test_c  # X[:, [0, 2, 4, 6]]
options["E"] = test_e  # X[:, [0, 2, 4, 6]]
# options["E"] = # X[:, [1, 3, 5, 7]]

investmentidx = np.arange(176, 352)
num_nbhds = np.prod(options['K'])
pop = puga.puga_run(data, options)
# results = debug.evaluate_debug(pop, data)
# best_result = min(results)
# print(f"\nBest result: {best_result}")



