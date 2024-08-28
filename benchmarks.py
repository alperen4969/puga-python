import deap.benchmarks as test_problems
import numpy as np
import random as random
import pymoo.gradient.toolbox as anp
from math import sin, cos, pi, exp, e, sqrt


def zdt1(individual):
    g = 1.0 + 9.0 * sum(individual[1:]) / (len(individual) - 1)
    f1 = individual[0]
    if not f1 >= 0:
        f1 = 0
    # print(f"{f1} ve {g}")
    f2 = g * (1 - sqrt(f1 / g))
    return f1, f2


def zdt2(x):
    n_var = 30
    f1 = x
    c = anp.sum(x[1:])
    g = 1.0 + 9.0 * c / (n_var - 1)
    f2 = g * (1 - anp.power((f1 * 1.0 / g), 2))

    # test = anp.column_stack([f1, f2])
    return f1.astype(float).tolist(), f2.astype(float).tolist()


def MOP4(x):
    a = 0.8
    b = 3
    z1 = np.sum(-10 * np.exp(-0.2 * np.sqrt(x[:-1] ** 2 + x[1:] ** 2)))
    z2 = np.sum(np.abs(x) ** a + 5 * np.sin(x) ** b)
    # return np.array([z1, z2])
    return z1.astype(float), z2.astype(float)


def evaluate_benchmarks(pop):
    results_pops = [eval_indv_bench(indiv) for indiv in pop]
    results_pops = np.array(results_pops)
    return results_pops


def eval_indv_bench(individual):
    individual = benchmark_problem(individual)
    return individual


def benchmark_problem(individual):
    """ZDT4 multiobjective function.
    g = 1 + 10 * (len(individual) - 1) + sum(xi ** 2 - 10 * cos(4 * pi * xi) for xi in individual[1:])
    f1 = individual[0]
    f2 = g * (1 - sqrt(f1 / g))
    """
    # individual["mean"] = test_problems.zdt4(individual["variables"][0])
    individual["mean"] = zdt1(individual["variables"][0])
    # individual["mean"] = test_problems.zdt2(individual["variables"][0])
    # individual["mean"] = MOP4(individual["variables"][0])
    # individual["mean"] = zdt2(individual["variables"][0])
    return individual


def pop_benchmark(pop_size, num_var):
    # indv = {}
    population = []
    # population = [indv] * pop_size
    for i in range(pop_size):
        # lower_bound = np.array([-10] * num_var).astype(int)
        lower_bound = np.array([0] * num_var).astype(float)
        # lower_bound[0] = 0  # ZDT4 için sanırım
        # upper_bound = np.array([10] * num_var).astype(int)  #  -10, 10 alt üst, ZDT4 için
        upper_bound = np.array([1] * num_var).astype(float)
        # upper_bound[0] = 1
        variables = np.array([lower_bound[j] + random.random() *
                              (upper_bound[j] - lower_bound[j]) for j in range(num_var)])
        variables[8] = 0  # bu baya test
        # variables[15:30] = 1
        variables = variables.reshape(1, num_var)
        population.append({
            "variables": variables,
            "mean": np.zeros((0, 2)),
            "rank": None,
            "crowding_distance": None
        })
    population = np.array(population)
    return population
