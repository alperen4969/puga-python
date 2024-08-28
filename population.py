import numpy as np
import pandas as pd
import scipy.io as sio
import random
import reshape_variables
import copy
import var_limit as var_limit


# varstr'ler reshape_variables.py çıktısı, array


def init_population(pop_size):
    num_var = 352
    pop_size = pop_size
    indv = {}
    population = [indv] * pop_size
    population = np.array(population)

    for i in range(pop_size):
        lower_bound = np.array([0] * 352)
        upper_bound = np.array(pd.read_excel('upperbound.xlsx').values.tolist()[0])
        variables = np.array([lower_bound[j] + random.random() *
                              (upper_bound[j] - lower_bound[j]) for j in range(len(lower_bound))])
        # variables = np.array(lower_bound[j] + random.random() *
        #                       (upper_bound[j] - lower_bound[j]) for j in range(len(lower_bound)))

        for v in range(num_var):
            variables[v] = round(variables[v])

        # limit in the lower and upper bound
        # variables = var_limit(variables, lower_bound, upper_bound)

        population[i]['variables'] = copy.deepcopy(variables)
        # variables_str struct yapısında matlabte
        population[i]['variables_str'] = reshape_variables.reshape_variables_v2(variables)  # v2 :/
        population[i]['mean'] = np.zeros((2, 1))
        population[i]['variance'] = np.zeros((2, 1))
        population[i]['cons'] = np.zeros((4, 1))
        population[i]['infr'] = np.zeros(1)
        population[i]['capacity'] = 0
        population[i]['rank'] = 0
        population[i]['pui'] = 0
        population[i]['pref'] = 0
        population[i]['feasibility'] = 0
        # population[i]['distance'] = 0
        population[i]['np_sets'] = np.empty((1, 1))
        population[i]['sp'] = np.empty((1, 1))
        population[i]['np'] = 0
        population[i]['nbhd'] = 0
        population[i]['twins'] = 0
        population[i]['num_viol'] = 0
        population[i]['parent'] = 0
        population[i]['viol_sum'] = 0

    return population


def pop_from_exist(pop_size, imported_data, opt):
    exist_pop = sio.loadmat("initial_py.mat")
    exist_pop = sio.loadmat("existed_pop.mat")  # 38 bu?
    existed_array_debug = exist_pop["temp_var"]
    # exist_pop_size = exist_pop["temp_var"].size
    exist_pop = sio.loadmat("28_existed.mat")  # 28 bu?  initials.mat : aynı 28'lik ama baş belası veri yapısı
    exist_pop_size = exist_pop["temp_var"].size  # : initialpops
    existed_array = exist_pop["temp_var"]
    num_var = 352
    # population = [{}]  # * exist_pop_size
    population = []

    for i in range(0, exist_pop_size):
        # i = 33  # for debug
        variables = existed_array[0, i]  # TEST TODO [0]
        # svariables = np.array(exist_pop["temp_var"][0,i])
        new_copy_variables = copy.deepcopy(variables)

        population.append({
            'variables': new_copy_variables,
            'variables_str': reshape_variables.reshape_variables(new_copy_variables),
            'mean': np.zeros((2, 1)),
            'variance': np.zeros((2, 1)),
            'cons': np.zeros((4, 1)),
            'infr': np.zeros(1),
            'capacity': 0,
            'rank': 0,
            'pui': 0,
            'pref': 0,
            'feasibility': 0,
            'distance': 0,
            'np_sets': np.empty((1, 1)),
            'sp': np.empty((1, 1)),
            'np': 0,
            'nbhd': 0,
            'twins': 0,
            'n_viol': 0,
            'parent': 0,
            'viol_sum': 0,
        })
    lack_pop_size = pop_size - 28  # 38
    if lack_pop_size > 0:
        lack_pop = new_init(lack_pop_size, imported_data, opt)
        # population.append(init_population(lack_pop_size))
        population = np.array(population)
        whole_pop = np.concatenate((population, lack_pop))
    else:
        whole_pop = population
    return whole_pop


# pop_from_exist()
# test = init_population(10)

def new_init(pop_size, imported_data, opt):
    num_var = 352
    pop_size = pop_size
    indv = {}
    population = []
    # population = [indv] * pop_size

    for i in range(pop_size):
        lower_bound = np.array([0] * 352)  # .astype(int)
        upper_bound = np.array(pd.read_excel('upperbound.xlsx').values.tolist()[0]).astype(int)
        # variables = np.array([lower_bound[j] + random.random() *
        #                       (upper_bound[j] - lower_bound[j]) for j in range(len(lower_bound))])
        ub_excel = (imported_data["capfactor"] * 8760)
        ub_excel = ub_excel.reshape(176)
        ub_inv = np.array(opt["ub"][176:352])
        ub_inv = ub_inv.reshape(176)
        ubx = np.hstack((ub_excel, ub_inv))  # axis = 1 for horizantally
        # variables = np.array([lower_bound[j] + random.random() *
        #                       (ubx[j] - lower_bound[j]) for j in range(len(lower_bound))])
        variables = np.array([lower_bound[j] + random.random() *
                              (ubx[j] - lower_bound[j]) for j in range(len(lower_bound))])
        # variables = np.array(lower_bound[j] + random.random() *
        #                       (upper_bound[j] - lower_bound[j]) for j in range(len(lower_bound)))

        # for v in range(num_var):
        #     variables[v] = round(variables[v])

        variables = variables.reshape(1, 352)
        variables = var_limit.var_limit(variables, imported_data['lb'], imported_data['ub'])

        # for i in range(0, pop_size):
        population.append({
            'variables': variables,
            'variables_str': reshape_variables.reshape_variables(variables),
            'mean': np.zeros((2, 1)),
            'variance': np.zeros((2, 1)),
            'cons': np.zeros((4, 1)),
            'infr': np.zeros(1),
            'capacity': 0,
            'rank': 0,
            'pui': 0,
            'pref': 0,
            'feasibility': 0,
            'distance': 0,
            'np_sets': np.empty((1, 1)),
            'sp': np.empty((1, 1)),
            'np': 0,
            'nbhd': 0,
            'twins': 0,
            'n_viol': 0,
            'parent': 0,
            'viol_sum': 0,
        })

    population = np.array(population)
    return population


def init_eps(pop_size, imported_data, opt):
    # old_pop = np.array(pd.read_excel("epsilon_output.xls").values)
    # old_pop = sio.loadmat("export_vars.mat")["temp_var"][0]
    old_pop = sio.loadmat("temp_28_eps.mat")["temp_28"][0]
    population = []
    for i in range(0, len(old_pop)):
        population.append({
            # 'var': old_pop[i],
            'var': np.array(old_pop[i], dtype=np.float32),
            'mean': np.zeros((2, 1)),
            'variance': np.zeros((1000, 1)),
            'cons': np.zeros((10, 1)),
            'infr': np.zeros(1),
            'rank': 0,
            'pref': 0,
            'feasibility': 0,
            'distance': 0,
            'np_sets': np.empty((1, 1)),
            'sp': np.empty((1, 1)),
            'np': 0,
            'nbhd': 0,
            'twins': 0,
            'n_viol': 0,
            'parent': 0,
            'viol_sum': 0,
        })

        # lack_pop_size = pop_size - 28
        # if lack_pop_size > 0:
        #     temp_arr = np.ones(lack_pop_size)
        #     lack_pop = init_gep(temp_arr, imported_data)
        #     population = np.array(population)
        #     whole_pop = np.concatenate((population, lack_pop))
        # else:
        #     whole_pop = population

    return population



