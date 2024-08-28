import copy
import random
import reshape_variables
import repair_operator
import numpy as np
import var_limit
import scipy.io as sio


def mutation_operator(pop, opt, state):
    fraction = opt['mutation_fraction']
    rand = random.random()
    for i in range(len(pop)):
        if rand < fraction:
            child = mutation_gaussian(pop[i], opt, state, fraction)
            pop[i] = child

    return pop


def mutation_operator_pui(pop, opt, state):
    ub_eps = sio.loadmat("ub_eps.mat")["ub"]
    lb_eps = sio.loadmat("lb_eps.mat")["lb"]
    fraction = opt['mutation_fraction']
    rand = random.random()
    for i in range(len(pop)):
        if rand < fraction:
            child = mutation_gaussian_pui(pop[i], opt, state, fraction)
            child['var'] = var_limit.var_limit_pui(child['var'], lb_eps, ub_eps)
            pop[i] = child

    return pop


def mutation_gaussian(parent, opt, state, fraction):
    scale = 0.10
    shrink = 0.50
    scale = scale - shrink * scale * state['currentGen'] / opt['max_gen']
    lb = opt['lb']
    ub = opt['ub']
    num_var = len(parent["variables"][0])
    scale = scale * (ub - lb)
    child = copy.deepcopy(parent)
    # len(child['variables'])>

    for i in range(num_var):
        # rand = random.random()
        # if rand < fraction:
        # -5, +5 - 0.0, 2.0
        randrng = random.uniform(0, 1)
        child['variables'][0, i] = parent['variables'][0, i] + scale[i] * randrng

    # child['variables'] = child['variables'].round()

    # if any(opt['vartype'] == 2):
    # change = [i for i, v in enumerate(opt['vartype']) if v == 2]
    # child['variables'][change] = [round(val) for val in child['variables'][change]]

    child['variables'] = var_limit(child['variables'], opt['lb'], opt['ub'])
    child['varstr'] = reshape_variables.reshape_variables(child['variables'])

    # if opt['inrepair'] == 1:
    #     child = repair_operator.repair_operator(child, opt)

    return child


def mutation_gaussian_pui(parent, opt, state, fraction):
    scale = 0.10
    shrink = 0.50
    scale = scale - shrink * scale * state['currentGen'] / opt['max_gen']
    ub_eps = sio.loadmat("ub_eps.mat")["ub"]
    lb_eps = sio.loadmat("lb_eps.mat")["lb"]
    num_var = 12
    scale = scale * (ub_eps - lb_eps)
    child = copy.deepcopy(parent)

    for i in range(num_var):
        randrng = random.uniform(0, 1)
        # child['var'][i] = parent['var'][i] + scale[i] * randrng
        child['var'] = parent['var'] + scale * randrng
    # child['var'] = var_limit.var_limit_pui(child['var'], opt['lb'], opt['ub'])

    return child
