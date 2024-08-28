import numpy as np
import objective
import EPS_CONS_objfun

def MOP4(x):
    a = 0.8
    b = 3
    z1 = np.sum(-10 * np.exp(-0.2 * np.sqrt(x[:-1]**2 + x[1:]**2)))
    z2 = np.sum(np.abs(x)**a + 5 * np.sin(x)**b)
    return np.array([z1, z2])


def evaluate(pop, import_data, opt):  # TODO: state, options
    fordebug = np.array(pop)
    results_pops = [eval_individual(indiv, import_data, opt) for indiv in pop]
    results_pops = np.array(results_pops)

    return results_pops


def eval_individual(individual, import_data, opt):
    # individual, cons = objective.gep_objective_function(individual, import_data)
    individual["mean"], individual["variance"], individual["cons"] = EPS_CONS_objfun.EPS_CONS_objfun(individual["var"], opt)
    # individual["mean"] = MOP4(individual)

    if individual["cons"] is not None:  # TODO, PUI YOKKEN BURASI AKTİF
        idx = [i for i, con in enumerate(individual["cons"]) if con]
        if idx:
            individual['n_viol'] = len(idx)
            individual['viol_sum'] = sum(abs(con) for con in individual["cons"])
            individual['feasibility'] = 0
        else:
            individual['n_viol'] = 0
            individual['viol_sum'] = 0
            individual['feasibility'] = 1
    return individual



