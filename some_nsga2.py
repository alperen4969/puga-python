import numpy as np
import stochastic_sort as stoc_sort
from calc_prcr import calc_pr_cr_v2, calc_pr_discrete


def dominates(p, q):
    p["mean"] = np.array(p["mean"])
    q["mean"] = np.array(q["mean"])
    # pr1, pr2 = 1, 1
    pr1, pr2 = prcr(p, q)
    if p["n_viol"] == 0 and q["n_viol"] == 0:
        log = all(p["mean"] <= q["mean"]) and any(p["mean"] < q["mean"])
        if log:
            if pr1 > pr2:
                return 1
            else:
                return 0
        else:
            return 0
    else:
        return 0
    # test = all(p["mean"] <= q["mean"])
    # np.all(x <= y) and np.any(x < y)
    return all(p["mean"] <= q["mean"]) and any(p["mean"] < q["mean"])


def non_dominated_sorting(pop):
    pop_size = len(pop)
    domination_set = [[] for _ in range(pop_size)]
    dominated_count = [0 for _ in range(pop_size)]
    F = [[]]
    for i in range(pop_size):
        for j in range(i + 1, pop_size):
            # if i dominates j
            if dominates(pop[i], pop[j]):
                domination_set[i].append(j)
                dominated_count[j] += 1
            # if j dominates i
            elif dominates(pop[j], pop[i]):
                domination_set[j].append(i)
                dominated_count[i] += 1
        # If i is not dominated at all
        if dominated_count[i] == 0:
            pop[i]['rank'] = 0
            F[0].append(i)
    # Pareto Counter
    k = 0
    while True:
        Q = []
        for i in F[k]:
            for j in domination_set[i]:
                dominated_count[j] -= 1
                if dominated_count[j] == 0:
                    pop[j]['rank'] = k + 1
                    Q.append(j)
        if not Q:
            break
        F.append(Q)
        k += 1
    return pop, F


def calc_crowding_distance(pop, F):
    pareto_count = len(F)
    n_obj = len(pop[0]["mean"])
    for k in range(pareto_count):
        costs = np.array([pop[i]["mean"] for i in F[k]])
        n = len(F[k])
        d = np.zeros((n, n_obj))
        for j in range(n_obj):
            idx = np.argsort(costs[:, j])
            d[idx[0], j] = np.inf
            d[idx[-1], j] = np.inf
            for i in range(1, n - 1):
                d[idx[i], j] = costs[idx[i + 1], j] - costs[idx[i - 1], j]
                d[idx[i], j] /= costs[idx[-1], j] - costs[idx[0], j]
        for i in range(n):
            temp = sum(d[i, :])
            pop[F[k][i]]['crowding_distance'] = sum(d[i, :])
    # pop = punish_twins(pop)
    return pop


def sort_population(pop):
    pop = sorted(pop, key=lambda x: (x['rank'], -x['crowding_distance']))
    max_rank = pop[-1]['rank']
    F = []
    for r in range(max_rank + 1):
        F.append([i for i in range(len(pop)) if pop[i]['rank'] == r])
    return pop, F


def truncate_population(pop, F):
    pop_size = int(len(pop) / 2)
    if len(pop) <= pop_size:
        return pop, F
    pop = pop[:pop_size]
    for k in range(len(F)):
        F[k] = [i for i in F[k] if i < pop_size]
    return pop, F


def prcr(p, q):
    ro = [0, 0]
    num_obj = 2
    guy1 = p
    guy2 = q
    pr1 = 1
    pr2 = 1
    for m in range(num_obj):
        # pr1 = pr1 * calc_pr_cr_v2(guy2, guy1, m, ro)
        pr1 = pr1 * calc_pr_discrete(guy2, guy1, m, ro)
        pr2 = pr2 * calc_pr_discrete(guy1, guy2, m, ro)
    return pr1, pr2


def punish_twins(pop):
    pop_size = len(pop)
    rank = [pop[m]["rank"] for m in range(len(pop))]
    rank = np.vstack(rank)
    max_rank = np.max(rank, 0)[0].astype(int)

    for i in range(0, pop_size - 1):
        for j in range(i + 1, pop_size):
            if (pop[i]["mean"][0] == pop[j]["mean"][0] and
                    pop[i]["mean"][1] == pop[j]["mean"][1]):
                pop[j]["rank"] = max_rank + 1
                pop[j]["twins"] = 1
    return pop
