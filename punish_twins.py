import copy

import numpy as np


def punishtwins(pop, opt):
    pop_size = len(pop)
    rank = [individual['rank'] for individual in pop]
    maxrank = max(rank)
    twins = []
    # i = 0
    # pop[i]['rank'] = 10
    pop = copy.deepcopy(pop)
    for i in range(pop_size - 1):
        for j in range(i + 1, pop_size):
            if pop[i]['mean'][0] == pop[j]['mean'][0] and pop[i]['mean'][1] == pop[j]['mean'][1]:
                pop[j]['rank'] = maxrank + 1
                twins.append(j)
                twins.append(i)
                pop[j]['twins'] = 1
    return pop


def punish_twins_pui(pop, opt):
    rank = [individual['rank'] for individual in pop]
    max_rank = max(rank)
    twins = []
    N = len(pop)

    for i in range(N - 1):
        if i not in twins:
            for j in range(i + 1, N):
                if j not in twins:
                    division = abs(1 - (pop[i]['var'] / pop[j]['var']))
                    division = np.nan_to_num(division)
                    viol = division < 0.01
                    test = np.sum(viol)
                    if np.sum(viol) > 12 * 0.9:
                        pop[j]['rank'] = max_rank + 1
                        pop[j]['twins'] = 1
                        twins.append(j)

    return pop
