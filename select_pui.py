import numpy as np


def select_op(pop):
    popsize = len(pop)
    pool = [None] * popsize  # pool: the individual index selected
    randnum = np.random.randint(0, popsize, 2 * popsize)
    j = 0
    for i in range(0, 2 * popsize, 2):
        p1 = randnum[i]
        p2 = randnum[i + 1]
        result = crowding_comp(pop[p1], pop[p2])
        if result == 1:
            pool[j] = p1
        else:
            pool[j] = p2
        j += 1

    newpop = [pop[i] for i in pool]
    return newpop


def crowding_comp(guy1, guy2):
    if guy1['PUI'] < guy2['PUI']:
        return 1
    else:
        return 0