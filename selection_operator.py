import random

debug_list_distance = []


def crowding_comp(guy1, guy2):
    # "crowding_"distance"
    # debug_list_distance.append(guy1["crowding_distance"])
    # debug_list_distance.append(guy2["crowding_distance"])
    # "crowding_"distance"
    # if guy1['distance'] > guy2['distance']:
    if guy1['crowding_distance'] > guy2['crowding_distance']:
        result = 1
    else:
        result = 0

    return result


# - - - - - - - - - - STEST
def crowding_comp_v2(guy1, guy2):
    if guy1['distance'] > guy2['distance']:
        return 1
    else:
        return 0


def select_operator(pop):
    popsize = 50
    pool = [0] * popsize
    randnum = [random.randint(0, popsize - 1) for _ in range(2 * popsize)]
    j = 0
    for i in range(0, 2 * popsize, 2):
        p1 = randnum[i]
        p2 = randnum[i + 1]
        result = crowding_comp_v2(pop[p1], pop[p2])
        pool[j] = p1 if result == 1 else p2
        j += 1
    newpop = [pop[i] for i in pool]

    return newpop


def select_op(pop, options):
    pop_size = options["pop_size"]
    pool = [0] * pop_size
    debug_list_select = []
    randnum = [random.randint(1, (pop_size - 1)) for _ in range(2 * pop_size)]

    j = 0
    for i in range(0, 2 * pop_size, 2):
        p1 = randnum[i]  # randnum[i]  # TODO: delete after debugging
        p2 = randnum[i + 1]  # randnum[i + 1]  # TODO: delete after debugging
        result = crowding_comp(pop[p1], pop[p2])
        if result == 1:
            pool[j] = p1
            # debug_list_select.append(result)
        else:
            pool[j] = p2
            # debug_list_select.append(result)
        j += 1
    new_pop = [pop[i] for i in pool]

    # print(f"SELECT: {debug_list_select}")
    # print(f"distances: {debug_list_distance}")

    return new_pop
