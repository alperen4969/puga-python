import numpy as np
import punish_twins
import pui_nbhd


def nd_sort(pop, options):
    """
    Find pareto fronts and front ranks of solutions (deterministic).
    """
    N = options["pop_size"]  # len(pop)
    num_viol = np.vstack([individual["num_viol"] for individual in pop])
    viol_sum = np.vstack([individual["viol_sum"] for individual in pop])
    mean = np.vstack([individual["mean"] for individual in pop])
    dom_mat = calc_domination_matrix_v2(num_viol, viol_sum, mean)

    # Initialize variables
    for individual in pop:
        individual["rank"] = 0
        individual["twins"] = 0
        individual["distance"] = 0
        individual["PUI"] = 0
        individual["pref"] = 0
        individual["npSet"] = []
        individual["sp"] = []

    # Compute np and sp of each individual
    ind = [{'np': 0, 'sp': []} for _ in range(N)]
    for p in range(N - 1):
        for q in range(p + 1, N):
            if dom_mat[p, q] == 1:
                ind[q]['np'] += 1
                ind[p]['sp'].append(q)
                # pop[q]["np_sets"].append(p)
                np.append(pop[q]["np_sets"], p)
            elif dom_mat[p, q] == -1:
                ind[p]['np'] += 1
                ind[q]['sp'].append(p)
                # pop[q]["np_sets"].append(p)
                np.append(pop[q]["np_sets"], p)
                # pop[p].npSet.append(q)

    # Find pareto fronts
    front = [{'f': []} for _ in range(N)]
    front[0]["f"] = []
    for i in range(N):
        if ind[i]['np'] == 0:
            pop[i]["rank"] = 1
            front[1]["f"].append(i)
            # front['f'][0] = i

    # Calculate pareto rank of each individual
    fid = 1
    while front[fid]['f']:
        Q = []
        for p in front[fid]['f']:
            for q in ind[p]['sp']:
                ind[q]['np'] = ind[q]['np'] - 1
                if ind[q]['np'] == 0:
                    pop[i]["rank"] = fid + 1
                    Q.append(q)

        front[fid]['f'] = Q
        fid += 1

    rank = [pop[m]["rank"] for m in range(len(pop))]
    rank = np.vstack(rank)
    max_rank = np.max(rank, 0).astype(int)
    r = max_rank + 1
    zero_ranked = 0
    zero_ranked = np.where(rank == 0)[0]  # matlab find
    zr = len(zero_ranked)
    if zero_ranked.any():
        for i in range(1, zr):
            pop[zero_ranked[i]].rank = r
        front[fid]["f"] = [zero_ranked]
        # print("buraya uğramaması lazım!")

    pop = punish_twins.punishtwins(pop, options)

    for i in range(N):
        pop[i]["np"] = ind[i]["np"]
        pop[i]["sp"] = ind[i]["sp"]
        ind[i]["np"] = 0
        ind[i]["sp"] = []

    pop = pui_nbhd.pui_nbhd(pop, options)
    pop = calc_crowding_distance(pop)

    return pop


def calc_domination_matrix(n_viol, viol_sum, mean):
    N = mean.shape[0]
    num_obj = mean.shape[1]
    dom_mat = np.zeros((N, N))
    for p in range(N - 1):
        for q in range(p + 1, N):
            if n_viol[p] == 0 and n_viol[q] == 0:
                pdomq = all(mean[p] < mean[q])
                qdomp = all(mean[p] > mean[q])
                if pdomq and not qdomp:
                    dom_mat[p, q] = 1
                elif not pdomq and qdomp:
                    dom_mat[p, q] = -1
            elif n_viol[p] == 0:
                dom_mat[p, q] = 1
            elif n_viol[q] == 0:
                dom_mat[p, q] = -1
            else:
                if viol_sum[p] < viol_sum[q]:
                    dom_mat[p, q] = 1
                elif viol_sum[p] > viol_sum[q]:
                    dom_mat[p, q] = -1

    return dom_mat - np.transpose(dom_mat)


def calc_domination_matrix_v2(n_viol, viol_sum, mean):
    N = mean.shape[0]
    num_obj = mean.shape[1]
    dom_mat = np.zeros((N, N))

    for p in range(N - 1):
        for q in range(p + 1, N):
            # Case 1: both feasible
            if n_viol[p] == 0 and n_viol[q] == 0:
                pdomq = False
                qdomp = False
                for i in range(num_obj):
                    if mean[p, i] < mean[q, i]:
                        pdomq = True
                    elif mean[p, i] > mean[q, i]:
                        qdomp = True

                if pdomq and not qdomp:
                    dom_mat[p, q] = 1
                elif not pdomq and qdomp:
                    dom_mat[p, q] = -1

            # Case 2: p is feasible, and q is infeasible
            elif n_viol[p] == 0 and n_viol[q] != 0:
                dom_mat[p, q] = 1

            # Case 3: q is feasible, and p is infeasible
            elif n_viol[p] != 0 and n_viol[q] == 0:
                dom_mat[p, q] = -1

            # Case 4: both infeasible
            else:
                if viol_sum[p] < viol_sum[q]:
                    dom_mat[p, q] = 1
                elif viol_sum[p] > viol_sum[q]:
                    dom_mat[p, q] = -1

    dom_mat = dom_mat - dom_mat.T
    return dom_mat


def calc_crowding_distance(pop):
    num_obj = 2
    col_idx = num_obj
    num_ind = len(pop)
    idx = np.arange(num_ind)
    mean = np.array([ind["mean"] for ind in pop])
    test = np.vstack(idx)
    mean = np.hstack((mean, test))
    # mean = np.concatenate((mean, idx[:, np.newaxis]), axis=1)

    # for m in range(num_obj):
    #     mean = np.sort(mean, axis=0)
    #     mean[:, num_obj] = np.append(np.zeros(1), np.diff(mean[:, m]))
    #     mean[:, num_obj] = mean[:, num_obj] / (mean[num_ind - 1, m] - mean[0, m])
    #     mean[:, num_obj] = np.append(mean[:, num_obj], np.zeros(1))
    #
    # for ind in pop:
    #     ind["distance"] = mean[ind.id, num_obj]

    for m in range(num_obj):
        first = mean[0, col_idx]
        last = mean[num_ind - 1, col_idx]
        mean = np.sort(mean, axis=0)

        pop[first.astype(int)]["distance"] = 0  # float('inf')
        pop[last.astype(int)]["distance"] = 0

        min_obj = mean[0, m]
        max_obj = mean[num_ind - 1, m]

        for i in range(2, num_ind - 1):
            idx = mean[i, col_idx]
            A = mean[i + 1, m]
            B = mean[i - 1, m]
            C = max_obj - min_obj

            pop[idx.astype(int)]["distance"] = (pop[idx.astype(int)]["distance"] + (mean[i + 1, m] - mean[i - 1, m])
                                               / (max_obj - min_obj))

    return pop


def calc_crowding_distance_v2(pop):
    numObj = len(pop[0].obj)
    for fid in range(len(front)):
        idx = front[fid]
        frontPop = [pop[i] for i in idx]

        numInd = len(idx)

        obj = [p.obj for p in frontPop]
        obj_with_idx = [(obj[i], idx[i]) for i in range(numInd)]
        for m in range(numObj):
            obj_with_idx.sort(key=lambda x: x[0][m])

            pop[obj_with_idx[0][1]].distance = float('inf')
            pop[obj_with_idx[numInd - 1][1]].distance = float('inf')

            minobj = obj_with_idx[0][0][m]
            maxobj = obj_with_idx[numInd - 1][0][m]

            for i in range(1, numInd - 1):
                idx = obj_with_idx[i][1]
                pop[idx].distance += (obj_with_idx[i + 1][0][m] - obj_with_idx[i - 1][0][m]) / (maxobj - minobj)

    return pop