import numpy as np
import punish_twins
import pui_nbhd
import nbhd
import scipy.io as sio


def ndsort_pui(pop, options):
    """
    Find pareto fronts and front ranks of solutions (deterministic).
    """
    # N = options["pop_size"]  # len(pop)
    # debug_mean = sio.loadmat("mean_56_eps.mat")["debug_mean"].T
    # debug_mean = np.vstack([m for m in debug_mean])
    debug_nviol = sio.loadmat("nviol_56_eps.mat")["debug_nviol"].T
    debug_violsum = sio.loadmat("violsum_56_eps.mat")["debug_violsum"].T
    # mean = debug_mean
    n_viol = debug_nviol
    viol_sum = debug_violsum

    n_viol = np.vstack([individual["n_viol"] for individual in pop])
    viol_sum = np.vstack([individual["viol_sum"] for individual in pop])
    mean = np.vstack([individual["mean"] for individual in pop])
    N = mean.shape[0]  # len(pop)
    dom_mat = calc_domination_matrix_v2(n_viol, viol_sum, mean)

    # Initialize variables
    for individual in pop:
        individual["rank"] = 0
        individual["twins"] = 0
        individual["distance"] = 0
        individual["PUI"] = 0
        individual["pref"] = 0
        individual["np_sets"] = []
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
                np.append(pop[p]["np_sets"], q)
                # pop[p].npSet.append(q)

    # Find pareto fronts
    front = [{'f': []} for _ in range(N)]
    front[0]["f"] = []
    for i in range(N):
        if ind[i]['np'] == 0:
            pop[i]["rank"] = 1
            front[0]["f"].append(i)
            # front['f'][0] = i

    fid = 0
    while front[fid]['f']:
        Q = []
        for p in front[fid]['f']:
            for q in ind[p]['sp']:
                ind[q]['np'] = ind[q]['np'] - 1
                if ind[q]['np'] == 0:
                    pop[q]["rank"] = fid + 1
                    Q.append(q)

        fid += 1
        front[fid]['f'] = Q
    front[fid].clear()

    # rank = [pop[m]["rank"] for m in range(len(pop))]
    # rank = np.vstack(rank)
    # max_rank = np.max(rank, 0).astype(int)
    # r = max_rank + 1
    # zero_ranked = 0
    # zero_ranked = np.where(rank == 0)[0]  # matlab find
    # zr = len(zero_ranked)
    # if zero_ranked.any():
    #     for i in range(1, zr):
    #         pop[zero_ranked[i]]["rank"] = r
    #     front[fid]["f"] = [zero_ranked]
        # print("buraya uğramaması lazım!")

    pop = punish_twins.punishtwins(pop, options)
    # pop = punish_twins.punish_twins_pui(pop, options)

    for i in range(N):
        pop[i]["np"] = ind[i]["np"]
        pop[i]["sp"] = ind[i]["sp"]
        ind[i]["np"] = 0
        ind[i]["sp"] = []

    # pop = pui_nbhd.pui_nbhd_eps(pop, options)  # CALC PUI
    # pop = pui_nbhd.pui_nbhd(pop, options)  # CALC PUI
    pop = pui_nbhd.pui_nbhd_v2(pop, options)  # CALC PUI
    # pop, options = nbhd.nbhd(pop, options)  # CALC PUI
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
            if n_viol[p] == 0 and n_viol[q] == 0:
                pdomq = False
                qdomp = False
                for i in range(num_obj):
                    if mean[p, i] < mean[q, i]:
                        pdomq = True
                    elif mean[p, i] > mean[q, i]:
                        qdomp = True
                if pdomq and (not qdomp):
                    dom_mat[p, q] = 1
                elif (not pdomq) and qdomp:
                    dom_mat[p, q] = -1
            elif n_viol[p] == 0 and n_viol[q] != 0:
                dom_mat[p, q] = 1
            elif n_viol[p] != 0 and n_viol[q] == 0:
                dom_mat[p, q] = -1
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

    for m in range(num_obj):
        first = mean[0, col_idx]
        last = mean[num_ind - 1, col_idx]
        # if m == 0:
        #     mean = np.sort(mean, axis=m)
        # # mean = np.sort(mean, axis=0)
        # if m == 1:
        #     mean = np.flip(mean)
        mean= mean[mean[:, m].argsort()]

        pop[first.astype(int)]["distance"] = float('inf')
        pop[last.astype(int)]["distance"] = float('inf')

        min_obj = mean[0, m]
        max_obj = mean[num_ind - 1, m]

        for i in range(1, num_ind - 1):
            idn = mean[i, col_idx]
            A = mean[i + 1, m]
            B = mean[i - 1, m]
            C = max_obj - min_obj

            pop[idn.astype(int)]["distance"] = (pop[idn.astype(int)]["distance"] + (mean[i + 1, m] - mean[i - 1, m])
                                               / (max_obj - min_obj))

    return pop

