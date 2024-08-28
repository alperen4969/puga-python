import numpy as np
from calc_prcr import calc_pr_cr_v2


def calc_crowding_distance(pop, opt):
    num_obj = 2
    num_ind = len(pop)
    # idx = np.vstack(np.arange(num_ind))
    idx = np.arange(num_ind)
    mean = np.array([ind["mean"] for ind in pop])
    mean = np.column_stack((mean, np.transpose(idx)))  # mean nedense 0. indexte
    pop = np.array(pop)

    for m in range(num_obj):
        # mean = np.sort(mean, m)
        # test = mean[:, 2].argsort()
        mean = mean[mean[:, 2].argsort()]
        col_idx = 2  # num_obj  # +1 vardı ama?, obj sayisina esit olmali

        some_index = mean[0, col_idx]
        pop[some_index.astype(int)]["distance"] = float('inf')
        pop[mean[num_ind-1, col_idx].astype(int)]["distance"] = float('inf')

        min_obj = mean[0, m]
        max_obj = mean[num_ind-1, m]

        for i in range(1, num_ind-1):
            idx = mean[i, col_idx]
            pop[idx.astype(int)]["distance"] = (pop[idx.astype(int)]["distance"] + (mean[i + 1, m] - mean[i - 1, m])
                                                / (max_obj - min_obj))

    return pop


def calc_domination_matrix(pop, opt):
    pop_size = len(pop)
    dom_mat = np.zeros((pop_size, pop_size))
    ro = opt["ro"]
    num_obj = 2

    for p in range(pop_size):
        for q in range(p + 1, pop_size):
            if pop[p]["n_viol"] == 0 and pop[q]["n_viol"] == 0:
                p_dom_q = False
                q_dom_p = False
                guy1 = pop[p]
                guy2 = pop[q]
                pr1 = 1
                pr2 = 1
                for m in range(num_obj):
                    pr1 = pr1 * calc_pr_cr_v2(guy2, guy1, m, ro)
                    pr2 = pr2 * calc_pr_cr_v2(guy1, guy2, m, ro)
                if np.all(guy1["mean"] < guy2["mean"]):
                    if pr1 > pr2:
                        p_dom_q = True
                        pop[q]["pui"] = pop[q]["pui"] + pr1
                        pop[p]["pref"] = pop[p]["pref"] + 1
                    elif pr2 > pr1:
                        pop[q]["pref"] = pop[q]["pref"] + 1
                elif np.all(guy2["mean"] < guy1["mean"]):
                    if pr2 > pr1:
                        q_dom_p = True
                        pop[p]["pui"] = pop[p]["pui"] + pr1
                        pop[q]["pref"] = pop[q]["pref"] + 1
                    elif pr1 > pr2:
                        pop[p]["pref"] = pop[p]["pref"] + 1
                if p_dom_q and not q_dom_p:
                    dom_mat[p, q] = 1
                elif not p_dom_q and q_dom_p:
                    dom_mat[p, q] = -1
            elif pop[p]["n_viol"] == 0 and pop[q]["n_viol"] != 0:
                dom_mat[p, q] = 1
                pop[q]["pui"] = pop[q]["pui"] + 1
            elif pop[p]["n_viol"] != 0 and pop[q]["n_viol"] == 0:
                dom_mat[p, q] = -1
                pop[p]["pui"] = pop[p]["pui"] + 1
            else:
                p_dom_q = False
                q_dom_p = False
                if pop[p]["viol_sum"] > pop[q]["viol_sum"]:
                    q_dom_p = True
                elif pop[p]["viol_sum"] < pop[q]["viol_sum"]:
                    p_dom_q = True

                if p_dom_q and not q_dom_p:
                    dom_mat[p, q] = 1
                elif not p_dom_q and q_dom_p:
                    dom_mat[p, q] = -1

    dom_mat = dom_mat - np.transpose(dom_mat)
    return pop, dom_mat


def stochastic_sort(pop, opt):
    len_pop = len(pop)
    for ind in pop:
        ind["pref"] = 0
        ind["rank"] = 0
        ind["twins"] = 0
        ind["np_set"] = []
        ind["sp"] = []

    pop, dom_mat = calc_domination_matrix(pop, opt)

    # compute np and sp of each indivudal
    ind = [{'np': 0, 'sp': []} for _ in range(len_pop)]
    for p in range(len_pop - 1):
        for q in range(p + 1, len_pop):
            if dom_mat[p, q] == 1:
                ind[q]['np'] += 1
                ind[p]['sp'].append(q)
                pop[q]["np_sets"].append(p)
                np.append(pop[q]["np_sets"], p)
            elif dom_mat[p, q] == -1:
                ind[p]['np'] += 1
                ind[q]['sp'].append(p)
                pop[p]["np_sets"].append(p)
                np.append(pop[q]["np_sets"], p)

    # front = {"f": np.empty((28, 0))}  # TODO, bu 28 problem çıkarabilir
    front = {"f": [[] for _ in range(28)]}  # TODO, bu 28 de dert olabilir

    for i in range(1, len_pop):
        if ind[i]["np"] == 0:
            pop[i]["rank"] = 1
            front["f"][0].append(i)
            # front["f"] = np.concatenate(front["f"], i, axis=1)

    fid = 0
    while front['f'][fid]:
        Q = []
        for p in front['f'][fid]:
            for q in ind[p]['sp']:
                ind[q]['np'] = ind[q]['np'] - 1
                if ind[q]['np'] == 0:
                    pop[i]["rank"] = fid + 1
                    Q.append(q)

        fid += 1
        front['f'][fid] = Q

    rank = [pop[m]["rank"] for m in range(len(pop))]
    rank = np.vstack(rank)
    max_rank = np.max(rank, 0).astype(int)
    r = max_rank + 1
    zero_ranked = 0
    zero_ranked = np.where(rank == 0)[0]
    zr = len(zero_ranked)
    if zero_ranked:
        for i in range(1, zr):
            pop[zero_ranked[i]].rank = r
        front["f"][fid] = [zero_ranked]
        print("buraya uğramaması lazım!")

    # punish twins
    rank = [pop[m]["rank"] for m in range(len(pop))]
    rank = np.vstack(rank)
    max_rank = np.max(rank, 0).astype(int)

    for i in range(0, len_pop - 1):
        for j in range(i + 1, len_pop):
            if (pop[i]["mean"][0] == pop[j]["mean"][0] and
                    pop[i]["mean"][1] == pop[j]["mean"][1]):
                pop[j]["rank"] = max_rank + 1
                pop[j]["twins"] = 1

    pop = calc_crowding_distance(pop, opt)
    return pop
