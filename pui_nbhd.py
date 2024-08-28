import calc_prcr
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra


def pui_nbhd(pop, opt):
    numObj = 2
    K = [10, 10]
    # ro = [0,0]
    ro = opt["ro"]
    nbhds = opt["nbhds"]
    num_nbhds = len(nbhds)

    for n in range(num_nbhds):
        members = nbhds[n]["members"]
        nm = len(members)
        PrMat = np.zeros((nm, nm))
        if len(members) > 0:
            for i in range(nm-1):
                for j in range(i+1, nm):
                    guy1 = pop[members[i]]
                    guy2 = pop[members[j]]
                    if guy2["twins"] == 0:
                        Pr1 = 1
                        Pr2 = 1
                        for m in range(numObj):
                            Pr1 = Pr1 * calc_prcr.calc_pr_discrete(guy2, guy1, m, ro)
                            # Pr1 = Pr1 * calc_prcr.calc_pr_cr_v2(guy2, guy1, m, ro)
                            Pr2 = Pr2 * calc_prcr.calc_pr_discrete(guy1, guy2, m, ro)
                            pass
                        PrMat[i, j] = Pr2
                        PrMat[j, i] = Pr1
                    else:
                        PrMat[i, j] = 0
                        PrMat[j, i] = float('inf')
            for i in members:
                pop[i]["PUI"] = np.sum(PrMat[i, :])

            # pop[27]["PUI"] = np.sum(PrMat[-1, :])  # 27 stands for last elemet
            return pop


def pui_nbhd_eps(opt, pop):
    num_obj = 2
    ro = opt.ro
    nbhds = opt.nbhds
    num_nbhds = len(nbhds)

    for n in range(num_nbhds):
        members = nbhds[n].members
        if members:
            nm = len(members)
            PrMat = np.zeros((nm, nm))

            for i in range(nm - 1):
                for j in range(i + 1, nm):
                    guy1 = pop[members[i]]
                    guy2 = pop[members[j]]

                    if guy2.twins == 0:
                        Pr1 = 1
                        Pr2 = 1

                        for m in range(num_obj):
                            Pr1 *= calc_prcr.calc_pr_discrete(guy2, guy1, m, ro)
                            Pr2 *= calc_prcr.calc_pr_discrete(guy1, guy2, m, ro)

                        PrMat[i, j] = Pr2
                        PrMat[j, i] = Pr1
                    else:
                        PrMat[i, j] = 0
                        PrMat[j, i] = np.inf

            for i in range(nm - 1):
                pop[members[i]].PUI = np.sum(PrMat[i, :])

            pop[members[-1]].PUI = np.sum(PrMat[-1, :])
            PUI = np.array([pop[m].PUI for m in members])
            mP = np.column_stack((members, PUI))
            mP = mP[mP[:, 1].argsort()]
            pref = mP[0]
            pop[pref].pref = 1

    return pop

# def calc_pr_discrete(guy1, guy2, m, ro):
#     return np.random.random()


def pui_nbhd_v2(pop, opt):
    num_obj = 2
    ro = [15, 15]
    nbhds = opt['nbhds']
    num_nbhds = len(nbhds)

    for n in range(num_nbhds):
        members = nbhds[n]['members']
        if members:
            nm = len(members)  # number of members in the neighborhood
            pr_mat = np.zeros((nm, nm))
            for i in range(nm - 1):
                for j in range(i + 1, nm):
                    guy1 = pop[members[i]]
                    guy2 = pop[members[j]]
                    if guy2['twins'] == 0:
                        pr1 = 1
                        pr2 = 1
                        for m in range(num_obj):
                            pr1 *= calc_prcr.calc_pr_discrete(guy2, guy1, m, ro)
                            pr2 *= calc_prcr.calc_pr_discrete(guy1, guy2, m, ro)
                        pr_mat[i, j] = pr2
                        pr_mat[j, i] = pr1
                    else:
                        pr_mat[i, j] = 0
                        pr_mat[j, i] = np.inf
                pop[members[i]]['PUI'] = np.sum(pr_mat[i, :])
            pop[members[-1]]['PUI'] = np.sum(pr_mat[-1, :])

            pui_values = [pop[member]['PUI'] for member in members]
            members_with_pui = list(zip(members, pui_values))
            members_with_pui.sort(key=lambda x: x[1])
            pref = members_with_pui[0][0]
            pop[pref]['pref'] = 1

    return pop
