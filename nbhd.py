from calc_prcr import calc_prcr
import numpy as np
import math
import scipy.io as sio

# mean not a mean, it is obj value: cost or emission, 0 or 1


def nbhd_old(pop, options):  # pui
    num_obj = 2
    ro = options["ro"]
    nbhds = options["nbhds"]
    num_nbhds = len(nbhds)
    for n in range(num_nbhds):
        members = nbhds[n].members
        if members:
            num_members = len(members)  # number of members in the neighborhood
            probability_matrix = [0] * num_members
            for i in range(num_members - 1):
                for j in range(i + 1, num_members):
                    guy1 = pop[members[i]]
                    guy2 = pop[members[j]]

                    if guy2.twins == 0:
                        probability_guy_1 = 1  # probability that guy1 dominates guy2
                        probability_guy_2 = 1  # probability that guy2 dominates guy1
                        for m in range(num_obj):
                            probability_guy_1 *= calc_prcr(guy2, guy1, m, ro)
                            probability_guy_2 *= calc_prcr(guy1, guy2, m, ro)
                        probability_matrix[i][j] = probability_guy_1
                        probability_matrix[j][i] = probability_guy_2
                    else:
                        probability_matrix[i][j] = 0

                pop[members[i]].pui = sum(probability_matrix[i])
            pop[members[-1]].pui = sum(probability_matrix[-1])
            pui = [pop[m]["PUI"] for m in members]
            members_pui = list(zip(members, pui))
            members_pui.sort(key=lambda x: x[1])
            pref = members_pui[0][0]
            pop[pref].pref = 1

    return pop


def nbhd_v2(pop, opt):
    k = opt['K']
    d = np.zeros(2)
    num_obj = 2
    mean = np.vstack([indv["mean"] for indv in pop])
    max_mean = np.zeros(2)
    min_mean = np.zeros(2)

    for i in range(num_obj):
        max_mean[i] = np.max(mean)
        min_mean[i] = np.min(mean)
        d[i] = max_mean[i] - min_mean[i] / k[i]

    num_nbhds = np.prod(k)
    nbhds = [{'center': np.zeros(num_obj), 'members': []} for _ in range(num_nbhds)]
    # nbhds = np.array({'center': np.zeros(numObj), 'members': []} for _ in range(num_nbhds))
    map_idx = np.arange(1, num_nbhds + 1)
    map_sub = np.reshape(map_idx, k)

    subs = np.ones(2)
    for i, individual in enumerate(pop):
        for m in range(num_obj):
            if m == 1:
                subs[0] = math.floor((max_mean[m] - individual["mean"][m]) / d[m])
                if subs[0] == 0:
                    subs[0] = 1
            elif m == 0:
                subs[1] = math.floor((individual["mean"][m] - min_mean[m]) / d[m])
                if subs[1] == 0:
                    subs[1] = 1

        s1 = subs[0]  # just two of them is enough for nsga-ii
        s2 = subs[1]
        # print(subs)
        # https://www.mathworks.com/help/matlab/matlab_prog/comma-separated-lists.html

        nbhd_int = s1.astype(int)
        # nbhd_int = mapsub[s1[0], s2[0], s3[0], s4[0], s5[0], s6[0]]
        # nbhd_int = mapsub(subs)
        pop[i]["nbhd"] = nbhd_int
        nbhds[nbhd_int]['members'].append(i)

    opt['nbhds'] = nbhds
    return opt['nbhds']


def nbhd(pop, opt):
    K = [10, 10]
    numObj = 2
    # export_means = sio.loadmat("export_means.mat")["temp_mean"][0]

    mean = np.array([p['mean'] for p in pop])
    # mean = np.array([p[0] for p in export_means])
    maxmean = np.zeros(numObj)
    minmean = np.zeros(numObj)
    D = np.zeros(numObj)

    for i in range(numObj):
        maxmean[i] = np.max(mean[:, i])
        minmean[i] = np.min(mean[:, i])
        D[i] = (maxmean[i] - minmean[i]) / K[i]  # Width of each cell in ith dimension (objective function)

    # Create Cells
    num_nbhds = np.prod(K)
    nbhds = [{'center': np.zeros(numObj), 'members': []} for _ in range(num_nbhds)]
    mapidx = np.arange(1, num_nbhds + 1)
    mapidx = np.reshape(mapidx, (1, 100))
    mapsub = mapidx.reshape(K)

    subs = np.ones(numObj, dtype=int)

    for i, p in enumerate(pop):
        for m in range(numObj):
            if m == 1:
                subs[0] = int((maxmean[m] - p['mean'][m]) / D[m])
                if subs[0] == 0:
                    subs[0] = 1
            elif m == 0:
                subs[1] = int((p['mean'][m] - minmean[m]) / D[m])
                if subs[1] == 0:
                    subs[1] = 1
            # else:
            #     subs[m] = int((p['mean'][m] - minmean[m]) / D[m])
            #     if subs[m] == 0:
            #         subs[m] = 1

        # s1, s2, s3, s4, s5, s6 = subs
        s1 = subs[0]
        s2 = subs[1]
        # nbhd = mapsub[s1 - 1, s2 - 1, s3 - 1, s4 - 1, s5 - 1, s6 - 1]
        nbhd = mapsub[s1 - 1, s2 - 1]
        pop[i]['nbhd'] = nbhd
        nbhds[nbhd - 1]['members'].append(i)
        temp = 0  # for debugging

    opt['nbhds'] = nbhds
    return pop, opt
