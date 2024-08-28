import numpy as np
import scipy.io as sio


def nbhd(opt):
    pop = []
    for i in range(0, len(pop)):
        pop.append({
            'mean': np.zeros((2, 1)),
        })
    K = [10, 10]
    numObj = 2
    export_means = sio.loadmat("export_means.mat")["temp_mean"][0]

    # mean = np.array([p['mean'] for p in pop])
    mean = np.array([p[0] for p in export_means])
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
            else:
                subs[m] = int((p['mean'][m] - minmean[m]) / D[m])
                if subs[m] == 0:
                    subs[m] = 1

        s1, s2, s3, s4, s5, s6 = subs
        nbhd = mapsub[s1 - 1, s2 - 1, s3 - 1, s4 - 1, s5 - 1, s6 - 1]
        pop[i]['nbhd'] = nbhd
        nbhds[nbhd - 1]['members'].append(i)

    # opt['nbhds'] = nbhds
    return pop, opt


# pop = np.zeros(40)
opt = np.zeros(40)
nbhd(opt)
