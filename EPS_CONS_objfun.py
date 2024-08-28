import numpy as np
from scipy.stats import norm


def EPS_CONS_objfun(x, opt):
    nObj = 2
    nCons = 10
    coeff_mean_1 = opt['coeff_mean_1']
    coeff_mean_2 = opt['coeff_mean_2']
    less = np.array((31000, 15000, 22000, 10000))
    less = np.reshape(less, (4, 1))
    # greater = opt['greater']
    greater = np.array((38400,19200,6400))
    less_var = opt['less_var']
    greater_var = np.array((3840, 1920, 640))
    # alpha = opt['alpha']
    alpha = [1, 1]

    # y = {'mean': np.zeros(nObj), 'var': np.zeros(nObj)}
    y = {'mean': [0] * nObj,
         'variance': [0] * nObj}
    cons = np.zeros(nCons)

    x1 = np.sum(x, axis=1)
    # x1_v2 = np.sum(x, axis=0)
    costpoints = np.zeros_like(opt["C"])
    for i in range(len(x1)):
        costpoints[:, i] = x1[i] * opt["C"][:, i]
    test3 = np.sum(costpoints, axis=1)
    y["variance"][0] = test3

    empoints = np.zeros_like(opt['E'])
    for i in range(len(x1)):
        empoints[:, i] = x1[i] * opt['E'][:, i]
    y['variance'][1] = np.sum(empoints, axis=1)

    # variance = y['var']
    col_0 = y['variance'][0]
    col_1 = y['variance'][1]
    var_all = np.vstack((col_0, col_1))
    var_all = var_all.T  # np.reshape(var_all, (1000, 2))
    y['variance'] = var_all

    meann = np.sum(var_all, axis=0) / 1000
    y['mean'][0] = meann[0]
    y['mean'][1] = meann[1]
    # eliti al, pui düşükse daha iyi bir sonuç gözcü arı, abc için
    if alpha[0] == 1:
        for i in range(len(less)):
            test_cum = np.sum(x[i, :])
            if np.sum(x[i, :]) > less[i]:
                cons[i] = np.sum(x[i, :]) - less[i]
    else:
        for i in range(len(less)):
            if 1 - norm.cdf(np.sum(x[i, :]), loc=less[i], scale=less_var[i]) < alpha[0]:
                cons[i] = np.sum(x[i, :]) - less[i]

    l = len(less)
    if alpha[1] == 1:
        for i in range(len(greater)):
            test_sum1 = np.sum(x[:, i])
            if np.sum(x[:, i]) < greater[i]:
                cons[l + i] = abs(np.sum(x[:, i]) - greater[i])
    else:
        pass
        # for i in range(len(greater)):
        #     if norm.cdf(np.sum(x[:, i]), loc=greater[i], scale=greater_var[i]) < alpha[1]:
        #         cons[l + i] = abs(np.sum(x[:, i]) - greater[i])

    if x[3, 1] != 0:
        cons[7] = abs(x[3, 1])

    if x[1, 0] != 0:
        cons[8] = abs(x[1, 0])

    if x[2, 1] != 0:
        cons[9] = abs(x[2, 1])

    return y["mean"], y["variance"], cons
