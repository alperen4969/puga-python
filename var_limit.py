import numpy as np


def var_limit(var, lb, ub):
    # var = np.array(var[0])
    var = np.array(var)
    if np.any(var < lb):
        change_pos = np.where(var[0, 176:] < lb[176:])  # [1]
        change_pos = np.array(change_pos)
        change_pos += 176
        change_pos = np.hstack(change_pos)
        var[0, change_pos] = lb[change_pos]
        #     change_pos = np.where(var < lb)[1]  # [1]
        #     change_pos = np.hstack(change_pos)
        #     var[0, change_pos] = lb[change_pos]

    if np.any(var > ub):
        change_pos = np.where(var[0, 176:] > ub[176:])  # [1]
        change_pos = np.array(change_pos)
        change_pos += 176
        change_pos = np.hstack(change_pos)
        var[0, change_pos] = ub[change_pos]
        # var[change_pos] = ub[change_pos]
    return var


def var_limit_pui(var, lb, ub):
    var = np.array(var)
    if np.any(var < lb):
        change_pos = np.where(var < lb)
        change_pos = np.array(change_pos)
        var[change_pos] = lb[change_pos]

    if np.any(var > ub):
        change_pos = np.where(var > ub)
        change_pos = np.array(change_pos)
        var[change_pos] = ub[change_pos]
    return var
