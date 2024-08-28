import numpy as np


def reshape_variables(variables):
    varstr = {
        # "generation" : np.transpose(np.reshape(variables[0][:176], (11, 16))),
        # "investment": np.transpose(np.reshape(variables[0][176:], (11, 16)))
        "generation": np.reshape(variables[0][:176], (16, 11)),  # variables[0][:176]
        "investment": np.reshape(variables[0][176:], (16, 11))
    }
    #  [0]'ları sil init_pop için
    return varstr


def reshape_variables_for_exist_pop(variables):
    varstr = {
        "generation" : np.transpose(np.reshape(variables[0][:176], (11, 16))),
        "investment": np.transpose(np.reshape(variables[0][176:], (11, 16)))
    }
    #  [0]'ları sil init_pop için
    return varstr


def reshape_variables_v2(variables):
    varstr = {
        "generation" : np.transpose(np.reshape(variables[:176], (11, 16))),
        "investment": np.transpose(np.reshape(variables[176:], (11, 16)))
    }
    #  [0]'ları sil init_pop için
    return varstr
"""
function var = reshapevarstr(varstr)
varstr.generation=permute(varstr.generation, [2,1]);
var(1:176) = reshape(varstr.generation,1,176);
varstr.investment=permute(varstr.investment, [2,1]);
var(177:352) = reshape(varstr.investment, 1,176);

function varstr = reshapeinitials(vararray)
varstr.generation = reshape(vararray(1:176),11,16);
varstr.generation = permute(varstr.generation, [2,1]);
varstr.investment = reshape(vararray(177:end),11,16);
varstr.investment = permute(varstr.investment, [2,1]);
end
"""