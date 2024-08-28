import numpy as np
import pandas as pd
import scipy.io as sio


def import_excel():
    import_data = {'years': 16, 'units': 11, 'popsize': 6, 'maxGen': 100, 'numVar': 176 * 2, 'sizeVar': [],
                   'numObj': 0, 'numCons': 0, 'lb': [], 'ub': [], 'coeff_mean_1': [], 'coeff_mean_2': [],
                   'coeff_var_1': [], 'coeff_var_2': [], 'less': [], 'greater': [], 'less_var': [], 'greater_var': [],
                   'alpha': 0.9, 'ro': 1, 'vartype': [], 'nameObj': [], 'nameVar': [], 'nameCons': [],
                   'outputfile': 'populations.txt', 'outputInterval': 1, 'plotInterval': 5,
                   'crossover': [['intermediate', 1.2]], 'mutation': [['gaussian', 0.1, 0.5]],
                   'crossoverFraction': 'auto', 'mutationFraction': 'auto', 'useParallel': 'no', 'poolsize': 0,
                   'refPoints': [], 'refWeight': [], 'refUseNormDistance': 'front', 'refEpsilon': 0.001}

    # please use relative path
    file_path = 'TRGEP_postPUI_input.xlsx'
    import_data['capfactor'] = np.array(pd.read_excel(file_path, sheet_name='capfactor', usecols='B:L', nrows=17).values)
    import_data['cap'] = np.array(pd.read_excel(file_path, sheet_name='XMAX', usecols='B', header=None,
                                       nrows=11).values.flatten())
    import_data['climit'] = np.array(pd.read_excel(file_path, sheet_name='climit', usecols='B', header=None,
                                          nrows=11).values.flatten())
    import_data['demand'] = np.array(pd.read_excel(file_path, sheet_name='demand', usecols='B', header=None,
                                          nrows=16).values.flatten())
    import_data['emrate'] = np.array(pd.read_excel(file_path, sheet_name='emrate', usecols='B:L').values)
    import_data['existing'] = np.array(pd.read_excel(file_path, sheet_name='existing', usecols='B', header=None,
                                            nrows=11).values.flatten())
    import_data['gencost'] = np.array(pd.read_excel(file_path, sheet_name='gencost', usecols='B:L', nrows=16
                                    ).values)
    import_data['hours'] = np.array(pd.read_excel(file_path, sheet_name='hours', usecols='B', header=None,
                                         nrows=11).values.flatten())
    import_data['invcost'] = np.array(pd.read_excel(file_path, sheet_name='invcost', usecols='B:L',  nrows=16,
                                    ).values)
    import_data['peak'] = np.array(pd.read_excel(file_path, sheet_name='peak', usecols='B', header=None,
                                        nrows=16).values.flatten())
    import_data['planned'] = np.array(pd.read_excel(file_path, sheet_name='planned', usecols='B:L', nrows = 16).values)
    import_data['omcost'] = np.array(pd.read_excel(file_path, sheet_name='omcost', usecols='B:L', nrows = 17).values)
    import_data['invcoststd'] = np.array(pd.read_excel(file_path, sheet_name='invcoststd', header=None, usecols='B',
                                              nrows=11).values.flatten())
    import_data['omcoststd'] = np.array(pd.read_excel(file_path, sheet_name='omcoststd', usecols='B', header=None,
                                             nrows=11).values.flatten())
    import_data['gencoststd'] = np.array(pd.read_excel(file_path, sheet_name='gencoststd', usecols='B', header=None,
                                              nrows=11).values.flatten())
    import_data['emratestd'] = np.array(pd.read_excel(file_path, sheet_name='emratestd', usecols='B', header=None,
                                             nrows=11).values.flatten())

    # import_data['lb'] = np.zeros(import_data['numVar'])
    climit = np.tile(import_data['climit'], (16, 1)).flatten()
    # import_data['ub'][:175] = np.zeros(176)
    # import_data['ub'][176:351] = np.zeros(176)
    investmentidx = np.arange(176, 352)
    # options_ub_nump = np.array(import_data['ub'])
    # options_ub_nump[investmentidx] = climit
    import_data['ub'] = sio.loadmat("ub.mat")["temp"][0]
    import_data['lb'] = np.zeros(import_data['numVar'])
    import_data['investmentidx'] = investmentidx
    import_data['genidx'] = np.arange(1, 177)

    # building a matrix for repair
    u = import_data['units']
    y = import_data['years']
    A = np.zeros((y, import_data['numVar'] // 2))
    row = 0
    for i in range(y):
        A[i, row * u:(row + 1) * u] = 1
        row += 1
    import_data['A'] = A

    ###
    # dataframe = pd.DataFrame(data)
    # with pd.ExcelWriter("cost-opt.xlsx", mode="a", engine="openpyxl",if_sheet_exists='overlay') as writer:
    # with pd.ExcelWriter("data-test.xlsx", mode="w", engine="openpyxl") as writer:
    #     dataframe.to_excel(writer, index=False, header=False, sheet_name="Sheet1")
    # dataframe.to_excel("data-test.xlsx", index=False, header=False)

    return import_data
