import numpy as np
from scipy.optimize import linprog
import reshape_variables


def repair_operator(population, import_data):
    pop_size = len(population)  # 28
    # generation = population[i]['varstr']['generation']
    # investment = population[i]['varstr']['investment']
    gen_decision = np.array(individual["variables_str"]["generation"])
    invest_decision = np.array(individual["variables_str"]["investment"])

    construction_limit = np.tile(import_data['climit'], (16, 1))
    cap = np.tile(import_data['cap'], (16, 1))

    for i in range(pop_size):


        # repair investment upper-bound
        violation = investment > construction_limit
        sum_violation = np.sum(violation)

        if sum_violation != 0:
            change = np.where(investment > construction_limit)
            for idx in change[0]:
                if np.random.rand() < 0.50:
                    investment[idx] = 0
                else:
                    investment[idx] = construction_limit[idx]

            investment_mw = investment * cap
            cuminvs = np.cumsum(investment_mw, axis=0)  # axis'i kontrol
            cumcapacity = cuminvs + np.cumsum(import_data["planned"], axis=0) + np.tile(import_data["existing"],
                                                                                        (16, 1))

            # repair peak-demand constraint
            violation = np.sum(import_data["capfactor"] * cumcapacity, axis=1) < import_data["peak"] * (1 + reserve)
            sum_violation = np.sum(violation)
            if sum_violation != 0:
                inf = np.transpose(np.nonzero(violation))
                constraint_peak_demand = (import_data["peak"] * (1 + reserve) -
                                          np.sum(import_data["capfactor"] * cumcapacity, axis=1)) * violation
                inf_num = inf.shape[0]

                for i in range(inf_num):
                    t = inf[i, 0]
                    u = 0
                    invcost = import_data["invcost"]
                    omcost = import_data["omcost"]
                    gencost = import_data["gencost"]
                    # MATLAB:
                    # order = [invcost(t,:)' omcost(t,:)' gencost(t,:)' (1:11)']
                    # order = sortrows(order, [1, 2, 3])
                    # order = order(:, 4)'
                    order = np.column_stack((invcost[t], omcost[t], gencost[t], np.arange(1, 12)))
                    order = order[np.lexsort((order[:, 2], order[:, 1], order[:, 0]))]
                    order = np.transpose(order[:, 3])

                    while constraint_peak_demand[t] > 0 and u < 11:
                        u += 1
                        j = order[u]
                        if investment[t, j] < construction_limit[t, j]:
                            increase = min(construction_limit[t, j] - investment[t, j],
                                           np.round(constraint_peak_demand[t] / cap[t, j]))  # unit
                            investment[t, j] += increase  # unit
                            constraint_peak_demand[t] -= cap[t, j] * increase * import_data["capfactor"][t, j]  # MW

            # gen_ub After Peak Repair
            investment_mw = investment * cap
            cuminvs = np.cumsum(investment_mw, axis=0)
            cumcapacity = cuminvs + np.cumsum(import_data["planned"], axis=0) + np.tile(import_data["existing"],
                                                                                        (16, 1))
            capacity = (import_data["capfactor"] * cumcapacity) * 8760
            gen_ub = capacity

            # repair gen_ub constraint violation
            if np.any(generation > gen_ub):
                change = generation > gen_ub
                generation[change] = gen_ub[change]

            # repair demand constraint violation
            violation = np.sum(generation, axis=1) < import_data["demand"]
            sum_violation = np.sum(violation)
            if sum_violation != 0:
                if np.random.rand() < 0.50:
                    f = import_data["emrate"].flatten()
                else:
                    f = import_data["gencost"].flatten()

                # - - - no idea - - - - # TODO
                # b = opt.demand;
                # b = -b;
                # ub = reshape(gen_ub, 1, 176);
                # lb = zeros(size(ub));
                # x0 = reshape(generation, 1, 176);
                # A = opt.A;
                # A = -A;
                # options = optimoptions('linprog', 'Display', 'none');
                # [gen, ~, exitflag] = linprog(f, A, b, [], [], lb, ub, x0, options);
                # if exitflag == 1
                #     gen = reshape(gen, 11, 16);
                #     x(i).varstr.generation = permute(gen, [2, 1]);
                # else
                #     fprintf('Cannot Repair... ');
                #     break;
                # end
                # - - - - - - - - - - - #

        population[i]['varstr']['investment'] = investment
        temp1 = np.array(population[i]['varstr']['generation']).flatten()
        temp2 = np.array(population[i]['varstr']['investment']).flatten()
        # test = np.hstack((temp1, temp2))
        # population[i]['var'] = reshape_variables.reshape_variables(population[i]['varstr'])
        population[i]['var'] = np.hstack((temp1, temp2))

        return population