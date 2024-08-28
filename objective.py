import numpy as np


# All objective function coefficients with uncertainty are assumed to be
# normal random variables with a mean and a standard deviation. Objective function
# evaluations (Pareto solutions) are treated as statistically independent random
# variables and are normally distributed.


def gep_objective_function(individual, import_data):  # evalueate each individual
    # gen_decision, invest_decision are NP_ARRAY
    # individual structure = decision variable structure
    gen_decision = np.array(individual["variables_str"]["generation"])
    invest_decision = np.array(individual["variables_str"]["investment"])

    # gen_decision = np.transpose(gen_decision)
    # invest_decision = np.transpose(invest_decision)

    cap = np.tile(import_data["cap"], (16, 1))
    # cap = np.tile(np.vstack(import_data["cap"]), (1, 16))
    investment_cap = np.multiply(invest_decision, cap)  # XMAX sheet
    # investment_cap = np.multiply(invest_decision, cap)
    # cap = np.tile(np.vstack(import_data["cap"]), (1, 16))
    # cap = np.tile(import_data["cap"], (16, 1))
    # investment_cap = invest_decision * cap  # XMAX sheet
    cumulative_invest = np.cumsum(investment_cap, axis=0)
    cumulative_planned = np.cumsum(import_data["planned"], axis=0)
    # cumulative_existing = np.matlib.repmat(import_data["existing"], 16, 1)
    cumulative_existing = np.tile(import_data["existing"], (16, 1))
    cumulative_capacity = cumulative_invest + cumulative_planned + cumulative_existing

    # cost calculation
    investment_cost = np.multiply((investment_cap + import_data["planned"]), import_data["invcost"])
    generation_cost = np.multiply(gen_decision, import_data["gencost"])
    operat_mainten_cost = np.multiply(import_data["omcost"], cumulative_capacity)
    total_cost = investment_cost + generation_cost + operat_mainten_cost
    total_cost = np.sum(total_cost[:])

    # cost-variance
    # std_inv_cost = np.repmat(import_data["invcoststd"], 16, 1);
    std_inv_cost = np.tile(import_data["invcoststd"], (16, 1))
    # std_operat_mainten_cost = np.repmat(import_data["invcoststd"], 16, 1);
    std_operat_mainten_cost = np.tile(import_data["omcoststd"], (16, 1))
    # std_gen_cost = np.repmat(import_data["invcoststd"], 16, 1);
    std_gen_cost = np.tile(import_data["gencoststd"], (16, 1))

    variance_investment_cost = (investment_cap * std_inv_cost) ** 2
    variance_operat_mainten_cost = (std_operat_mainten_cost * cumulative_capacity) ** 2
    variance_gen_cost = (gen_decision * std_gen_cost) ** 2

    # variance_investment_cost = np.multiply((np.multiply(investment_cap, std_inv_cost)))
    # variance_operat_mainten_cost = np.multiply((np.multiply(cumulative_capacity,
    #                                                         std_operat_mainten_cost)))
    # variance_gen_cost = np.multiply((np.multiply(gen_decision, std_gen_cost)))

    total_cost_variance = (variance_investment_cost +
                           variance_operat_mainten_cost + variance_gen_cost);
    total_cost_variance = np.sum(total_cost_variance[:])

    # emissions calculation
    em_rate_std = np.tile(import_data["emratestd"], (16, 1))
    emission = np.multiply(import_data["emrate"], gen_decision)
    emission = np.sum(emission[:])
    emission_variance = np.multiply(em_rate_std, gen_decision)
    emission_variance = np.sum(emission_variance[:])

    # structure y, import_data["nObj"]
    n_obj = 2
    y = {'mean': [0] * n_obj,
         'variance': [0] * n_obj}
    y['mean'][0] = total_cost
    y['mean'][1] = emission
    y['variance'][0] = total_cost_variance
    y['variance'][1] = emission_variance
    individual['mean'] = y['mean']
    individual['variance'] = y['variance']

    #  CONSTRAINTS
    # constraints = np.zeros((4, 1))  # num_cons
    constraints = [0, 0, 0, 0]

    # 1. peak_demand
    reserve = 0.1
    viol_peak_demand = (np.sum(import_data["capfactor"] * cumulative_capacity, axis=1)
                        < import_data["peak"] * (1 + reserve))
    sum_viol__peak_demand = np.sum(viol_peak_demand)

    if sum_viol__peak_demand != 0:
        peak_demand_cons = (import_data["peak"] * (1 + reserve) -
                            np.sum(import_data["capfactor"] * cumulative_capacity, axis=1)) * viol_peak_demand
        constraints[0] = np.sum(peak_demand_cons)

    # 2. demand
    viol_demand = np.sum(gen_decision, axis=1) < import_data["demand"]
    sum_viol_demand = np.sum(viol_demand)
    sum_viol_demand = round(sum_viol_demand, 2)
    if sum_viol_demand != 0:
        demand_cons = (import_data["demand"] - np.sum(gen_decision, axis=1)) * viol_demand
        constraints[1] = np.sum(demand_cons)

    # 3. generation limit
    # test = import_data["capfactor"] * cumulative_capacity * 8760
    viol_gen = gen_decision > (import_data["capfactor"] * cumulative_capacity * 8760)
    sum_viol_gen = np.sum(viol_gen)

    if sum_viol_gen != 0:
        gen_cons = (gen_decision - import_data["capfactor"] * cumulative_capacity * 8760) * viol_gen
        constraints[2] = np.sum(gen_cons)

    # 4. construction limit
    viol_construction = invest_decision > import_data['climit']
    sum_viol_construction = np.sum(viol_construction)

    if sum_viol_construction != 0:
        cons_construction = (invest_decision - import_data['climit']) * viol_construction
        constraints[3] = np.sum(cons_construction)

    constraints = np.round(constraints, 4)
    individual["cons"] = constraints

    return individual, constraints

# indv = {"position": points[:, i], "cost": [], "trial": 0, "is_dominated": False, "grid_index": [], "grid_subindex": []}