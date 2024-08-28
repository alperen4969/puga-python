import copy

import numpy as np
import crossover_operator
import crossover_pui
import evaluate
import extract_pop
import ndsort_pui
import mutation_operator
import debug
import nbhd
import nd_sort
import select_pui
import stochastic_sort
import population
import repair_operator
import selection_operator
import matplotlib.pyplot as plt
from plot_tool import plot_result
import some_nsga2
import benchmarks
import EPS_CONS_objfun


# TODO : plot

# https://visualgit.readthedocs.io/en/latest/pages/naming_convention.html
# https://peps.python.org/pep-0008/

def puga_run(imported_data, options):
    all_pops_results = []
    all_pops_emission = []
    all_pops_resultsX = []
    # ndsorta ekle
    # güncel test problemleri
    #
    pop_size = options["pop_size"]  # POP_SIZE
    num_generation = 1
    # all_populations = [0] * pop_size
    # new_pop = population.new_init(pop_size, imported_data, options)  # individual
    pop = population.init_eps(28, imported_data, options)  # individual
    # new_pop = population.pop_from_exist(pop_size, imported_data, options)  # individual   # TODO OK
    # new_pop = benchmarks.pop_benchmark(pop_size, num_var=30)   # TODO BENCHMARK
    # new_pop = benchmarks.evaluate_benchmarks(new_pop)  # TODO BENCHMARK
    pop = evaluate.evaluate(pop, imported_data, options)  # TODO OK
    pop, options = nbhd.nbhd(pop, options)
    # obj_values = debug.debug_obj_values(new_pop, imported_data)
    # obj_values = np.array([new_pop[m]["mean"] for m in range(len(new_pop))])
    # pop = stochastic_sort.stochastic_sort(new_pop, options)
    pop = ndsort_pui.ndsort_pui(pop, options)
    # pop, F = some_nsga2.non_dominated_sorting(new_pop)
    # pop = some_nsga2.calc_crowding_distance(pop, F)
    # obj_values = debug.debug_obj_values(new_pop, imported_data)

    pop_size = len(pop)
    rank_vec = np.vstack([individual["rank"] for individual in pop])
    rank_vec = np.sort(rank_vec)
    front_count = rank_vec[pop_size - 1]
    debug_front = np.where(rank_vec == 1)
    firstFrontCount = len(np.where(rank_vec == 1)[0])
    print(f'Iteration {num_generation}, Frontiers: {front_count}, First_Count: {firstFrontCount}')

    state = {
        "currentGen": num_generation,
    }

    # pop = [individual(num_var, num_obj, num_cons) for _ in range(pop_size)]

    # pf_costs = np.array([pop[i]["mean"] for i in F[0]])
    # xAxis = [0, ]
    # yAxis = [pf_costs[-1], ]
    # plt.ion()
    # graph = plt.plot(xAxis, yAxis)[0]
    # plt.grid(True)
    # plt.pause(1)

    # PUGA iterations
    while num_generation < options["max_gen"]:  # TODO: termination criterion may be add
        # print(f'Iteration {num_generation}')
        num_generation += 1
        state["currentGen"] = num_generation
        # new_pop = selection_operator.select_op(pop, options)  # WARNING bu neden kapalı ya?  # AÇ BUNU
        new_pop = select_pui.select_op(pop)  # WARNING bu neden kapalı ya?
        # new_pop = crossover_operator.crossover_op(new_pop, options, state)
        # new_pop = copy.deepcopy(pop)
        new_pop = crossover_pui.crossover_op_pui(new_pop, options, state)  # AÇ BUNU
        # obj_values = debug.debug_obj_values(new_pop, imported_data)
        # TODO, randum kısmı eklenmeli MATLAB'tan
        # new_pop = mutation_operator.mutation_operator(new_pop, options, state)
        # new_pop = mutation_operator.mutation_operator_pui(new_pop, options, state)
        new_pop = evaluate.evaluate(pop, imported_data, options)  # TODO new_pop olacak arguman, pui
        new_pop = copy.deepcopy(new_pop)
        combinepop = np.concatenate((pop, new_pop))
        # np.vstack([individual["viol_sum"] for individual in new_pop]) for PANEL
        combinepop, options = nbhd.nbhd(combinepop, options)
        combinepop = ndsort_pui.ndsort_pui(combinepop, options)
        # pop = extract_pop.extract_pop(combinepop, options)
        pop = extract_pop.extract_pop_pui(combinepop)
        debug_test = 0

        pop_size = len(pop)
        rank_vec = np.vstack([individual["rank"] for individual in pop])
        rank_vec = np.sort(rank_vec)
        front_count = rank_vec[pop_size - 1]
        first_front_count = len(np.where(rank_vec == 1)[0])
        print(f'Iteration {num_generation}, Frontiers: {front_count}, First_Count: {first_front_count}')

        # all_sols = np.array([pop[i]["mean"] for i in range(len(pop))])
        # plt.grid()
        # plt.xlabel('f1 - Cost')
        # plt.ylabel('f2 - Emission')
        # plt.scatter(all_sols[:, 0], all_sols[:, 1], c="red", s=4)
        # plt.show(block=True)

        # obj_values = debug.debug_obj_values(new_pop, imported_data)
        # new_pop = evaluate.evaluate(new_pop, imported_data, options)  # TODO OK
        # new_pop = benchmarks.evaluate_benchmarks(new_pop)  # TODO BENCHMARK
        # obj_values = debug.debug_obj_values(new_pop, imported_data)

        # result = debug.evaluate(new_pop, imported_data)
        # plot_result(result, options)

        # repair operator
        pops_br = []  # TODO: pops_br idk
        if num_generation > options['repgen']:  # and options['repair'] == 1:
            # combine_pop_br = pop + new_pop
            # pops_br[num_generation] = combine_pop_br
            # new_pop = repair_operator.repair_operator(new_pop, imported_data)
            # new_pop = evaluate.evaluate(new_pop, imported_data)
            pass

        # Combine the new population and old population
        # combinepop = np.concatenate((pop, new_pop))
        # Front rank and PUI sorting
        # combinepop = stochastic_sort.stochastic_sort(combinepop, options)
        # combinepop, F = some_nsga2.non_dominated_sorting(combinepop)
        # combinepop = some_nsga2.calc_crowding_distance(combinepop, F)
        # pop = combinepop
        # all_populations[num_generation] = combinepop

        # Extract the next population
        # pop, F = some_nsga2.sort_population(combinepop)
        # results = np.array([pop[i]["mean"] for i in F[0]])
        # all_pops_results.append(results)
        # pop, F = some_nsga2.truncate_population(pop, F)
        # pop = extract_pop.extract_pop(combinepop, options)
        # print(pop[0]["variables"])
        # all_pops_emission.append(emission)
        # print("gen   >    ", num_generation)
        # print(f'Iteration {num_generation}: Number of Pareto Members = {len(F[0])}')
        # print(F[0])

        # pf_costs = np.array([pop[i]["mean"] for i in F[0]])
        # xAxis.append(num_generation)
        # yAxis.append(pf_costs[1])
        # graph.remove()
        # graph = plt.plot(xAxis, yAxis, color='g', linewidth=2)[0]
        # plt.pause(0.05)

    # plot_result(all_pops_results, options)
    # result = debug.evaluate(pop, imported_data)
    # plot_result(result, options)

    # plt.ioff()
    # plt.close()
    # pop = new_pop
    # pf_costs = np.array([pop[i]["mean"] for i in F[0]])

    all_sols = np.array([pop[i]["mean"] for i in range(len(pop))])
    plt.grid()
    plt.xlabel('f1 - Cost')
    plt.ylabel('f2 - Emission')
    plt.scatter(all_sols[:, 0], all_sols[:, 1], c="red", s=4)
    plt.show(block=True)
    # plt.scatter(pf_costs[:, 0], pf_costs[:, 1], c="green", s=15, alpha=0.6)
    # existed_pop_initial = population.pop_from_exist(pop_size, imported_data, options)
    # existed_pop_initial = evaluate.evaluate(existed_pop_initial, imported_data, options)
    # all_initial_pop = np.array([existed_pop_initial[i]["mean"] for i in range(len(pop))])
    # plt.scatter(all_initial_pop[:, 0], all_initial_pop[:, 1], c="purple", s=15, alpha=0.3)
    # plt.show(block=True)
    # pop, F = some_nsga2.sort_population(pop)
    # print(pop)
    # print(pf_costs)
    return pop
