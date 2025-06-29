import pandas as pd
import numpy as np
import time
import copy

# ----------------------------
# Non-dominated Sorting
# ----------------------------
def non_dominated_sorting(pop_size, chroms_obj_record):
    s, n, rank, front = {}, {}, {}, {0: []}
    for p in range(pop_size * 2):
        s[p] = []
        n[p] = 0
        for q in range(pop_size * 2):
            if ((chroms_obj_record[p][0] > chroms_obj_record[q][0] and chroms_obj_record[p][1] < chroms_obj_record[q][1])
                or (chroms_obj_record[p][0] >= chroms_obj_record[q][0] and chroms_obj_record[p][1] < chroms_obj_record[q][1])
                or (chroms_obj_record[p][0] > chroms_obj_record[q][0] and chroms_obj_record[p][1] <= chroms_obj_record[q][1])):
                s[p].append(q)
            elif ((chroms_obj_record[p][0] < chroms_obj_record[q][0] and chroms_obj_record[p][1] > chroms_obj_record[q][1])
                  or (chroms_obj_record[p][0] <= chroms_obj_record[q][0] and chroms_obj_record[p][1] > chroms_obj_record[q][1])
                  or (chroms_obj_record[p][0] < chroms_obj_record[q][0] and chroms_obj_record[p][1] >= chroms_obj_record[q][1])):
                n[p] += 1
        if n[p] == 0:
            rank[p] = 0
            front[0].append(p)

    i = 0
    while front[i]:
        Q = []
        for p in front[i]:
            for q in s[p]:
                n[q] -= 1
                if n[q] == 0:
                    rank[q] = i + 1
                    Q.append(q)
        i += 1
        front[i] = Q

    del front[i]
    return front

# ----------------------------
# Crowding Distance
# ----------------------------
def calculate_crowding_distance(front, chroms_obj_record):
    distance = {m: 0 for m in front}
    for o in range(2):
        obj = {m: chroms_obj_record[m][o] for m in front}
        sorted_keys = sorted(obj, key=obj.get)
        distance[sorted_keys[0]] = distance[sorted_keys[-1]] = float('inf')
        for i in range(1, len(front) - 1):
            if len(set(obj.values())) == 1:
                continue
            distance[sorted_keys[i]] += (
                (obj[sorted_keys[i + 1]] - obj[sorted_keys[i - 1]]) /
                (obj[sorted_keys[-1]] - obj[sorted_keys[0]])
            )
    return distance

# ----------------------------
# Selection
# ----------------------------
def selection(pop_size, front, chroms_obj_record, total_chromosome):
    N, new_pop = 0, []
    for i in range(len(front)):
        N += len(front[i])
        if N > pop_size:
            distance = calculate_crowding_distance(front[i], chroms_obj_record)
            sorted_cdf = sorted(distance, key=distance.get, reverse=True)
            for j in sorted_cdf:
                if len(new_pop) == pop_size:
                    break
                new_pop.append(j)
            break
        else:
            new_pop.extend(front[i])

    population_list = [total_chromosome[n] for n in new_pop]
    return population_list, new_pop

# ----------------------------
# Genetic Algorithm Main Function
# ----------------------------
def Genetic_Algorithm(pop_size, generations, fitness_fn, num_tasks, num_nodes, make_span_fn, cost_fn):
    population = []
    for _ in range(pop_size):
        individual = list(np.random.permutation(num_tasks))
        population.append([gene % num_nodes for gene in individual])

    best_list, best_obj = [], []

    for _ in range(generations):
        # Crossover
        parents = copy.deepcopy(population)
        offspring = []
        shuffle = list(np.random.permutation(pop_size))
        for i in range(0, pop_size, 2):
            p1, p2 = parents[shuffle[i]], parents[shuffle[i+1]]
            c1, c2 = p1[:], p2[:]
            cut = sorted(np.random.choice(num_tasks, 2, replace=False))
            c1[cut[0]:cut[1]], c2[cut[0]:cut[1]] = p2[cut[0]:cut[1]], p1[cut[0]:cut[1]]
            offspring.extend([c1, c2])

        # Mutation
        mutation_rate = 0.01
        mutation_size = round(num_tasks * 0.4)
        for child in offspring:
            if np.random.rand() <= mutation_rate:
                pos = list(np.random.choice(num_tasks, mutation_size, replace=False))
                last = child[pos[0]]
                for i in range(mutation_size - 1):
                    child[pos[i]] = child[pos[i+1]]
                child[pos[-1]] = last

        # Fitness Calculation
        total_population = parents + offspring
        chrom_fitness = {
            i: [
                fitness_fn(make_span_fn(total_population[i]), cost_fn(total_population[i])),
                cost_fn(total_population[i])
            ]
            for i in range(pop_size * 2)
        }

        front = non_dominated_sorting(pop_size, chrom_fitness)
        population, selected = selection(pop_size, front, chrom_fitness, total_population)
        selected_fitness = [chrom_fitness[i] for i in selected]

        # Preserve best
        if not best_list:
            best_list, best_obj = population[:], selected_fitness[:]
        else:
            total = population + best_list
            total_fit = selected_fitness + best_obj
            new_front = non_dominated_sorting(pop_size, {i: fit for i, fit in enumerate(total_fit)})
            best_list, best_idx = selection(pop_size, new_front, {i: fit for i, fit in enumerate(total_fit)}, total)
            best_obj = [total_fit[i] for i in best_idx]

    # Return best individual and its fitness
    best_solution = best_list[0]
    best_score = fitness_fn(make_span_fn(best_solution), cost_fn(best_solution))
    return best_solution, best_score
