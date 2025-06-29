import random
import numpy as np

def differential_evolution(pop_size, generations, utility_fn, cr, bounds, num_tasks, make_span, total_cost):
    """
    Differential Evolution Algorithm for Task Allocation Optimization

    Args:
        pop_size (int): Population size
        generations (int): Number of iterations
        utility_fn (function): Fitness function combining time and cost
        cr (float): Crossover rate
        bounds (tuple): (min, max) bounds for node index values
        num_tasks (int): Number of tasks
        make_span (function): Function to calculate makespan of a chromosome
        total_cost (function): Function to calculate cost of a chromosome

    Returns:
        best (array): Best chromosome found
        best_fitness (float): Utility score of the best chromosome
    """

    # Step 1: Initialize population with random task-node allocations
    population = np.random.randint(bounds[0], bounds[1], (pop_size, num_tasks))

    # Step 2: Run optimization loop
    for _ in range(generations):
        for i in range(pop_size):
            # Select three distinct candidates
            a, b, c = random.sample(range(pop_size), 3)
            mutant = population[a] + 0.5 * (population[b] - population[c])
            trial = np.copy(population[i])

            # Apply crossover
            for j in range(num_tasks):
                if random.uniform(0, 1) < cr:
                    trial[j] = mutant[j]

            # Clip to valid range
            trial = np.clip(trial, bounds[0], bounds[1])

            # Evaluate fitness
            trial_fitness = utility_fn(make_span(trial), total_cost(trial))
            current_fitness = utility_fn(make_span(population[i]), total_cost(population[i]))

            # Selection: accept trial if better
            if trial_fitness > current_fitness:
                population[i] = trial

    # Step 3: Select best solution from final population
    fitness_scores = [utility_fn(make_span(ind), total_cost(ind)) for ind in population]
    best_idx = np.argmax(fitness_scores)
    best = population[best_idx]
    best_fitness = fitness_scores[best_idx]

    return best, best_fitness
