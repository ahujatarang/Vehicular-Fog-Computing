import random
import numpy as np


def PSO(pop_size, generations, f, NoOfTask, TotalNode, makeSpan, totalCost):
    def List2Matrix(task):
        mat = [[0 for _ in range(NoOfTask)] for _ in range(TotalNode)]
        for Ti in range(NoOfTask):
            mat[task[Ti]][Ti] = 1
        return mat

    def Matrix2List(task):
        return [row.index(1) for row in zip(*task)]

    def velocityMatrix(rng):
        return [[round(random.uniform(-rng, rng), 8) for _ in range(NoOfTask)] for _ in range(TotalNode)]

    # PSO hyperparameters
    c1 = 1.5  # cognitive component
    c2 = 1.5  # social component
    w = 0.6   # inertia weight
    Vmax = 28

    # Initialize population
    population = []
    velocities = []
    position_matrices = []

    for _ in range(pop_size):
        individual = [i % TotalNode for i in np.random.permutation(NoOfTask)]
        population.append(individual)
        velocities.append(velocityMatrix(Vmax))
        position_matrices.append(List2Matrix(individual))

    # Initialize personal and global bests
    pbest = population.copy()
    gbest = random.choice(population)

    for _ in range(generations):
        for i in range(pop_size):
            current = Matrix2List(position_matrices[i])
            if f(makeSpan(current), totalCost(current)) > f(makeSpan(pbest[i]), totalCost(pbest[i])):
                pbest[i] = current

            if f(makeSpan(pbest[i]), totalCost(pbest[i])) > f(makeSpan(gbest), totalCost(gbest)):
                gbest = pbest[i]

        # Velocity and position update
        for i in range(pop_size):
            pbest_matrix = List2Matrix(pbest[i])
            gbest_matrix = List2Matrix(gbest)

            for Ni in range(TotalNode):
                for Ti in range(NoOfTask):
                    velocities[i][Ni][Ti] = (
                        w * velocities[i][Ni][Ti]
                        + c1 * 0.4 * (pbest_matrix[Ni][Ti] - position_matrices[i][Ni][Ti])
                        + c2 * 0.6 * (gbest_matrix[Ni][Ti] - position_matrices[i][Ni][Ti])
                    )

            # Update position using max velocity per task
            for Ti in range(NoOfTask):
                max_index = np.argmax([velocities[i][Ni][Ti] for Ni in range(TotalNode)])
                for Ni in range(TotalNode):
                    position_matrices[i][Ni][Ti] = 1 if Ni == max_index else 0

    return gbest, f(makeSpan(gbest), totalCost(gbest))
