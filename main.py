import random
import time
import pandas as pd
import numpy as np

# Import metaheuristic algorithms from algorithms folder
from algorithms.differential_evolution import differential_evolution as DE
from algorithms.genetic_algorithm import Genetic_Algorithm as GA
from algorithms.pso_modified import PSO


class Simulation:
    def __init__(self, num_tasks, cloud_nodes, fog_nodes):
        self.TaskDetails = []
        self.NodeDetails = []
        self.eTimeList = []
        self.costList = []
        self.minTotalcost = 0
        self.minmakespan = 0
        self.alphaVal = 0.5

        self.generate_data(num_tasks, cloud_nodes, fog_nodes)

        self.TotalNode = len(self.NodeDetails)
        self.NoOfTask = len(self.TaskDetails)

        print("\n--- Node Details ---")
        self.show_node_details()

        print("\n--- Task Details ---")
        self.show_task_details()

        print("\n--- Optimal Reference Values ---")
        self.calculate_min_makespan()
        self.calculate_min_total_cost()

        self.run_simulation_loop()

    def generate_data(self, num_tasks, num_cloud_nodes, num_fog_nodes):
        total_nodes = num_cloud_nodes + num_fog_nodes

        # Generate Cloud Node Specs
        for _ in range(num_cloud_nodes):
            node = [
                random.randint(3000, 5000),                 # CPU (MIPS)
                round(random.uniform(0.7, 1.0), 4),         # CPU cost
                round(random.uniform(0.02, 0.05), 5),       # Memory cost
                round(random.uniform(0.05, 0.1), 4)         # Bandwidth cost
            ]
            self.NodeDetails.append(node)

        # Generate Fog Node Specs
        for _ in range(num_fog_nodes):
            node = [
                random.randint(500, 1500),
                round(random.uniform(0.1, 0.4), 4),
                round(random.uniform(0.01, 0.03), 5),
                round(random.uniform(0.01, 0.02), 4)
            ]
            self.NodeDetails.append(node)

        # Generate Task Specs
        for _ in range(num_tasks):
            task = [
                random.randint(1, 100),   # Instructions
                random.randint(50, 200),  # Memory
                random.randint(10, 100),  # Input Size
                random.randint(10, 100)   # Output Size
            ]
            self.TaskDetails.append(task)

        # Execution Time Matrix
        for node in self.NodeDetails:
            row = [round((task[0] * 1e3) / node[0], 4) for task in self.TaskDetails]
            self.eTimeList.append(row)

        # Cost Matrix
        for i, node in enumerate(self.NodeDetails):
            row = []
            for task in self.TaskDetails:
                cpu_cost = node[1] * self.eTimeList[i][self.TaskDetails.index(task)]
                mem_cost = node[2] * task[1]
                bw_cost = node[3] * (task[2] + task[3])
                total = round(cpu_cost + mem_cost + bw_cost, 4)
                row.append(total)
            self.costList.append(row)

    def show_node_details(self):
        for idx, node in enumerate(self.NodeDetails, start=1):
            print(f"Node_{idx}: CPU={node[0]} MIPS | CPU Cost={node[1]} | Mem Cost={node[2]} | BW Cost={node[3]}")

    def show_task_details(self):
        for idx, task in enumerate(self.TaskDetails, start=1):
            print(f"Task_{idx}: Instr={task[0]} | Mem={task[1]} MB | Input={task[2]} MB | Output={task[3]} MB")

    def calculate_min_makespan(self):
        total_instructions = sum([task[0] for task in self.TaskDetails])
        total_cpu = sum([node[0] for node in self.NodeDetails])
        self.minmakespan = round((total_instructions * 1e3) / total_cpu, 4)
        print(f"Min Makespan: {self.minmakespan}")

    def calculate_min_total_cost(self):
        transposed = list(zip(*self.costList))
        self.minTotalcost = round(sum([min(col) for col in transposed]), 4)
        print(f"Min Total Cost: {self.minTotalcost}")

    def utility_function(self, makespan, total_cost):
        alpha = self.alphaVal
        return (alpha * self.minmakespan / makespan) + ((1 - alpha) * self.minTotalcost / total_cost)

    def total_cost(self, chrom):
        return round(sum([self.costList[node][i] for i, node in enumerate(chrom)]), 4)

    def makespan(self, chrom):
        span = []
        for node in range(self.TotalNode):
            indices = [i for i, n in enumerate(chrom) if n == node]
            total = sum([self.eTimeList[node][i] for i in indices])
            span.append(total)
        return max(span)

    def show_task_allocation(self, chrom):
        print("\n--- Task Allocation ---")
        for i, node in enumerate(chrom, start=1):
            print(f"Task_{i} → Node_{node + 1}")

    def show_grouped_allocation(self, chrom):
        print("\n--- Node-wise Task Grouping ---")
        for node in range(self.TotalNode):
            tasks = [f"Task_{i+1}" for i, n in enumerate(chrom) if n == node]
            if tasks:
                print(f"Node_{node+1}: {', '.join(tasks)}")

    def run_simulation_loop(self):
        while True:
            print("\nChoose an Algorithm:")
            print("1. Genetic Algorithm")
            print("2. Particle Swarm Optimization")
            print("3. Differential Evolution")
            print("4. Exit")

            try:
                choice = int(input("Enter choice (1–4): "))
            except ValueError:
                print("Invalid input.")
                continue

            if choice == 4:
                print("Exiting simulation.")
                break

            start = time.time()
            if choice == 1:
                print("\nRunning Genetic Algorithm...")
                best, best_fitness = GA(100, 500, self.utility_function, self.NoOfTask, self.TotalNode, self.makespan, self.total_cost)
            elif choice == 2:
                print("\nRunning Modified PSO...")
                best, best_fitness = PSO(100, 500, self.utility_function, self.NoOfTask, self.TotalNode, self.makespan, self.total_cost)
            elif choice == 3:
                print("\nRunning Differential Evolution...")
                best, best_fitness = DE(100, 500, self.utility_function, 0.5, (0, self.TotalNode - 1), self.NoOfTask, self.makespan, self.total_cost)
            else:
                print("Invalid choice.")
                continue

            elapsed = time.time() - start
            print(f"\nExecution Time: {elapsed:.2f} seconds")
            print(f"Best Chromosome: {list(best)}")
            print(f"Utility Score: {best_fitness:.4f}")
            print(f"Total Cost: {self.total_cost(best)}")
            print(f"Makespan: {self.makespan(best)}")

            self.show_task_allocation(best)
            self.show_grouped_allocation(best)


if __name__ == "__main__":
    print("\n=== Vehicular Fog Computing Resource Allocation ===")
    cloudN = int(input("Enter number of Cloud Nodes: "))
    fogN = int(input("Enter number of Fog Nodes: "))
    taskN = int(input("Enter number of Tasks: "))
    Simulation(taskN, cloudN, fogN)
