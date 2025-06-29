# Task Allocation in Vehicular Fog Computing using Metaheuristic Algorithms

This project simulates a **Vehicular Fog Computing (VFC)** environment and evaluates how different metaheuristic algorithms perform in **task scheduling**. The objective is to minimize execution time and cost by optimally allocating tasks to fog and cloud nodes.

---

## 🚀 Project Overview

- Simulates a 3-layer VFC architecture: Vehicles → Fog Nodes → Cloud Nodes  
- Implements three metaheuristic algorithms:
  - **Genetic Algorithm (GA)**
  - **Modified Particle Swarm Optimization (PSO)**
  - **Differential Evolution (DE)**
- Compares algorithms based on task scheduling efficiency  
- Uses a synthetically generated dataset representing tasks and node resources

---

## 📂 Dataset

We created and published a custom dataset modeling realistic fog–cloud task scenarios.  
📎 [View on Kaggle](https://www.kaggle.com/datasets/sachin26240/vehicularfogcomputing)

---

## 🧠 Technologies Used

- Python (Jupyter Notebook)  
- Numpy, Pandas  
- Matplotlib  
- Custom GA, PSO, and DE implementations

---

## 📊 Key Features

- Simulates execution time and resource cost for multiple task loads  
- Fitness evaluation and optimization loop for each algorithm  
- Gantt chart visualization of task allocation  
- Performance comparison across models using plots

---

## 📁 Repository Structure

```
vehicular-fog-computing/
├── main.py                             # Console-based simulation driver
├── algorithms/                         # Folder for metaheuristic algorithms
│   ├── genetic_algorithm.py            # Genetic Algorithm
│   ├── pso_modified.py                 # Modified PSO
│   └── differential_evolution.py       # Differential Evolution
├── README.md
```
---

## 🏁 How to Run

Step 1: Install required packages

```bash
pip install numpy pandas
```

Step 2: Run the simulation script

```bash
python main.py
```

---

## 📌 Notes

This simulation is based on synthetic data and is intended for research/academic use.

Algorithms can be extended to support real-time vehicular mobility and QoS-aware modeling.



