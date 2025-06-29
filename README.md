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
vehicular-fog-computing-ml/
│
├── main.py # Optional entry point (can run all models)
├── data/
│ └── task_generator.py # Task and node setup
├── algorithms/
│ ├── genetic_algorithm.py # Genetic Algorithm logic
│ ├── pso_modified.py # Modified PSO implementation
│ └── differential_evolution.py # Differential Evolution
├── utils/
│ ├── fitness.py # Time and cost fitness functions
│ └── gantt.py # Gantt chart generation
├── plots/
│ └── comparisons.py # Model performance plots
├── CloudFogComputing.ipynb # Notebook version for demonstration
├── README.md
└── assets/ # Visualizations & screenshots
```
---

## 🏁 How to Run

### ▶️ Run the Notebook (for demonstration)
```bash
pip install numpy pandas matplotlib
jupyter notebook CloudFogComputing_v3.ipynb
```

### 🐍 Run Modular Scripts (once modules are created)
```bash
python algorithms/genetic_algorithm.py
```

Or:

```bash
python main.py
```

---

## 📌 Notes

This simulation is based on synthetic data and is intended for research/academic use.

Algorithms can be extended to support real-time vehicular mobility and QoS-aware modeling.



