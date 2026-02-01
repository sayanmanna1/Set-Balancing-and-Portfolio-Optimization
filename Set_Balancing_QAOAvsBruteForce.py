#!/usr/bin/env python
# coding: utf-8

# In[15]:


from scipy.sparse import csc_array as spcsc
from scipy import sparse
from scipy.sparse import kron as spkrn
from scipy.sparse.linalg import expm
from scipy.sparse.linalg import eigs
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes, mark_inset
import random
import numpy as np
import os
import sys
from scipy.optimize import minimize, Bounds
from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister
from qiskit.circuit import Parameter
from qiskit import QuantumCircuit
from qiskit.visualization import plot_histogram
from qiskit_aer import qasm_simulator
from qiskit_aer import AerSimulator
import networkx as nx
from tqdm import tqdm
from numpy import pi
from qiskit_optimization import QuadraticProgram
import matplotlib as mpl
from qiskit_optimization import QuadraticProgram
from qiskit_optimization.problems import QuadraticObjective
from qiskit_optimization import QuadraticProgram
from qiskit_optimization.translators import from_docplex_mp

from docplex.mp.model import Model
import numpy as np


from qiskit_algorithms.utils import algorithm_globals
from qiskit.circuit.library import TwoLocal
from qiskit.result import QuasiDistribution
from qiskit_aer.primitives import Sampler
from qiskit_algorithms import NumPyMinimumEigensolver, QAOA, SamplingVQE
from qiskit_algorithms.optimizers import COBYLA
from qiskit_optimization.algorithms import MinimumEigenOptimizer
import numpy as np
import matplotlib.pyplot as plt
import datetime

algorithm_globals.random_seed = 1234


# In[26]:


import numpy as np
from itertools import product
import time
import matplotlib.pyplot as plt

def brute_force_set_balancing(A):
    n = A.shape[1]
    min_val = float('inf')
    best_b_list = []

    for b_tuple in product([-1, 1], repeat=n):
        b = np.array(b_tuple)
        objective_val = np.linalg.norm(A @ b) ** 2

        if objective_val < min_val:
            min_val = objective_val
            best_b_list = [b.copy()]
        elif objective_val == min_val:
            best_b_list.append(b.copy())

    return min_val

n_values = list(range(10, 20))  # You can expand this if needed
avg_objective_brute_force = []
avg_objective_qaoa = []

# QAOA settings
cobyla = COBYLA()
cobyla.set_options(maxiter=500)
qaoa_mes = QAOA(sampler=Sampler(), optimizer=cobyla, reps=5)
qaoa = MinimumEigenOptimizer(qaoa_mes)

num_trials = 50

for n in n_values:
    total_brute = 0.0
    total_qaoa = 0.0

    for _ in tqdm(range(num_trials), desc=f"Running trials for n={n}"):
        A = np.random.randint(0, 2, size=(n, n))
        Q = A.T @ A

        # --- Brute Force ---
        brute_val = brute_force_set_balancing(A)
        total_brute += brute_val

        # --- QAOA ---
        mdl = Model(name="SetBalancing")
        b = {j: mdl.binary_var(name=f"b_{j}") for j in range(n)}
        b_vars = {j: 2 * b[j] - 1 for j in range(n)}
        quadratic_terms = mdl.sum(Q[i, j] * b_vars[i] * b_vars[j] for i in range(n) for j in range(n))
        mdl.minimize(quadratic_terms)
        mod = from_docplex_mp(mdl)

        result = qaoa.solve(mod)
        qaoa_val = result.fval
        total_qaoa += qaoa_val

    # Append averages
    avg_objective_brute_force.append(total_brute / num_trials)
    avg_objective_qaoa.append(total_qaoa / num_trials)

    print(f"\nn = {n}")
    print(f"Average Brute Force Objective: {total_brute / num_trials}")
    print(f"Average QAOA Objective:       {total_qaoa / num_trials}")
    print("="*40)

# Plot: Average Energy vs n
plt.figure(figsize=(10, 6))

# Brute Force
plt.plot(n_values, avg_objective_brute_force, marker='o', linestyle='-', color='blue', label='Brute Force')

# QAOA
plt.plot(n_values, avg_objective_qaoa, marker='s', linestyle='--', color='purple', label='QAOA')

# Aesthetics
plt.title("Average Minimum Energy vs n(=m) (50 trials per n)")
plt.xlabel("n (columns of A)")
plt.ylabel("Average Minimum Energy")
plt.grid(True)
plt.legend()
plt.xticks(n_values)
plt.tight_layout()

# Save and show
plt.savefig("'/scratch/21mt3fp30/SetB/Brute_vs_qaoa.png'")
#plt.show()


# In[ ]:




