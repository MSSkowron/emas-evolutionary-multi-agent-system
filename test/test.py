import json
import time
from threading import Thread

import numpy as np

import emas
import evolution_strategy
import gde3
import genetic_algorithm
import nsgaii_float
import omopso
import smpso
import spea2

from rastrigin import rastrigin
from rastrigin import LB as rastrigin_LB
from rastrigin import UB as rastrigin_UB

from sphere import sphere
from sphere import LB as sphere_LB
from sphere import UB as sphere_UB

from schwefel import schwefel
from schwefel import LB as schwefel_LB
from schwefel import UB as schwefel_UB

from schaffer import schaffer
from schaffer import LB as schaffer_LB
from schaffer import UB as schaffer_UB


algorithms = [
    emas,
    evolution_strategy,
    gde3,
    genetic_algorithm,
    nsgaii_float,
    omopso,
    smpso,
    spea2
]

functions = [
    {"func": rastrigin, "LB": rastrigin_LB, "UB": rastrigin_UB},
    {"func": sphere, "LB": sphere_LB, "UB": sphere_UB},
    {"func": schwefel, "LB": schwefel_LB, "UB": schwefel_UB},
    {"func": schaffer, "LB": schaffer_LB, "UB": schaffer_UB}
]

# Define constants
NUM_TESTS = 10
DIMENSIONS = 100
NUM_AGENTS = 20
MAX_FITNESS_EVALS = 5000
RESULTS_DIR = 'results/'

# Initialize results and threads structures
results = [{"name": alg.__name__, "functions": [{"name": function["func"].__name__, "results": [None] * NUM_TESTS} for function in functions]}
           for alg in algorithms]
threads = [{"name": alg.__name__, "functions": [{"name": function["func"].__name__, "threads": [None] * NUM_TESTS} for function in functions]}
           for alg in algorithms]


# Function to run an algorithm
def run_algorithm(algorithm, function, LB, UB, dimensions, num_agents, max_fitness_evals, results, alg_idx, function_idx, test_idx):
    result = algorithm.run(dimensions, function, LB, UB,
                           num_agents, max_fitness_evals)
    results[alg_idx]["functions"][function_idx]["results"][test_idx] = result


# Generate unique ID for this run
run_id = str(time.time())

# Start threads
for alg_idx, algorithm in enumerate(algorithms):
    for func_idx, function in enumerate(functions):
        for test_idx in range(NUM_TESTS):
            threads[alg_idx]["functions"][func_idx]["threads"][test_idx] = Thread(
                target=run_algorithm,
                args=(algorithm,
                      function["func"], function["LB"], function["UB"],
                      DIMENSIONS, NUM_AGENTS, MAX_FITNESS_EVALS,
                      results,
                      alg_idx, func_idx, test_idx)
            )
            threads[alg_idx]["functions"][func_idx]["threads"][test_idx].start()

# Join threads
for algo in threads:
    for func in algo["functions"]:
        for test_thread in func["threads"]:
            test_thread.join()

# Save results to files
for algo in results:
    try:
        file_path = f'{RESULTS_DIR}{run_id}_{algo["name"]}.json'
        with open(file_path, 'w+') as file:
            json.dump(algo, file, indent=4)
            file.write('\n')
    except Exception as e:
        print(f"Error while saving results to file {file_path}: {e}")
