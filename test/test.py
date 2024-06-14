import json
import time
import os
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
import matplotlib.pyplot as plt
import math

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
NUM_TESTS = 2
DIMENSIONS = 10
NUM_AGENTS = 20
MAX_FITNESS_EVALS = 1000
AMOUNT_OF_BOXPLOTS = 5  # from 1 to MAX_FITNESS_EVALS//100
RESULTS_DIR = 'results'
PLOTS_DIR = 'plots'

# Initialize results and threads structures
results = [{"name": alg.__name__, "functions": [{"name": function["func"].__name__, "results": [None] * NUM_TESTS, "avg": []} for function in functions], "labels": []}
           for alg in algorithms]
threads = [{"name": alg.__name__, "functions": [{"name": function["func"].__name__, "threads": [None] * NUM_TESTS} for function in functions]}
           for alg in algorithms]


# Function to run an algorithm
def run_algorithm(algorithm, function, LB, UB, dimensions, num_agents, max_fitness_evals, results, alg_idx, function_idx, test_idx):
    result = algorithm.run(dimensions, function, LB, UB,
                           num_agents, max_fitness_evals)
    results[alg_idx]["labels"] = result[0]
    results[alg_idx]["functions"][function_idx]["results"][test_idx] = result[1]


def perform_calculations(run_id):
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
    for algorithm in threads:
        for function in algorithm["functions"]:
            for thread in function["threads"]:
                thread.join()

    # Calculate average results
    for algorithm in results:
        for function in algorithm["functions"]:
            function["avg"] = list(np.average(
                np.array(function["results"]), axis=0))

    # Save results to files
    for algorithm in results:
        file_path = os.path.join(
            RESULTS_DIR, f'{run_id}_{algorithm["name"]}.json')
        try:
            with open(file_path, 'w') as file:
                json.dump(algorithm, file, indent=4)
        except Exception as e:
            print(f"Error while saving results to file {file_path}: {e}")


def plot_results(run_id, labels, results, avg, alg_name, func_name, every_nth_box=math.ceil((MAX_FITNESS_EVALS//100)/AMOUNT_OF_BOXPLOTS)):
    fig, ax = plt.subplots()

    # for i, seqence in enumerate(res[0]["results"]):
    #    ax.plot(seqence[0], seqence[1], label=f"Test {i+1}")

    box_data_x = np.array(labels)
    box_data_y = np.array(results)
    box_avg_y = np.array(avg)

    ax.plot(box_data_x, box_avg_y, label="Average")
    ax.boxplot(list(box_data_y.T[::every_nth_box]), positions=list(box_data_x[::every_nth_box]), widths=[
               MAX_FITNESS_EVALS*0.03 for _ in range(math.ceil((MAX_FITNESS_EVALS/100)/every_nth_box))])

    ax.set_title(alg_name)
    ax.set_xlabel("Number of fitness evaluations")
    ax.set_ylabel("Fitness")
    # ax.legend()

    # plt.show()
    file_path = os.path.join(
        PLOTS_DIR, f'{run_id}_plot_{alg_name}_{func_name}.png')
    plt.savefig(file_path)


def plot_comparison(run_id, results, every_nth_box=math.ceil((MAX_FITNESS_EVALS//100)/AMOUNT_OF_BOXPLOTS)):
    # 3D array of results:
    #         func1 func2 func3 func4
    # alg1: [  [lbs],   [],   [],   [] ] = row_of_functions
    # alg2: [  [lbs],   [],   [],   [] ]
    # alg3: [  [lbs],   [],   [],   [] ]
    # alg4: [  [lbs],   [],   [],   [] ]
    # ... ...
    # alg8: [  [lbs],   [],   [],   [] ]
    #
    # lbs := labels

    for func_idx, function in enumerate(functions):
        # 2D array of results for every function {func1,... func4}
        #         func
        # alg1:   [labels]
        # alg2:   [labels]
        # alg3:   [labels]
        # alg4:   [labels]
        # ... ...
        # alg8:   [labels]
        #
        # lbs := labels
        func_name = function["func"].__name__
        avg_results = []
        fig, ax = plt.subplots()
        labels = np.array(results[0]["labels"])
        for algorithm in results:
            row = np.array(*list(map(lambda func: func["avg"], filter(
                lambda func: func["name"] == func_name, algorithm["functions"]))))
            avg_results.append(row)
            ax.plot(labels, row, label=algorithm["name"])

        avg_results = np.array(avg_results)

        ax.boxplot(list(avg_results.T[::every_nth_box]), positions=list(labels[::every_nth_box]), widths=[
                   MAX_FITNESS_EVALS*0.03 for _ in range(math.ceil((MAX_FITNESS_EVALS/100)/every_nth_box))])
        ax.set_title("Comparison of algorithms for func: "+func_name)
        ax.set_xlabel("Number of fitness evaluations")
        ax.set_ylabel("Fitness")
        ax.legend(fontsize="6", loc="upper right")
        file_path = os.path.join(
            PLOTS_DIR, f'{run_id}_plot_comparison_all_{func_name}.png')
        plt.savefig(file_path)

        fig, ax = plt.subplots()

        ax.plot(labels, np.average(avg_results, axis=0),
                label="Average of algorithms")
        ax.boxplot(list(avg_results.T[::every_nth_box]), positions=list(labels[::every_nth_box]), widths=[
                   MAX_FITNESS_EVALS*0.03 for _ in range(math.ceil((MAX_FITNESS_EVALS/100)/every_nth_box))])
        ax.set_title("Comparison of algorithms for func: "+func_name)
        ax.set_xlabel("Number of fitness evaluations")
        ax.set_ylabel("Fitness")
        ax.legend(fontsize="6", loc="upper right")
        file_path = os.path.join(
            PLOTS_DIR, f'{run_id}_plot_comparison_avg_{func_name}.png')
        plt.savefig(file_path)


if __name__ == "__main__":
    # Generate unique ID for this run
    run_id = str(time.time())

    perform_calculations(run_id)

    for algorithmn in results:
        for function in algorithmn["functions"]:
            plot_results(run_id, algorithmn["labels"], function["results"],
                         function["avg"], algorithmn["name"], function["name"])

    plot_comparison(run_id, results)
