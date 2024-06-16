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

from rosenbrock import rosenbrock
from rosenbrock import LB as rosenbrock_LB
from rosenbrock import UB as rosenbrock_UB

from concurrent.futures import ThreadPoolExecutor

import logging

logging.disable()

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
    {"func": rastrigin, "LB": rastrigin_LB, "UB": rastrigin_UB, "dimensions":100},
    {"func": sphere, "LB": sphere_LB, "UB": sphere_UB, "dimensions":100},
    {"func": schwefel, "LB": schwefel_LB, "UB": schwefel_UB, "dimensions":100},
    {"func": rosenbrock, "LB": rosenbrock_LB, "UB": rosenbrock_UB, "dimensions":100}
]

# Define constants
NUM_TESTS = 10
NUM_AGENTS = 20
MAX_FITNESS_EVALS = 1000
AMOUNT_OF_BOXPLOTS = 5  # from 1 to MAX_FITNESS_EVALS//100
RESULTS_DIR = 'results'
PLOTS_DIR = 'plots'

# Initialize results and threads structures
results = [
    {
        "name": alg.__name__,
        "functions": [
            {"name": function["func"].__name__,
                "results": [None] * NUM_TESTS, "avg": []}
            for function in functions
        ],
        "labels": []
    }
    for alg in algorithms
]
threads = [
    {
        "name": alg.__name__,
        "functions": [
            {"name": function["func"].__name__, "threads": [None] * NUM_TESTS}
            for function in functions
        ]
    }
    for alg in algorithms
]


# Function to run an algorithm
def run_algorithm(algorithm, function, LB, UB, dimensions, num_agents, max_fitness_evals, results, alg_idx, function_idx, test_idx):
    print(
        f"Starting {algorithm.__name__} on {function.__name__} test {test_idx+1}/{NUM_TESTS}")
    start_time = time.time()
    result = algorithm.run(dimensions, function, LB, UB,
                           num_agents, max_fitness_evals)
    end_time = time.time()
    results[alg_idx]["labels"] = result[0]
    results[alg_idx]["functions"][function_idx]["results"][test_idx] = result[1]
    print(f"Finished {algorithm.__name__} on {function.__name__} test {test_idx+1}/{NUM_TESTS} in {round(end_time-start_time, 2)} seconds")


def perform_calculations(run_id):
    num_algorithms = len(algorithms)
    num_functions = len(functions)
    max_workers = num_algorithms * num_functions * NUM_TESTS

    with ThreadPoolExecutor() as executor:
        future_to_test = {}
        for alg_idx, algorithm in enumerate(algorithms):
            for func_idx, function in enumerate(functions):
                for test_idx in range(NUM_TESTS):
                    future = executor.submit(run_algorithm, algorithm,
                                             function["func"], function["LB"], function["UB"],
                                             function["dimensions"], NUM_AGENTS, MAX_FITNESS_EVALS,
                                             results,
                                             alg_idx, func_idx, test_idx)
                    future_to_test[future] = (alg_idx, func_idx, test_idx)

        for future in future_to_test:
            future.result()

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

    box_data_x = np.array(labels)
    box_data_y = np.array(results)
    box_avg_y = np.array(avg)

    ax.plot(box_data_x, box_avg_y, label="Average")
    ax.boxplot(list(box_data_y.T[::every_nth_box]), positions=list(box_data_x[::every_nth_box]), widths=[
               MAX_FITNESS_EVALS*0.03 for _ in range(math.ceil((MAX_FITNESS_EVALS/100)/every_nth_box))])

    ax.set_title(f"{alg_name} for function {func_name}")
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
        labels = np.array(results[0]["labels"])
        fig, ax = plt.subplots()
        avg_results = []

        for algorithm in results:
            alg_results = np.array(
                [function["avg"] for function in algorithm["functions"] if function["name"] == func_name])
            avg_results.append(alg_results)
            ax.plot(labels, alg_results[0], label=algorithm["name"])

        avg_results = np.vstack(avg_results)

        ax.boxplot(list(avg_results.T[::every_nth_box]), positions=list(labels[::every_nth_box]), widths=[
            MAX_FITNESS_EVALS*0.03 for _ in range(len(labels[::every_nth_box]))
        ])
        ax.set_title(f"Comparison of algorithms for {func_name}")
        ax.set_xlabel("Number of fitness evaluations")
        ax.set_ylabel("Fitness")
        ax.legend(fontsize="6", loc="upper right")

        file_path = os.path.join(
            PLOTS_DIR, f'{run_id}_plot_comparison_all_{func_name}.png')
        plt.savefig(file_path)
        plt.close(fig)

        # Plot the average of all algorithms
        fig, ax = plt.subplots()
        ax.plot(labels, np.mean(avg_results, axis=0),
                label="Average of algorithms")
        ax.boxplot(list(avg_results.T[::every_nth_box]), positions=list(labels[::every_nth_box]), widths=[
            MAX_FITNESS_EVALS*0.03 for _ in range(len(labels[::every_nth_box]))
        ])
        ax.set_title(f"Comparison of algorithms for {func_name}")
        ax.set_xlabel("Number of fitness evaluations")
        ax.set_ylabel("Fitness")
        ax.legend(fontsize="6", loc="upper right")

        file_path = os.path.join(
            PLOTS_DIR, f'{run_id}_plot_comparison_avg_{func_name}.png')
        plt.savefig(file_path)
        plt.close(fig)

        for algorithm in results:
            fig, ax = plt.subplots()
            row = np.array(
                [function["avg"] for function in algorithm["functions"] if function["name"] == func_name])

            ax.plot(labels, row[0], label="Average of algor ithms")
            ax.boxplot(list(avg_results.T[::every_nth_box]), positions=list(labels[::every_nth_box]), widths=[
                       MAX_FITNESS_EVALS*0.03 for _ in range(math.ceil((MAX_FITNESS_EVALS/100)/every_nth_box))])
            ax.set_title("Comparison of algorithms for func: " +
                         func_name+" for algo: "+algorithm["name"])
            ax.set_xlabel("Number of fitness evaluations")
            ax.set_ylabel("Fitness")
            ax.legend(fontsize="6", loc="upper right")
            file_path = os.path.join(
                PLOTS_DIR, f'{run_id}_plot_comparison_avg_{func_name}_alg_{algorithm["name"]}.png')
            plt.savefig(file_path)
            plt.close(fig)


if __name__ == "__main__":
    run_id = str(time.time())

    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(PLOTS_DIR, exist_ok=True)

    perform_calculations(run_id)

    for algorithmn in results:
        for function in algorithmn["functions"]:
            plot_results(run_id, algorithmn["labels"], function["results"],
                         function["avg"], algorithmn["name"], function["name"])

    plot_comparison(run_id, results)
