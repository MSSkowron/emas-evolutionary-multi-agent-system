from jmetal.algorithm.singleobjective.genetic_algorithm import GeneticAlgorithm
from jmetal.operator import PolynomialMutation, SBXCrossover
from problem import Rastrigin, Sphere, Schwefel, Rosenbrock
from jmetal.util.comparator import DominanceComparator
from jmetal.util.solution import print_function_values_to_file, print_variables_to_file, get_non_dominated_solutions, print_function_values_to_file
from jmetal.util.termination_criterion import StoppingByEvaluations
from jmetal.lab.visualization import Plot
from jmetal.util.observer import Observer, VisualizerObserver
import matplotlib.pyplot as plt
import time

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



class PrintObjectivesObserver(Observer):
    def __init__(self, frequency: int = 1, data=[[], []]) -> None:
        """Show the number of evaluations, best fitness and computing time.

        :param frequency: Display frequency."""
        self.display_frequency = frequency
        self.data = data

    def update(self, *args, **kwargs):
        global ga_data
        evaluations = kwargs["EVALUATIONS"]
        solutions = kwargs["SOLUTIONS"]

        if (evaluations % self.display_frequency) == 0 and solutions:
            if type(solutions) == list:
                fitness = solutions[0].objectives
            else:
                fitness = solutions.objectives

            self.data[0].append(evaluations)
            self.data[1].append(fitness)


def solve(problem, numberOfAgents, maxNumberOfFitnessEvaluations):
    algorithm = GeneticAlgorithm(
        problem=problem,
        population_size=numberOfAgents,
        offspring_population_size=1,
        mutation=PolynomialMutation(1.0 / problem.number_of_variables(), 20.0),
        crossover=SBXCrossover(0.9, 5.0),
        termination_criterion=StoppingByEvaluations(
            max_evaluations=maxNumberOfFitnessEvaluations),
    )
    data = [[], []]
    algorithm.observable.register(observer=PrintObjectivesObserver(100, data))

    algorithm.run()

    return data


def run(dimensions, function, lowerBound, upperBound, numberOfAgents, maxNumberOfFitnessEvaluations):
    problem = None
    if function.__name__ == "rastrigin":
        problem = Rastrigin(lowerBound, upperBound, dimensions)
    elif function.__name__ == "sphere":
        problem = Sphere(lowerBound, upperBound, dimensions)
    elif function.__name__ == "schwefel":
        problem = Schwefel(lowerBound, upperBound, dimensions)
    elif function.__name__ == "rosenbrock":
        problem = Rosenbrock(lowerBound, upperBound, dimensions)
    else:
        raise ValueError("Function not supported")

    data = solve(problem, numberOfAgents,
                 maxNumberOfFitnessEvaluations)
    data[1] = [t[0] for t in data[1]]
    return data


if __name__ == "__main__":
    print(run(100, rastrigin, rastrigin_LB, rastrigin_UB, 20, 1000))
