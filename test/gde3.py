from jmetal.algorithm.multiobjective.gde3 import GDE3
from problem import Rastrigin, Sphere, Schwefel, Schaffer
from jmetal.util.termination_criterion import StoppingByEvaluations
from jmetal.util.observer import Observer
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

from schaffer import schaffer
from schaffer import LB as schaffer_LB
from schaffer import UB as schaffer_UB


class PrintObjectivesObserver(Observer):
    def __init__(self, frequency: int = 1, data=[[], []]) -> None:
        """Show the number of evaluations, best fitness and computing time.

        :param frequency: Display frequency."""
        self.display_frequency = frequency
        self.data = data

    def update(self, *args, **kwargs):
        global gde3_data
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
    algorithm = GDE3(
        problem=problem,
        population_size=numberOfAgents,
        cr=0.5,
        f=0.5,
        termination_criterion=StoppingByEvaluations(
            maxNumberOfFitnessEvaluations)
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
    elif function.__name__ == "schaffer":
        problem = Schaffer(lowerBound, upperBound, dimensions)
    else:
        raise ValueError("Function not supported")

    data = solve(problem, numberOfAgents,
                 maxNumberOfFitnessEvaluations)
    data[1] = [t[0] for t in data[1]]
    return data


if __name__ == "__main__":
    print(run(100, rastrigin, rastrigin_LB, rastrigin_UB, 20, 1000))
