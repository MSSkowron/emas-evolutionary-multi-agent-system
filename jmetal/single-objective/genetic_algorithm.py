from jmetal.algorithm.singleobjective.genetic_algorithm import GeneticAlgorithm
from jmetal.operator import PolynomialMutation, SBXCrossover
from jmetal.problem.singleobjective.unconstrained import Rastrigin
from jmetal.util.comparator import DominanceComparator
from jmetal.util.solution import print_function_values_to_file, print_variables_to_file, get_non_dominated_solutions, print_function_values_to_file
from jmetal.util.termination_criterion import StoppingByEvaluations
from jmetal.lab.visualization import Plot
from jmetal.util.observer import Observer, VisualizerObserver
import matplotlib.pyplot as plt
import time

dimensions = 100
numberOfAgents = 20
maxNumberOfFitnessEvaluations = 100000
algorithm_name = "GeneticAlgorithm"

data = [[],[]]

class PrintObjectivesObserver(Observer):
    def __init__(self, frequency: int = 1) -> None:
        """Show the number of evaluations, best fitness and computing time.

        :param frequency: Display frequency."""
        self.display_frequency = frequency

    def update(self, *args, **kwargs):
        global data
        evaluations = kwargs["EVALUATIONS"]
        solutions = kwargs["SOLUTIONS"]

        if (evaluations % self.display_frequency) == 0 and solutions:
            if type(solutions) == list:
                fitness = solutions[0].objectives
            else:
                fitness = solutions.objectives

            print("Evaluations: {}. fitness: {}".format(evaluations, fitness))
            data[0].append(evaluations)
            data[1].append(fitness)

def solve(problem):
    global maxNumberOfFitnessEvaluations

    algorithm = GeneticAlgorithm(
        problem=problem,
        population_size=numberOfAgents,
        offspring_population_size=1,
        mutation=PolynomialMutation(1.0 / problem.number_of_variables(), 20.0),
        crossover=SBXCrossover(0.9, 5.0),
        termination_criterion=StoppingByEvaluations(max_evaluations=maxNumberOfFitnessEvaluations),
    )

    algorithm.observable.register(observer=PrintObjectivesObserver(100))

    algorithm.run()

    print("Algorithm (continuous problem): " + algorithm.get_name())
    print("Problem: " + problem.name())
    print("Computing time: " + str(algorithm.total_computing_time))

def show_data():

    plt.plot(data[0], data[1])
    plt.xlabel('evaluations')
    plt.ylabel('fitness')
    plt.title("Best fitness: "+str(data[1][len(data[1])-1]))

    file_name = "results/"+algorithm_name+"_"+str(round(time.time()))
    plt.savefig(file_name+".png")

    plt.show()


if __name__ == "__main__":
    
    solve(Rastrigin(dimensions))
    show_data()
