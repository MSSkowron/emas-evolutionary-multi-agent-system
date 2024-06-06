from jmetal.algorithm.multiobjective.smpso import SMPSO
from jmetal.util.archive import CrowdingDistanceArchive
from jmetal.util.termination_criterion import StoppingByEvaluations
from jmetal.problem.singleobjective.unconstrained import Rastrigin
from jmetal.util.observer import Observer, VisualizerObserver
import matplotlib.pyplot as plt
import time
from jmetal.operator import PolynomialMutation

dimensions = 100
swarm_size = 100
maxNumberOfFitnessEvaluations = 2000
algorithm_name = "SMPSO"

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
    global swarm_size, maxNumberOfFitnessEvaluations

    algorithm = SMPSO(
        problem=problem,
        swarm_size=swarm_size,
        mutation=PolynomialMutation(probability=1.0 / problem.number_of_variables(), distribution_index=20),
        leaders=CrowdingDistanceArchive(100),
        termination_criterion=StoppingByEvaluations(max_evaluations=maxNumberOfFitnessEvaluations)
    )
    algorithm.observable.register(observer=PrintObjectivesObserver(100))
    algorithm.run()
    solutions = algorithm.get_result()

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