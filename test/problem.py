import numpy as np

from jmetal.core.problem import FloatProblem
from jmetal.core.solution import FloatSolution


class Sphere(FloatProblem):
    def __init__(self, lowerBound, upperBound, dimensions):
        super(Sphere, self).__init__()

        self.obj_directions = [self.MINIMIZE]
        self.obj_labels = ["f(x)"]

        self.lower_bound = [lowerBound for _ in range(dimensions)]
        self.upper_bound = [upperBound for _ in range(dimensions)]

        FloatSolution.lower_bound = self.lower_bound
        FloatSolution.upper_bound = self.upper_bound

    def number_of_objectives(self) -> int:
        return 1

    def number_of_constraints(self) -> int:
        return 0

    def number_of_variables(self) -> int:
        return len(self.lower_bound)

    def evaluate(self, solution: FloatSolution) -> FloatSolution:
        solution.objectives[0] = np.sum([xi ** 2 for xi in solution.variables])

        return solution

    def name(self) -> str:
        return "Sphere"


class Rastrigin(FloatProblem):
    def __init__(self, lowerBound, upperBound, dimensions):
        super(Rastrigin, self).__init__()

        self.obj_directions = [self.MINIMIZE]
        self.obj_labels = ["f(x)"]

        self.lower_bound = [lowerBound for _ in range(dimensions)]
        self.upper_bound = [upperBound for _ in range(dimensions)]

        FloatSolution.lower_bound = self.lower_bound
        FloatSolution.upper_bound = self.upper_bound

    def number_of_objectives(self) -> int:
        return 1

    def number_of_constraints(self) -> int:
        return 0

    def number_of_variables(self) -> int:
        return len(self.lower_bound)

    def evaluate(self, solution: FloatSolution) -> FloatSolution:
        a = 10.0

        solution.objectives[0] = a * len(solution.variables) + np.sum(
            [(xi ** 2 - a * np.cos(2 * np.pi * xi)) for xi in solution.variables])

        return solution

    def name(self) -> str:
        return "Rastrigin"


class Schwefel(FloatProblem):
    def __init__(self, lowerBound, upperBound, dimensions):
        super(Schwefel, self).__init__()

        self.obj_directions = [self.MINIMIZE]
        self.obj_labels = ["f(x)"]

        self.lower_bound = [lowerBound for _ in range(dimensions)]
        self.upper_bound = [upperBound for _ in range(dimensions)]

        FloatSolution.lower_bound = self.lower_bound
        FloatSolution.upper_bound = self.upper_bound

    def number_of_objectives(self) -> int:
        return 1

    def number_of_constraints(self) -> int:
        return 0

    def number_of_variables(self) -> int:
        return len(self.lower_bound)

    def evaluate(self, solution: FloatSolution) -> FloatSolution:
        solution.objectives[0] = 418.9829 * len(solution.variables) - np.sum([(x_i * np.sin(np.sqrt(np.abs(x_i))))
                                                                              for x_i in solution.variables])

        return solution

    def name(self) -> str:
        return "Schwefel"


class Rosenbrock(FloatProblem):
    def __init__(self, lowerBound, upperBound, dimensions):
        super(Rosenbrock, self).__init__()

        self.obj_directions = [self.MINIMIZE]
        self.obj_labels = ["f(x)"]

        self.lower_bound = [lowerBound for _ in range(dimensions)]
        self.upper_bound = [upperBound for _ in range(dimensions)]

        FloatSolution.lower_bound = self.lower_bound
        FloatSolution.upper_bound = self.upper_bound

    def number_of_objectives(self) -> int:
        return 1

    def number_of_constraints(self) -> int:
        return 0

    def number_of_variables(self) -> int:
        return len(self.lower_bound)

    def evaluate(self, solution: FloatSolution) -> FloatSolution:
        a = 10.0

        solution.objectives[0] = np.sum([100*np.square(solution.variables[i+1]-np.square(solution.variables[i])) + np.square(solution.variables[i]-1) for i in range(len(solution.variables)-1)])


        return solution

    def name(self) -> str:
        return "Rosenbrock"

